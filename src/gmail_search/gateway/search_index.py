"""Trusted owner-index bindings and cancellation-safe generation lifetimes."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import math
from pathlib import Path
from numbers import Integral, Real
import re
import time
import weakref

import numpy as np


class IndexUnavailable(RuntimeError):
    def __init__(self):
        super().__init__('Owner search index is unavailable.')


class PendingIndex(IndexUnavailable):
    """Only an explicitly registered, genuinely unbuilt owner index."""


class IndexBusy(IndexUnavailable):
    """Bounded index capacity is occupied, including draining native work."""


def _text(value, limit=256):
    return type(value) is str and 0 < len(value) <= limit and '\x00' not in value and not any(0xD800<=ord(c)<=0xDFFF for c in value)


def _dimensions(value):
    return type(value) is int and 1 <= value <= 8192


@dataclass(frozen=True)
class IndexBinding:
    owner_id: str
    generation: str
    model: str
    dimensions: int
    source_id: str

    def __post_init__(self):
        if (not _text(self.owner_id,2048) or not _text(self.generation,128)
                or not _text(self.model) or not _dimensions(self.dimensions) or not _text(self.source_id,512)):
            raise ValueError('Invalid trusted index binding.')


@dataclass(frozen=True)
class LoadedIndex:
    """Trusted ownership transfer to the registry occurs only on publish success."""
    binding: IndexBinding
    searcher: object = field(repr=False,compare=False)
    resource_id: str

    def __post_init__(self):
        if (type(self.binding) is not IndexBinding or not _text(self.resource_id,1024)
                or not callable(getattr(self.searcher,'search',None)) or not callable(getattr(self.searcher,'close',None))):
            raise ValueError('Invalid trusted index resource.')


@dataclass(eq=False)
class _Generation:
    resource: LoadedIndex
    refs: int = 0
    searches: int = 0
    retired: bool = False
    close_task: asyncio.Task | None = None
    close_failed: bool = False


async def _drain(task):
    """Repeated cancellation never acknowledges teardown while work survives."""
    cancelled=False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled=True
        except BaseException:
            break
    if cancelled:
        # Retrieve any failure even when cancellation wins, avoiding task leaks.
        if not task.cancelled():task.exception()
        raise asyncio.CancelledError
    return task.result()


def _vector(value,dimensions):
    if not isinstance(value,(list,tuple,np.ndarray)):
        raise ValueError('Invalid query vector.')
    if isinstance(value,np.ndarray) and (value.ndim!=1 or value.dtype.kind not in 'fiu'):
        raise ValueError('Invalid query vector.')
    if len(value)!=dimensions:
        raise ValueError('Invalid query vector.')
    if isinstance(value,(list,tuple)) and any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in value):
        raise ValueError('Invalid query vector.')
    raw=np.asarray(value)
    if raw.ndim!=1 or raw.dtype.kind not in 'fiu' or raw.size!=dimensions:
        raise ValueError('Invalid query vector.')
    with np.errstate(over='ignore',invalid='ignore'):
        vector=np.array(raw,dtype=np.float32,copy=True)
    if not np.isfinite(vector).all() or not np.any(vector) or np.max(np.abs(vector))>1e6:
        raise ValueError('Invalid query vector.')
    return vector


def _result(value,top_k):
    if not isinstance(value,(tuple,list)) or len(value)!=2:
        raise IndexUnavailable()
    ids,scores=value
    if (not isinstance(ids,(tuple,list,np.ndarray)) or not isinstance(scores,(tuple,list,np.ndarray))
            or len(ids)!=len(scores) or len(ids)>top_k):
        raise IndexUnavailable()
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral) or not 0<int(v)<2**63 for v in ids):
        raise IndexUnavailable()
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) or not math.isfinite(float(v)) or abs(float(v))>1e6 for v in scores):
        raise IndexUnavailable()
    if len(set(map(int,ids)))!=len(ids):
        raise IndexUnavailable()
    return list(map(int,ids)),list(map(float,scores))


class _Lease:
    def __init__(self,registry,generation):
        self._registry=registry
        self._generation=generation
        self.binding=generation.resource.binding
        self._closed=False
        self._jobs=set()

    async def search(self,vector,*,top_k,absolute_deadline):
        registry=self._registry;registry._check_loop()
        if self._closed or registry._closed:raise IndexUnavailable()
        if type(top_k) is not int or not 1<=top_k<=10000:
            raise ValueError('Invalid candidate limit.')
        if isinstance(absolute_deadline,bool) or not isinstance(absolute_deadline,Real) or not math.isfinite(absolute_deadline):
            raise ValueError('Invalid operation deadline.')
        if absolute_deadline<=time.monotonic():raise TimeoutError()
        vector=_vector(vector,self.binding.dimensions)
        generation=self._generation
        if generation.searches or registry._searches>=registry._max_searches:raise IndexBusy()
        generation.searches+=1;registry._searches+=1
        caller=asyncio.current_task();self._jobs.add(caller)
        worker=asyncio.create_task(asyncio.to_thread(generation.resource.searcher.search,vector,top_k=top_k))
        try:
            try:
                async with asyncio.timeout_at(min(absolute_deadline,time.monotonic()+30)):
                    result=await asyncio.shield(worker)
            except BaseException:
                # Native ScaNN has no kill/cancel API. Hold the generation and
                # capacity until its thread exits, even past the deadline.
                try:await _drain(worker)
                except asyncio.CancelledError:raise
                except Exception:pass
                raise
            return _result(result,top_k)
        except (asyncio.CancelledError,TimeoutError,IndexUnavailable):
            raise
        except Exception:
            raise IndexUnavailable() from None
        finally:
            self._jobs.discard(caller)
            generation.searches-=1;registry._searches-=1
            registry._changed.set()
            await registry._close_if_ready(generation)

    async def _finish(self):
        self._closed=True
        async def finish():
            for task in tuple(self._jobs):task.cancel()
            for task in tuple(self._jobs):
                try:await _drain(task)
                except (asyncio.CancelledError,Exception):pass
            self._generation.refs-=1
            self._registry._leases-=1
            self._registry._changed.set()
            await self._registry._close_if_ready(self._generation)
        await _drain(asyncio.create_task(finish()))


class OwnerIndexRegistry:
    """One event-loop registry of explicitly bound, trusted immutable indexes.

    Native parsers are not sandboxed. Registered resources must come from trusted
    owner-specific provisioning. Candidate IDs always require owner SQL hydration.
    No loading occurs during acquire, and no disk/global-path fallback exists.
    """
    def __init__(self,*,max_generations=16,max_leases=8,max_searches=2,max_bindings=4096):
        for value in (max_generations,max_leases,max_searches,max_bindings):
            if type(value) is not int or value<1:raise ValueError('Invalid trusted index capacity.')
        self._max_generations=max_generations;self._max_leases=max_leases
        self._max_searches=max_searches;self._max_bindings=max_bindings
        self._active={};self._pending={};self._resident=set()
        self._bindings={};self._sources={};self._resources={}
        self._leases=0;self._searches=0;self._closed=False;self._loop=None
        self._changed=asyncio.Event()

    def _check_loop(self):
        loop=asyncio.get_running_loop()
        if self._loop is None:self._loop=loop
        if self._loop is not loop:raise IndexUnavailable()

    async def mark_pending(self,owner_id,model,dimensions):
        self._check_loop()
        IndexBinding(owner_id,'pending',model,dimensions,'pending')
        if self._closed or owner_id in self._active:raise IndexUnavailable()
        if owner_id not in self._pending and len(self._pending)>=self._max_bindings:raise IndexBusy()
        if owner_id in self._pending and self._pending[owner_id]!=(model,dimensions):raise IndexUnavailable()
        self._pending[owner_id]=(model,dimensions)

    async def publish(self,resource: LoadedIndex):
        self._check_loop()
        if self._closed or type(resource) is not LoadedIndex:raise IndexUnavailable()
        binding=resource.binding;key=(binding.owner_id,binding.generation)
        old=self._active.get(binding.owner_id)
        if old is not None and old.resource is resource:return
        prior=self._resources.get(id(resource.searcher))
        if prior is not None and prior() is resource.searcher:raise IndexUnavailable()
        if key in self._bindings or any(source in self._sources for source in (binding.source_id,resource.resource_id)):
            raise IndexUnavailable()
        if binding.owner_id in self._pending and self._pending[binding.owner_id]!=(binding.model,binding.dimensions):
            raise IndexUnavailable()
        if len(self._resident)>=self._max_generations or len(self._bindings)>=self._max_bindings:raise IndexBusy()
        # The weak reference prevents object reuse while avoiding retention of
        # closed multi-GB searchers. Source/generation tombstones remain bounded.
        try:ref=weakref.ref(resource.searcher)
        except TypeError:raise IndexUnavailable() from None
        self._resources[id(resource.searcher)]=ref
        self._bindings[key]=binding
        for source in (binding.source_id,resource.resource_id):self._sources[source]=binding
        current=_Generation(resource);self._resident.add(current)
        self._active[binding.owner_id]=current;self._pending.pop(binding.owner_id,None)
        if old is not None:
            old.retired=True
            self._schedule_close(old)

    @asynccontextmanager
    async def acquire(self,owner_id):
        self._check_loop()
        if self._closed or not _text(owner_id,2048):raise IndexUnavailable()
        generation=self._active.get(owner_id)
        if generation is None:
            if owner_id in self._pending:raise PendingIndex()
            raise IndexUnavailable()
        if self._leases>=self._max_leases:raise IndexBusy()
        generation.refs+=1;self._leases+=1
        lease=_Lease(self,generation)
        try:yield lease
        finally:await lease._finish()

    def _schedule_close(self,generation):
        if not generation.retired or generation.refs or generation.searches:return
        if generation not in self._resident:return
        async def close():
            try:await asyncio.to_thread(generation.resource.searcher.close)
            except Exception:
                generation.close_failed=True
                return
            generation.close_failed=False
            self._resident.discard(generation)
            self._changed.set()
        if generation.close_task is None:
            generation.close_task=asyncio.create_task(close())
        return generation.close_task

    async def _close_if_ready(self,generation):
        task=self._schedule_close(generation)
        if task is None:return
        await _drain(task)
        if generation.close_failed:raise IndexUnavailable()

    async def aclose(self):
        self._check_loop();self._closed=True;self._active.clear();self._pending.clear()
        for generation in self._resident:generation.retired=True
        async def finish():
            while self._resident:
                self._changed.clear()
                for generation in tuple(self._resident):
                    if generation.close_failed:generation.close_task=None
                    await self._close_if_ready(generation)
                if self._resident:await self._changed.wait()
        await _drain(asyncio.create_task(finish()))


@dataclass(frozen=True)
class IndexLoadLimits:
    """Trusted host resource policy, never derived from a run request."""
    max_total_bytes: int = 32 * 1024**3
    max_files: int = 4096
    max_ids: int = 2_000_000
    max_shards: int = 256
    max_json_bytes: int = 64 * 1024**2
    max_metadata_bytes: int = 4 * 1024**2
    max_rerank_pool: int = 10000

    def __post_init__(self):
        if any(type(value) is not int or value<1 for value in self.__dict__.values()):
            raise ValueError('Invalid trusted index load limit.')


def _generation_files(root,limits):
    import os,stat
    if not root.is_absolute() or root.resolve(strict=True)!=root or not root.is_dir():
        raise IndexUnavailable()
    result={};total=0;directories=0
    for directory,names,files in os.walk(root,followlinks=False):
        directories+=1
        if directories>limits.max_files:raise IndexUnavailable()
        for path in [Path(directory), *(Path(directory)/name for name in names+files)]:
            info=path.lstat()
            if info.st_uid!=os.geteuid() or info.st_mode&0o022 or stat.S_ISLNK(info.st_mode):raise IndexUnavailable()
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):raise IndexUnavailable()
            if stat.S_ISREG(info.st_mode):
                result[path]=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
                total+=info.st_size
                if len(result)>limits.max_files or total>limits.max_total_bytes:raise IndexUnavailable()
    return result


def _read(path,maximum):
    with path.open('rb') as stream:data=stream.read(maximum+1)
    if len(data)>maximum:raise IndexUnavailable()
    return data


def _json_file(path,maximum):
    import json
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise IndexUnavailable()
            out[key]=value
        return out
    def invalid(_):raise IndexUnavailable()
    return json.loads(_read(path,maximum),object_pairs_hook=pairs,parse_constant=invalid)


def _ids(path,limits):
    value=_json_file(path,limits.max_json_bytes)
    if (type(value) is not list or len(value)>limits.max_ids
            or any(type(v) is not int or not 0<v<2**63 for v in value) or len(set(value))!=len(value)):
        raise IndexUnavailable()
    return value


def _docids(path,ids,limits):
    import io,pickle,pickletools
    raw=_read(path,limits.max_json_bytes)
    allowed={'PROTO','FRAME','EMPTY_LIST','MARK','SHORT_BINUNICODE','BINUNICODE','BINUNICODE8','MEMOIZE','APPEND','APPENDS','STOP'}
    count=0
    for opcode,arg,_ in pickletools.genops(raw):
        count+=1
        if opcode.name not in allowed or count>limits.max_ids*4+16:raise IndexUnavailable()
        if opcode.name=='PROTO' and arg not in (4,5):raise IndexUnavailable()
        if opcode.name=='FRAME' and arg>len(raw):raise IndexUnavailable()
    class StringsOnly(pickle.Unpickler):
        def find_class(self,module,name):raise IndexUnavailable()
        def persistent_load(self,pid):raise IndexUnavailable()
    stream=io.BytesIO(raw);result=StringsOnly(stream).load()
    if (stream.tell()!=len(raw) or type(result) is not list or len(result)!=len(ids)
            or any(type(v) is not str or not re.fullmatch('[1-9][0-9]{0,18}',v) or int(v)>=2**63 for v in result)
            or set(result)!=set(map(str,ids))):
        raise IndexUnavailable()
    return result


def _finite_array(path,*,shape=None,raw=False,max_rows=2_000_000,block_rows=256):
    array=np.memmap(path,dtype=np.float32,mode='r',shape=shape) if raw else np.load(path,mmap_mode='r',allow_pickle=False,max_header_size=16384)
    try:
        if (array.dtype.kind not in 'fiu' or array.ndim not in (1,2) or len(array)>max_rows
                or array.ndim==2 and array.shape[1]>8192 or shape is not None and array.shape!=shape):
            raise IndexUnavailable()
        # np.load mmap checks file length before mapping; scan finite values in
        # bounded blocks rather than materializing a whole mailbox matrix.
        for start in range(0,len(array),block_rows):
            if not np.isfinite(array[start:start+block_rows]).all():raise IndexUnavailable()
        return array.shape
    finally:
        if getattr(array,'_mmap',None) is not None:array._mmap.close()


def _native_inputs(directory,ids,index_dim,is_docid,root,files,limits):
    from google.protobuf import text_format
    from scann.scann_ops import scann_assets_pb2
    from scann.proto import scann_pb2
    assets_text=_read(directory/'scann_assets.pbtxt',limits.max_metadata_bytes).decode('utf-8')
    assets=scann_assets_pb2.ScannAssets();text_format.Parse(assets_text,assets)
    config=scann_pb2.ScannConfig();config.ParseFromString(_read(directory/'scann_config.pb',limits.max_metadata_bytes))
    if config.distance_measure.distance_measure!='DotProductDistance':raise IndexUnavailable()
    def check_proto(message):
        for descriptor,value in message.ListFields():
            values=value if descriptor.is_repeated else (value,)
            if len(values)>limits.max_ids:raise IndexUnavailable()
            for item in values:
                if descriptor.type==descriptor.TYPE_MESSAGE:check_proto(item)
                elif descriptor.type==descriptor.TYPE_STRING:
                    if descriptor.name!='distance_measure' or item not in ('DotProductDistance','SquaredL2Distance'):raise IndexUnavailable()
                elif descriptor.type==descriptor.TYPE_BYTES:raise IndexUnavailable()
                elif isinstance(item,float) and not math.isfinite(item):
                    # The installed serializer uses NaN for this disabled
                    # fixed-point option; it is not vector/index data.
                    if not (math.isnan(item) and message.DESCRIPTOR.full_name=='research_scann.FixedPoint'
                            and descriptor.name=='noise_shaping_threshold' and message.enabled is False):
                        raise IndexUnavailable()
                elif isinstance(item,int) and not isinstance(item,bool):
                    ceiling=8192 if 'dim' in descriptor.name else 10000 if 'neighbor' in descriptor.name else max(limits.max_ids,8192)
                    if abs(item)>ceiling:raise IndexUnavailable()
    check_proto(config)
    unknown_free=scann_pb2.ScannConfig();unknown_free.CopyFrom(config);unknown_free.DiscardUnknownFields()
    if unknown_free.SerializeToString()!=config.SerializeToString():raise IndexUnavailable()
    if not 0<len(assets.assets)<=limits.max_files:raise IndexUnavailable()
    kinds={scann_assets_pb2.ScannAsset.AssetType.Name(asset.asset_type) for asset in assets.assets}
    if len(kinds)!=len(assets.assets):raise IndexUnavailable()
    if config.HasField('partitioning') and not {'PARTITIONER','TOKENIZATION_NPY'}<=kinds:raise IndexUnavailable()
    if config.HasField('hash') and not {'AH_CENTERS','AH_DATASET_NPY'}<=kinds:raise IndexUnavailable()
    if (config.HasField('brute_force') or config.HasField('exact_reordering')) and not kinds&{'DATASET_NPY','INT8_DATASET_NPY','BF16_DATASET_NPY'}:
        raise IndexUnavailable()
    used=set()
    from pathlib import Path
    for asset in assets.assets:
        path=Path(asset.asset_path)
        if not path.is_absolute():path=directory/path
        if path.resolve(strict=True)!=path or not path.is_relative_to(root) or path not in files or path in used:
            raise IndexUnavailable()
        used.add(path)
        kind=scann_assets_pb2.ScannAsset.AssetType.Name(asset.asset_type)
        if kind.endswith('_NPY'):
            shape=_finite_array(path,max_rows=limits.max_ids)
            if kind in ('DATASET_NPY','INT8_DATASET_NPY','AH_DATASET_NPY','AH_DATASET_SOAR_NPY','BF16_DATASET_NPY'):
                if len(shape)!=2 or shape[0]!=len(ids) or shape[1]>8192:raise IndexUnavailable()
                if kind in ('DATASET_NPY','INT8_DATASET_NPY','BF16_DATASET_NPY') and shape[1]!=index_dim:raise IndexUnavailable()
        elif kind not in ('AH_CENTERS','PARTITIONER') or files[path][2]>limits.max_json_bytes:
            raise IndexUnavailable()
    pickle_path=directory/'scann_docids.pkl'
    if pickle_path.exists()!=is_docid:raise IndexUnavailable()
    docids=_docids(pickle_path,ids,limits) if is_docid else None
    return assets_text,docids


def load_scann_index(binding: IndexBinding,index_dir,*,limits: IndexLoadLimits | None=None) -> LoadedIndex:
    """Load a complete trusted generation, without global lookup or fallback.

    This synchronous administrator operation must run off the event loop.
    Owner/model provenance is supplied by trusted provisioning; legacy files do
    not establish it. Files must stay immutable through all active leases. Native
    ScaNN parsing is not sandboxed; these checks are not a parser security boundary.
    """
    from pathlib import Path
    if type(binding) is not IndexBinding:raise ValueError('Trusted index binding required.')
    limits=limits or IndexLoadLimits();root=Path(index_dir)
    loaded=None
    try:
        files=_generation_files(root,limits)
        ids=_ids(root/'ids.json',limits)
        if not ids and set(files)-{root/'ids.json',root/'manifest.json'}:
            raise IndexUnavailable()
        manifest=_json_file(root/'manifest.json',limits.max_metadata_bytes) if root/'manifest.json' in files else None
        index_dim=binding.dimensions;entries=[]
        if manifest is not None:
            allowed={'format_version','num_shards','dimensions','shard_size','shards','watermark','reorder_pool','index_dim','manual_rerank'}
            if (type(manifest) is not dict or set(manifest)-allowed or manifest.get('format_version',1) not in (1,2)
                    or type(manifest.get('dimensions')) is not int or manifest.get('dimensions')!=binding.dimensions or type(manifest.get('num_shards')) is not int
                    or not 0<=manifest['num_shards']<=limits.max_shards):raise IndexUnavailable()
            index_dim=manifest.get('index_dim',binding.dimensions)
            if type(index_dim) is not int or not 1<=index_dim<=binding.dimensions:raise IndexUnavailable()
            shards=manifest.get('shards',[{'dir':f'shard_{i}'} for i in range(manifest['num_shards'])])
            if type(shards) is not list or len(shards)!=manifest['num_shards']:raise IndexUnavailable()
            combined=[];names=set()
            for entry in shards:
                if type(entry) is not dict or set(entry)-{'dir','key','count','first_id','last_id','sealed','docids'}:raise IndexUnavailable()
                name=entry.get('dir')
                if type(name) is not str or not re.fullmatch('shard_[0-9]+',name) or name in names:raise IndexUnavailable()
                names.add(name)
                shard_ids=_ids(root/name/'ids.json',limits)
                if not shard_ids or ('count' in entry and (type(entry['count']) is not int or entry['count']!=len(shard_ids))):raise IndexUnavailable()
                if (type(entry.get('docids',False)) is not bool
                        or 'first_id' in entry and entry['first_id']!=shard_ids[0]
                        or 'last_id' in entry and entry['last_id']!=shard_ids[-1]):raise IndexUnavailable()
                combined.extend(shard_ids)
                if len(combined)>limits.max_ids:raise IndexUnavailable()
                entries.append((root/name,shard_ids,entry.get('docids',False)))
            if combined!=ids:raise IndexUnavailable()
            if manifest.get('manual_rerank') is not None:
                cfg=manifest['manual_rerank']
                if (type(cfg) is not dict or set(cfg)-{'ah_dim','full_dim','reorder_pool','corpus_per_shard','corpus_full_path','ah_only'}
                        or any(type(cfg[key]) is not bool for key in ('corpus_per_shard','ah_only') if key in cfg)
                        or type(cfg.get('full_dim')) is not int or type(cfg.get('ah_dim')) is not int
                        or cfg.get('full_dim')!=binding.dimensions or cfg.get('ah_dim')!=index_dim
                        or type(cfg.get('reorder_pool')) is not int or not 1<=cfg['reorder_pool']<=limits.max_rerank_pool):raise IndexUnavailable()
                if cfg.get('corpus_per_shard') is True:
                    corpus=[(directory/'corpus_full.f32',len(shard_ids)) for directory,shard_ids,_ in entries]
                else:
                    filename=cfg.get('corpus_full_path','corpus_full.memmap')
                    if filename!='corpus_full.memmap':raise IndexUnavailable()
                    corpus=[(root/filename,len(ids))]
                for path,count in corpus:
                    if path not in files or files[path][2]!=count*binding.dimensions*4:raise IndexUnavailable()
                    _finite_array(path,shape=(count,binding.dimensions),raw=True,max_rows=limits.max_ids)
        elif ids:entries=[(root,ids,False)]
        native_inputs=[_native_inputs(directory,shard_ids,index_dim,docid,root,files,limits) for directory,shard_ids,docid in entries]
        from scann.scann_ops.py import scann_ops_pybind
        from gmail_search.index.searcher import ScannSearcher
        class StrictSearcher(ScannSearcher):
            def __init__(self,path,dimensions,validated_entries,validated_inputs,validated_manifest):
                self._validated_entries=validated_entries
                self._validated_inputs=validated_inputs
                self._validated_manifest=validated_manifest
                try:super().__init__(path,dimensions)
                except BaseException:
                    self.close()
                    raise
                finally:
                    self._validated_entries=self._validated_inputs=self._validated_manifest=None
            def _populate(self):
                for (directory,shard_ids,is_docid),(assets_text,docids) in zip(self._validated_entries,self._validated_inputs):
                    native=None
                    try:
                        native=scann_ops_pybind.ScannSearcher(scann_ops_pybind.scann_pybind.ScannNumpy(str(directory),assets_text),docids=docids)
                        self._shards.append((native,shard_ids));self._shard_docids.append(is_docid)
                        self._shard_keys.append(directory.name)
                        if native.size()!=len(shard_ids):raise IndexUnavailable()
                        native.set_num_threads(1)
                    finally:native=None
            def _load_legacy_single(self):self._populate()
            def _load_sharded(self,current,prev=None):
                if current!=self._validated_manifest:raise IndexUnavailable()
                self._populate()
            def close(self):
                super().close()
                self.embedding_ids=[];self._id_to_pos={};self._shard_ids_np=None
        loaded=StrictSearcher(root,binding.dimensions,entries,native_inputs,manifest)
        if (loaded.embedding_ids!=ids or len(loaded._shards)!=len(entries) or loaded._index_dim!=index_dim
                or manifest is not None and manifest.get('manual_rerank') is not None and loaded._manual_rerank!=manifest['manual_rerank']
                or _generation_files(root,limits)!=files):raise IndexUnavailable()
        info=root.stat()
        return LoadedIndex(binding,loaded,f'scann:{info.st_dev}:{info.st_ino}')
    except Exception:
        if loaded is not None:loaded.close()
        raise IndexUnavailable() from None
