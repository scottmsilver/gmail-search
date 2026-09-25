"""Owner index binding and native-work lifetime; no production indexes loaded."""
import asyncio
import threading
import time

import numpy as np
import pytest

from gmail_search.gateway import search_index as indexes


class FakeIndex:
    def __init__(self, result=None, gate=None):
        self.result=([1],[.75]) if result is None else result
        self.gate=gate
        self.started=threading.Event()
        self.closed=False
        self.close_thread=None
        self.calls=0
    def search(self,vector,top_k):
        self.calls+=1;self.started.set()
        if self.gate is not None:assert self.gate.wait(5)
        assert not self.closed
        return self.result
    def close(self):
        self.close_thread=threading.get_ident();self.closed=True


def binding(owner='alice',generation='g1',source=None):
    return indexes.IndexBinding(owner,generation,'synthetic-model',4,source or owner+'-'+generation)


def loaded(owner='alice',generation='g1',fake=None,source=None):
    return indexes.LoadedIndex(binding(owner,generation,source),fake or FakeIndex(),source or owner+'-'+generation)


def test_explicit_pending_and_unknown_owner_have_distinct_failures():
    async def run():
        registry=indexes.OwnerIndexRegistry()
        with pytest.raises(indexes.IndexUnavailable):
            async with registry.acquire('unknown'):pass
        await registry.mark_pending('alice','synthetic-model',4)
        with pytest.raises(indexes.PendingIndex):
            async with registry.acquire('alice'):pass
        await registry.publish(loaded())
        async with registry.acquire('alice') as lease:
            assert lease.binding==binding()
            assert await lease.search([1,0,0,0],top_k=3,absolute_deadline=time.monotonic()+1)==([1],[.75])
        await registry.aclose()
    asyncio.run(run())


def test_binding_generation_source_and_searcher_reuse_refused():
    async def run():
        registry=indexes.OwnerIndexRegistry()
        first=loaded();await registry.publish(first)
        for resource in (
            indexes.LoadedIndex(binding('bob'),first.searcher,'bob-g1'),
            loaded('bob',source=first.resource_id),
            loaded('alice',source='different-source'),
        ):
            with pytest.raises(indexes.IndexUnavailable):await registry.publish(resource)
        assert not first.searcher.closed
        await registry.aclose()
    asyncio.run(run())


def test_rotation_pins_old_generation_until_lease_drains():
    async def run():
        registry=indexes.OwnerIndexRegistry()
        old=loaded();new=loaded(generation='g2');await registry.publish(old)
        async with registry.acquire('alice') as lease:
            await registry.publish(new)
            assert not old.searcher.closed and lease.binding.generation=='g1'
            async with registry.acquire('alice') as current:assert current.binding.generation=='g2'
        assert old.searcher.closed and old.searcher.close_thread!=threading.get_ident()
        assert not new.searcher.closed
        await registry.aclose();assert new.searcher.closed
    asyncio.run(run())


@pytest.mark.parametrize('vector',[[1,2],[float('nan'),0,0,0],[float('inf'),0,0,0],[[1,0,0,0]],[0,0,0,0],[True,0,0,0]])
def test_invalid_vector_never_reaches_native(vector):
    async def run():
        registry=indexes.OwnerIndexRegistry();resource=loaded();await registry.publish(resource)
        async with registry.acquire('alice') as lease:
            with pytest.raises(ValueError):await lease.search(vector,top_k=2,absolute_deadline=time.monotonic()+1)
        assert resource.searcher.calls==0
        await registry.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('result',[([1,2],[.1]),([1,1],[.1,.2]),([True],[.1]),([-1],[.1]),([1],[float('nan')]),([1,2],[.1,.2])])
def test_bad_or_oversized_native_result_is_unavailable(result):
    async def run():
        registry=indexes.OwnerIndexRegistry();await registry.publish(loaded(fake=FakeIndex(result)))
        async with registry.acquire('alice') as lease:
            with pytest.raises(indexes.IndexUnavailable):await lease.search([1,0,0,0],top_k=1,absolute_deadline=time.monotonic()+1)
        await registry.aclose()
    asyncio.run(run())


def test_repeated_cancellation_drain_precedes_release_and_close():
    async def run():
        gate=threading.Event();fake=FakeIndex(gate=gate)
        registry=indexes.OwnerIndexRegistry();await registry.publish(loaded(fake=fake))
        async def query():
            async with registry.acquire('alice') as lease:
                await lease.search([1,0,0,0],top_k=1,absolute_deadline=time.monotonic()+2)
        task=asyncio.create_task(query())
        assert await asyncio.to_thread(fake.started.wait,1)
        await registry.publish(loaded(generation='g2'))
        task.cancel();await asyncio.sleep(.01);task.cancel();await asyncio.sleep(.01)
        assert not task.done() and not fake.closed
        gate.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert fake.closed
        await registry.aclose()
    asyncio.run(run())


def test_deadline_and_background_query_exit_drain_native():
    async def run():
        gate=threading.Event();fake=FakeIndex(gate=gate)
        registry=indexes.OwnerIndexRegistry();await registry.publish(loaded(fake=fake))
        async def query():
            async with registry.acquire('alice') as lease:
                await lease.search([1,0,0,0],top_k=1,absolute_deadline=time.monotonic()+.03)
        task=asyncio.create_task(query());assert await asyncio.to_thread(fake.started.wait,1)
        await asyncio.sleep(.06);assert not task.done()
        gate.set()
        with pytest.raises(TimeoutError):await task
        await registry.aclose();assert fake.closed
    asyncio.run(run())


def test_retained_generation_and_active_lease_limits():
    async def run():
        registry=indexes.OwnerIndexRegistry(max_generations=2,max_leases=1)
        await registry.publish(loaded())
        async with registry.acquire('alice'):
            await registry.publish(loaded(generation='g2'))
            with pytest.raises(indexes.IndexBusy):await registry.publish(loaded(generation='g3'))
            with pytest.raises(indexes.IndexBusy):
                async with registry.acquire('alice'):pass
        await registry.aclose()
    asyncio.run(run())


def build_native_index(path,ids,dimensions=4,docids=False):
    scann=pytest.importorskip('scann')
    import json
    path.mkdir()
    vectors=np.eye(dimensions,dtype=np.float32)[:len(ids)]
    native=scann.scann_ops_pybind.builder(vectors,2,'dot_product').set_n_training_threads(1).score_brute_force().build(docids=list(map(str,ids)) if docids else None)
    native.serialize(str(path))
    (path/'ids.json').write_text(json.dumps(ids))
    return vectors


@pytest.mark.parametrize('sharded',[False,True])
def test_strict_loader_searches_real_serialized_format(tmp_path,sharded):
    import json
    path=tmp_path/'generation'
    if sharded:
        path.mkdir()
        build_native_index(path/'shard_0',[11,12],docids=True)
        (path/'ids.json').write_text('[11,12]')
        (path/'manifest.json').write_text(json.dumps({'format_version':2,'num_shards':1,'dimensions':4,'shards':[{'dir':'shard_0','key':'11_12_2','count':2,'first_id':11,'last_id':12,'sealed':False,'docids':True}]}))
    else:build_native_index(path,[11,12])
    seal_generation(path)
    resource=indexes.load_scann_index(binding(),path)
    async def run():
        registry=indexes.OwnerIndexRegistry();await registry.publish(resource)
        async with registry.acquire('alice') as lease:
            ids,scores=await lease.search([1,0,0,0],top_k=1,absolute_deadline=time.monotonic()+1)
            assert ids==[11] and scores[0]==pytest.approx(1.0)
        await registry.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('fault',['missing_shard','foreign_asset','symlink','duplicate_ids','dimension','pickle_global','file_bytes','nan_dataset','manual_corpus'])
def test_loader_rejects_partial_foreign_or_oversized_assets(tmp_path,fault):
    import json,pickle
    path=tmp_path/'generation';build_native_index(path,[11,12])
    if fault=='missing_shard':
        (path/'manifest.json').write_text(json.dumps({'num_shards':1,'dimensions':4,'shards':[{'dir':'missing','count':2}]}))
    elif fault=='foreign_asset':
        text=(path/'scann_assets.pbtxt').read_text()
        (path/'scann_assets.pbtxt').write_text(text.replace(str(path),str(tmp_path/'foreign')))
    elif fault=='symlink':
        source=path/'ids.json';source.rename(tmp_path/'ids.json');source.symlink_to(tmp_path/'ids.json')
    elif fault=='duplicate_ids':(path/'ids.json').write_text('[11,11]')
    elif fault=='dimension':
        (path/'manifest.json').write_text(json.dumps({'num_shards':1,'dimensions':8,'shards':[{'dir':'shard_0','count':2}]}))
    elif fault=='pickle_global':(path/'scann_docids.pkl').write_bytes(pickle.dumps(eval))
    elif fault=='file_bytes':
        seal_generation(path)
        with pytest.raises(indexes.IndexUnavailable):indexes.load_scann_index(binding(),path,limits=indexes.IndexLoadLimits(max_total_bytes=1))
        return
    elif fault=='nan_dataset':
        dataset=np.load(path/'dataset.npy');dataset[0,0]=np.nan;np.save(path/'dataset.npy',dataset)
    else:
        (path/'manifest.json').write_text(json.dumps({'num_shards':1,'dimensions':4,'shards':[{'dir':'shard_0','count':2}],'manual_rerank':{'full_dim':4,'ah_dim':2,'reorder_pool':2,'corpus_per_shard':True}}))
    seal_generation(path)
    with pytest.raises(indexes.IndexUnavailable):indexes.load_scann_index(binding(),path)


def seal_generation(path):
    path.chmod(0o700)
    for item in path.rglob('*'):
        if not item.is_symlink():item.chmod(0o700 if item.is_dir() else 0o600)


def test_nested_vectors_are_rejected_before_numpy_conversion(monkeypatch):
    def forbidden(_):raise AssertionError('conversion must not run')
    monkeypatch.setattr(indexes.np,'asarray',forbidden)
    with pytest.raises(ValueError):indexes._vector([[1,2]]*4,4)
    with pytest.raises(ValueError):indexes._vector(np.array(1),4)


def test_background_search_is_cancelled_and_drained_on_lease_exit():
    async def run():
        gate=threading.Event();fake=FakeIndex(gate=gate)
        registry=indexes.OwnerIndexRegistry();await registry.publish(loaded(fake=fake))
        async def borrower():
            async with registry.acquire('alice') as lease:
                asyncio.create_task(lease.search([1,0,0,0],top_k=1,absolute_deadline=time.monotonic()+2))
                assert await asyncio.to_thread(fake.started.wait,1)
        task=asyncio.create_task(borrower())
        assert await asyncio.to_thread(fake.started.wait,1)
        await asyncio.sleep(.01);assert not task.done()
        gate.set();await task
        await registry.aclose();assert fake.closed
    asyncio.run(run())


def test_close_failure_retains_capacity_until_successful_retry():
    class FailingClose(FakeIndex):
        fail=True
        def close(self):
            if self.fail:raise RuntimeError('native close failed')
            super().close()
    async def run():
        fake=FailingClose();registry=indexes.OwnerIndexRegistry(max_generations=1)
        await registry.publish(loaded(fake=fake))
        with pytest.raises(indexes.IndexUnavailable):await registry.aclose()
        assert not fake.closed and len(registry._resident)==1
        fake.fail=False
        await registry.aclose();assert fake.closed and not registry._resident
    asyncio.run(run())


def test_real_docid_order_and_manual_rerank_corpus(tmp_path):
    import json
    path=tmp_path/'generation';path.mkdir()
    build_native_index(path/'shard_0',[12,11],dimensions=2,docids=True)
    # Mutable native order differs from canonical ids/corpus order.
    (path/'shard_0/ids.json').write_text('[11,12]')
    (path/'ids.json').write_text('[11,12]')
    full=np.array([[0,.8,0,.6],[.8,0,.6,0]],dtype=np.float32)
    full.tofile(path/'shard_0/corpus_full.f32')
    manifest={'format_version':2,'num_shards':1,'dimensions':4,'index_dim':2,
        'shards':[{'dir':'shard_0','count':2,'docids':True}],
        'manual_rerank':{'full_dim':4,'ah_dim':2,'reorder_pool':2,'corpus_per_shard':True,'ah_only':True}}
    (path/'manifest.json').write_text(json.dumps(manifest));seal_generation(path)
    resource=indexes.load_scann_index(binding(),path)
    assert resource.searcher._manual_rerank==manifest['manual_rerank']
    ids,scores=resource.searcher.search(np.array([.8,0,.6,0],dtype=np.float32),top_k=1)
    assert ids==[12] and scores[0]==pytest.approx(1)
    resource.searcher.close()
    assert resource.searcher._corpus_files is None
    full[0,0]=np.nan;full.tofile(path/'shard_0/corpus_full.f32')
    with pytest.raises(indexes.IndexUnavailable):indexes.load_scann_index(binding(),path)


def test_sharded_pickle_globals_and_external_config_are_rejected(tmp_path):
    import json,pickle
    from scann.proto import scann_pb2
    path=tmp_path/'generation';path.mkdir()
    build_native_index(path/'shard_0',[11,12],docids=True)
    (path/'ids.json').write_text('[11,12]')
    (path/'manifest.json').write_text(json.dumps({'num_shards':1,'dimensions':4,'shards':[{'dir':'shard_0','count':2,'docids':True}]}))
    pickle_path=path/'shard_0/scann_docids.pkl';original=pickle_path.read_bytes()
    pickle_path.write_bytes(pickle.dumps(eval));seal_generation(path)
    with pytest.raises(indexes.IndexUnavailable):indexes.load_scann_index(binding(),path)
    pickle_path.write_bytes(original)
    config_path=path/'shard_0/scann_config.pb';config=scann_pb2.ScannConfig();config.ParseFromString(config_path.read_bytes())
    config.input_output.preprocessed_artifacts_dir=str(tmp_path/'foreign')
    config_path.write_bytes(config.SerializeToString())
    with pytest.raises(indexes.IndexUnavailable):indexes.load_scann_index(binding(),path)


def test_same_physical_generation_cannot_bind_two_owners(tmp_path):
    path=tmp_path/'generation';build_native_index(path,[11,12]);seal_generation(path)
    alice=indexes.load_scann_index(binding(),path)
    bob=indexes.load_scann_index(binding('bob'),path)
    async def run():
        registry=indexes.OwnerIndexRegistry();await registry.publish(alice)
        with pytest.raises(indexes.IndexUnavailable):await registry.publish(bob)
        bob.searcher.close()
        async with registry.acquire('alice') as lease:
            assert (await lease.search([1,0,0,0],top_k=1,absolute_deadline=time.monotonic()+1))[0]==[11]
        await registry.aclose()
    asyncio.run(run())


def test_numpy_shape_and_loader_id_limits_precede_native_work(tmp_path,monkeypatch):
    path=tmp_path/'wide.npy';np.save(path,np.zeros((1,8193),dtype=np.float32))
    def forbidden(_):raise AssertionError('unbounded finite scan')
    monkeypatch.setattr(indexes.np,'isfinite',forbidden)
    with pytest.raises(indexes.IndexUnavailable):indexes._finite_array(path)
    generation=tmp_path/'generation';generation.mkdir();(generation/'ids.json').write_text('[1,2]');seal_generation(generation)
    with pytest.raises(indexes.IndexUnavailable):indexes.load_scann_index(binding(),generation,limits=indexes.IndexLoadLimits(max_ids=1))


def test_built_empty_index_and_stray_native_assets_are_distinguished(tmp_path):
    empty=tmp_path/'empty';empty.mkdir();(empty/'ids.json').write_text('[]');seal_generation(empty)
    resource=indexes.load_scann_index(binding(),empty)
    assert resource.searcher.search(np.array([1,0,0,0],dtype=np.float32),top_k=1)==([],[])
    resource.searcher.close()
    partial=tmp_path/'partial';build_native_index(partial,[11,12]);(partial/'ids.json').write_text('[]');seal_generation(partial)
    with pytest.raises(indexes.IndexUnavailable):indexes.load_scann_index(binding(),partial)


def test_real_ah_index_and_missing_required_asset_preflight(tmp_path,monkeypatch):
    import json
    scann=pytest.importorskip('scann')
    from google.protobuf import text_format
    from scann.scann_ops import scann_assets_pb2
    from scann.scann_ops.py import scann_ops_pybind
    path=tmp_path/'generation';path.mkdir()
    vectors=np.random.default_rng(3).normal(size=(120,16)).astype(np.float32)
    vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    native=scann.scann_ops_pybind.builder(vectors,10,'dot_product').set_n_training_threads(1).tree(num_leaves=4,num_leaves_to_search=4,training_sample_size=120).score_ah(2,anisotropic_quantization_threshold=.2).reorder(20).build()
    native.serialize(str(path));(path/'ids.json').write_text(json.dumps(list(range(1,121))));seal_generation(path)
    bound=indexes.IndexBinding('alice','ah1','synthetic-model',16,'alice-ah1')
    resource=indexes.load_scann_index(bound,path)
    ids,scores=resource.searcher.search(vectors[0],top_k=2)
    assert ids[0]==1 and scores[0]==pytest.approx(1,abs=1e-5)
    resource.searcher.close()
    metadata=path/'scann_assets.pbtxt';assets=scann_assets_pb2.ScannAssets();text_format.Parse(metadata.read_text(),assets)
    keep=[asset for asset in assets.assets if asset.asset_type!=scann_assets_pb2.ScannAsset.PARTITIONER]
    del assets.assets[:];assets.assets.extend(keep);metadata.write_text(text_format.MessageToString(assets))
    called=[]
    def forbidden(*args):called.append(True);raise RuntimeError('native constructor must not run')
    monkeypatch.setattr(scann_ops_pybind.scann_pybind,'ScannNumpy',forbidden)
    with pytest.raises(indexes.IndexUnavailable):indexes.load_scann_index(bound,path)
    assert not called


class CountingIndex(FakeIndex):
    """Holds each search until `release`, recording the peak overlap."""
    def __init__(self,release):
        super().__init__(gate=release)
        self.active=0;self.peak=0;self.lock=threading.Lock()
    def search(self,vector,top_k):
        with self.lock:self.active+=1;self.peak=max(self.peak,self.active)
        try:return super().search(vector,top_k)
        finally:
            with self.lock:self.active-=1


def test_one_index_serves_several_searches_at_once_up_to_its_limit():
    # A parent plus three subagents each search the same owner's index.
    async def run():
        release=threading.Event()
        fake=CountingIndex(release)
        registry=indexes.OwnerIndexRegistry(max_searches=6,max_searches_per_index=2)
        await registry.publish(loaded(fake=fake))
        async def one():
            async with registry.acquire('alice') as lease:
                return await lease.search([1,0,0,0],top_k=3,absolute_deadline=time.monotonic()+5)
        tasks=[asyncio.create_task(one()) for _ in range(3)]
        for _ in range(200):
            if fake.active==2:break
            await asyncio.sleep(.01)
        assert fake.active==2  # two run together; the third waits for a slot
        release.set()
        assert await asyncio.gather(*tasks)==[([1],[.75])]*3
        assert fake.peak==2
        await registry.aclose()
    asyncio.run(run())
