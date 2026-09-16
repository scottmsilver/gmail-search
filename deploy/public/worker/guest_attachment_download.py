"""Private guest files for a persistent tool core; no CLI or config activation.

Quotas are per instance, not durable per run. Only the outer VM disk/resource
ceiling constrains an adversarial same-UID shell or repeated process restarts.
"""
import asyncio
from dataclasses import dataclass
import errno
import math
import os
import secrets
import stat

from guest_attachment_transport import DownloadError, DownloadCleanupError, GuestRawTransport, _drain

MAX_FILES=20
MAX_DISK_BYTES=64*1024**2
MAX_BATCH_BYTES=20*1024**2
MAX_BATCH_ITEMS=20


@dataclass
class _File:
    batch: object
    temporary: str
    final: str
    size: int=0
    written: int=0
    fd: int|None=None
    identity: tuple|None=None
    temporary_exists: bool=False
    final_exists: bool=False
    slot: bool=True
    committed: bool=False
    quarantined: bool=False
    close_uncertain: bool=False
    sha256: str=''


class GuestAttachmentDownloader:
    def __init__(self,workspace_fd,capability,*,slots):
        # The persistent core must supply its one shared two-socket semaphore.
        # _value can catch an obviously invalid configuration; it cannot prove
        # the original capacity of a semaphore that is already in use.
        if (type(workspace_fd) is not int or workspace_fd<0
                or type(slots) not in (asyncio.Semaphore,asyncio.BoundedSemaphore)
                or not 0<=slots._value<=2):
            raise DownloadError()
        self._transport=GuestRawTransport(capability)
        self._slots=slots
        self._workspace_fd=self._directory_fd=None
        self._directory_name='gms-downloads-'+secrets.token_hex(16)
        self._records={}
        self._bytes=0
        self._batches=set()
        self._closed=False
        self._poisoned=False
        self._loop=None
        self._close_task=None
        self._uncertain_directory_fds=set()
        try:
            self._workspace_fd=os.dup(workspace_fd)
            self._private(os.fstat(self._workspace_fd))
            os.mkdir(self._directory_name,0o700,dir_fd=self._workspace_fd)
            self._directory_fd=os.open(self._directory_name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,
                                       dir_fd=self._workspace_fd)
            info=os.fstat(self._directory_fd); self._private(info)
            self._directory_identity=(info.st_dev,info.st_ino)
        except (OSError,ValueError):
            for fd in (self._directory_fd,self._workspace_fd):
                if fd is not None: os.close(fd)
            raise DownloadError() from None

    @staticmethod
    def _private(info):
        if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700:
            raise DownloadError()

    @property
    def usage(self):
        return {'files':len(self._records),'bytes':self._bytes,
                'quarantined':sum(record.quarantined for record in self._records.values())}

    def _directory(self):
        info=os.fstat(self._directory_fd); self._private(info)
        named=os.stat(self._directory_name,dir_fd=self._workspace_fd,follow_symlinks=False)
        self._private(named)
        if (info.st_dev,info.st_ino)!=self._directory_identity or (named.st_dev,named.st_ino)!=self._directory_identity:
            raise DownloadError()

    def _reserve(self,batch):
        if self._closed or self._poisoned or len(self._records)>=MAX_FILES:
            raise DownloadError()
        self._directory()
        key=secrets.token_hex(16)
        if key in self._records:
            raise DownloadError()
        record=_File(batch,'.'+key+'.partial','attachment-'+key+'.bin')
        self._records[key]=record
        return record

    def _create(self,record,header,budget):
        size=header['size_bytes']
        if budget[0]+size>MAX_BATCH_BYTES:
            raise DownloadError()
        # Attempted payload charges survive failed hash/IO within this batch.
        budget[0]+=size
        if self._bytes+size>MAX_DISK_BYTES:
            raise DownloadError()
        self._bytes+=size; record.size=size; record.sha256=header['sha256']
        self._directory()
        record.fd=os.open(record.temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,
                          0o600,dir_fd=self._directory_fd)
        record.temporary_exists=True
        info=os.fstat(record.fd)
        record.identity=(info.st_dev,info.st_ino)
        self._file_info(record,info,size=0)

    @staticmethod
    def _file_info(record,info,*,size):
        if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600
                or info.st_nlink!=1 or info.st_size!=size or (info.st_dev,info.st_ino)!=record.identity):
            raise DownloadError()

    async def _write(self,record,chunk):
        if type(chunk) is not bytes or not 0<len(chunk)<=65536 or record.written+len(chunk)>record.size:
            raise DownloadError()
        offset=0
        while offset<len(chunk):
            count=os.write(record.fd,chunk[offset:])
            if count<=0: raise DownloadError()
            offset+=count; record.written+=count
            await asyncio.sleep(0)

    def _close_file(self,record):
        if record.fd is not None:
            try: info=os.fstat(record.fd)
            except OSError as error:
                if error.errno!=errno.EBADF: raise
                record.fd=None
                return
            if record.close_uncertain:
                # Even equal dev/inode cannot identify an open-file handle:
                # the descriptor number may have been reopened to the same
                # inode after a close with an uncertain acknowledgment. Never
                # retry a still-valid uncertain number. Teardown is required.
                raise DownloadCleanupError()
            if record.identity is not None and (info.st_dev,info.st_ino)!=record.identity:
                # A lost close acknowledgment may leave a number later reused
                # for an unrelated descriptor. Never close that replacement.
                record.fd=None
                return
            try: os.close(record.fd)
            except OSError:
                record.close_uncertain=True
                raise
            record.fd=None

    def _rollback(self,record):
        failed=False
        # Retain a still-owned descriptor while checking/removing its names.
        # Completed tentative files and uncertain close paths have only saved
        # identities; same-UID pathname mutation is not a security boundary.
        for name,field in ((record.temporary,'temporary_exists'),(record.final,'final_exists')):
            if not getattr(record,field): continue
            try:
                info=os.stat(name,dir_fd=self._directory_fd,follow_symlinks=False)
                if (info.st_dev,info.st_ino)!=record.identity or not stat.S_ISREG(info.st_mode):
                    raise DownloadCleanupError()
                os.unlink(name,dir_fd=self._directory_fd)
                setattr(record,field,False)
            except FileNotFoundError:
                setattr(record,field,False)
            except (OSError,DownloadError):
                failed=True
        try: self._close_file(record)
        except (OSError,DownloadError): failed=True
        if failed:
            record.quarantined=True; self._poisoned=True
            raise DownloadCleanupError()

    def _forget(self,record):
        for key,item in tuple(self._records.items()):
            if item is record:
                del self._records[key]; self._bytes-=record.size
                break
        if record.slot:
            record.slot=False; self._slots.release()

    async def _one(self,attachment_id,batch,budget,deadline):
        record=None
        acquired=False
        success=False
        failure=None
        cleanup_failed=False
        try:
            async with asyncio.timeout_at(deadline):
                if self._closed or self._poisoned: raise DownloadError()
                await self._slots.acquire(); acquired=True
                record=self._reserve(batch)
                async def header(value): self._create(record,value,budget)
                async def chunk(value): await self._write(record,value)
                await self._transport.receive(attachment_id,deadline=deadline,on_header=header,on_chunk=chunk)
                self._file_info(record,os.fstat(record.fd),size=record.size)
                if record.written!=record.size: raise DownloadError()
                if asyncio.get_running_loop().time()>=deadline: raise TimeoutError()
                self._publish(record)
                if asyncio.get_running_loop().time()>=deadline: raise TimeoutError()
                if asyncio.current_task().cancelling(): raise asyncio.CancelledError()
                self._close_file(record)
                success=True
        except asyncio.CancelledError:
            failure=asyncio.CancelledError
        except TimeoutError:
            failure=TimeoutError
        except DownloadCleanupError:
            failure=DownloadCleanupError; cleanup_failed=True
        except Exception:
            failure=DownloadError
        finally:
            if record is not None:
                if not success:
                    try: self._rollback(record)
                    except DownloadCleanupError: cleanup_failed=True
                    if cleanup_failed:
                        record.quarantined=True; self._poisoned=True
                    else: self._forget(record)
                if success and record.slot:
                    record.slot=False; self._slots.release()
            elif acquired:
                self._slots.release()
        if failure in (asyncio.CancelledError,TimeoutError): raise failure()
        if failure is not None: return {'error':str(DownloadCleanupError() if cleanup_failed else DownloadError())}
        return record

    def _publish(self,record):
        self._directory()
        self._file_info(record,os.fstat(record.fd),size=record.size)
        info=os.stat(record.temporary,dir_fd=self._directory_fd,follow_symlinks=False)
        self._file_info(record,info,size=record.size)
        # link() fails if destination exists; rename() would overwrite it.
        os.link(record.temporary,record.final,src_dir_fd=self._directory_fd,dst_dir_fd=self._directory_fd,
                follow_symlinks=False)
        record.final_exists=True
        final=os.stat(record.final,dir_fd=self._directory_fd,follow_symlinks=False)
        held=os.fstat(record.fd)
        if ((final.st_dev,final.st_ino)!=(held.st_dev,held.st_ino) or (held.st_dev,held.st_ino)!=record.identity
                or not stat.S_ISREG(final.st_mode) or final.st_nlink!=2 or held.st_nlink!=2):
            raise DownloadError()
        os.unlink(record.temporary,dir_fd=self._directory_fd); record.temporary_exists=False
        self._file_info(record,os.stat(record.final,dir_fd=self._directory_fd,follow_symlinks=False),size=record.size)
        self._file_info(record,os.fstat(record.fd),size=record.size)
        self._directory()
        return {'relative_path':self._directory_name+'/'+record.final,'size_bytes':record.size,'sha256':record.sha256}

    def _discard_batch(self,batch):
        for record in tuple(self._records.values()):
            if record.batch is not batch or record.committed or record.quarantined: continue
            try: self._rollback(record)
            except DownloadCleanupError: continue
            self._forget(record)

    async def download_many(self,attachment_ids,*,deadline):
        loop=asyncio.get_running_loop()
        if (self._closed or self._poisoned or (self._loop is not None and self._loop is not loop)
                or type(attachment_ids) is not tuple or not 1<=len(attachment_ids)<=MAX_BATCH_ITEMS
                or any(type(aid) is not int or not 0<aid<=9223372036854775807 for aid in attachment_ids)
                or len(set(attachment_ids))!=len(attachment_ids) or type(deadline) not in (int,float)
                or not math.isfinite(deadline)):
            raise DownloadError()
        self._loop=loop
        deadline=min(deadline,loop.time()+30)
        if loop.time()>=deadline: raise TimeoutError()
        owner=asyncio.current_task(); self._batches.add(owner)
        batch=object(); budget=[0]; tasks=[]; finished=False
        try:
            tasks=[asyncio.create_task(self._one(aid,batch,budget,deadline)) for aid in attachment_ids]
            joined=asyncio.gather(*tasks)
            try:
                values=await asyncio.shield(joined)
            finally:
                async def cleanup():
                    for task in tasks:
                        if not task.done() and not task.cancelling(): task.cancel()
                    await asyncio.gather(*tasks,return_exceptions=True)
                    # Retrieve aggregate failure even after outer cancellation.
                    await asyncio.gather(joined,return_exceptions=True)
                await _drain(cleanup())
            if loop.time()>=deadline: raise TimeoutError()
            if owner.cancelling(): raise asyncio.CancelledError()
            results=[]
            for value in values:
                if isinstance(value,dict): results.append(value); continue
                results.append({'relative_path':self._directory_name+'/'+value.final,
                                'size_bytes':value.size,'sha256':value.sha256})
            if loop.time()>=deadline: raise TimeoutError()
            if owner.cancelling(): raise asyncio.CancelledError()
            for record in self._records.values():
                if record.batch is batch and record.final_exists and not record.quarantined:
                    record.committed=True
            finished=True
            return tuple(results)
        finally:
            if not finished: self._discard_batch(batch)
            self._batches.discard(owner)

    def _close_directory(self,field):
        fd=getattr(self,field)
        if fd is None:return
        if field in self._uncertain_directory_fds:
            try:os.fstat(fd)
            except OSError as error:
                if error.errno!=errno.EBADF:raise DownloadCleanupError() from None
                setattr(self,field,None)
                self._uncertain_directory_fds.remove(field)
                return
            # A new handle can refer to the same inode. Never infer ownership
            # of an uncertain descriptor number from directory identity.
            raise DownloadCleanupError()
        try:os.close(fd)
        except OSError:
            self._uncertain_directory_fds.add(field)
            raise DownloadCleanupError() from None
        setattr(self,field,None)

    async def aclose(self):
        loop=asyncio.get_running_loop()
        if (self._loop is not None and self._loop is not loop) or asyncio.current_task() in self._batches:
            raise DownloadCleanupError()
        self._loop=loop
        self._closed=True
        async def close():
            tasks=[task for task in self._batches if task is not asyncio.current_task()]
            for task in tasks:
                if not task.done() and not task.cancelling(): task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            await self._transport.aclose()
            for record in tuple(self._records.values()):
                if not record.committed:
                    self._rollback(record); self._forget(record)
            self._close_directory('_directory_fd')
            self._close_directory('_workspace_fd')
        # Concurrent callers share one cleanup owner. Only an acknowledged
        # failure permits a later explicit retry; cancellation cannot fork it.
        if self._close_task is None or (self._close_task.done() and not self._close_task.cancelled()
                                       and self._close_task.exception() is not None):
            self._close_task=asyncio.create_task(close())
        await _drain(self._close_task)
