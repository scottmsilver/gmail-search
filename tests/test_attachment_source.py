import hashlib
from pathlib import Path

import pytest

from gmail_search.gateway.attachment_sandbox import AttachmentDenied


@pytest.mark.asyncio
async def test_attachment_loader_has_no_symlink_or_cross_owner_fallback(tmp_path):
    from gmail_search.gateway.attachment_source import AttachmentLocator, OwnerAttachmentSource
    root = tmp_path / 'attachments'
    directory = root / 'owners' / hashlib.sha256(b'alice').hexdigest() / 'message'
    directory.mkdir(parents=True)
    source = directory / 'file.pdf'
    source.write_bytes(b'opaque bytes')
    record = AttachmentLocator('alice', 1, 'application/pdf', str(source))
    async def locate(owner, aid):
        assert (owner, aid) == ('alice', 1)
        return record
    loader = OwnerAttachmentSource(root, locate=locate)
    result = await loader.load('alice', 1)
    assert result.data == b'opaque bytes'
    source.unlink()
    source.symlink_to(tmp_path / 'outside.pdf')
    (tmp_path / 'outside.pdf').write_bytes(b'forbidden')
    with pytest.raises(AttachmentDenied):
        await loader.load('alice', 1)
    source.unlink()
    record = AttachmentLocator('bob', 1, 'application/pdf', str(source))
    with pytest.raises(AttachmentDenied):
        await loader.load('alice', 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', [False, True])
@pytest.mark.parametrize('kind', ['foreign_path', 'traversal', 'intermediate_symlink', 'fifo', 'large', 'hardlink'])
async def test_attachment_loader_rejects_unsafe_storage(tmp_path, kind, raw):
    from gmail_search.gateway.attachment_source import AttachmentLocator, OwnerAttachmentSource
    import os
    root = tmp_path / 'attachments'
    directory = root / 'owners' / hashlib.sha256(b'alice').hexdigest() / 'message'
    directory.mkdir(parents=True)
    source = directory / 'file.pdf'
    if kind == 'fifo':
        os.mkfifo(source)
    elif kind == 'large':
        with source.open('wb') as stream:
            stream.truncate(11 * 1024**2)
    elif kind == 'intermediate_symlink':
        directory.rmdir()
        outside = tmp_path / 'outside'
        outside.mkdir()
        (outside / 'file.pdf').write_bytes(b'forbidden')
        directory.symlink_to(outside, target_is_directory=True)
    else:
        source.write_bytes(b'opaque')
        if kind == 'hardlink':
            os.link(source, directory / 'second')
        elif kind == 'foreign_path':
            source = root / 'owners' / hashlib.sha256(b'bob').hexdigest() / 'message' / 'file.pdf'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'foreign')
        elif kind == 'traversal':
            source = Path(str(directory / '..' / 'message' / 'file.pdf'))
    async def locate(owner, aid):
        return AttachmentLocator(owner, aid, 'application/pdf', str(source), expected_size=6)
    loader = OwnerAttachmentSource(root, locate=locate)
    with pytest.raises(AttachmentDenied):
        await (loader.load_raw if raw else loader.load)('alice', 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', [False, True])
@pytest.mark.parametrize('cancel', [False, True])
async def test_changed_or_cancelled_read_closes_descriptor(tmp_path, monkeypatch, cancel, raw):
    import asyncio
    import os
    from gmail_search.gateway.attachment_source import AttachmentLocator, OwnerAttachmentSource
    root = tmp_path / 'attachments'
    path = root / 'owners' / hashlib.sha256(b'alice').hexdigest() / 'message' / 'file.pdf'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'x' * 100000)
    async def locate(owner, aid):
        return AttachmentLocator(owner, aid, 'application/pdf', str(path), expected_size=100000)
    original_read = os.read
    observed = []
    def changing_read(fd, count):
        chunk = original_read(fd, count)
        if not observed:
            observed.append(fd)
            if cancel:
                asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
            else:
                with path.open('ab') as stream:
                    stream.write(b'changed')
        return chunk
    monkeypatch.setattr(os, 'read', changing_read)
    loader = OwnerAttachmentSource(root, locate=locate)
    with pytest.raises(asyncio.CancelledError if cancel else AttachmentDenied):
        await (loader.load_raw if raw else loader.load)('alice', 1)
    with pytest.raises(OSError):
        os.fstat(observed[0])


@pytest.mark.asyncio
@pytest.mark.parametrize('data', [b'', b'opaque office bytes'])
async def test_raw_source_allows_generic_and_empty_but_parser_stays_closed(tmp_path, data):
    from gmail_search.gateway.attachment_source import AttachmentLocator, OwnerAttachmentSource
    root = tmp_path/'attachments'
    path = root/'owners'/hashlib.sha256(b'alice').hexdigest()/'message'/'file.docx'
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    async def locate(owner, aid):
        return AttachmentLocator(owner, aid, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', str(path), expected_size=len(data))
    loader = OwnerAttachmentSource(root, locate=locate)
    assert callable(getattr(loader, 'load_raw', None)), 'raw bytes need a separate entry point'
    result = await loader.load_raw('alice', 1)
    assert result.data == data and result.owner_id == 'alice'
    from gmail_search.gateway.attachment_sandbox import AttachmentInput
    assert type(result) is not AttachmentInput
    with pytest.raises(AttachmentDenied):
        await loader.load('alice', 1)


@pytest.mark.asyncio
async def test_raw_source_requires_matching_declared_size(tmp_path):
    from gmail_search.gateway.attachment_source import AttachmentLocator, OwnerAttachmentSource
    root = tmp_path/'attachments'
    path = root/'owners'/hashlib.sha256(b'alice').hexdigest()/'message'/'file.bin'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'123')
    async def locate(owner, aid):
        return AttachmentLocator(owner, aid, 'application/octet-stream', str(path), expected_size=2)
    loader = OwnerAttachmentSource(root, locate=locate)
    assert callable(getattr(loader, 'load_raw', None))
    with pytest.raises(AttachmentDenied):
        await loader.load_raw('alice', 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('mime', ['text/plain\r\nInjected: x', 'text/plain; charset=utf8', '', 'x'*256])
async def test_raw_source_rejects_noncanonical_mime_before_open(tmp_path, monkeypatch, mime):
    from gmail_search.gateway.attachment_source import AttachmentLocator, OwnerAttachmentSource
    path = tmp_path/'owners'/hashlib.sha256(b'alice').hexdigest()/'message'/'file.bin'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'')
    def forbidden_open(*args, **kwargs):
        pytest.fail('invalid MIME reached filesystem open')
    monkeypatch.setattr('os.open', forbidden_open)
    async def locate(owner, aid):
        return AttachmentLocator(owner, aid, mime, str(path), expected_size=0)
    with pytest.raises(AttachmentDenied):
        await OwnerAttachmentSource(tmp_path, locate=locate).load_raw('alice', 1)


@pytest.mark.asyncio
async def test_empty_qualified_mime_is_raw_only(tmp_path):
    from gmail_search.gateway.attachment_source import AttachmentLocator, OwnerAttachmentSource
    path = tmp_path/'owners'/hashlib.sha256(b'alice').hexdigest()/'message'/'file.pdf'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'')
    async def locate(owner, aid):
        return AttachmentLocator(owner, aid, 'application/pdf', str(path), expected_size=0)
    loader = OwnerAttachmentSource(tmp_path, locate=locate)
    assert (await loader.load_raw('alice', 1)).data == b''
    with pytest.raises(AttachmentDenied):
        await loader.load('alice', 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('size', [None, True, -1, 10485761, '0'])
async def test_invalid_raw_declared_size_rejected_before_open(tmp_path, monkeypatch, size):
    from gmail_search.gateway.attachment_source import AttachmentLocator, OwnerAttachmentSource
    path = tmp_path/'owners'/hashlib.sha256(b'alice').hexdigest()/'message'/'file.bin'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'')
    def forbidden_open(*args, **kwargs):
        pytest.fail('invalid declared size reached filesystem open')
    monkeypatch.setattr('os.open', forbidden_open)
    async def locate(owner, aid):
        return AttachmentLocator(owner, aid, 'application/octet-stream', str(path), expected_size=size)
    with pytest.raises(AttachmentDenied):
        await OwnerAttachmentSource(tmp_path, locate=locate).load_raw('alice', 1)
