"""The host handles bounded opaque bytes; parsers execute only in its backend."""
import json

import pytest

from gmail_search.gateway.attachment_sandbox import AttachmentSandbox, AttachmentInput, AttachmentDenied


class Backend:
    def __init__(self):
        self.calls = []
        self.result = json.dumps({'text': 'synthetic', 'pages': [], 'truncated': False}).encode()
    def parse(self, data, mime_type, options):
        self.calls.append((data, mime_type, options))
        return self.result


def test_owner_binding_and_reauthorization_before_return():
    backend = Backend()
    active = [True]
    source = AttachmentInput('alice', 12, 'application/pdf', b'%PDF-synthetic')
    service = AttachmentSandbox(backend, load=lambda owner, attachment: source,
                                authorized=lambda owner, attachment: active[0] and owner == 'alice')
    assert service.parse('alice', 12).text == 'synthetic'
    with pytest.raises(AttachmentDenied):
        service.parse('bob', 12)
    assert len(backend.calls) == 1
    def revoke(*args):
        active[0] = False
        return backend.result
    backend.parse = revoke
    with pytest.raises(AttachmentDenied):
        service.parse('alice', 12)


def test_foreign_loader_result_and_oversized_inputs_never_reach_backend():
    backend = Backend()
    for source in [AttachmentInput('bob', 12, 'application/pdf', b'x'),
                   AttachmentInput('alice', 13, 'application/pdf', b'x'),
                   AttachmentInput('alice', 12, 'application/pdf', b'x' * (10*1024**2+1))]:
        service = AttachmentSandbox(backend, load=lambda *_: source, authorized=lambda *_: True)
        with pytest.raises(AttachmentDenied):
            service.parse('alice', 12)
    assert backend.calls == []


@pytest.mark.parametrize('result', [b'{"path":"/etc/passwd"}', b'{"text":NaN,"pages":[],"truncated":false}',
    b'{"text":"x","pages":[{"number":1,"width":100000,"height":100000,"png":"eA=="}],"truncated":false}',
    b'x'*(8*1024**2+1)])
def test_untrusted_guest_results_are_bounded_and_schema_checked(result):
    backend = Backend()
    backend.result = result
    source = AttachmentInput('alice', 12, 'application/pdf', b'%PDF-x')
    service = AttachmentSandbox(backend, load=lambda *_: source, authorized=lambda *_: True)
    with pytest.raises(AttachmentDenied):
        service.parse('alice', 12)


def test_options_cannot_select_paths_or_unbounded_rendering():
    backend = Backend()
    service = AttachmentSandbox(backend, load=lambda *_: AttachmentInput('alice', 12, 'application/pdf', b'x'),
                                authorized=lambda *_: True)
    for options in [dict(dpi=301), dict(pages=tuple(range(1, 10))), dict(pages=('../x',)), dict(dpi=True)]:
        with pytest.raises(AttachmentDenied):
            service.parse('alice', 12, **options)
    assert backend.calls == []
