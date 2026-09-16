import json

import pytest

from gmail_search.gateway.attachment_rpc import encode_frame, response_header
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import Registry, AccessDenied


class Transport:
    def __init__(self):
        self.calls = []
        self.fail_stop = False
        self.done = True
    def exchange(self, header, payload, cancelled):
        encode_frame(header, payload)
        self.calls.append(header)
        if header['op'] == 'stop' and self.fail_stop:
            raise RuntimeError('synthetic lost acknowledgement')
        status = 'stopped' if header['op'] == 'stop' else 'done' if header['op'] == 'poll' and self.done else 'running'
        data = b'{"text":"synthetic","pages":[],"truncated":false}' if status == 'done' else b''
        response = response_header(header, status, payload=data)
        encode_frame(response, data, response=True)
        return response, data


def bound(tmp_path, transport):
    from gmail_search.gateway.attachment_remote import SSHAttachmentBackend
    registry = Registry(tmp_path / 'remote.sqlite', is_active=lambda _: True)
    run = registry.start_run('alice', 'conversation', request_key='attachment', writer=False)
    capabilities = Capabilities(registry)
    token = capabilities.issue(run.run_id, audience='attachment', operations={'parse'}).secret
    def authorize():
        return capabilities.authorize(token, audience='attachment', operation='parse')
    backend = SSHAttachmentBackend(registry, transport=transport, poll_interval=.01)
    return backend, registry, run, capabilities, token, authorize


def test_remote_job_binds_context_and_reauthenticates_before_renewal(tmp_path):
    transport = Transport()
    backend, registry, run, _, token, authorize = bound(tmp_path, transport)
    checks = []
    def checked():
        checks.append(True)
        return authorize()
    job = backend.start_authorized(b'opaque', 'application/pdf', {'dpi':100,'pages':[]}, lease=run, attachment_id=1, authorize=checked)
    try:
        assert json.loads(job.wait())['text'] == 'synthetic'
        assert len(checks) >= 2
        assert transport.calls[0]['context']['owner_id'] == 'alice'
        assert token not in json.dumps(transport.calls)
        with registry._transaction() as db:
            assert db.execute('SELECT state FROM attachment_remote_jobs').fetchone()['state'] != 'stopped'
    finally:
        job.stop()
    assert transport.calls[-1]['op'] == 'stop'
    with registry._transaction() as db:
        assert db.execute('SELECT state FROM attachment_remote_jobs').fetchone()['state'] == 'stopped'


def test_revocation_stops_renewal_but_still_allows_stop(tmp_path):
    transport = Transport()
    backend, _, run, capabilities, token, authorize = bound(tmp_path, transport)
    capabilities.revoke(token)
    job = backend.start_authorized(b'opaque', 'application/pdf', {'dpi':100,'pages':[]}, lease=run, attachment_id=1, authorize=authorize)
    with pytest.raises(AccessDenied):
        job.wait()
    job.stop()
    assert [header['op'] for header in transport.calls] == ['stop']


def test_lost_stop_ack_retains_durable_binding_for_restart_reconciliation(tmp_path):
    from gmail_search.gateway.attachment_remote import SSHAttachmentBackend, RemoteAttachmentUnavailable
    transport = Transport()
    backend, registry, run, _, _, authorize = bound(tmp_path, transport)
    job = backend.start_authorized(b'opaque', 'application/pdf', {'dpi':100,'pages':[]}, lease=run, attachment_id=1, authorize=authorize)
    job.wait()
    transport.fail_stop = True
    with pytest.raises(RemoteAttachmentUnavailable):
        job.stop()
    replacement = SSHAttachmentBackend(registry, transport=transport)
    with pytest.raises(RemoteAttachmentUnavailable):
        replacement.start_authorized(b'opaque','application/pdf',{'dpi':100,'pages':[]},lease=run,attachment_id=1,authorize=authorize)
    transport.fail_stop = False
    replacement.reconcile()
    assert transport.calls[-1]['op'] == 'stop'
    job.stop()
