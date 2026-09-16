#!/usr/bin/env python3
"""Synthetic fixtures only, run within the clean outer VM. No host parsing."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,'/opt/gmail-worker')
from attachment_backend import AttachmentFirecrackerBackend
from attachment_sandbox import validate_result, AttachmentDenied
from firecracker_backend import SyntheticAttachmentBackend, process_start

backend = AttachmentFirecrackerBackend()
fixture = Path('/home/worker/attachment-fixtures')
results = []
for name,mime,accepted in [('hello.pdf','application/pdf',True),('huge.pdf','application/pdf',False),
                         ('long-text.pdf','application/pdf',True),
                         ('malformed.pdf','application/pdf',False),('tiny.png','image/png',True),
                         ('huge.png','image/png',False),('hello.zip','application/zip',True),
                         ('many.zip','application/zip',True)]:
    raw = backend.parse((fixture/name).read_bytes(),mime,{'dpi':100,'pages':[]})
    try:
        parsed=validate_result(raw)
    except AttachmentDenied:
        assert not accepted,(name,raw)
        results.append(dict(fixture=name,rejected=True))
        continue
    assert accepted,name
    if name in ('hello.pdf','hello.zip'):
        assert 'SYNTHETIC ATTACHMENT 42' in parsed.text
        assert len(parsed.pages)==1
    if name=='many.zip':
        assert parsed.truncated and not parsed.pages and not parsed.text
    if name=='long-text.pdf':
        assert parsed.truncated and len(parsed.text.encode())==200000
        results.append(dict(fixture=name,text_bytes=200000,truncated=True))
        continue
    results.append(dict(fixture=name,text=parsed.text,pages=[dict(width=p.width,height=p.height,sha256=hashlib.sha256(p.png).hexdigest()) for p in parsed.pages],truncated=parsed.truncated))
# A separate controller exits with a parser VM awaiting its input. Independent
# lease supervisor must kill and reap the whole VMM without any controller call.
script='''import sys,time,uuid,json,os
from types import SimpleNamespace
sys.path.insert(0,"/opt/gmail-worker")
from firecracker_backend import SyntheticAttachmentBackend,ROOT
from attachment_backend import Limits
handle=uuid.uuid4().hex
backend=SyntheticAttachmentBackend()
deadline=time.time()+8
backend.launch(handle,SimpleNamespace(run_id=uuid.uuid4().hex,deadline=deadline,lease_expires=deadline),Limits())
print(json.dumps(dict(handle=handle,state=json.loads((ROOT/"runs"/handle/"state.json").read_text()))),flush=True)
os._exit(0)
'''
started=json.loads(subprocess.check_output(['/usr/bin/python3','-c',script],timeout=10))
handle,pid=started['handle'],started['state']['pid']
end=time.monotonic()+12
while time.monotonic()<end and process_start(pid) is not None:
    time.sleep(.1)
assert process_start(pid) is None,'parser watchdog failed to reap orphan VMM'
assert handle not in SyntheticAttachmentBackend().inventory()
results.append(dict(orphan_watchdog_killed=True,evidence=started['state']['evidence']))
assert not SyntheticAttachmentBackend().inventory()
print(json.dumps(dict(status='passed',results=results),indent=2))
