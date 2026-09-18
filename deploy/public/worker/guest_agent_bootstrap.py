"""Fixed full-agent envelope, separate from historical tool-only bootstraps."""
import json
import re
import socket
import struct
import time

from guest_run_bootstrap import _pairs
from guest_tool_config import RAW_PROFILE, parse_tool_config

PROFILE='mail-agent-pi-v1'
CLAUDE_PROFILE='mail-agent-claude-v1'
# Closed set: which agent runs is chosen by the trusted controller, never by
# the guest or the model. Anything else is refused before a runner starts.
PROFILES=frozenset({PROFILE,CLAUDE_PROFILE})
MAX_FRAME=32768
MAX_PROMPT=16384
_TOKEN=re.compile(r'[a-f0-9]{64}\Z',re.ASCII)


def validate_config(value):
    try:
        if (type(value) is not dict or set(value)!={'version','profile','prompt','tool_config',
                'inference_capability','events_capability'} or type(value['version']) is not int
                or value['version']!=1 or value['profile'] not in PROFILES):
            raise ValueError()
        prompt=value['prompt']
        if (type(prompt) is not str or not prompt.strip() or '\x00' in prompt
                or not 0<len(prompt.encode('utf-8'))<=MAX_PROMPT):
            raise ValueError()
        config=parse_tool_config(value['tool_config'])
        if config.profile!=RAW_PROFILE or config.version!=3:
            raise ValueError()
        for key in ('inference_capability','events_capability'):
            if type(value[key]) is not str or not _TOKEN.fullmatch(value[key]):raise ValueError()
        # Snapshot bounded JSON primitives; callers cannot mutate a shared config.
        raw=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
        if len(raw)>MAX_FRAME:raise ValueError()
        return json.loads(raw)
    except (ValueError,TypeError,UnicodeError,RecursionError):
        raise ValueError('Invalid agent bootstrap.') from None


def encode_config(value):
    raw=json.dumps(validate_config(value),ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return struct.pack('!I',len(raw))+raw


def read_config(read):
    def exact(size):
        data=bytearray()
        while len(data)<size:
            part=read(size-len(data))
            if not part or len(part)>size-len(data):raise ValueError('Invalid agent bootstrap.')
            data.extend(part)
        return bytes(data)
    size=struct.unpack('!I',exact(4))[0]
    if not 0<size<=MAX_FRAME:raise ValueError('Invalid agent bootstrap.')
    try:
        value=json.loads(exact(size).decode('utf-8'),object_pairs_hook=_pairs)
        value=validate_config(value)
    except (ValueError,TypeError,UnicodeError,RecursionError):
        raise ValueError('Invalid agent bootstrap.') from None
    if read(1)!=b'':raise ValueError('Invalid agent bootstrap.')
    return value


def receive():
    end=time.monotonic()+5
    with socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM) as peer:
        peer.settimeout(5)
        peer.connect((2,8002))
        def read(size):
            remaining=end-time.monotonic()
            if remaining<=0:raise TimeoutError('Agent bootstrap deadline.')
            peer.settimeout(remaining)
            return peer.recv(size)
        return read_config(read)
