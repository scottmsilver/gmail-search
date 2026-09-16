"""One-shot synthetic-only capability bootstrap; no paths or command selectors."""
import json
import socket
import struct
import time

from guest_tool_config import LEGACY_PROFILE, READ_PROFILE, RAW_PROFILE, parse_tool_config, write_capability_file


def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('Invalid bootstrap')
        result[key]=value
    return result


def validate_config(value, *, expected_profile=LEGACY_PROFILE):
    if (type(value) is not dict or type(value.get('runtime')) is not str
            or value['runtime'] not in ('pi','claude') or type(value.get('version')) is not int):
        raise ValueError('Invalid bootstrap')
    if value['version']==1 and set(value)=={'version','runtime','capabilities'}:
        config=parse_tool_config(value['capabilities'])
        if config.version!=1:raise ValueError('Invalid bootstrap')
    elif value['version'] in (2,3) and set(value)=={'version','runtime','tool_profile','capabilities'}:
        config=parse_tool_config({key:value[key] for key in ('version','tool_profile','capabilities')})
    else:raise ValueError('Invalid bootstrap')
    if expected_profile not in (LEGACY_PROFILE,READ_PROFILE,RAW_PROFILE) or config.profile!=expected_profile:
        raise ValueError('Invalid bootstrap')
    return value


def tool_config(value, *, expected_profile=LEGACY_PROFILE):
    """Validate bootstrap and retain its exact tool profile when persisting."""
    value=validate_config(value,expected_profile=expected_profile)
    return parse_tool_config(value['capabilities'] if value['version']==1 else
                             {key:value[key] for key in ('version','tool_profile','capabilities')})


def persist_config(root,value, *, expected_profile=LEGACY_PROFILE):
    write_capability_file(root,tool_config(value,expected_profile=expected_profile))


def read_config(read, *, expected_profile=LEGACY_PROFILE):
    def exact(size):
        data=bytearray()
        while len(data)<size:
            part=read(size-len(data))
            if not part or len(part)>size-len(data):raise ValueError('Invalid bootstrap')
            data.extend(part)
        return bytes(data)
    size=struct.unpack('!I',exact(4))[0]
    if not 0<size<=4096:raise ValueError('Invalid bootstrap')
    value=json.loads(exact(size),object_pairs_hook=_pairs)
    validate_config(value,expected_profile=expected_profile)
    if read(1)!=b'':raise ValueError('Invalid bootstrap')
    return value


def receive(*, expected_profile=LEGACY_PROFILE):
    end=time.monotonic()+5
    with socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM) as peer:
        peer.settimeout(5)
        peer.connect((2,8002))
        def read(size):
            remaining=end-time.monotonic()
            if remaining<=0:raise TimeoutError('Bootstrap deadline')
            peer.settimeout(remaining)
            return peer.recv(size)
        return read_config(read,expected_profile=expected_profile)
