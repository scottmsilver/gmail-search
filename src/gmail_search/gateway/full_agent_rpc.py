"""Fixed full-agent RPC framing; stdlib-only and copied unchanged to worker."""
import hashlib
import json
import math
import re
import struct

MAX_HEADER=4096
MAX_INPUT=32772
MAX_OUTPUT=4096
COMMON={'version','op','request_id','handle','context_sha256','payload_size'}
CONTEXT={'run_id','owner_id','conversation_id','fence','workspace_version','deadline'}
ZERO='0'*32


class ProtocolError(ValueError):pass


def require(ok):
    if not ok:raise ProtocolError('Invalid fixed agent RPC')


def canonical(value):
    try:return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    except (ValueError,TypeError,RecursionError):raise ProtocolError('Invalid fixed agent RPC') from None


def hex_value(value,n):require(type(value) is str and re.fullmatch('[a-f0-9]{'+str(n)+'}',value) is not None)


def timestamp(value):require(type(value) in (int,float) and math.isfinite(value) and 0<value<2**53)


def context_digest(value):
    require(type(value) is dict and set(value)==CONTEXT)
    hex_value(value['run_id'],32)
    for key in ('owner_id','conversation_id'):
        text=value[key];require(type(text) is str and 0<len(text.encode('utf-8'))<=512 and all(ord(c)>=32 and ord(c)!=127 for c in text))
    for key in ('fence','workspace_version'):require(type(value[key]) is int and 0<=value[key]<2**63)
    timestamp(value['deadline'])
    return hashlib.sha256(canonical(value)).hexdigest()


def header(value,response=False):
    require(type(value) is dict and type(value.get('version')) is int and value['version']==1)
    require(value.get('op') in ('launch','renew','stop','inventory'))
    extra={'status','code','handles'} if response else {'context','input_sha256','lease_expires'} if value['op']=='launch' else {'renew_seq','lease_expires'} if value['op']=='renew' else set()
    require(set(value)==COMMON|extra)
    hex_value(value['request_id'],32);hex_value(value['handle'],32);hex_value(value['context_sha256'],64)
    size=value['payload_size'];require(type(size) is int and 0<=size<=MAX_INPUT)
    if value['op']=='inventory':require(value['handle']==ZERO and value['context_sha256']=='0'*64)
    else:require(value['handle']!=ZERO)
    if response:
        require(size==0 and value['status'] in ('running','stopped','ok','error'))
        require(type(value['code']) is str and re.fullmatch('[a-z_]{1,40}',value['code']) is not None)
        handles=value['handles'];require(type(handles) is list and len(handles)<=16 and len(set(handles))==len(handles))
        for item in handles:hex_value(item,32);require(item!=ZERO)
        require(not handles or value['op']=='inventory')
    elif value['op']=='launch':
        require(4<size<=MAX_INPUT);require(context_digest(value['context'])==value['context_sha256']);hex_value(value['input_sha256'],64);timestamp(value['lease_expires'])
        require(value['lease_expires']<=value['context']['deadline'])
    else:
        require(size==0)
        if value['op']=='renew':
            require(type(value['renew_seq']) is int and 0<value['renew_seq']<2**63);timestamp(value['lease_expires'])


def encode_frame(value,payload=b'',*,response=False):
    header(value,response);require(type(payload) is bytes and len(payload)==value['payload_size'])
    if not response and value['op']=='launch':require(hashlib.sha256(payload).hexdigest()==value['input_sha256'])
    raw=canonical(value);require(0<len(raw)<=MAX_HEADER)
    require(not response or 4+len(raw)<=MAX_OUTPUT)
    return struct.pack('!I',len(raw))+raw+payload


def pairs(items):
    result={}
    for key,value in items:require(key not in result);result[key]=value
    return result


def read_frame(read,*,response=False):
    def exact(size):
        result=bytearray()
        while len(result)<size:
            part=read(size-len(result));require(type(part) is bytes and 0<len(part)<=size-len(result));result.extend(part)
        return bytes(result)
    size=struct.unpack('!I',exact(4))[0];require(0<size<=MAX_HEADER)
    try:value=json.loads(exact(size),object_pairs_hook=pairs,parse_constant=lambda _:require(False))
    except (ValueError,UnicodeError,RecursionError):raise ProtocolError('Invalid fixed agent RPC') from None
    header(value,response);payload=exact(value['payload_size']);encode_frame(value,payload,response=response)
    return value,payload


def reply(request,status,code='ok',handles=()):
    value={k:request[k] for k in COMMON};value.update(payload_size=0,status=status,code=code,handles=sorted(handles));encode_frame(value,response=True)
    return value,b''
