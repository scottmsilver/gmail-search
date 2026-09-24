"""Closed guest tool profiles, shared by trusted bootstrap and stdio adapters.

These are capability envelopes, never URLs, owner identities or model arguments.
Historical bare three-token files retain the four-tool legacy profile.
"""
from dataclasses import dataclass, field
import json
import os
import re
import stat
from types import MappingProxyType

LEGACY_PROFILE='legacy-mail-v1'
READ_PROFILE='mail-read-v2'
RAW_PROFILE='mail-raw-mcp-v3'
LEGACY_TOOLS=('describe_schema','sql_query_batch','get_thread_batch','publish_artifact_batch')
READ_TOOLS=LEGACY_TOOLS+('search_emails_batch','find_facts','query_emails_batch','get_attachment_batch')
# v3 adds `judge`: typed Jev judgments the agent asks instead of guessing.
RAW_TOOLS=READ_TOOLS+('judge',)
_TOKEN=re.compile(r'[a-f0-9]{64}\Z',re.ASCII)


class ConfigError(ValueError):
    def __init__(self):
        super().__init__('Invalid guest tool configuration.')


@dataclass(frozen=True)
class GuestToolConfig:
    version: int
    profile: str
    capabilities: object = field(repr=False)

    def __post_init__(self):
        if (type(self.version) is not int or type(self.profile) is not str
                or (self.version,self.profile) not in ((1,LEGACY_PROFILE),(2,READ_PROFILE),(3,RAW_PROFILE))):
            raise ConfigError()
        audiences={'sql','retrieval','artifact'} | ({'attachment'} if self.version>=2 else set())
        caps=self.capabilities
        if (type(caps) is not dict or set(caps)!=audiences
                or any(type(token) is not str or not _TOKEN.fullmatch(token) for token in caps.values())):
            raise ConfigError()
        object.__setattr__(self,'capabilities',MappingProxyType(dict(caps)))

    @property
    def tool_names(self):
        return {1:LEGACY_TOOLS,2:READ_TOOLS,3:RAW_TOOLS}[self.version]


def parse_tool_config(value):
    if type(value) is GuestToolConfig:
        return value
    if type(value) is not dict:
        raise ConfigError()
    if set(value)=={'sql','retrieval','artifact'}:
        return GuestToolConfig(1,LEGACY_PROFILE,value)
    if set(value)!={'version','tool_profile','capabilities'} or type(value['version']) is not int or value['version'] not in (2,3):
        raise ConfigError()
    return GuestToolConfig(value['version'],value['tool_profile'],value['capabilities'])


def persisted_payload(config):
    config=parse_tool_config(config)
    caps=dict(config.capabilities)
    return caps if config.version==1 else dict(version=config.version,tool_profile=config.profile,capabilities=caps)


def write_capability_file(root,config):
    """Create one private file in an existing trusted private run directory.

    Never overwrite/reuse a previous run's file. A failed write leaves startup
    failed; callers must not launch tools until this function has returned.
    """
    payload=json.dumps(persisted_payload(config),separators=(',',':')).encode('ascii')
    if len(payload)>4096:
        raise ConfigError()
    directory=fd=None
    try:
        directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        info=os.fstat(directory)
        if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700:
            raise ConfigError()
        fd=os.open('capabilities.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,
                   0o600,dir_fd=directory)
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_uid!=os.getuid():
            raise ConfigError()
        os.fchmod(fd,0o600)
        remaining=memoryview(payload)
        while remaining:
            written=os.write(fd,remaining)
            if written<=0:
                raise ConfigError()
            remaining=remaining[written:]
    finally:
        if fd is not None:os.close(fd)
        if directory is not None:os.close(directory)
