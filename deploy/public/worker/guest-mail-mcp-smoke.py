#!/usr/bin/env python3
"""Native Claude typed MCP qualification in one immutable synthetic guest."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
sys.path.insert(0,'/tmp/runtime')
from guest_run_bootstrap import receive

ROOT=Path('/tmp/runtime')
RUN=Path('/tmp/gms-run')
MODEL='claude-sonnet-4-6'
PROMPT='SYNTHETIC_MCP_TEST: Use the mail MCP tools to inspect schema, query messages, fetch shared-thread, and publish claude-mcp.csv. Use Bash only to create the CSV from retrieved data. Then report SYNTHETIC_MCP_COMPLETE.'


def unprivileged():
    os.setgroups([]);os.setgid(1000);os.setuid(1000)
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
    resource.setrlimit(resource.RLIMIT_NPROC,(128,128))


def main():
    config=receive()
    if config['runtime']!='claude':raise ValueError('Native Claude qualification only')
    RUN.mkdir(mode=0o700)
    home,cwd=RUN/'home',RUN/'work'
    home.mkdir(mode=0o700);cwd.mkdir(mode=0o700)
    for path in (RUN,home,cwd):os.chown(path,1000,1000)
    capfile=RUN/'capabilities.json'
    fd=os.open(capfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(config['capabilities'],stream)
    os.chown(capfile,1000,1000)
    del config
    mcp={'mcpServers':{'mail':{'type':'stdio','command':'/usr/bin/env',
         'args':['-i','PATH=/usr/bin:/bin','LANG=C.UTF-8','/usr/bin/python3','-I','/tmp/runtime/guest_mail_mcp.py']}}}
    bridge=subprocess.Popen(['/usr/bin/python3',str(ROOT/'guest-vsock-bridge.py')],stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,preexec_fn=unprivileged)
    time.sleep(.3)
    env={'HOME':str(home),'PATH':'/tmp/runtime/bin:/usr/bin:/bin','LANG':'C.UTF-8','TERM':'dumb',
         'ANTHROPIC_API_KEY':'synthetic-not-a-real-key','ANTHROPIC_BASE_URL':'http://127.0.0.1:18080',
         'DISABLE_PROMPT_CACHING':'1','CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1',
         'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS':'1','CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK':'1',
         'CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL':'1','CLAUDE_CODE_DISABLE_AUTO_MEMORY':'1',
         'CLAUDE_CODE_DISABLE_THINKING':'1','CLAUDE_CONFIG_DIR':str(home/'claude')}
    argv=['/tmp/runtime/bin/claude','--bare','--setting-sources','','--strict-mcp-config','--mcp-config',
          json.dumps(mcp,separators=(',',':')),'--disable-slash-commands','--no-session-persistence','--model',MODEL,
          '--tools','Bash','--allowedTools','Bash,mcp__mail__*','--dangerously-skip-permissions','--max-turns','8',
          '--output-format','json','--system-prompt','You are a synthetic local MCP test.','-p',PROMPT]
    try:
        proc=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,preexec_fn=unprivileged,timeout=95)
        status={'runtime':'claude','returncode':proc.returncode,'completion':b'SYNTHETIC_MCP_COMPLETE' in proc.stdout,
                'csv_exists':(cwd/'claude-mcp.csv').is_file()}
        print('INNER_MAIL_MCP_RESULT '+json.dumps(status),flush=True)
        assert status['returncode']==0 and status['completion'] and status['csv_exists']
        print('INNER_MAIL_MCP_PASS',flush=True)
    finally:
        capfile.unlink(missing_ok=True)
        bridge.terminate();bridge.wait(timeout=5)


if __name__=='__main__':
    try:main()
    except Exception:
        print('INNER_MAIL_MCP_FAILED',flush=True)
        raise SystemExit(1)
