#!/usr/bin/env python3
"""Fixed synthetic workflow invoked by the actual model CLI's Bash tool."""
import csv
import json
import os
from pathlib import Path
import subprocess


def invoke(name, arguments):
    proc=subprocess.run(['/usr/bin/python3','-I','/tmp/runtime/guest_mail_tool_cli.py'],
        input=json.dumps({'name':name,'arguments':arguments}).encode(),capture_output=True,timeout=40,check=True)
    return json.loads(proc.stdout)


def main():
    runtime=os.environ['GMS_SYNTHETIC_RUNTIME']
    expected={'pi':('ALICE_ONLY_SUBJECT','ALICE_ONLY_BODY'),
              'claude':('BOB_ONLY_SUBJECT','BOB_ONLY_BODY')}[runtime]
    schema=invoke('describe_schema',{})
    assert 'messages' in schema and 'users' not in schema
    sql=invoke('sql_query_batch',{'queries':['SELECT id, subject FROM messages ORDER BY id','SELECT count(*) AS total FROM messages']})
    rows=sql['results'][0]['result']['rows']
    assert rows==[['shared-message',expected[0]]], 'SQL owner isolation failed'
    assert sql['results'][1]['result']['rows']==[[1]]
    thread=invoke('get_thread_batch',{'thread_ids':['shared-thread'],'body_format':'text'})
    messages=thread['results'][0]['result']['messages']
    assert len(messages)==1 and messages[0]['body_text']==expected[1]
    target=Path('/tmp/gms-run/work')/(runtime+'-mail.csv')
    with target.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['id','subject','body'])
        writer.writerow([rows[0][0],rows[0][1],messages[0]['body_text']])
    result=invoke('publish_artifact_batch',{'items':[{'path':target.name,'name':target.name}]})
    receipt=result['results'][0]['result']
    assert 'id' in receipt and receipt['size']==target.stat().st_size
    print(json.dumps({'runtime':runtime,'schema':True,'sql_count':1,'thread_count':1,'owner_subject':expected[0],
                      'artifact_id':receipt['id'],'artifact_bytes':receipt['size'],'workflow':'INNER_MAIL_WORKFLOW_PASS'}),flush=True)


if __name__=='__main__':main()
