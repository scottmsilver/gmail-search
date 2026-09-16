from gmail_search.gateway import browser_prompt


def test_followup_includes_server_answer_and_excludes_tool_payload_and_duplicate_question():
    rows=[{'role':'user','parts':[{'type':'text','text':'Find invoices'}]},
          {'role':'assistant','parts':[{'type':'data-deep-stage','data':{'private':'TOOL_PAYLOAD'}},
                                     {'type':'text','text':'Found two invoices'}]},
          {'role':'user','parts':[{'type':'text','text':'Total them'}]}]
    prompt=browser_prompt.build_prompt(rows,'Total them')
    assert 'Found two invoices' in prompt and 'Find invoices' in prompt
    assert prompt.count('Total them')==1
    assert 'TOOL_PAYLOAD' not in prompt


def test_prompt_keeps_latest_context_with_exact_utf8_budget():
    rows=[{'role':'user','parts':[{'type':'text','text':'OLD'+('é'*9000)}]},
          {'role':'assistant','parts':[{'type':'text','text':'Recent answer'}]}]
    prompt=browser_prompt.build_prompt(rows,'next')
    assert 'Recent answer' in prompt and 'OLD' not in prompt
    assert len(prompt.encode())<=16384
    question='é'*8192
    assert browser_prompt.build_prompt(rows,question)==question


def test_first_question_is_unchanged():
    assert browser_prompt.build_prompt([], 'hello')=='hello'
    assert browser_prompt.build_prompt([{'role':'user','parts':[{'type':'text','text':'hello'}]}], 'hello')=='hello'
