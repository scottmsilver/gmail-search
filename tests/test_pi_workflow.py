import json

import pytest

from gmail_search.agents import pi_workflow as wf


def test_workflow_config_is_private_and_session_specific(tmp_path):
    models = tmp_path / "models.json"
    models.write_text('{"providers": {}}')
    mcp = tmp_path / "mcp.json"
    mcp.write_text('{"mcpServers":{"gmail":{"bearerToken":"!cat .session-token"}}}')
    one = wf.prepare_workflow(tmp_path, "eval-one", "s1", models_path=models, mcp_path=mcp, timeout=60)
    two = wf.prepare_workflow(tmp_path, "eval-two", "s2", models_path=models, mcp_path=mcp, timeout=60)
    assert one.env["PI_CODING_AGENT_DIR"] != two.env["PI_CODING_AGENT_DIR"]
    assert one.directory.stat().st_mode & 0o777 == 0o700
    config = json.loads((one.directory / "mcp.json").read_text())
    assert config["mcpServers"]["gmail"]["bearerToken"] == "!cat /workspaces/eval-one/.session-token"
    # pi-subagents resolves mcp: selectors from the agent directory, independent
    # of the adapter's explicit config-path option.
    assert json.loads((one.directory / "agent/mcp.json").read_text()) == config
    settings = json.loads((one.directory / "agent/settings.json").read_text())
    assert settings["subagents"]["modelScope"]["allow"] == ["inherit"]
    assert settings["subagents"]["disableBuiltins"] is True
    assert "telemetry.ts" in str(settings)


def test_unknown_profile_rejected(monkeypatch):
    monkeypatch.setenv("GMAIL_PI_WORKFLOW_PROFILE", "typo")
    with pytest.raises(ValueError, match="profile"):
        wf.resolve_profile(None)
    assert wf.resolve_profile("baseline") == "baseline"


def test_child_usage_is_deduplicated_and_parent_not_counted(tmp_path):
    events = [
        {"type": "message_end", "pi_session_id": "p", "parent": True, "event_id": "p:1", "usage": {"input": 999}},
        {"type": "message_end", "pi_session_id": "c", "parent": False, "event_id": "c:1", "provider": "openrouter", "model": "meta/muse-spark-1.3", "usage": {"input": 12, "output": 3, "cacheRead": 5, "cost": {"total": 0.02}}},
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events + events) + '\n{"partial"')
    loaded, complete = wf.read_trace(path)
    assert not complete
    totals = wf.child_usage(loaded)
    assert len(totals) == 1
    assert totals[0][0] == "openrouter/meta/muse-spark-1.3"
    assert totals[0][1].input_tokens == 12
    assert totals[0][1].cost_usd == 0.02


def test_unknown_usage_is_not_reported_as_free():
    totals = wf.child_usage([{"type": "message_end", "parent": False, "pi_session_id": "c", "event_id": "c:1", "provider": "p", "model": "m", "usage": {"input": 1}}])
    assert totals[0][1].cost_usd is None


def test_trace_completeness_requires_every_observed_session_to_close(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("")
    assert wf.read_trace(path) == ([], False)
    rows = [{"type": "session_start", "pi_session_id": "parent"},
            {"type": "session_end", "pi_session_id": "parent"},
            {"type": "message_end", "pi_session_id": "child"}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert wf.read_trace(path)[1] is False
    rows.extend([{"type": "session_start", "pi_session_id": "child"},
                 {"type": "session_end", "pi_session_id": "child"}])
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert wf.read_trace(path)[1] is True


def test_summary_usage_accounted_once_and_gap_prevents_complete(tmp_path):
    records = [
        {'type': 'session_start', 'pi_session_id': 'p'},
        {'type': 'summary_usage', 'parent': True, 'pi_session_id': 'p', 'event_id': 'summary:1', 'provider': 'p', 'model': 'm', 'usage': {'input': 5, 'output': 2, 'cost': {'total': .1}}},
        {'type': 'accounting_gap', 'pi_session_id': 'p'},
        {'type': 'session_end', 'pi_session_id': 'p'},
    ]
    path = tmp_path / 'trace.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in records))
    assert wf.read_trace(path)[1] is False
    assert wf.parent_usage(records)[0][1].input_tokens == 5
