# Pi Deep Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth deep-mode backend named `pi` that runs one deep-analysis turn through the Pi agent harness in its own container, driven over pi's RPC protocol, with the existing MCP tools bridged by a pi extension.

**Architecture:** `service.py` routes `backend == "pi"` to `runtime_pi.pi_run()`, which spawns `docker exec -i pi-sandbox pi --mode rpc …` and maps pi's JSONL events into the same `agent_events` rows the UI already renders. A TypeScript extension inside the container turns every tool on the MCP tools server into a pi tool, injecting `session_id` from the environment. The event emitters shared with `claude_native` move into `agents/deep_events.py`.

**Tech Stack:** Python 3.12 / asyncio subprocess, FastAPI, psycopg; TypeScript extension on `@earendil-works/pi-coding-agent` + `@modelcontextprotocol/sdk` + `typebox`, tested with `bun test`; Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-02-pi-deep-backend-design.md`

## Global Constraints

- **No commits without Scott's password.** Every task ends with "run the suite and report"; Scott commits. Never `git commit` or `git push` on your own.
- **No hard-coded server URLs in code.** The MCP URL comes from `GMS_MCP_URL`; the extension throws if it is unset.
- **Small, well-named functions** (project CLAUDE.md). Nothing over ~30 lines; extract helpers.
- **Formatter strips unused imports.** Use inline imports or `# noqa: F401` where an import is only used indirectly.
- **Existing backends untouched.** `adk`, `claude_code`, `claude_native` keep their behaviour; the only edit to their code is the emitter extraction in Task 1 and the stream-helper extraction in Task 6, both pure refactors.
- **Container is the sandbox.** Model-driven shell runs only inside `pi-sandbox`. Pi is launched with `--no-extensions --no-skills --no-context-files --no-prompt-templates -e <our extension>`.
- **Secrets never on argv or in logs.** The service token and API key reach pi only through container env.
- **Env-var configuration, not config.yaml.** The agents package reads env vars everywhere (`GMAIL_CLAUDEBOX_*`); pi settings follow that pattern: `GMAIL_PI_MODEL`, `GMAIL_PI_THINKING`, `GMAIL_PI_CONTAINER`, `GMAIL_PI_HARD_TIMEOUT`, `GMAIL_PI_IDLE_TIMEOUT`, `GMAIL_PI_EXTENSION_PATH`. (Deviation from spec §4.4, which named `config.yaml` keys; env vars match the surrounding code.)
- **Every check-in updates README.md** (Task 11).
- Run Python tests with `uv run pytest -q`; run extension tests with `cd deploy/pi/extensions/gmail-tools && bun test`.

## File map

| Path | Responsibility |
|---|---|
| `src/gmail_search/agents/deep_events.py` (new) | Event emitters shared by `claude_native` and `pi`: plan, retriever/evidence, analyst/analysis, writer/final, error, auto-publish sweep. Moved from `runtime_claude_native.py`. |
| `src/gmail_search/agents/pi_protocol.py` (new) | Pure functions over pi RPC records: tool-call entries, assistant text, usage stats, argv builder. No I/O. |
| `src/gmail_search/agents/pi_rpc.py` (new) | `PiRpcClient`: spawn subprocess, send commands, read JSONL records with timeouts, request/response, abort/close. |
| `src/gmail_search/agents/runtime_pi.py` (new) | `pi_run()`: register session, drive one turn, map events to `agent_events`, cost, finalize. `PI_INSTRUCTION`. |
| `src/gmail_search/agents/runtime_claude_native.py` (modify) | Import emitters from `deep_events`; keep old private names as aliases. |
| `src/gmail_search/agents/cost.py` (modify) | `record_agent_cost(..., usd_override=None)`. |
| `src/gmail_search/agents/service.py` (modify) | `_VALID_BACKENDS`, `_record_cost` extras, `_stream_task_events` + `_finish_single_agent_turn` helpers, `pi` branch. |
| `src/gmail_search/agents/gc.py` (modify) | Delete `deploy/pi/sessions/<conv>.jsonl` when a conversation workspace is pruned. |
| `scripts/run_deep_compare.py` (modify) | Accept `pi` in `--backends`. |
| `deploy/pi/{Dockerfile,docker-compose.yml,setup.sh,start.sh,stop.sh,README.md,.gitignore}` (new) | Container. |
| `deploy/pi/extensions/gmail-tools/{index.ts,mcp-bridge.ts,mcp-bridge.test.ts,package.json,tsconfig.json}` (new) | MCP bridge extension. |
| `web/lib/config.ts`, `web/lib/chatSettings.ts`, `web/app/api/chat/route.ts`, `web/components/ModelPicker.tsx` (modify) | `"pi"` in the `DeepBackend` union and a fourth picker button. |
| `tests/test_deep_events.py`, `tests/test_pi_protocol.py`, `tests/test_pi_rpc.py`, `tests/fakes/fake_pi.py`, `tests/test_runtime_pi.py` (new); `tests/test_agent_service.py`, `tests/test_agent_gc.py`, `tests/test_agent_cost.py` (modify) | Tests. |
| `README.md` (modify) | Backend list, env knobs, module map, deploy list. |

---

### Task 0: Spike — confirm pi behaviour before writing repo code

Throwaway. Nothing from this task is kept except the notes file.

**Files:**
- Create: `docs/superpowers/plans/2026-09-02-pi-spike-notes.md`
- Scratch only: `~/.local/pi-spike/` (delete when done)

**Interfaces:**
- Produces: five recorded answers that later tasks read: (1) whether `--session <new path>` creates the file, (2) the exact `pi-coding-agent` version, (3) the pi model id string for the Claude model to use as `GMAIL_PI_MODEL` default, (4) the shape of `get_session_stats.data`, (5) the shape of `tool_execution_end.result` for an extension tool.

- [ ] **Step 1: Install pi into a scratch prefix**

```bash
mkdir -p ~/.local/pi-spike && cd ~/.local/pi-spike
npm init -y >/dev/null
npm install --ignore-scripts --save-exact @earendil-works/pi-coding-agent @modelcontextprotocol/sdk typebox
npx pi --version
npm view @earendil-works/pi-coding-agent version
```
Record both version strings in the notes file under "Versions".

- [ ] **Step 2: Log in or set a key**

Either `export ANTHROPIC_API_KEY=...` for the shell, or run `npx pi` interactively once and use `/login` → Claude Pro/Max, then `/exit`. Record which path was used.

- [ ] **Step 3: Answer question 1 (session file creation)**

```bash
cd ~/.local/pi-spike && rm -f /tmp/claude-1000/pi-spike-session.jsonl
printf '%s\n' '{"id":"p1","type":"prompt","message":"Reply with the single word OK."}' \
  | npx pi --mode rpc --no-extensions --no-skills --no-context-files --no-prompt-templates \
      --no-builtin-tools --session /tmp/claude-1000/pi-spike-session.jsonl 2>/dev/null \
  | jq -c 'select(.type=="agent_end" or .type=="response")'
ls -la /tmp/claude-1000/pi-spike-session.jsonl
```
Record: did the file get created? If pi exits before `agent_end` because stdin closed, re-run with `(printf ...; sleep 60) | npx pi ...` and note that the driver must keep stdin open until `agent_end`.

- [ ] **Step 4: Answer question 5 and 2 (model ids)**

```bash
(printf '%s\n' '{"id":"m","type":"get_available_models"}'; sleep 3) \
  | npx pi --mode rpc --no-extensions --no-skills --no-context-files --no-session 2>/dev/null \
  | jq -c 'select(.type=="response") | .data' | head -c 4000
```
Record the exact `provider/id` strings for the Sonnet and Opus entries under "Model ids". Pick the Sonnet one as the default for `GMAIL_PI_MODEL`.

- [ ] **Step 5: Write the throwaway MCP bridge extension**

`~/.local/pi-spike/gmail-tools.ts`:
```ts
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

export default async function (pi: ExtensionAPI) {
  const url = process.env.GMS_MCP_URL!;
  const token = process.env.GMAIL_MCP_SERVICE_TOKEN;
  const sessionId = process.env.GMS_SESSION_ID!;
  const transport = new StreamableHTTPClientTransport(
    new URL(url),
    token ? { requestInit: { headers: { Authorization: `Bearer ${token}` } } } : undefined,
  );
  const client = new Client({ name: "gmail-tools-spike", version: "0.0.0" });
  await client.connect(transport);
  const { tools } = await client.listTools();
  for (const t of tools) {
    const schema = structuredClone(t.inputSchema) as any;
    delete schema.properties?.session_id;
    schema.required = (schema.required ?? []).filter((r: string) => r !== "session_id");
    pi.registerTool({
      name: t.name,
      label: t.name,
      description: t.description ?? "",
      parameters: Type.Unsafe<Record<string, unknown>>(schema),
      async execute(_id, params) {
        const res: any = await client.callTool({ name: t.name, arguments: { session_id: sessionId, ...params } });
        const text = (res.content ?? []).filter((c: any) => c.type === "text").map((c: any) => c.text).join("\n");
        return { content: [{ type: "text", text }], details: res, isError: !!res.isError };
      },
    });
  }
  pi.registerCommand("tools", {
    description: "list tools",
    handler: async (_a, ctx) => ctx.ui.notify(pi.getAllTools().map((t) => t.name).join(","), "info"),
  });
}
```

- [ ] **Step 6: Register a session on the MCP server and answer questions 3 and 4 (real run)**

The MCP server must be running (`python -m gmail_search.agents.mcp_tools_server`). Register a throwaway session the same way `register_session_via_admin` does (see `runtime_claude.py:87`); the simplest route is a tiny Python one-liner in the repo venv:
```bash
cd /home/ssilver/development/gmail-search && uv run python - <<'PY'
import asyncio
from gmail_search.agents import runtime_claude as rc
asyncio.run(rc.register_session_via_admin("spike-1", evidence_records=None, conversation_id=None, workspace="spike", user_id=None))
print("registered")
PY
mkdir -p deploy/claudebox/workspaces/spike
```
Then run one real question and capture everything:
```bash
cd deploy/claudebox/workspaces/spike
source ../../.env   # for GMAIL_MCP_SERVICE_TOKEN
export GMS_MCP_URL=http://127.0.0.1:7878/mcp GMS_SESSION_ID=spike-1
(printf '%s\n' '{"id":"p1","type":"prompt","message":"List every hotel I stayed at in 2025 with check-in dates. Make a bar chart of nights per hotel with python and publish it."}'; \
 while ! grep -q '"agent_end"' /tmp/claude-1000/pi-spike-events.jsonl 2>/dev/null; do sleep 2; done; \
 printf '%s\n' '{"id":"s","type":"get_session_stats"}'; sleep 2) \
 | npx --prefix ~/.local/pi-spike pi --mode rpc --no-extensions --no-skills --no-context-files --no-prompt-templates \
     -e ~/.local/pi-spike/gmail-tools.ts --session /tmp/claude-1000/pi-spike-conv.jsonl \
     --model "<sonnet id from step 4>" 2>/tmp/claude-1000/pi-spike-stderr.log \
 | tee /tmp/claude-1000/pi-spike-events.jsonl >/dev/null
jq -c 'select(.type=="tool_execution_start") | {toolName, args}' /tmp/claude-1000/pi-spike-events.jsonl
jq -c 'select(.type=="tool_execution_end") | {toolName, isError, result: (.result|tostring|.[0:300])}' /tmp/claude-1000/pi-spike-events.jsonl | head -5
jq -c 'select(.type=="response" and .command=="get_session_stats") | .data' /tmp/claude-1000/pi-spike-events.jsonl
```
Record in the notes: the tool names called, whether a bash chart and a `publish_artifact_batch` call happened, the full `get_session_stats.data` object, one `tool_execution_end.result` object for an extension tool, and the wall time. Then unregister:
```bash
cd /home/ssilver/development/gmail-search && uv run python -c "import asyncio; from gmail_search.agents import runtime_claude as rc; asyncio.run(rc.unregister_session_via_admin('spike-1'))"
```

- [ ] **Step 7: Compare pi's cost against ours**

```bash
cd /home/ssilver/development/gmail-search && uv run python -c "
from gmail_search.agents.cost import estimate_agent_cost_usd
print(estimate_agent_cost_usd('<model id>', <input tokens>, <output tokens>))"
```
Record both numbers in a two-row table under "Cost".

- [ ] **Step 8: Write the notes file and clean up**

`docs/superpowers/plans/2026-09-02-pi-spike-notes.md` with headings: Versions, Session file creation, Model ids, Stats shape, Tool result shape, Real run (transcript summary), Cost. Then `rm -rf ~/.local/pi-spike deploy/claudebox/workspaces/spike`. Report the five answers to Scott before continuing. **Stop here if question 1 or the real run failed** and discuss.

---

### Task 1: Extract shared deep-mode event emitters into `deep_events.py`

**Files:**
- Create: `src/gmail_search/agents/deep_events.py`
- Modify: `src/gmail_search/agents/runtime_claude_native.py:161-388`
- Test: `tests/test_deep_events.py`

**Interfaces:**
- Produces (all in `gmail_search.agents.deep_events`):
  - `emit_plan_event(conn, session_id: str, *, agent_name: str = "native", approach: str) -> None`
  - `emit_retriever_events(conn, session_id, tool_calls: list[dict], *, skip_per_tool_emission: bool = False) -> None`
  - `emit_analyst_events(conn, session_id, tool_calls: list[dict], *, skip_per_tool_emission: bool = False) -> None`
  - `emit_writer_and_final(conn, session_id, text: str) -> None`
  - `emit_error(conn, session_id, exc: BaseException, *, agent_name: str = "native") -> None`
  - `sweep_and_extend_final_text(conn, *, session_id, workspace, conversation_id, turn_started_at: float, base_text: str) -> str`
  - `RETRIEVAL_TOOL_NAMES: frozenset[str]`

- [ ] **Step 1: Write the failing test**

`tests/test_deep_events.py`:
```python
"""deep_events: emitters shared by the single-agent deep backends."""

from __future__ import annotations

import json
from typing import Any

from gmail_search.agents import deep_events


class _FakeConn:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._seq = 0

    def execute(self, sql: str, params: tuple):
        if "insert into agent_events" in sql.lower():
            self._seq += 1
            session_id, _again, agent_name, kind, payload_json = params
            self.events.append({"agent_name": agent_name, "kind": kind, "payload": json.loads(payload_json)})
            return _Row({"seq": self._seq})
        return _Row(None)

    def commit(self) -> None:
        pass


class _Row:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


def test_plan_event_uses_supplied_agent_name_and_approach():
    conn = _FakeConn()
    deep_events.emit_plan_event(conn, "s1", agent_name="pi", approach="single pi loop")
    assert conn.events == [
        {"agent_name": "pi", "kind": "plan", "payload": {"native_mode": True, "approach": "single pi loop"}}
    ]


def test_retriever_events_emit_tool_calls_then_evidence():
    conn = _FakeConn()
    calls = [
        {"name": "search_emails", "args": {"q": "delta"}},
        {"name": "search_emails", "response": {"results": [{"cite_ref": "t1"}, {"cite_ref": "t2"}]}},
    ]
    deep_events.emit_retriever_events(conn, "s1", calls)
    kinds = [e["kind"] for e in conn.events]
    assert kinds == ["tool_call", "evidence"]
    assert conn.events[1]["payload"]["cite_refs"] == ["t1", "t2"]
    assert conn.events[1]["payload"]["summary"] == "Retrieval calls: 1× search_emails."


def test_retriever_events_skip_per_tool_when_streamed():
    conn = _FakeConn()
    calls = [{"name": "search_emails", "args": {"q": "x"}}]
    deep_events.emit_retriever_events(conn, "s1", calls, skip_per_tool_emission=True)
    assert [e["kind"] for e in conn.events] == ["evidence"]


def test_analyst_events_silent_without_run_code():
    conn = _FakeConn()
    deep_events.emit_analyst_events(conn, "s1", [{"name": "search_emails", "args": {}}])
    assert conn.events == []


def test_analyst_events_collect_artifact_ids():
    conn = _FakeConn()
    calls = [
        {"name": "run_code", "args": {"code": "print(1)"}},
        {"name": "run_code", "response": {"artifacts": [{"id": 7}]}},
    ]
    deep_events.emit_analyst_events(conn, "s1", calls, skip_per_tool_emission=True)
    assert [e["kind"] for e in conn.events] == ["analysis"]
    assert conn.events[0]["payload"]["artifact_ids"] == [7]
    assert conn.events[0]["payload"]["called_run_code"] is True


def test_writer_and_final_carry_same_text():
    conn = _FakeConn()
    deep_events.emit_writer_and_final(conn, "s1", "answer")
    assert [(e["agent_name"], e["kind"]) for e in conn.events] == [("writer", "draft"), ("root", "final")]
    assert all(e["payload"] == {"text": "answer"} for e in conn.events)


def test_error_event_uses_agent_name():
    conn = _FakeConn()
    deep_events.emit_error(conn, "s1", RuntimeError("boom"), agent_name="pi")
    assert conn.events == [{"agent_name": "pi", "kind": "error", "payload": {"message": "boom"}}]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_deep_events.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'gmail_search.agents.deep_events'`

- [ ] **Step 3: Create `deep_events.py` by moving the code**

Move these from `runtime_claude_native.py` verbatim, renaming the public ones (drop the leading underscore) and adding the two keyword parameters:

```python
"""Event emitters shared by the single-agent deep backends
(`claude_native`, `pi`).

Both backends run ONE agent loop with every tool available and then
synthesize the orchestrator's event vocabulary (`plan` / `tool_call` /
`evidence` / `code_run` / `analysis` / `draft` / `final`) so the UI's
deep-mode panels render without change. This module owns that
synthesis; the runtimes own only how they drive their agent.
"""

from __future__ import annotations

import logging

from gmail_search.agents.orchestration import _artifact_ids_from_tool_calls, _cite_refs_from_tool_calls
from gmail_search.agents.session import append_event

logger = logging.getLogger(__name__)

# Names of the retrieval tools the UI surfaces under the "retriever"
# agent. `run_code` is treated separately as the Analyst's tool.
RETRIEVAL_TOOL_NAMES = frozenset({"search_emails", "query_emails", "get_thread", "sql_query"})


def _is_args_entry(tc: dict) -> bool:
    return "args" in tc and "response" not in tc


def _is_response_entry(tc: dict) -> bool:
    return "response" in tc


def _retrieval_args_entries(tool_calls: list[dict]) -> list[dict]:
    return [tc for tc in tool_calls if _is_args_entry(tc) and tc.get("name") in RETRIEVAL_TOOL_NAMES]


def _run_code_response_entries(tool_calls: list[dict]) -> list[dict]:
    return [tc for tc in tool_calls if _is_response_entry(tc) and tc.get("name") == "run_code"]


def _has_run_code(tool_calls: list[dict]) -> bool:
    return any(tc.get("name") == "run_code" for tc in tool_calls)


def _retriever_summary(retrieval_calls: list[dict]) -> str:
    if not retrieval_calls:
        return "No retrieval tools invoked."
    by_name: dict[str, int] = {}
    for tc in retrieval_calls:
        n = str(tc.get("name") or "?")
        by_name[n] = by_name.get(n, 0) + 1
    parts = [f"{count}× {name}" for name, count in sorted(by_name.items())]
    return f"Retrieval calls: {', '.join(parts)}."


def _analyst_summary(run_code_calls: list[dict], artifact_ids: list[int]) -> str:
    if not run_code_calls:
        return "No code execution."
    art_blurb = f", produced {len(artifact_ids)} artifact(s)" if artifact_ids else ""
    return f"Ran {len(run_code_calls)} code block(s){art_blurb}."


def emit_plan_event(conn, session_id: str, *, agent_name: str = "native", approach: str) -> None:
    append_event(
        conn,
        session_id=session_id,
        agent_name=agent_name,
        kind="plan",
        payload={"native_mode": True, "approach": approach},
    )
```
…followed by `emit_retriever_events`, `emit_analyst_events`, `emit_writer_and_final`, `emit_error` (with `agent_name` keyword, default `"native"`), and `sweep_and_extend_final_text`, each copied from `runtime_claude_native.py` with only the name changes. Keep the docstrings.

- [ ] **Step 4: Point `runtime_claude_native.py` at the new module**

Replace the moved definitions (from `_RETRIEVAL_TOOL_NAMES` through `_sweep_and_extend_final_text`) with:
```python
from gmail_search.agents.deep_events import (  # noqa: F401
    RETRIEVAL_TOOL_NAMES as _RETRIEVAL_TOOL_NAMES,
    emit_analyst_events as _emit_analyst_events,
    emit_error as _emit_error,
    emit_retriever_events as _emit_retriever_events,
    emit_writer_and_final as _emit_writer_and_final,
    sweep_and_extend_final_text as _sweep_and_extend_final_text,
)


def _emit_plan_event(conn, session_id: str) -> None:
    from gmail_search.agents.deep_events import emit_plan_event

    emit_plan_event(conn, session_id, approach="single-agent claude code loop with all tools")
```
Remove the now-unused imports of `_artifact_ids_from_tool_calls` / `_cite_refs_from_tool_calls` from `runtime_claude_native.py` if nothing else there uses them (check with `grep -n "_cite_refs_from_tool_calls\|_artifact_ids_from_tool_calls" src/gmail_search/agents/runtime_claude_native.py`).

- [ ] **Step 5: Run the new and existing tests**

Run: `uv run pytest tests/test_deep_events.py tests/test_runtime_claude_native.py tests/test_agent_service.py -q`
Expected: all PASS. If `test_runtime_claude_native.py` monkeypatches any `native._emit_*` name, the alias keeps that working; if it fails, check the alias list against the names the test patches.

- [ ] **Step 6: Run the full suite and report**

Run: `uv run pytest -q`
Expected: PASS. Report to Scott; do not commit.

---

### Task 2: `pi_protocol.py` — pure parsing of pi RPC records

**Files:**
- Create: `src/gmail_search/agents/pi_protocol.py`
- Test: `tests/test_pi_protocol.py`

**Interfaces:**
- Produces:
  - `RESPONSE_CLIP_CHARS = 4000`
  - `tool_call_args_entry(ev: dict) -> dict` → `{"name": str, "args": dict}` from a `tool_execution_start` record
  - `tool_call_response_entry(ev: dict, *, clip: int = RESPONSE_CLIP_CHARS) -> dict` → `{"name", "response": {"text": str, "is_error": bool}}` from a `tool_execution_end` record
  - `assistant_text(ev: dict) -> str | None` → joined text blocks from an assistant `message_end`, else None
  - `bash_as_run_code(start: dict, end: dict, *, clip: int = RESPONSE_CLIP_CHARS) -> list[dict]` → `[{"name":"run_code","args":{"code": cmd}}, {"name":"run_code","response":{"stdout": text, "artifacts": []}}]`
  - `@dataclass UsageStats(input_tokens: int, output_tokens: int, cache_read_tokens: int, cache_write_tokens: int, cost_usd: float | None)`
  - `usage_from_stats_response(resp: dict) -> UsageStats`
  - `build_pi_argv(*, container: str, session_id: str, workspace: str, session_path: str | None, extension_path: str, model: str, thinking: str | None, system_prompt: str) -> list[str]`

- [ ] **Step 1: Write the failing tests**

`tests/test_pi_protocol.py`:
```python
"""Pure parsers for pi RPC records. No I/O."""

from __future__ import annotations

from gmail_search.agents import pi_protocol as pp


def test_tool_call_args_entry_copies_name_and_args():
    ev = {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "search_emails_batch", "args": {"searches": [{"query": "x"}]}}
    assert pp.tool_call_args_entry(ev) == {"name": "search_emails_batch", "args": {"searches": [{"query": "x"}]}}


def test_tool_call_args_entry_wraps_non_dict_args():
    ev = {"type": "tool_execution_start", "toolName": "bash", "args": "ls"}
    assert pp.tool_call_args_entry(ev) == {"name": "bash", "args": {"value": "ls"}}


def test_tool_call_response_entry_joins_text_and_clips():
    ev = {
        "type": "tool_execution_end",
        "toolName": "bash",
        "isError": False,
        "result": {"content": [{"type": "text", "text": "a" * 10}, {"type": "text", "text": "b"}]},
    }
    entry = pp.tool_call_response_entry(ev, clip=6)
    assert entry["name"] == "bash"
    assert entry["response"]["text"] == "aaaaaa"
    assert entry["response"]["is_error"] is False


def test_tool_call_response_entry_flags_error():
    ev = {"type": "tool_execution_end", "toolName": "x", "isError": True, "result": {"content": []}}
    assert pp.tool_call_response_entry(ev)["response"]["is_error"] is True


def test_assistant_text_joins_text_blocks_only():
    ev = {
        "type": "message_end",
        "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Hi"}, {"type": "text", "text": " there"}]},
    }
    assert pp.assistant_text(ev) == "Hi there"


def test_assistant_text_none_for_user_or_empty():
    assert pp.assistant_text({"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "q"}]}}) is None
    assert pp.assistant_text({"type": "message_end", "message": {"role": "assistant", "content": [{"type": "toolCall"}]}}) is None
    assert pp.assistant_text({"type": "turn_end"}) is None


def test_bash_as_run_code_produces_args_then_response():
    start = {"type": "tool_execution_start", "toolName": "bash", "args": {"command": "python plot.py"}}
    end = {"type": "tool_execution_end", "toolName": "bash", "isError": False, "result": {"content": [{"type": "text", "text": "saved chart.png"}]}}
    assert pp.bash_as_run_code(start, end) == [
        {"name": "run_code", "args": {"code": "python plot.py"}},
        {"name": "run_code", "response": {"stdout": "saved chart.png", "artifacts": []}},
    ]


def test_usage_from_stats_response():
    resp = {"type": "response", "command": "get_session_stats", "success": True,
            "data": {"tokens": {"input": 10, "output": 4, "cacheRead": 3, "cacheWrite": 1}, "cost": 0.25}}
    u = pp.usage_from_stats_response(resp)
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens, u.cost_usd) == (10, 4, 3, 1, 0.25)


def test_usage_from_stats_response_tolerates_missing_fields():
    u = pp.usage_from_stats_response({"type": "response", "success": True, "data": {}})
    assert (u.input_tokens, u.output_tokens, u.cost_usd) == (0, 0, None)


def test_build_pi_argv_shape():
    argv = pp.build_pi_argv(
        container="pi-sandbox", session_id="s1", workspace="deep-conv-c1",
        session_path="/sessions/c1.jsonl", extension_path="/opt/gmail-tools",
        model="anthropic/claude-x", thinking="medium", system_prompt="SYS",
    )
    assert argv[:3] == ["docker", "exec", "-i"]
    assert "-e" in argv and "GMS_SESSION_ID=s1" in argv
    assert argv[argv.index("-w") + 1] == "/workspaces/deep-conv-c1"
    assert "pi-sandbox" in argv
    tail = argv[argv.index("pi"):]
    assert tail[:3] == ["pi", "--mode", "rpc"]
    for flag in ("--no-extensions", "--no-skills", "--no-context-files", "--no-prompt-templates"):
        assert flag in tail
    assert tail[tail.index("-e") + 1] == "/opt/gmail-tools"
    assert tail[tail.index("--session") + 1] == "/sessions/c1.jsonl"
    assert tail[tail.index("--model") + 1] == "anthropic/claude-x"
    assert tail[tail.index("--thinking") + 1] == "medium"
    assert tail[tail.index("--system-prompt") + 1] == "SYS"


def test_build_pi_argv_without_session_uses_no_session():
    argv = pp.build_pi_argv(container="c", session_id="s", workspace="w", session_path=None,
                            extension_path="/x", model="m", thinking=None, system_prompt="p")
    assert "--no-session" in argv and "--session" not in argv and "--thinking" not in argv
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pi_protocol.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `pi_protocol.py`**

```python
"""Pure helpers over pi's RPC/JSONL records.

Everything here is a function of its inputs: no subprocess, no DB,
no clock. `pi_rpc.py` owns the process; `runtime_pi.py` owns the turn.
Shapes follow packages/coding-agent/docs/rpc.md in the pi repo.
"""

from __future__ import annotations

from dataclasses import dataclass

RESPONSE_CLIP_CHARS = 4000

_PI_ISOLATION_FLAGS = ("--no-extensions", "--no-skills", "--no-context-files", "--no-prompt-templates")


def _as_dict(value) -> dict:
    return dict(value) if isinstance(value, dict) else {"value": value}


def _text_of_content(content) -> str:
    if not isinstance(content, list):
        return ""
    return "".join(str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def tool_call_args_entry(ev: dict) -> dict:
    return {"name": str(ev.get("toolName") or ""), "args": _as_dict(ev.get("args") or {})}


def tool_call_response_entry(ev: dict, *, clip: int = RESPONSE_CLIP_CHARS) -> dict:
    result = ev.get("result") if isinstance(ev.get("result"), dict) else {}
    text = _text_of_content(result.get("content"))[:clip]
    return {"name": str(ev.get("toolName") or ""), "response": {"text": text, "is_error": bool(ev.get("isError"))}}


def assistant_text(ev: dict) -> str | None:
    if ev.get("type") != "message_end":
        return None
    message = ev.get("message") if isinstance(ev.get("message"), dict) else {}
    if message.get("role") != "assistant":
        return None
    text = _text_of_content(message.get("content"))
    return text or None


def bash_as_run_code(start: dict, end: dict, *, clip: int = RESPONSE_CLIP_CHARS) -> list[dict]:
    """Present a pi `bash` call in the `run_code` shape the analyst
    panel understands. `artifacts` is always empty: files reach the
    UI through `publish_artifact_batch` or the auto-publish sweep."""
    command = str(_as_dict(start.get("args") or {}).get("command") or "")
    stdout = tool_call_response_entry(end, clip=clip)["response"]["text"]
    return [
        {"name": "run_code", "args": {"code": command}},
        {"name": "run_code", "response": {"stdout": stdout, "artifacts": []}},
    ]


@dataclass(frozen=True)
class UsageStats:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None


def usage_from_stats_response(resp: dict) -> UsageStats:
    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    cost = data.get("cost")
    return UsageStats(
        input_tokens=int(tokens.get("input") or 0),
        output_tokens=int(tokens.get("output") or 0),
        cache_read_tokens=int(tokens.get("cacheRead") or 0),
        cache_write_tokens=int(tokens.get("cacheWrite") or 0),
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
    )


def _session_flags(session_path: str | None) -> list[str]:
    return ["--session", session_path] if session_path else ["--no-session"]


def _thinking_flags(thinking: str | None) -> list[str]:
    return ["--thinking", thinking] if thinking else []


def build_pi_argv(
    *,
    container: str,
    session_id: str,
    workspace: str,
    session_path: str | None,
    extension_path: str,
    model: str,
    thinking: str | None,
    system_prompt: str,
) -> list[str]:
    """argv for one turn. Secrets are NOT here — the service token and
    provider key live in the container's own environment."""
    return [
        "docker", "exec", "-i",
        "-e", f"GMS_SESSION_ID={session_id}",
        "-w", f"/workspaces/{workspace}",
        container,
        "pi", "--mode", "rpc",
        *_PI_ISOLATION_FLAGS,
        "-e", extension_path,
        *_session_flags(session_path),
        "--model", model,
        *_thinking_flags(thinking),
        "--system-prompt", system_prompt,
    ]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_pi_protocol.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Full suite and report**

Run: `uv run pytest -q`. Report; no commit.

---

### Task 3: `pi_rpc.py` — subprocess client with a fake pi for tests

**Files:**
- Create: `src/gmail_search/agents/pi_rpc.py`
- Create: `tests/fakes/__init__.py` (empty), `tests/fakes/fake_pi.py`
- Test: `tests/test_pi_rpc.py`

**Interfaces:**
- Produces (`gmail_search.agents.pi_rpc`):
  - `class PiRpcError(RuntimeError)`
  - `class PiRpcClient` with `@classmethod async spawn(argv: list[str]) -> PiRpcClient`; `async send(command: dict) -> None`; `async read_record(timeout: float) -> dict | None` (None on EOF; raises `asyncio.TimeoutError`); `async request(command: dict, *, timeout: float) -> dict` (adds an `id`, returns the matching `response` record, stashes any other records in `self.stray: list[dict]`); `async abort_and_close(*, grace: float = 5.0) -> None`; `async close() -> None`; property `returncode`.
- `tests/fakes/fake_pi.py`: a stand-in pi process. Reads `FAKE_PI_SCRIPT` (path to a JSON file: `{"events": [...records...], "stats": {...data...}, "delay_before_end": 0.0}`); on a `prompt` command prints each event line, sleeping `delay_before_end` before `agent_end`; on `get_session_stats` prints a response with the given data; on `abort` prints `agent_end` and exits.

- [ ] **Step 1: Write the fake pi**

`tests/fakes/fake_pi.py`:
```python
"""Scripted stand-in for `pi --mode rpc`, driven by FAKE_PI_SCRIPT."""

from __future__ import annotations

import json
import os
import sys
import time


def _load_script() -> dict:
    with open(os.environ["FAKE_PI_SCRIPT"]) as f:
        return json.load(f)


def _emit(record: dict) -> None:
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


def _handle_prompt(cmd: dict, script: dict) -> None:
    _emit({"id": cmd.get("id"), "type": "response", "command": "prompt", "success": True})
    for ev in script.get("events", []):
        if ev.get("type") == "agent_end":
            time.sleep(float(script.get("delay_before_end", 0.0)))
        _emit(ev)


def _handle_stats(cmd: dict, script: dict) -> None:
    _emit({"id": cmd.get("id"), "type": "response", "command": "get_session_stats", "success": True, "data": script.get("stats", {})})


def main() -> None:
    script = _load_script()
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        if not line:
            continue
        cmd = json.loads(line)
        kind = cmd.get("type")
        if kind == "prompt":
            _handle_prompt(cmd, script)
        elif kind == "get_session_stats":
            _handle_stats(cmd, script)
        elif kind == "abort":
            _emit({"id": cmd.get("id"), "type": "response", "command": "abort", "success": True})
            _emit({"type": "agent_end", "messages": []})
            return
    return


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing tests**

`tests/test_pi_rpc.py`:
```python
"""PiRpcClient against the scripted fake pi process."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from gmail_search.agents import pi_rpc

FAKE_PI = Path(__file__).parent / "fakes" / "fake_pi.py"


def _write_script(tmp_path: Path, events: list[dict], stats: dict | None = None, delay: float = 0.0) -> Path:
    p = tmp_path / "script.json"
    p.write_text(json.dumps({"events": events, "stats": stats or {}, "delay_before_end": delay}))
    return p


def _argv() -> list[str]:
    return [sys.executable, str(FAKE_PI)]


def test_prompt_streams_events_until_agent_end(tmp_path, monkeypatch):
    script = _write_script(tmp_path, [{"type": "agent_start"}, {"type": "agent_end", "messages": []}])
    monkeypatch.setenv("FAKE_PI_SCRIPT", str(script))

    async def run():
        client = await pi_rpc.PiRpcClient.spawn(_argv())
        await client.send({"id": "p1", "type": "prompt", "message": "hi"})
        seen = []
        while True:
            rec = await client.read_record(timeout=5.0)
            assert rec is not None
            seen.append(rec["type"])
            if rec["type"] == "agent_end":
                break
        await client.close()
        return seen

    assert asyncio.run(run()) == ["response", "agent_start", "agent_end"]


def test_request_returns_matching_response_and_stashes_strays(tmp_path, monkeypatch):
    script = _write_script(tmp_path, [{"type": "agent_end", "messages": []}], stats={"cost": 0.5})
    monkeypatch.setenv("FAKE_PI_SCRIPT", str(script))

    async def run():
        client = await pi_rpc.PiRpcClient.spawn(_argv())
        await client.send({"type": "prompt", "message": "hi"})  # response + agent_end become strays
        resp = await client.request({"type": "get_session_stats"}, timeout=5.0)
        await client.close()
        return resp, [r["type"] for r in client.stray]

    resp, strays = asyncio.run(run())
    assert resp["command"] == "get_session_stats" and resp["data"] == {"cost": 0.5}
    assert strays == ["response", "agent_end"]


def test_read_record_times_out_when_idle(tmp_path, monkeypatch):
    script = _write_script(tmp_path, [{"type": "agent_end", "messages": []}], delay=2.0)
    monkeypatch.setenv("FAKE_PI_SCRIPT", str(script))

    async def run():
        client = await pi_rpc.PiRpcClient.spawn(_argv())
        await client.send({"type": "prompt", "message": "hi"})
        await client.read_record(timeout=5.0)  # prompt response
        with pytest.raises(asyncio.TimeoutError):
            await client.read_record(timeout=0.2)
        await client.abort_and_close(grace=3.0)
        return client.returncode

    assert asyncio.run(run()) == 0


def test_read_record_returns_none_on_eof(tmp_path, monkeypatch):
    script = _write_script(tmp_path, [])
    monkeypatch.setenv("FAKE_PI_SCRIPT", str(script))

    async def run():
        client = await pi_rpc.PiRpcClient.spawn(_argv())
        await client.close()  # closes stdin → fake exits
        return await client.read_record(timeout=5.0)

    assert asyncio.run(run()) is None


def test_malformed_line_is_skipped(tmp_path, monkeypatch):
    script = _write_script(tmp_path, [{"type": "agent_end", "messages": []}])
    monkeypatch.setenv("FAKE_PI_SCRIPT", str(script))

    async def run():
        client = await pi_rpc.PiRpcClient.spawn(_argv())
        client._inject_line(b"not json\n")
        await client.send({"type": "prompt", "message": "hi"})
        first = await client.read_record(timeout=5.0)
        await client.close()
        return first["type"]

    assert asyncio.run(run()) == "response"
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_pi_rpc.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `pi_rpc.py`**

```python
"""Minimal client for `pi --mode rpc` over a subprocess's stdin/stdout.

Protocol (pi repo, packages/coding-agent/docs/rpc.md): one JSON object
per line each way, LF-delimited only. Commands may carry an `id`; the
matching `response` record echoes it. Agent events stream on stdout
in between. asyncio's `readline()` splits on b"\\n" only, which is
exactly what the protocol requires.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging

logger = logging.getLogger(__name__)

_ids = itertools.count(1)


class PiRpcError(RuntimeError):
    """The pi process died or answered a request with success=false."""


class PiRpcClient:
    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc
        self._pending: list[bytes] = []  # test seam for injected lines
        self.stray: list[dict] = []

    @classmethod
    async def spawn(cls, argv: list[str]) -> "PiRpcClient":
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return cls(proc)

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    def _inject_line(self, line: bytes) -> None:
        self._pending.append(line)

    async def send(self, command: dict) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(command).encode() + b"\n")
        await self._proc.stdin.drain()

    async def _next_line(self, timeout: float) -> bytes:
        if self._pending:
            return self._pending.pop(0)
        assert self._proc.stdout is not None
        return await asyncio.wait_for(self._proc.stdout.readline(), timeout)

    async def read_record(self, timeout: float) -> dict | None:
        """Next parsed record, None on EOF. Malformed lines are logged
        and skipped. Raises asyncio.TimeoutError when nothing arrives
        within `timeout` seconds."""
        while True:
            line = await self._next_line(timeout)
            if not line:
                return None
            record = _parse_line(line)
            if record is not None:
                return record

    async def request(self, command: dict, *, timeout: float) -> dict:
        req_id = f"req-{next(_ids)}"
        await self.send({**command, "id": req_id})
        while True:
            rec = await self.read_record(timeout)
            if rec is None:
                raise PiRpcError(f"pi exited before answering {command.get('type')}")
            if rec.get("type") == "response" and rec.get("id") == req_id:
                if not rec.get("success", False):
                    raise PiRpcError(f"pi rejected {command.get('type')}: {rec.get('error')}")
                return rec
            self.stray.append(rec)

    async def close(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.is_closing():
            self._proc.stdin.close()
        try:
            await asyncio.wait_for(self._proc.wait(), 5.0)
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()

    async def abort_and_close(self, *, grace: float = 5.0) -> None:
        """Ask pi to stop, wait up to `grace` for it to exit, then kill."""
        try:
            await self.send({"type": "abort"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), grace)
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()
        await self._drain_stderr()

    async def _drain_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        try:
            tail = await asyncio.wait_for(self._proc.stderr.read(4000), 1.0)
        except asyncio.TimeoutError:
            return
        if tail:
            logger.warning("pi stderr tail: %s", tail.decode(errors="replace"))


def _parse_line(line: bytes) -> dict | None:
    text = line.rstrip(b"\r\n")
    if not text:
        return None
    try:
        record = json.loads(text)
    except ValueError:
        logger.warning("pi rpc: skipping malformed line: %r", text[:200])
        return None
    return record if isinstance(record, dict) else None
```

Note on `abort_and_close` and `docker exec`: killing the host-side `docker exec` client does not always kill the in-container process. Task 5 adds a best-effort in-container `pkill` after a kill.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_pi_rpc.py -q`
Expected: PASS (5 tests). If `test_read_record_times_out_when_idle` is flaky on a slow machine, raise the fake's delay to 3.0 in that test.

- [ ] **Step 6: Full suite and report**

Run: `uv run pytest -q`. Report; no commit.

---

### Task 4: Cost sink accepts a provider-reported USD figure and extra token fields

**Files:**
- Modify: `src/gmail_search/agents/cost.py:86-121`
- Modify: `src/gmail_search/agents/service.py:724-747` (`_record_cost`)
- Test: `tests/test_agent_cost.py` (add), `tests/test_agent_service.py` (add)

**Interfaces:**
- Produces: `record_agent_cost(conn, *, session_id, agent_name, model, input_tokens, output_tokens, usd_override: float | None = None) -> float` — when `usd_override` is not None it is stored and returned instead of the pricing-table estimate.
- `service._record_cost(*, agent_name, model, input_tokens, output_tokens, usd_override=None, **extra)` — `extra` keys are merged into the `cost` event payload.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_cost.py` (create the file if it does not exist, with the same `_FakeConn` pattern as `tests/test_deep_events.py` but capturing `record_cost` via monkeypatch):
```python
def test_record_agent_cost_uses_override_when_given(monkeypatch):
    from gmail_search.agents import cost as cost_mod

    captured = {}

    def fake_record_cost(conn, **kw):
        captured.update(kw)

    monkeypatch.setattr(cost_mod, "record_cost", fake_record_cost)
    usd = cost_mod.record_agent_cost(
        object(), session_id="s1", agent_name="pi", model="anthropic/x",
        input_tokens=1000, output_tokens=10, usd_override=0.42,
    )
    assert usd == 0.42
    assert captured["estimated_cost_usd"] == 0.42
    assert captured["operation"] == "deep_pi"


def test_record_agent_cost_estimates_without_override(monkeypatch):
    from gmail_search.agents import cost as cost_mod

    monkeypatch.setattr(cost_mod, "record_cost", lambda conn, **kw: None)
    usd = cost_mod.record_agent_cost(object(), session_id="s1", agent_name="x", model="gemini-2.5-flash",
                                     input_tokens=1_000_000, output_tokens=0)
    assert usd == 0.075
```
And in `tests/test_agent_service.py`, a test that the sink merges extras into the event payload. The sink is a closure inside `_real_run`, so test it through the `pi` branch in Task 6 instead; here only the `cost.py` tests are added.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_agent_cost.py -q`
Expected: FAIL with `TypeError: record_agent_cost() got an unexpected keyword argument 'usd_override'`

- [ ] **Step 3: Implement**

In `cost.py`, change the signature and first line of the body:
```python
def record_agent_cost(
    conn,
    *,
    session_id: str,
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    usd_override: float | None = None,
) -> float:
    """... (existing docstring) ...

    `usd_override`, when given, is a provider-reported figure (pi's
    `get_session_stats.cost`) and replaces the pricing-table estimate,
    which only knows Gemini rates."""
    usd = usd_override if usd_override is not None else estimate_agent_cost_usd(model, input_tokens, output_tokens)
```
In `service.py` `_record_cost`:
```python
    def _record_cost(
        *,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        usd_override: float | None = None,
        **extra,
    ) -> None:
        nonlocal turn_cost_usd
        usd = record_agent_cost(
            conn,
            session_id=session_id,
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd_override=usd_override,
        )
        turn_cost_usd += usd
        append_event(
            conn,
            session_id=session_id,
            agent_name=agent_name,
            kind="cost",
            payload={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "usd": round(usd, 5),
                "turn_total_usd": round(turn_cost_usd, 5),
                **extra,
            },
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_agent_cost.py tests/test_agent_service.py -q`
Expected: PASS

- [ ] **Step 5: Full suite and report**

Run: `uv run pytest -q`. Report; no commit.

---

### Task 5: `runtime_pi.py` — `pi_run()`

**Files:**
- Create: `src/gmail_search/agents/runtime_pi.py`
- Test: `tests/test_runtime_pi.py`

**Interfaces:**
- Consumes: `PiRpcClient` (Task 3), `pi_protocol.*` (Task 2), `deep_events.*` (Task 1), `runtime_claude.register_session_via_admin / unregister_session_via_admin / _fetch_structured_tool_calls / _tool_calls_from_side_channel`, `session.finalize_session`, `session.append_event`, `store.db.get_connection`.
- Produces:
  - `async pi_run(*, db_path: Path, session_id: str, workspace: str, conversation_id: str | None, question: str, model: str | None, cost_sink: Callable[..., None] | None, user_id: str | None = None) -> None`
  - `PI_INSTRUCTION: str`
  - `pi_model() -> str`, `pi_thinking() -> str | None`, `pi_container() -> str`, `pi_extension_path() -> str`, `hard_timeout_seconds() -> float`, `idle_timeout_seconds() -> float`, `session_path_for(conversation_id: str | None) -> str | None`
  - `@dataclass TurnOutcome(final_text: str, local_tool_calls: list[dict], usage: UsageStats | None)`
  - `async drive_turn(client, question: str, *, on_tool_event: Callable[[str, dict], Awaitable[None]], hard_timeout: float, idle_timeout: float) -> TurnOutcome` — the loop, separated so tests can feed a fake client.
  - Test seam: `_spawn_client(argv) -> PiRpcClient` module function that tests monkeypatch.

- [ ] **Step 1: Write the failing tests**

`tests/test_runtime_pi.py`:
```python
"""pi_run: one deep turn through a fake PiRpcClient."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from gmail_search.agents import runtime_claude as rc
from gmail_search.agents import runtime_pi


class _FakeConn:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.finalized: list[dict[str, Any]] = []
        self.closed = False
        self._seq = 0

    def execute(self, sql: str, params: tuple):
        low = sql.lower()
        if "insert into agent_events" in low:
            self._seq += 1
            session_id, _a, agent_name, kind, payload_json = params
            self.events.append({"agent_name": agent_name, "kind": kind, "payload": json.loads(payload_json)})
            return _Row({"seq": self._seq})
        if "update agent_sessions" in low:
            status, final_answer, session_id = params
            self.finalized.append({"status": status, "final_answer": final_answer})
        return _Row(None)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _Row:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeClient:
    """Replays scripted records; records commands sent."""

    def __init__(self, records: list[dict], stats: dict | None = None, *, hang_after: int | None = None) -> None:
        self._records = list(records)
        self._stats = stats or {}
        self._hang_after = hang_after
        self.sent: list[dict] = []
        self.aborted = False
        self.closed = False
        self.stray: list[dict] = []
        self._served = 0

    async def send(self, command: dict) -> None:
        self.sent.append(command)

    async def read_record(self, timeout: float) -> dict | None:
        if self._hang_after is not None and self._served >= self._hang_after:
            await asyncio.sleep(timeout)
            raise asyncio.TimeoutError()
        if not self._records:
            return None
        self._served += 1
        return self._records.pop(0)

    async def request(self, command: dict, *, timeout: float) -> dict:
        self.sent.append(command)
        return {"type": "response", "command": command["type"], "success": True, "data": self._stats}

    async def abort_and_close(self, *, grace: float = 5.0) -> None:
        self.aborted = True
        self.closed = True

    async def close(self) -> None:
        self.closed = True


def _happy_records() -> list[dict]:
    return [
        {"type": "response", "command": "prompt", "success": True},
        {"type": "agent_start"},
        {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "search_emails_batch", "args": {"searches": [{"query": "hotel"}]}},
        {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "search_emails_batch", "isError": False,
         "result": {"content": [{"type": "text", "text": "{\"results\": []}"}]}},
        {"type": "tool_execution_start", "toolCallId": "c2", "toolName": "bash", "args": {"command": "python plot.py"}},
        {"type": "tool_execution_end", "toolCallId": "c2", "toolName": "bash", "isError": False,
         "result": {"content": [{"type": "text", "text": "wrote chart.png"}]}},
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "Final answer"}]}},
        {"type": "agent_end", "messages": []},
    ]


def _install(monkeypatch, client: _FakeClient, *, side_channel: list[dict] | None = None):
    conn = _FakeConn()
    monkeypatch.setattr(runtime_pi, "get_connection", lambda _p: conn)
    monkeypatch.setattr(runtime_pi, "_spawn_client", _make_spawn(client))
    calls: dict[str, list] = {"register": [], "unregister": []}

    async def fake_register(session_id, **kw):
        calls["register"].append({"session_id": session_id, **kw})

    async def fake_unregister(session_id):
        calls["unregister"].append(session_id)

    async def fake_fetch(session_id):
        return side_channel or []

    monkeypatch.setattr(rc, "register_session_via_admin", fake_register)
    monkeypatch.setattr(rc, "unregister_session_via_admin", fake_unregister)
    monkeypatch.setattr(rc, "_fetch_structured_tool_calls", fake_fetch)
    monkeypatch.setattr(runtime_pi, "sweep_and_extend_final_text", lambda conn, **kw: kw["base_text"])

    async def fake_kill(session_path):
        calls.setdefault("kill", []).append(session_path)

    monkeypatch.setattr(runtime_pi, "_kill_stray_pi", fake_kill)
    return conn, calls


def _make_spawn(client):
    async def _spawn(argv):
        _spawn.argv = argv
        return client
    return _spawn


def _run(**overrides):
    kwargs = dict(db_path=Path("/tmp/x.db"), session_id="s1", workspace="deep-conv-c1", conversation_id="c1",
                  question="hotels?", model="anthropic/claude-test", cost_sink=None, user_id="u1")
    kwargs.update(overrides)
    asyncio.run(runtime_pi.pi_run(**kwargs))


def test_happy_path_emits_full_event_sequence(monkeypatch):
    client = _FakeClient(_happy_records(), stats={"tokens": {"input": 100, "output": 20, "cacheRead": 5, "cacheWrite": 1}, "cost": 0.03})
    side = [{"name": "search_emails_batch", "args": {"searches": []}, "response": {"results": [{"cite_ref": "r1"}]}}]
    conn, calls = _install(monkeypatch, client, side_channel=side)
    costs: list[dict] = []
    _run(cost_sink=lambda **kw: costs.append(kw))

    kinds = [e["kind"] for e in conn.events]
    assert kinds[0] == "plan"
    assert kinds.count("tool_call") == 4  # start+end for search, start+end for bash
    assert "evidence" in kinds and "analysis" in kinds
    assert kinds[-2:] == ["draft", "final"]
    assert conn.events[-1]["payload"]["text"] == "Final answer"
    assert conn.finalized == [{"status": "done", "final_answer": "Final answer"}]
    assert costs == [{"agent_name": "pi", "model": "anthropic/claude-test", "input_tokens": 100, "output_tokens": 20,
                      "usd_override": 0.03, "cache_read_tokens": 5, "cache_write_tokens": 1}]
    assert calls["register"][0]["workspace"] == "deep-conv-c1" and calls["register"][0]["user_id"] == "u1"
    assert calls["unregister"] == ["s1"]
    assert client.sent[0]["type"] == "prompt" and client.sent[0]["message"] == "hotels?"
    assert client.closed and conn.closed


def test_argv_uses_conversation_session_path(monkeypatch):
    client = _FakeClient(_happy_records())
    _install(monkeypatch, client)
    monkeypatch.setenv("GMAIL_PI_CONTAINER", "pi-test")
    _run()
    argv = runtime_pi._spawn_client.argv
    assert argv[argv.index("--session") + 1] == "/sessions/c1.jsonl"
    assert "pi-test" in argv and "GMS_SESSION_ID=s1" in argv


def test_no_conversation_runs_without_session(monkeypatch):
    client = _FakeClient(_happy_records())
    _install(monkeypatch, client)
    _run(conversation_id=None)
    assert "--no-session" in runtime_pi._spawn_client.argv


def test_idle_timeout_aborts_and_emits_error(monkeypatch):
    client = _FakeClient(_happy_records()[:2], hang_after=2)
    conn, calls = _install(monkeypatch, client)
    monkeypatch.setenv("GMAIL_PI_IDLE_TIMEOUT", "0.2")
    _run()
    assert client.aborted
    assert calls["kill"] == ["/sessions/c1.jsonl"]
    assert conn.events[-1]["kind"] == "error" and "idle" in conn.events[-1]["payload"]["message"]
    assert conn.finalized == [{"status": "error", "final_answer": None}]


def test_eof_before_agent_end_is_an_error(monkeypatch):
    client = _FakeClient(_happy_records()[:3])
    conn, calls = _install(monkeypatch, client)
    _run()
    assert conn.events[-1]["kind"] == "error"
    assert calls["unregister"] == ["s1"]


def test_extension_error_is_logged_not_fatal(monkeypatch, caplog):
    records = _happy_records()
    records.insert(2, {"type": "extension_error", "extensionPath": "/opt/gmail-tools", "event": "tool_call", "error": "kaboom"})
    client = _FakeClient(records)
    conn, _ = _install(monkeypatch, client)
    _run()
    assert conn.finalized[0]["status"] == "done"
    assert "kaboom" in caplog.text


def test_session_path_for_rejects_bad_ids():
    assert runtime_pi.session_path_for("abc-123") == "/sessions/abc-123.jsonl"
    assert runtime_pi.session_path_for("../etc") is None
    assert runtime_pi.session_path_for(None) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_runtime_pi.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `runtime_pi.py`**

```python
"""Single-agent deep-analysis turn through the Pi agent harness.

One `pi --mode rpc` process per turn, inside the `pi-sandbox`
container, driven over stdin/stdout. Tool calls stream to
`agent_events` as they happen; the MCP side channel supplies the full
structured responses afterwards, exactly as `claude_native` does.
Public entry point: `pi_run()`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from gmail_search.agents import pi_protocol as pp
from gmail_search.agents.deep_events import (
    emit_analyst_events,
    emit_error,
    emit_plan_event,
    emit_retriever_events,
    emit_writer_and_final,
    sweep_and_extend_final_text,
)
from gmail_search.agents.pi_rpc import PiRpcClient, PiRpcError
from gmail_search.agents.session import append_event, finalize_session
from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)

AGENT_NAME = "pi"
_DEFAULT_MODEL = "<sonnet id from spike notes>"
_DEFAULT_THINKING = "medium"
_DEFAULT_CONTAINER = "pi-sandbox"
_DEFAULT_EXTENSION_PATH = "/opt/gmail-tools"
_DEFAULT_HARD_TIMEOUT = 900.0
_DEFAULT_IDLE_TIMEOUT = 300.0
_ABORT_GRACE = 5.0
_STATS_TIMEOUT = 15.0
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

PI_INSTRUCTION = """(NATIVE_INSTRUCTION from runtime_claude_native.py with these edits:
  1. Delete the sentence 'Always pass the `session_id` provided in this prompt as the first arg.'
  2. Replace the paragraph beginning 'Sub-agents via the `Task` tool' with:
     'There is no sub-agent tool. Heavy fan-out goes through larger batch calls.'
  3. In '# Before you finish', item 1: replace '(via Bash, run_code, external commands, anything)'
     with '(via bash, python, anything)'.
  4. Add under '# Tools': '- `bash` — run shell/python inside your workspace to compute, chart
     (matplotlib is installed) and write files. Publish any file the user should see.')"""


# ── Settings (env, matching the GMAIL_CLAUDEBOX_* pattern) ────────────


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        v = float(raw) if raw else default
    except ValueError:
        return default
    return v if v > 0 else default


def pi_model() -> str:
    return os.environ.get("GMAIL_PI_MODEL") or _DEFAULT_MODEL


def pi_thinking() -> str | None:
    raw = os.environ.get("GMAIL_PI_THINKING", _DEFAULT_THINKING).strip().lower()
    return None if raw in ("", "off", "none") else raw


def pi_container() -> str:
    return os.environ.get("GMAIL_PI_CONTAINER") or _DEFAULT_CONTAINER


def pi_extension_path() -> str:
    return os.environ.get("GMAIL_PI_EXTENSION_PATH") or _DEFAULT_EXTENSION_PATH


def hard_timeout_seconds() -> float:
    return _env_float("GMAIL_PI_HARD_TIMEOUT", _DEFAULT_HARD_TIMEOUT)


def idle_timeout_seconds() -> float:
    return _env_float("GMAIL_PI_IDLE_TIMEOUT", _DEFAULT_IDLE_TIMEOUT)


def session_path_for(conversation_id: str | None) -> str | None:
    """Deterministic in-container session file per conversation. No
    conversation (one-off probes) → ephemeral run."""
    if not conversation_id or not _SESSION_ID_RE.match(conversation_id):
        return None
    return f"/sessions/{conversation_id}.jsonl"


async def _spawn_client(argv: list[str]) -> PiRpcClient:
    return await PiRpcClient.spawn(argv)


# ── Turn loop ───────────────────────────────────────────────────────


@dataclass
class TurnOutcome:
    final_text: str
    local_tool_calls: list[dict] = field(default_factory=list)
    usage: pp.UsageStats | None = None


class _TurnState:
    def __init__(self) -> None:
        self.final_text = ""
        self.local_tool_calls: list[dict] = []
        self.open_bash: dict[str, dict] = {}


ToolEventSink = Callable[[str, dict], Awaitable[None]]


async def drive_turn(
    client,
    question: str,
    *,
    on_tool_event: ToolEventSink,
    hard_timeout: float,
    idle_timeout: float,
) -> TurnOutcome:
    """Send the prompt, consume events until `agent_end`, then fetch
    usage. Raises PiRpcError on EOF, idle timeout or hard timeout; the
    caller aborts the client."""
    started = time.monotonic()
    state = _TurnState()
    await client.send({"type": "prompt", "message": question})
    while True:
        remaining = hard_timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise PiRpcError(f"hard timeout after {hard_timeout:.0f}s")
        try:
            rec = await client.read_record(min(idle_timeout, remaining))
        except asyncio.TimeoutError as exc:
            raise PiRpcError(f"idle timeout: no event for {idle_timeout:.0f}s") from exc
        if rec is None:
            raise PiRpcError("pi exited before agent_end")
        if rec.get("type") == "agent_end":
            break
        await _handle_record(rec, state, on_tool_event)
    usage = await _fetch_usage(client)
    return TurnOutcome(final_text=state.final_text, local_tool_calls=state.local_tool_calls, usage=usage)


async def _handle_record(rec: dict, state: _TurnState, on_tool_event: ToolEventSink) -> None:
    kind = rec.get("type")
    if kind == "tool_execution_start":
        await on_tool_event("tool_call", pp.tool_call_args_entry(rec))
        if rec.get("toolName") == "bash":
            state.open_bash[str(rec.get("toolCallId"))] = rec
    elif kind == "tool_execution_end":
        await on_tool_event("tool_call", pp.tool_call_response_entry(rec))
        start = state.open_bash.pop(str(rec.get("toolCallId")), None)
        if start is not None:
            state.local_tool_calls.extend(pp.bash_as_run_code(start, rec))
    elif kind == "message_end":
        text = pp.assistant_text(rec)
        if text:
            state.final_text = text
    elif kind == "extension_error":
        logger.error("pi extension error: %s", rec)


async def _fetch_usage(client) -> pp.UsageStats | None:
    try:
        resp = await client.request({"type": "get_session_stats"}, timeout=_STATS_TIMEOUT)
    except (PiRpcError, asyncio.TimeoutError) as exc:
        logger.warning("get_session_stats failed (cost not recorded): %s", exc)
        return None
    return pp.usage_from_stats_response(resp)


# ── Glue: DB events, side channel, cost ─────────────────────────────


def _make_tool_event_sink(conn, session_id: str) -> ToolEventSink:
    async def _sink(kind: str, payload: dict) -> None:
        try:
            append_event(conn, session_id=session_id, agent_name=AGENT_NAME, kind=kind, payload=payload)
        except Exception:
            logger.exception("streaming append_event failed for session %s", session_id)

    return _sink


def _report_cost(cost_sink, model: str, usage: pp.UsageStats | None) -> None:
    if cost_sink is None or usage is None:
        return
    try:
        cost_sink(
            agent_name=AGENT_NAME,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            usd_override=usage.cost_usd,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("cost_sink failed (non-fatal): %s", exc)


async def _side_channel_tool_calls(session_id: str) -> list[dict]:
    from gmail_search.agents import runtime_claude as rc

    records = await rc._fetch_structured_tool_calls(session_id)
    return rc._tool_calls_from_side_channel(records)


def _build_argv(session_id: str, workspace: str, conversation_id: str | None, model: str) -> list[str]:
    return pp.build_pi_argv(
        container=pi_container(),
        session_id=session_id,
        workspace=workspace,
        session_path=session_path_for(conversation_id),
        extension_path=pi_extension_path(),
        model=model,
        thinking=pi_thinking(),
        system_prompt=PI_INSTRUCTION,
    )


async def _kill_stray_pi(session_path: str | None) -> None:
    """Killing the host-side `docker exec` client does not always kill
    the pi process inside the container. Best effort: pkill by the
    session path, which is unique per conversation. Ephemeral runs
    (no session path) are left to exit on their own."""
    if not session_path:
        return
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", pi_container(), "pkill", "-f", session_path,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), 10.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pkill of stray pi for %s failed: %s", session_path, exc)


async def _run_turn(conn, *, session_id: str, workspace: str, conversation_id: str | None, question: str, model: str) -> TurnOutcome:
    client = await _spawn_client(_build_argv(session_id, workspace, conversation_id, model))
    try:
        outcome = await drive_turn(
            client,
            question,
            on_tool_event=_make_tool_event_sink(conn, session_id),
            hard_timeout=hard_timeout_seconds(),
            idle_timeout=idle_timeout_seconds(),
        )
    except BaseException:
        await client.abort_and_close(grace=_ABORT_GRACE)
        await _kill_stray_pi(session_path_for(conversation_id))
        raise
    await client.close()
    return outcome


def _finish_ok(conn, *, session_id, workspace, conversation_id, turn_started_at, outcome: TurnOutcome, side_calls: list[dict]) -> None:
    all_calls = side_calls + outcome.local_tool_calls
    emit_retriever_events(conn, session_id, all_calls, skip_per_tool_emission=True)
    emit_analyst_events(conn, session_id, all_calls, skip_per_tool_emission=True)
    final_text = sweep_and_extend_final_text(
        conn,
        session_id=session_id,
        workspace=workspace,
        conversation_id=conversation_id,
        turn_started_at=turn_started_at,
        base_text=outcome.final_text,
    )
    emit_writer_and_final(conn, session_id, final_text)
    finalize_session(conn, session_id, status="done", final_answer=final_text)


def _finish_error(conn, session_id: str, exc: BaseException) -> None:
    logger.exception("pi_run failed for session %s: %s", session_id, exc)
    try:
        emit_error(conn, session_id, exc, agent_name=AGENT_NAME)
    except Exception:
        logger.exception("failed to emit error event for %s", session_id)
    try:
        finalize_session(conn, session_id, status="error")
    except Exception:
        logger.exception("failed to finalize session %s on error path", session_id)


async def pi_run(
    *,
    db_path: Path,
    session_id: str,
    workspace: str,
    conversation_id: str | None,
    question: str,
    model: str | None,
    cost_sink: Callable[..., None] | None,
    user_id: str | None = None,
) -> None:
    """Run one deep-mode turn through pi. Same contract as
    `runtime_claude_native.native_run` minus resume plumbing: the
    session file path is derived from `conversation_id`."""
    from gmail_search.agents import runtime_claude as rc

    turn_started_at = time.time()
    resolved_model = model or pi_model()
    conn = get_connection(db_path)
    registered = False
    try:
        await rc.register_session_via_admin(
            session_id, evidence_records=None, conversation_id=conversation_id, workspace=workspace, user_id=user_id
        )
        registered = True
        emit_plan_event(conn, session_id, agent_name=AGENT_NAME, approach="single pi agent loop with all tools")
        outcome = await _run_turn(
            conn, session_id=session_id, workspace=workspace, conversation_id=conversation_id, question=question, model=resolved_model
        )
        _report_cost(cost_sink, resolved_model, outcome.usage)
        side_calls = await _side_channel_tool_calls(session_id)
        _finish_ok(
            conn, session_id=session_id, workspace=workspace, conversation_id=conversation_id,
            turn_started_at=turn_started_at, outcome=outcome, side_calls=side_calls,
        )
    except Exception as exc:  # noqa: BLE001
        _finish_error(conn, session_id, exc)
    finally:
        if registered:
            try:
                await rc.unregister_session_via_admin(session_id)
            except Exception:
                logger.exception("unregister_session failed for %s", session_id)
        try:
            conn.close()
        except Exception:
            logger.exception("closing conn for session %s failed", session_id)
```
Write `PI_INSTRUCTION` as the full edited text, not the placeholder description above: copy `NATIVE_INSTRUCTION` and apply the four edits. Replace `_DEFAULT_MODEL` with the Sonnet id from the spike notes.

Note on `model`: `service.py` passes `None` for this backend (Task 6), so `pi_model()` decides; a caller (the compare script) may pass an explicit pi id.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_runtime_pi.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Full suite and report**

Run: `uv run pytest -q`. Report; no commit.

---

### Task 6: Wire the `pi` backend into `service.py`

**Files:**
- Modify: `src/gmail_search/agents/service.py:241` (`_VALID_BACKENDS`), `:766-960` (native branch → helpers + pi branch)
- Test: `tests/test_agent_service.py`

**Interfaces:**
- Produces:
  - `_VALID_BACKENDS = ("adk", "claude_code", "claude_native", "pi")`
  - `async def _stream_task_events(poll_conn, session_id: str, task: asyncio.Task) -> AsyncIterator[str]` — polls `fetch_events_after` every 0.1 s, yields SSE frames, returns once `task.done()` and the tail is drained.
  - `def _finish_single_agent_turn(conn, task: asyncio.Task, *, conversation_id, session_id, runner_name: str) -> str | None` — returns the failure frame from `_surface_deep_failure` when the task raised, else the `persist_ok` frame when `_persist_rich_assistant_message` succeeded, else None.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_service.py` (next to `test_real_run_claude_native_routes_to_native_run`):
```python
def test_real_run_pi_backend_routes_to_pi_run(monkeypatch, tmp_path):
    """backend="pi" must ensure the workspace, call pi_run with the
    turn's kwargs (model=None so runtime_pi picks GMAIL_PI_MODEL),
    skip the claudebox credential preflight, and never build the
    orchestrator."""
    import asyncio

    pi_calls: list[dict] = []

    async def fake_pi_run(*, db_path, session_id, workspace, conversation_id, question, model, cost_sink, user_id=None):
        pi_calls.append({"session_id": session_id, "workspace": workspace, "conversation_id": conversation_id,
                         "question": question, "model": model, "has_cost_sink": cost_sink is not None, "user_id": user_id})

    import gmail_search.agents.runtime_pi as rp

    monkeypatch.setattr(rp, "pi_run", fake_pi_run)

    workspace_dirs: list[str] = []
    monkeypatch.setattr(service, "_ensure_workspace_dir", lambda w: workspace_dirs.append(w))

    def _preflight_must_not_run():
        raise AssertionError("credential preflight must be skipped for pi")

    import gmail_search.claudebox_creds as creds

    monkeypatch.setattr(creds, "credentials_health", _preflight_must_not_run)

    class _OrchestratorMustNotRun:
        def __init__(self, *a, **kw):
            raise AssertionError("Orchestrator should not be constructed for pi")

    import gmail_search.agents.orchestration as orch_mod

    monkeypatch.setattr(orch_mod, "Orchestrator", _OrchestratorMustNotRun)

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(service, "get_connection", lambda _p: _FakeConn())
    monkeypatch.setattr(service, "fetch_events_after", lambda *a, **kw: [])
    monkeypatch.setattr(service, "_persist_rich_assistant_message", lambda *a, **kw: True)

    frames: list[str] = []

    async def consume():
        async for frame in service._real_run(tmp_path / "x.db", "sess-PI", "what happened",
                                             default_model="opus", backend="pi", conversation_id="conv-9", user_id="u1"):
            frames.append(frame)

    asyncio.run(consume())

    assert workspace_dirs == ["deep-conv-conv-9"]
    assert pi_calls == [{"session_id": "sess-PI", "workspace": "deep-conv-conv-9", "conversation_id": "conv-9",
                         "question": "what happened", "model": None, "has_cost_sink": True, "user_id": "u1"}]
    assert any("persist_ok" in f for f in frames)


def test_deep_backend_accepts_pi():
    assert service._deep_backend("pi") == "pi"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_agent_service.py -k "pi" -q`
Expected: `test_deep_backend_accepts_pi` FAILS (`adk` returned); the routing test fails because `pi` falls back to adk and the orchestrator guard raises.

- [ ] **Step 3: Extract the polling loop and the finish step from the native branch**

Add two module-level helpers above `_real_run`:
```python
async def _stream_task_events(poll_conn, session_id: str, task: "asyncio.Task") -> AsyncIterator[str]:
    """Mirror `agent_events` rows into SSE frames until `task` finishes,
    then drain whatever landed after the last poll."""
    import asyncio

    last_seq = 0
    while True:
        for ev in fetch_events_after(poll_conn, session_id, after_seq=last_seq):
            last_seq = max(last_seq, ev.seq)
            yield _sse(ev.kind, {"seq": ev.seq, "agent": ev.agent_name, "payload": ev.payload})
        if task.done():
            break
        await asyncio.sleep(0.1)
    for ev in fetch_events_after(poll_conn, session_id, after_seq=last_seq):
        yield _sse(ev.kind, {"seq": ev.seq, "agent": ev.agent_name, "payload": ev.payload})


def _finish_single_agent_turn(conn, task: "asyncio.Task", *, conversation_id: str | None, session_id: str, runner_name: str) -> str | None:
    """After a single-agent runner task ends: surface an uncaught
    exception as a visible failure, else persist the rich assistant
    message and return the `persist_ok` frame (None if persist failed)."""
    exc = task.exception()
    if exc is not None:
        logger.exception(f"{runner_name} raised in session {session_id}: {exc}")
        return _surface_deep_failure(
            conn,
            conversation_id=conversation_id,
            session_id=session_id,
            reason="the analysis run failed (see server logs); please retry.",
            final_answer=f"Deep analysis failed ({runner_name} raised).",
        )
    if _persist_rich_assistant_message(conn, conversation_id=conversation_id, session_id=session_id):
        return _sse("persist_ok", {"session_id": session_id})
    return None
```
Then in the `claude_native` branch replace the inline `last_seq` loop and the `exc = native_task.exception()` block with:
```python
        try:
            async for frame in _stream_task_events(poll_conn, session_id, native_task):
                yield frame
            tail = _finish_single_agent_turn(conn, native_task, conversation_id=conversation_id, session_id=session_id, runner_name="native_run")
            if tail is not None:
                yield tail
        finally:
            ...(unchanged poll_conn/conn close)...
```
Run `uv run pytest tests/test_agent_service.py -q` — the existing native tests must still pass before continuing.

- [ ] **Step 4: Add `pi` to the valid backends and the branch**

```python
_VALID_BACKENDS = ("adk", "claude_code", "claude_native", "pi")
```
The credential preflight block is guarded by `if backend in ("claude_native", "claude_code"):` — leave it; `pi` is not in that tuple, so it is skipped. Directly after `workspace = _claudebox_workspace_for(conversation_id, session_id)` and before `if backend == "claude_native":`, add:
```python
    if backend == "pi":
        from gmail_search.agents.runtime_pi import pi_run

        _ensure_workspace_dir(workspace)
        pi_task = asyncio.create_task(
            pi_run(
                db_path=db_path,
                session_id=session_id,
                workspace=workspace,
                conversation_id=conversation_id,
                question=question,
                model=None,
                cost_sink=_record_cost,
                user_id=user_id,
            )
        )
        try:
            async for frame in _stream_task_events(poll_conn, session_id, pi_task):
                yield frame
            tail = _finish_single_agent_turn(conn, pi_task, conversation_id=conversation_id, session_id=session_id, runner_name="pi_run")
            if tail is not None:
                yield tail
        finally:
            try:
                poll_conn.close()
            except Exception:
                logger.exception(f"closing poll_conn for session {session_id} failed")
            conn.close()
        return
```
Update the `_real_run` docstring's backend sentence to list `pi`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_agent_service.py tests/test_agent_service_race.py -q`
Expected: PASS including the two new tests.

- [ ] **Step 6: Full suite and report**

Run: `uv run pytest -q`. Report; no commit.

---

### Task 7: Prune pi session files with their workspaces

**Files:**
- Modify: `src/gmail_search/agents/gc.py:146-240`
- Test: `tests/test_agent_gc.py`

**Interfaces:**
- Produces: `_PI_SESSIONS_ROOT = "deploy/pi/sessions"`; `prune_conversation_workspaces(conn, *, retention_days=..., workspaces_root=None, pi_sessions_root: str | None = None)`; helper `_delete_pi_session_file(sessions_root: Path, conversation_id: str) -> bool`.

- [ ] **Step 1: Write the failing test**

No test covers `prune_conversation_workspaces` yet, so add a self-contained one to `tests/test_agent_gc.py`. The function calls `conn.execute(sql, params).fetchall()` for the stale-id SELECT (rows subscriptable by `"id"`), `conn.execute(...).rowcount` for the DELETE, and `conn.commit()`:
```python
class _FakeStaleConn:
    """Answers the stale-conversation SELECT with the ids given and
    accepts the mapping DELETE."""

    def __init__(self, stale_ids: list[str]) -> None:
        self._stale_ids = stale_ids
        self.committed = False

    def execute(self, sql: str, params=None):
        if sql.lstrip().upper().startswith("SELECT"):
            return _Result(rows=[{"id": i} for i in self._stale_ids], rowcount=len(self._stale_ids))
        return _Result(rows=[], rowcount=len(self._stale_ids))

    def commit(self) -> None:
        self.committed = True


class _Result:
    def __init__(self, rows, rowcount):
        self._rows = rows
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows


def test_prune_conversation_workspaces_removes_pi_session_file(tmp_path):
    from gmail_search.agents import gc

    ws_root = tmp_path / "workspaces"
    (ws_root / "deep-conv-stale1").mkdir(parents=True)
    (ws_root / "deep-conv-fresh2").mkdir(parents=True)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "stale1.jsonl").write_text("{}\n")
    (sessions / "fresh2.jsonl").write_text("{}\n")

    result = gc.prune_conversation_workspaces(
        _FakeStaleConn(["stale1"]), retention_days=30, workspaces_root=str(ws_root), pi_sessions_root=str(sessions)
    )

    assert result.dirs_deleted == 1
    assert not (ws_root / "deep-conv-stale1").exists() and (ws_root / "deep-conv-fresh2").exists()
    assert not (sessions / "stale1.jsonl").exists()
    assert (sessions / "fresh2.jsonl").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_agent_gc.py -k pi_session -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'pi_sessions_root'`

- [ ] **Step 3: Implement**

In `gc.py` next to `_WORKSPACES_ROOT`:
```python
_PI_SESSIONS_ROOT = "deploy/pi/sessions"


def _delete_pi_session_file(sessions_root, conversation_id: str) -> bool:
    """Remove the pi session transcript for a pruned conversation so
    the next deep turn starts a fresh session instead of resuming a
    workspace that no longer exists."""
    from pathlib import Path as _Path

    path = _Path(sessions_root) / f"{conversation_id}.jsonl"
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("prune_conversation_workspaces: failed to remove %s: %s", path, exc)
        return False
    return True
```
Add `pi_sessions_root: str | None = None` to `prune_conversation_workspaces`, and inside the scandir loop, right after `dirs_deleted += 1`:
```python
        _delete_pi_session_file(pi_sessions_root or _PI_SESSIONS_ROOT, conv_id)
```
Extend the docstring with one sentence about the pi session file.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_agent_gc.py -q`
Expected: PASS

- [ ] **Step 5: Full suite and report**

Run: `uv run pytest -q`. Report; no commit.

---

### Task 8: The `gmail-tools` pi extension (TypeScript)

**Files:**
- Create: `deploy/pi/extensions/gmail-tools/package.json`, `tsconfig.json`, `mcp-bridge.ts`, `index.ts`, `mcp-bridge.test.ts`, `.gitignore` (`node_modules/`)

**Interfaces:**
- Produces (`mcp-bridge.ts`):
  - `stripSessionId(schema: JsonSchema): JsonSchema` — deep-copies, removes `properties.session_id` and the `required` entry.
  - `toPiResult(res: McpCallResult): PiToolResult` — `{content: [{type:"text", text}], details: res, isError}`.
  - `registerMcpTools(pi: {registerTool: Function}, client: {listTools, callTool}, opts: {sessionId: string}): Promise<string[]>` — returns registered tool names.
- `index.ts` default export: async extension factory reading `GMS_MCP_URL` (required), `GMAIL_MCP_SERVICE_TOKEN` (optional), `GMS_SESSION_ID` (required).

- [ ] **Step 1: Scaffold the package with pinned deps**

`package.json` (versions: pi from the spike notes; the other two from `npm view <pkg> version` at the time, written exactly):
```json
{
  "name": "gmail-tools-pi-extension",
  "private": true,
  "type": "module",
  "scripts": { "test": "bun test" },
  "dependencies": {
    "@modelcontextprotocol/sdk": "<exact>",
    "typebox": "<exact>"
  },
  "devDependencies": {
    "@earendil-works/pi-coding-agent": "<exact from spike>",
    "@types/node": "<exact>",
    "bun-types": "<exact>",
    "zod": "<exact>"
  }
}
```
`typebox` stays in `dependencies` (the Dockerfile installs with `--omit=dev`); everything else is test-only.
`tsconfig.json`:
```json
{ "compilerOptions": { "target": "ES2022", "module": "ESNext", "moduleResolution": "Bundler", "strict": true, "allowImportingTsExtensions": true, "noEmit": true, "types": ["node", "bun-types"] } }
```
Run: `cd deploy/pi/extensions/gmail-tools && npm install --ignore-scripts` (creates `package-lock.json`; keep it).

- [ ] **Step 2: Write the failing tests**

`mcp-bridge.test.ts`:
```ts
import { describe, expect, test } from "bun:test";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { z } from "zod";
import { registerMcpTools, stripSessionId, toPiResult } from "./mcp-bridge.ts";

describe("stripSessionId", () => {
  test("removes the property and the required entry without mutating input", () => {
    const schema = {
      type: "object",
      properties: { session_id: { type: "string" }, queries: { type: "array" } },
      required: ["session_id", "queries"],
    };
    const out = stripSessionId(schema);
    expect(out.properties).toEqual({ queries: { type: "array" } });
    expect(out.required).toEqual(["queries"]);
    expect(schema.required).toEqual(["session_id", "queries"]);
  });
});

describe("toPiResult", () => {
  test("joins text blocks and carries isError", () => {
    const res = { content: [{ type: "text", text: "a" }, { type: "text", text: "b" }], isError: true };
    expect(toPiResult(res)).toEqual({ content: [{ type: "text", text: "a\nb" }], details: res, isError: true });
  });
});

async function linkedClient(): Promise<Client> {
  const server = new McpServer({ name: "fake", version: "0" });
  server.tool("echo", { session_id: z.string(), text: z.string() }, async ({ session_id, text }) => ({
    content: [{ type: "text", text: `${session_id}:${text}` }],
  }));
  const [a, b] = InMemoryTransport.createLinkedPair();
  await server.connect(a);
  const client = new Client({ name: "t", version: "0" });
  await client.connect(b);
  return client;
}

describe("registerMcpTools", () => {
  test("registers every tool, hides session_id, injects it on call", async () => {
    const registered: any[] = [];
    const pi = { registerTool: (d: any) => registered.push(d) };
    const names = await registerMcpTools(pi, await linkedClient(), { sessionId: "S1" });
    expect(names).toEqual(["echo"]);
    expect(Object.keys(registered[0].parameters.properties)).toEqual(["text"]);
    const result = await registered[0].execute("id", { text: "hi" }, new AbortController().signal);
    expect(result.content[0].text).toBe("S1:hi");
    expect(result.isError).toBe(false);
  });
});
```
Add `zod` to devDependencies (exact version) since the fake server uses it; the SDK already depends on it.

- [ ] **Step 3: Run to verify failure**

Run: `cd deploy/pi/extensions/gmail-tools && bun test`
Expected: FAIL, cannot resolve `./mcp-bridge.ts`

- [ ] **Step 4: Implement `mcp-bridge.ts`**

```ts
import { Type } from "typebox";

export type JsonSchema = { type?: string; properties?: Record<string, unknown>; required?: string[]; [k: string]: unknown };
type McpContent = { type: string; text?: string };
export type McpCallResult = { content?: McpContent[]; isError?: boolean; [k: string]: unknown };
export type PiToolResult = { content: { type: "text"; text: string }[]; details: unknown; isError: boolean };

type McpTool = { name: string; description?: string; inputSchema: JsonSchema };
type McpClientLike = {
  listTools(): Promise<{ tools: McpTool[] }>;
  callTool(req: { name: string; arguments: Record<string, unknown> }, schema?: undefined, opts?: { signal?: AbortSignal }): Promise<unknown>;
};
type PiLike = { registerTool(def: unknown): void };

export function stripSessionId(schema: JsonSchema): JsonSchema {
  const out = structuredClone(schema);
  if (out.properties) delete out.properties.session_id;
  out.required = (out.required ?? []).filter((r) => r !== "session_id");
  return out;
}

export function toPiResult(res: McpCallResult): PiToolResult {
  const text = (res.content ?? []).filter((c) => c.type === "text").map((c) => c.text ?? "").join("\n");
  return { content: [{ type: "text", text }], details: res, isError: Boolean(res.isError) };
}

function toolDefinition(tool: McpTool, client: McpClientLike, sessionId: string) {
  return {
    name: tool.name,
    label: tool.name,
    description: tool.description ?? "",
    parameters: Type.Unsafe<Record<string, unknown>>(stripSessionId(tool.inputSchema)),
    async execute(_id: string, params: Record<string, unknown>, signal?: AbortSignal) {
      const res = (await client.callTool({ name: tool.name, arguments: { session_id: sessionId, ...params } }, undefined, { signal })) as McpCallResult;
      return toPiResult(res);
    },
  };
}

export async function registerMcpTools(pi: PiLike, client: McpClientLike, opts: { sessionId: string }): Promise<string[]> {
  const { tools } = await client.listTools();
  for (const tool of tools) pi.registerTool(toolDefinition(tool, client, opts.sessionId));
  return tools.map((t) => t.name);
}
```

- [ ] **Step 5: Implement `index.ts`**

```ts
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { registerMcpTools } from "./mcp-bridge.ts";

function requiredEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`${name} is not set; the gmail-tools extension cannot start`);
  return v;
}

function transportFor(url: string, token: string | undefined) {
  const init = token ? { requestInit: { headers: { Authorization: `Bearer ${token}` } } } : undefined;
  return new StreamableHTTPClientTransport(new URL(url), init);
}

export default async function gmailTools(pi: ExtensionAPI) {
  const url = requiredEnv("GMS_MCP_URL");
  const sessionId = requiredEnv("GMS_SESSION_ID");
  const client = new Client({ name: "gmail-tools-bridge", version: "0.1.0" });
  await client.connect(transportFor(url, process.env.GMAIL_MCP_SERVICE_TOKEN));
  await registerMcpTools(pi, client, { sessionId });
  pi.on("session_shutdown", async () => {
    await client.close();
  });
}
```

- [ ] **Step 6: Run tests**

Run: `cd deploy/pi/extensions/gmail-tools && bun test`
Expected: 3 passing. Also `npx tsc --noEmit` must be clean.

- [ ] **Step 7: Report**

Report; no commit.

---

### Task 9: `deploy/pi/` container

**Files:**
- Create: `deploy/pi/Dockerfile`, `docker-compose.yml`, `setup.sh`, `start.sh`, `stop.sh`, `README.md`, `.gitignore`

**Interfaces:**
- Produces: a running container named `pi-sandbox` in which `pi --version` works, `/opt/gmail-tools` holds the extension with its `node_modules`, `/workspaces` is the shared claudebox workspaces mount, `/sessions` is `deploy/pi/sessions`.

- [ ] **Step 1: Dockerfile**

```dockerfile
# Pi agent harness sandbox for gmail-search deep mode.
# The whole `pi` process runs in here; the container boundary is the
# only thing between model-driven bash and the host.
FROM node:24-bookworm-slim

ARG PI_VERSION=<exact from spike notes>

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       bash ca-certificates git ripgrep procps \
       python3 python3-pandas python3-matplotlib python3-numpy \
  && rm -rf /var/lib/apt/lists/*

RUN npm install -g --ignore-scripts "@earendil-works/pi-coding-agent@${PI_VERSION}"

COPY extensions/gmail-tools/package.json extensions/gmail-tools/package-lock.json /opt/gmail-tools/
RUN cd /opt/gmail-tools && npm ci --ignore-scripts --omit=dev
COPY extensions/gmail-tools/*.ts /opt/gmail-tools/

# node:24 ships a `node` user at uid 1000; rename it so the workspace
# files it writes are owned by the host user (also uid 1000).
RUN usermod -l pi -d /home/pi -m node && groupmod -n pi node \
  && mkdir -p /home/pi/.pi/agent /sessions /workspaces \
  && chown -R pi:pi /home/pi /sessions /workspaces

ENV MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
USER pi
WORKDIR /workspaces
ENTRYPOINT ["sleep", "infinity"]
```
`procps` is for `pkill` (stray-process cleanup). `typebox` must be in `dependencies`, not devDependencies, for `--omit=dev` to keep it.

- [ ] **Step 2: docker-compose.yml**

```yaml
services:
  pi-sandbox:
    build:
      context: .
    image: gmail-search-pi:local
    container_name: pi-sandbox
    restart: unless-stopped
    environment:
      GMS_MCP_URL: ${GMS_MCP_URL}
      GMAIL_MCP_SERVICE_TOKEN: ${GMAIL_MCP_SERVICE_TOKEN:-}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
    volumes:
      - ../claudebox/workspaces:/workspaces
      - ./sessions:/sessions
      - ./pi-agent:/home/pi/.pi/agent
    extra_hosts:
      - host.docker.internal:host-gateway
```

- [ ] **Step 3: setup.sh, start.sh, stop.sh, .gitignore**

`setup.sh`:
```bash
#!/usr/bin/env bash
# Prepare deploy/pi: sessions + pi-agent dirs, .env with GMS_MCP_URL.
# Never copies Claude Code credentials — pi authenticates on its own
# (ANTHROPIC_API_KEY in .env, or `/login` inside the container).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"
say() { printf '[setup] %s\n' "$*"; }

ensure_dirs() {
  mkdir -p ./sessions ./pi-agent ../claudebox/workspaces
  say "sessions/, pi-agent/, workspaces ready"
}

ensure_env() {
  if [[ ! -f ./.env ]]; then
    printf 'GMS_MCP_URL=http://host.docker.internal:7878/mcp\nGMAIL_MCP_SERVICE_TOKEN=\nANTHROPIC_API_KEY=\n' > ./.env
    chmod 600 ./.env
    say "wrote .env template — fill in GMAIL_MCP_SERVICE_TOKEN (and ANTHROPIC_API_KEY unless you /login)"
  else
    say ".env present"
  fi
}

ensure_dirs
ensure_env
```
`start.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"
say() { printf '[start] %s\n' "$*"; }
bash ./setup.sh
say "docker compose up -d --build"
docker compose -f ./docker-compose.yml up -d --build
if docker exec pi-sandbox pi --version >/dev/null 2>&1; then
  say "pi $(docker exec pi-sandbox pi --version) — Ready."
else
  say "pi did not answer --version. recent logs:"
  docker compose -f ./docker-compose.yml logs --tail 40 pi-sandbox || true
  exit 1
fi
```
`stop.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
docker compose -f ./docker-compose.yml down
```
`.gitignore`:
```
.env
sessions/
pi-agent/
extensions/gmail-tools/node_modules/
```
`chmod +x setup.sh start.sh stop.sh`.

- [ ] **Step 4: README.md for the deploy dir**

Cover: layout table (same rows as spec §4.1), start/stop, logging in (`docker exec -it pi-sandbox pi` then `/login`, tokens land in `pi-agent/auth.json`; or `ANTHROPIC_API_KEY` in `.env`), the service token (same paragraph as claudebox's README: tenantless service token, scoping by registered session), and a smoke test:
```bash
docker exec -i -e GMS_SESSION_ID=smoke pi-sandbox pi --mode rpc --no-extensions --no-skills \
  --no-context-files --no-session --no-tools <<'EOF' | jq -c 'select(.type=="message_end")'
{"type":"prompt","message":"Say only the word OK."}
EOF
```
(Without a registered session the extension is deliberately not loaded here.)

- [ ] **Step 5: Bring it up and smoke-test the extension**

```bash
bash deploy/pi/start.sh
docker exec pi-sandbox ls /opt/gmail-tools/node_modules/@modelcontextprotocol >/dev/null && echo deps-ok
```
Then register a throwaway session as in Task 0 step 6 and run one exec through the real driver path:
```bash
cd /home/ssilver/development/gmail-search && uv run python - <<'PY'
import asyncio, json
from gmail_search.agents import runtime_claude as rc, runtime_pi as rp, pi_protocol as pp
from gmail_search.agents.pi_rpc import PiRpcClient

async def main():
    await rc.register_session_via_admin("smoke-1", evidence_records=None, conversation_id=None, workspace="smoke", user_id=None)
    argv = pp.build_pi_argv(container=rp.pi_container(), session_id="smoke-1", workspace="smoke", session_path=None,
                            extension_path=rp.pi_extension_path(), model=rp.pi_model(), thinking=None,
                            system_prompt="Call describe_schema once, then reply with the word OK.")
    client = await PiRpcClient.spawn(argv)
    outcome = await rp.drive_turn(client, "go", on_tool_event=lambda k, p: _p(k, p), hard_timeout=120, idle_timeout=60)
    await client.close()
    print("FINAL:", outcome.final_text[:200]); print("USAGE:", outcome.usage)
    await rc.unregister_session_via_admin("smoke-1")

async def _p(kind, payload): print(kind, json.dumps(payload)[:160])
asyncio.run(main())
PY
```
Expected: one `tool_call` line naming `describe_schema`, a FINAL line, and a USAGE line with non-zero tokens. `mkdir -p deploy/claudebox/workspaces/smoke` first if the exec fails on `-w`.

- [ ] **Step 6: Report**

Report the smoke output; no commit.

---

### Task 10: Web: `pi` in the picker

**Files:**
- Modify: `web/lib/config.ts:40`, `web/lib/chatSettings.ts:71-78`, `web/app/api/chat/route.ts:273,540`, `web/components/ModelPicker.tsx:187-198`

- [ ] **Step 1: Extend the type**

`web/lib/config.ts`:
```ts
export type DeepBackend = "adk" | "claude_code" | "claude_native" | "pi";
```

- [ ] **Step 2: Accept it when parsing saved settings**

`web/lib/chatSettings.ts` around line 71:
```ts
    const deepBackend: DeepBackend =
      parsed.deepBackend === "claude_code" ||
      parsed.deepBackend === "claude_native" ||
      parsed.deepBackend === "pi" ||
      parsed.deepBackend === "adk"
        ? parsed.deepBackend
        : current.deepBackend;
```
Read lines 77-80: if the following expression gates model choices on `claude_code || claude_native`, leave `pi` out of it (the Gemini model list still applies to the chat path; the pi backend ignores `model`).

- [ ] **Step 3: Widen the two unions in `route.ts`**

Lines 273 and 540: `"adk" | "claude_code" | "claude_native" | "pi"`. At line 551 the `deep === true && (claude_code || claude_native)` check chooses a Claude model list; do not add `pi` there.

- [ ] **Step 4: Add the button**

After the Claude Native button in `ModelPicker.tsx`:
```tsx
                <button
                  type="button"
                  onClick={() => switchDeepBackend("pi")}
                  className={
                    deepBackend === "pi"
                      ? "flex-1 rounded bg-teal-700 text-white px-2 py-1 font-medium hover:bg-teal-600"
                      : "flex-1 rounded bg-neutral-100 text-neutral-700 px-2 py-1 hover:bg-neutral-200"
                  }
                  title="Pi harness: single agent in the pi-sandbox container, billed per token via the provider set in GMAIL_PI_MODEL. Model picker does not apply."
                >
                  Pi
                </button>
```

- [ ] **Step 5: Type-check and lint**

Run: `cd web && bun run lint && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 6: End-to-end from the UI**

With `serve`, the MCP server, the web app and `pi-sandbox` running: toggle Deep, choose Pi, ask "how many hotel confirmations did I get in 2025, by month, with a chart". Expected in the UI: tool calls streaming under the disclosure, a chart chip, a final answer, and a cost line. Record the `agent_events` kinds for the session:
```bash
uv run python -c "
from gmail_search.store.db import get_connection; from pathlib import Path
c=get_connection(Path('data/gmail.db'))
print([r[0] for r in c.execute('select kind from agent_events where session_id=%s order by seq',('<session id from the UI>',)).fetchall()])"
```
Expected: starts with `plan`, contains `tool_call`s, `evidence`, `cost`, ends with `draft`, `final`.

- [ ] **Step 7: Report**

Report; no commit.

---

### Task 11: README and compare script

**Files:**
- Modify: `README.md` (Deep analysis mode section ~line 475; env knobs ~line 542; module map ~lines 323-345; deploy list ~line 360), `scripts/run_deep_compare.py:102-113,226`

- [ ] **Step 1: README edits**

- Deep analysis section: add a bullet "**Backends** — `adk` (Gemini orchestrator), `claude_code` (claudebox + orchestrator), `claude_native` (one Claude Code loop, plan-billed), `pi` (one [Pi harness](https://github.com/earendil-works/pi) loop in `deploy/pi/`, per-token billed via `GMAIL_PI_MODEL`). Pick per turn in the model picker or default with `GMAIL_DEEP_BACKEND`."
- Env knobs line: append `GMAIL_PI_MODEL`, `GMAIL_PI_THINKING`, `GMAIL_PI_CONTAINER`, `GMAIL_PI_HARD_TIMEOUT` (default 900), `GMAIL_PI_IDLE_TIMEOUT` (default 300), `GMAIL_PI_EXTENSION_PATH`.
- Module map under `agents/`: add `deep_events.py — event synthesis shared by claude_native + pi`, `pi_protocol.py / pi_rpc.py / runtime_pi.py — Pi harness deep backend`.
- Deploy list: add `pi/ — Pi agent harness sandbox (deep mode, per-token)`.
- Commands table: unchanged; say so in the report.

- [ ] **Step 2: Compare script**

In `_scope_model_envs`, add `elif backend == "pi": pass  # model comes from GMAIL_PI_MODEL`. Change the `--backends` help string to list `pi`. Since `_run_one_turn` sets `GMAIL_DEEP_BACKEND`, nothing else changes.

- [ ] **Step 3: Run the comparison**

Run: `uv run python scripts/run_deep_compare.py --backends claude_native,pi --timeout 900 --out scripts/deep_compare_pi.json`
Expected: a JSON report with per-backend latency, cost and final text for each canonical query. Publish the comparison as an artifact for Scott (table: query × backend → seconds, USD, tool calls, answer length).

- [ ] **Step 4: Report**

Report; no commit.

---

### Task 12: Audit

- [ ] **Step 1: Dependency scan**

```bash
uv run pip-audit
cd deploy/pi/extensions/gmail-tools && npm audit --omit=dev
docker exec pi-sandbox sh -c 'cd /usr/local/lib/node_modules/@earendil-works/pi-coding-agent && npm audit --omit=dev' || true
```
Summarise direct-impact vs informational findings.

- [ ] **Step 2: Codex review**

```bash
codex exec --sandbox read-only "Review this diff for a new deep-mode backend that drives the pi agent harness in a container. Files: src/gmail_search/agents/{pi_protocol,pi_rpc,runtime_pi,deep_events,service,cost,gc}.py, deploy/pi/{Dockerfile,docker-compose.yml,setup.sh,start.sh}, deploy/pi/extensions/gmail-tools/{index.ts,mcp-bridge.ts}. Concerns: (1) argv construction in pi_protocol.build_pi_argv — can session_id, workspace or conversation_id inject flags or paths? (2) runtime_pi.session_path_for — path traversal into /sessions. (3) pi_rpc abort/kill — orphaned in-container processes, stdin/stdout deadlocks. (4) The extension injects GMS_SESSION_ID into every MCP call — can the model override it via params? (5) Secrets: does anything log argv, env, or headers? (6) docker-compose mounts and the uid-1000 rename — host file ownership and escape surface. (7) service.py _finish_single_agent_turn — any behaviour change for claude_native. Rate @modelcontextprotocol/sdk, typebox and @earendil-works/pi-coding-agent on reputation, maintenance, license, attack surface." $(git diff --name-only)
```
Fix concrete findings in the same session or list them with a justification.

- [ ] **Step 3: Final report to Scott**

Full test run, audit summary, the compare artifact link, and the exact list of files changed. Wait for the password before committing.
