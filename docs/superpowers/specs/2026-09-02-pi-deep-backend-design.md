# Pi deep-analysis backend — design

**Date:** 2026-09-02
**Status:** Draft for review
**Decision owner:** Scott

## 1. Goal

Add a fourth deep-mode backend, `pi`, that runs one deep-analysis turn
through the [Pi agent harness](https://github.com/earendil-works/pi)
(`@earendil-works/pi-coding-agent`) instead of Claude Code inside
claudebox. It is a **choice in the picker, not a replacement**: `adk`,
`claude_code` and `claude_native` keep working exactly as today.

What the user gets:

- A deep backend that is billed per token through whichever provider
  pi is pointed at, with no Claude Code credential copying and no
  token-expiry preflight.
- Live tool-call streaming, cost and resume delivered by pi's RPC
  protocol rather than by tailing Claude Code's JSONL transcript.
- The same UI. Every event row the deep-mode panels render today is
  produced in the same shape.

## 2. Non-goals

- Retiring claudebox or ADK. Nothing on those paths changes.
- Replacing the chat-mode loop in `web/app/api/chat/route.ts`.
- Subagents. Pi has no built-in `Task` tool. The batch tools are the
  fan-out mechanism for the first cut; `pi-subagents` is a later
  experiment if a question set shows the need.
- Multi-user hardening beyond what the MCP server already enforces.
  Deep mode is single-host, single-operator today and stays so.

## 3. Constraints that shaped the design

**Plan billing is not reachable from pi.** Pi's provider docs say
Claude Pro/Max login in a third-party harness draws from Anthropic
"extra usage" and is billed per token, not against plan limits. Only
Claude Code itself uses the plan. That is why `claude_native` stays and
`pi` sits beside it.

**Pi has no MCP client.** Tools reach the model only through a pi
extension (TypeScript, `pi.registerTool`). The MCP server in
`agents/mcp_tools_server.py` owns session registration, per-user
scoping, tool-call persistence, blob URL rewriting and artifact
publishing. All of that stays in Python; one extension bridges it.

**Pi has no permission system.** With `bash` enabled, the pi process
must run in a container, as claudebox does today. The deep prompt
relies on shell access for charts and file work, so `bash` stays on
and the container boundary is the sandbox.

**Pi is TypeScript; this repo's agent code is Python.** The Python
service drives pi as a subprocess over stdin/stdout. No new HTTP
service and no Node code outside the extension.

## 4. Architecture

```
FastAPI serve (host)
  service.py ── backend == "pi" ──► runtime_pi.py::pi_run()
                                         │
                                         │ docker exec -i pi-sandbox pi --mode rpc …
                                         ▼
                              ┌─────────────────────────────┐
                              │ container: pi-sandbox       │
                              │  pi (RPC over stdin/stdout) │
                              │  bash / read / write / …    │
                              │  extension: gmail-tools ────┼──► http://host.docker.internal:7878/mcp
                              │  /workspaces/<conv>  (mount)│         (existing MCP tools server,
                              │  /sessions/<conv>.jsonl     │          service-token auth)
                              └─────────────────────────────┘
                                         │ JSONL events
                                         ▼
                        agent_events rows (tool_call, evidence, analysis,
                        draft, final, cost)  ──► existing SSE poller ──► UI
```

### 4.1 Container: `deploy/pi/`

| File | Purpose |
|---|---|
| `Dockerfile` | `node:24-bookworm-slim` + `@earendil-works/pi-coding-agent` at a pinned version, installed with `--ignore-scripts`; `python3` plus the analyst stack (`pandas`, `matplotlib`, `numpy`) so charts work as they do in claudebox; `ripgrep`, `git`; non-root user `pi` (uid 1000). |
| `docker-compose.yml` | Service `pi-sandbox`, `restart: unless-stopped`, entrypoint `sleep infinity`. Mounts: `../claudebox/workspaces:/workspaces` (shared with claudebox so the host-side auto-publish sweep and `_ensure_workspace_dir` work unchanged), `./sessions:/sessions`, `./extensions:/ext:ro`, `./pi-agent:/home/pi/.pi/agent` (settings and `auth.json`). `extra_hosts: host.docker.internal:host-gateway`. Env: `GMAIL_MCP_SERVICE_TOKEN`, `GMS_MCP_URL`, and either `ANTHROPIC_API_KEY` or nothing when `auth.json` holds pi OAuth tokens. |
| `start.sh` / `stop.sh` | Mirror claudebox's scripts. `start.sh` builds the image, brings compose up, then runs `docker exec pi-sandbox pi --version` as the readiness check. |
| `setup.sh` | Creates `sessions/`, `pi-agent/`, and `.env`; never copies Claude Code credentials. |
| `README.md` | Layout, start/stop, how to log pi in (`docker exec -it pi-sandbox pi` then `/login`), smoke test. |

The container is long-lived and idle. Each turn is one `docker exec`.
Nothing listens on a port; the only inbound path is the exec, and the
only outbound paths are the MCP server on the host and the model
provider.

### 4.2 Extension: `deploy/pi/extensions/gmail-tools/`

A pi extension (`index.ts` + `package.json` with pinned
`@modelcontextprotocol/sdk` and `typebox`) that:

1. On load, opens a Streamable HTTP MCP client to `GMS_MCP_URL`
   with `Authorization: Bearer $GMAIL_MCP_SERVICE_TOKEN`.
2. Calls `tools/list` and registers each tool with `pi.registerTool`,
   copying name, description and JSON Schema. The `session_id`
   property is removed from the schema and injected from the
   `GMS_SESSION_ID` env var on every call, so the model never sees or
   forges it. The rest of the tool guidance in the system prompt loses
   the "always pass `session_id`" instruction.
3. Maps MCP results to pi results: text content passes through;
   `isError` becomes a pi error result; the raw MCP payload is kept in
   `details` for logging.
4. Exposes nothing else. No commands, no UI, no renderers.

The MCP server is unchanged. It already records every call to
`agent_events` and scopes each call to the registered session, so tool
persistence and tenant scoping need no new code.

### 4.3 Driver: `src/gmail_search/agents/runtime_pi.py`

Public entry point `pi_run(...)` with the same keyword signature as
`native_run` (db_path, session_id, workspace, conversation_id,
question, model, cost_sink, user_id). No `resume` or `on_session_uuid`
arguments; see §4.5.

Sequence per turn:

1. `register_session_via_admin(...)` exactly as `native_run` does.
2. Emit the synthetic `plan` event.
3. Spawn the subprocess:

   ```
   docker exec -i
     -e GMS_SESSION_ID=<session_id>
     -w /workspaces/<workspace>
     pi-sandbox
     pi --mode rpc
        --session /sessions/<conversation_id>.jsonl
        --no-extensions --no-skills --no-context-files --no-prompt-templates
        -e /ext/gmail-tools
        --model <pi model id> [--thinking <level>]
        --system-prompt "<PI_INSTRUCTION>"
   ```

   `--no-context-files` matters: a `CLAUDE.md` or `AGENTS.md` written
   into the workspace by a previous turn must not become system prompt.

4. Write one `prompt` command with the question. Read stdout line by
   line, splitting on `\n` only (the RPC spec forbids readers that
   split on Unicode separators).
5. Map events as they arrive:

   | pi event | agent_events row |
   |---|---|
   | `tool_execution_start` | `tool_call` `{name, args}` |
   | `tool_execution_end` | `tool_call` `{name, response}` (response clipped to the same cap `jsonl_tail` uses) |
   | `message_end` for an assistant message | accumulate text blocks; last one is the final answer |
   | `agent_end` | stop reading |
   | `extension_error` | log at error level; continue |

   Built-in tools (`bash`, `read`, `write`, …) are recorded too, with
   `bash` mapped to the `run_code` name so the existing analyst panel
   logic in `runtime_claude_native` lights up without change.

6. Send `get_session_stats`, read the response, and call
   `cost_sink(agent_name="pi", model=<model>, input_tokens, output_tokens)`.
   Cache-read, cache-write and pi's own `cost` figure go into the
   `cost` event payload as extra keys so the comparison phase can check
   our pricing table against pi's.
7. Close stdin, wait for exit with a short grace period, kill on
   overrun.
8. Reuse the event emitters from `runtime_claude_native`:
   `_emit_retriever_events`, `_emit_analyst_events`,
   `_sweep_and_extend_final_text`, `_emit_writer_and_final`,
   `_emit_error`. These move into a small shared module
   `agents/deep_events.py` so neither runtime imports the other; this
   is the one refactor the change makes to existing code.
9. `finalize_session(status="done")`, unregister in `finally`.

Timeouts: a hard wall-clock cap (default 900 s, env
`GMAIL_PI_HARD_TIMEOUT`) and an idle cap on time since the last event
(default 300 s, env `GMAIL_PI_IDLE_TIMEOUT`). On either, send
`{"type":"abort"}`, wait 5 s, then terminate the exec. The turn ends
with an `error` event, as today.

Cancellation: if the FastAPI task is cancelled (client disconnect),
the same abort path runs. The pi session file keeps whatever was
persisted, so a retry resumes rather than restarts.

### 4.4 Backend wiring

- `service.py`: add `"pi"` to `_VALID_BACKENDS`; the claudebox
  credential preflight and workspace helpers are shared with
  `claude_native` except that the preflight is skipped for `pi`; call
  `pi_run` in a branch parallel to the `claude_native` one, with the
  same exception guard.
- `web/components/ModelPicker.tsx`: fourth button, label "Pi".
- `web/app/api/agent/analyze/route.ts`: no change; it proxies the
  `backend` field through and `service.py` validates it.
- Config: `deep.pi_model` (default set in `config.py`, overridable in
  `config.yaml`) and `deep.pi_thinking`. The UI model picker's Gemini
  choices do not apply to this backend; the picker shows the configured
  pi model as read-only text for now. `GMAIL_DEEP_BACKEND=pi` selects
  it by default.
- README: Commands table unchanged; add `pi` to the deep-backend list,
  the env-var table, the daemon list and the module map.

### 4.5 Resume without a UUID column

Claude Code assigns its own session UUID, which forced the
`conversation_claude_session` table, the advisory lock and the
first-turn callback. Pi takes the session **file path** as input, so
the path is derived deterministically:
`/sessions/<conversation_id>.jsonl`. Every turn of a conversation
passes the same path; the first turn creates the file, later turns
append. No new table, no lock, no callback.

The spike (§6) must confirm that `--session <path>` creates a missing
file rather than erroring. If it does not, the driver touches an empty
file first.

Stale-workspace recovery stays as it is: if the workspace directory
was pruned, the session file is deleted alongside it in
`prune_conversation_workspaces`.

### 4.6 System prompt

`PI_INSTRUCTION` starts from `NATIVE_INSTRUCTION` with three edits:
drop the `session_id` sentence, replace the `Task` subagent paragraph
with a note that heavy fan-out goes through batch tool calls, and name
`bash` rather than `run_code` as the way to produce charts and files
under the workspace before publishing them.

## 5. Security

- **Boundary.** Model-driven shell runs only inside `pi-sandbox`. The
  container mounts exactly three host paths: the shared workspaces
  tree, its own sessions directory, and the read-only extension
  directory. No Docker socket, no host home.
- **Tenant scoping.** The extension carries the tenantless service
  token, the same choice claudebox made and for the same reason: the
  registered session, not the transport, decides whose mailbox a call
  sees. `GMS_SESSION_ID` is set per exec; a stale or forged id fails
  registration lookup in the MCP server.
- **Prompt injection surface.** `--no-context-files`, `--no-skills`,
  `--no-extensions` (except the one we pass) and
  `--no-prompt-templates` keep files the model wrote in a previous
  turn from becoming instructions in the next.
- **Secrets.** `ANTHROPIC_API_KEY` or pi's `auth.json` live in the
  container only. The driver never logs the exec argv with env values,
  and the extension never logs headers.
- **Supply chain.** Two new npm dependencies (`pi-coding-agent` in the
  image, `@modelcontextprotocol/sdk` + `typebox` in the extension),
  all exact-pinned, installed with `--ignore-scripts`. `npm audit` on
  the extension and the image's global install run as part of the
  session audit; codex reviews the extension, the driver's subprocess
  handling and the argv construction.

## 6. Spike (phase 0)

Throwaway. Answers these before any repo code is written:

1. Does `pi --mode rpc --session <new path>` create the file?
2. Does the MCP-bridge extension load in the container with
   `--no-extensions -e /ext/gmail-tools`, and do all eight tools appear
   in `get_available_tools`-equivalent output (`pi.getAllTools()` via a
   one-line debug command)?
3. Does a real question ("list every hotel I stayed at in 2025 with
   dates") complete end to end with batch tool calls, a bash-generated
   chart, and a `publish_artifact_batch` call?
4. What does `get_session_stats` report, and how does its `cost` line
   up with `store/cost.py` for the same model?
5. Which exact pi model ids exist for the Claude models we use
   (`get_available_models`)?

Exit criterion: one transcript showing 3 and a table for 4 and 5,
pasted into the implementation plan. Anything built in the spike is
discarded except the extension, which is rewritten with tests.

## 7. Phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 Spike | Local pi install, throwaway extension, one real run | §6 answers |
| 1 Container + extension | `deploy/pi/*`, extension with unit tests against a fake MCP server | `start.sh` reaches Ready; extension tests pass |
| 2 Driver | `runtime_pi.py`, `deep_events.py` refactor, tests with a scripted fake `pi` process (mirrors `test_runtime_claude.py`) | All existing native tests still pass; new driver tests cover happy path, tool mapping, idle timeout, hard timeout, abort on cancel, `extension_error` |
| 3 Wiring | Backend enum, picker button, route validator, config keys, README | A deep turn from the UI on backend `pi` streams tool calls and ends with a final answer and cost |
| 4 Parity + compare | `run_deep_compare.py` on a fixed question set, `pi` vs `claude_native`; cost and latency table | Report reviewed; any UI regression fixed |
| 5 Audit | pip-audit / npm audit, codex review of changed files | Findings fixed or explicitly punted |

## 8. Testing

- **Extension.** Vitest against an in-process fake MCP server (the SDK
  ships one): tool listing, `session_id` stripping and injection, error
  mapping.
- **Driver.** A Python fake `pi` script that replays a scripted JSONL
  event stream and answers `get_session_stats`; tests patch the exec
  argv builder to point at it. Covers every row in the event map,
  timeouts, cancellation, malformed lines.
- **Service.** Extend `test_agent_service.py` with a `backend="pi"`
  case that asserts `pi_run` is called with the right workspace and
  that the credential preflight is skipped.
- **Integration (marked slow, needs the container).** One real turn
  through the MCP server with a cheap model.

## 9. Open questions and deferred work

- Whether to expose provider choice (Gemini through pi-ai) in the
  picker. Deferred until the compare phase shows whether it is worth a
  second per-token option.
- `pi-subagents` for heavy fan-out. Deferred; measure first.
- Whether claudebox can eventually be retired for the `claude_code`
  orchestrated backend too. Out of scope; `claude_native` must stay for
  plan billing regardless.
- Cache-token pricing in `store/cost.py`. The cost event will carry
  the figures; adding them to the pricing table is a follow-up once the
  compare phase shows the gap.
