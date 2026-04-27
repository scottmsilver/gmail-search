# Per-conversation sessions — plan (REVISED)

**Date**: 2026-04-26
**Author**: Claude (Opus 4.7) with Scott
**Status**: revised after adversarial + codex review; see "Review history" at end

## Goal

Each chat thread in the UI = its own persistent claudebox session. User can:
- Run multiple conversations concurrently and switch between them while live.
- Resume a conversation later and pick up where it left off.
- Inspect a conversation's full debug detail (tool calls, artifacts, raw JSONL).

## Why this needed a v2

The first pass made three load-bearing wrong claims that codex caught:
1. "Side channel is drained before unregister." False — drain happens after the run completes, and unregister is in `finally`. If the orchestrator crashes between, the call log is lost. Plus there's a hard cap (`_MAX_CALLS_PER_SESSION`) that silently drops calls.
2. "Pass `resume=conversation_id` and append to the same JSONL." Asserted as fact with zero test coverage. Claude Code's `--resume` behavior is unverified.
3. First-turn UUID race "solved" by a side-table — but no transactional claim, so two concurrent first-turn requests both proceed.

Plus 3 smaller misses: conversation_id sanitization (migration problem, not "open question"); GC by mtime nukes idle-but-active conversations; effort estimate ignored most of the actual UI work.

## Concept (unchanged)

A **conversation** = `conversations.id` row. Multiple turns per conversation; each turn = an `agent_sessions` row. After this work:
- One claudebox **workspace dir** per conversation, stable across turns.
- One Claude **session UUID** per conversation, threaded via `--resume` so the JSONL appends.
- One `/work` mount per conversation (already true via `conversation_id`).

---

## Phase 0 — RESULTS (2026-04-26 ~21:43 PT)

Two empirical tests against the live claudebox container (`localhost:8765`, workspace `phase0-resume-test-1777264959`):

**Test 1 — same workspace, two calls, second uses `resume=<sessionId>`:**
- ✅ Same JSONL appended (1842 B / 5 lines → 3811 B / 10 lines, same `28a89f49-...jsonl`).
- ✅ Memory intact ("what word did I just ask you to say?" → "**PINEAPPLE**").
- ✅ Real cache hit on resume: call 1 had `cacheCreationInputTokens=16751, cacheReadInputTokens=0` ($0.063); call 2 had `cacheCreationInputTokens=5631, cacheReadInputTokens=11142` ($0.025). ~70% of the original 16k context cached and read on resume.
- ℹ️ Response key is `sessionId` (camelCase), not `session_id`. Update code references accordingly.

**Test 2 — fresh workspace, single call with `resume=<uuid we invented>`:**
- ❌ **Fails**: `"No conversation found with session ID: 187c7826-...".` HTTP 200 but result text is the error. The preferred path of "pre-allocate UUID at conversation creation, always resume" is **not viable** — Claude Code's `--resume` requires a UUID it actually wrote first.

**Implications for the plan:**
1. ✅ The JSONL-append assumption is solid. Phase 5 debug pane gets ONE JSONL per conversation as designed.
2. ❌ Drop the "pre-allocate UUID at conversation creation" preferred path. **Must always do**: first turn with no `resume` → capture `sessionId` from response → persist to `conversation_claude_session` → subsequent turns resume.
3. ⚠️ The first-turn UUID race is REAL and needs serialization. Postgres advisory lock keyed on `hashtext(conversation_id)`, held only across the first-turn establishment critical section, is the cleanest fix. Once a row exists in `conversation_claude_session`, no lock is needed (subsequent turns just SELECT and pass `resume`).
4. ⚠️ I did NOT test whether `--system-prompt` / `--append-system-prompt` re-apply or stick. Claudebox `_build_args` always re-injects them (`api_server.py:230-239`), so worst case the original is overridden by the new — probably fine. Worth a follow-up test if the conversation prompt changes mid-stream.

---

## Phase 0 — Empirically validate `--resume` (gates everything)

This is 15 minutes of work and decides the whole shape of the project. Do this BEFORE writing any other code.

**Test**:
1. Pick a fresh workspace dir inside the running claudebox.
2. POST to `/run` with no `resume`. Capture the `session_id` from claudebox's response (it's surfaced in the JSON).
3. Note the JSONL file Claude Code wrote to (under `claude-config/projects/-workspaces-<workspace>/`).
4. POST again with `resume=<that session_id>` and a follow-up prompt.
5. Verify the SAME JSONL file got new lines appended (not a new file created).
6. Note: did `--system-prompt` and `--append-system-prompt` get re-applied, or are they remembered from the original?
7. Note: cost behaviour — does the resumed turn read the prior context (cache hit), or replay it (cache miss)?

**Decision matrix**:
- ✅ JSONL appends + system prompt re-applied + cache hits → plan proceeds as written.
- ⚠️ JSONL is a new file → debug pane needs to stitch by `claude_session_uuid` instead of by file. Adds ~½ day.
- ⚠️ System prompt is sticky from first run → we need to detect a prompt change and force a non-resume turn. Adds another ~½ day.
- ❌ `--resume` reliably fails (e.g., requires a strict UUID format we can't produce, or refuses on non-clean exit) → drop the JSONL-append goal, keep stable workspace dir only, stitch JSONLs by `claude_session_uuid` mapping at read time. Sidebar/debug pane still works, just with a per-turn JSONL list instead of one stream.

**Output of Phase 0**: a one-paragraph note (in this file) saying which branch we're taking. Then continue.

---

## Phase 1 — Stable per-conversation workspace + atomic UUID claim

### 1a. Sanitize + persist conversation IDs

**The problem**: `conversations.id` is `TEXT` and accepts arbitrary strings (`server.py:1176-1217`). The persistent `/work` resolver only accepts `[A-Za-z0-9_-]` and rejects everything else (`sandbox.py:297-305`). Two conversations with different ids could collide after sanitization.

**Fix**: at conversation creation time (server.py upsert path), require `conversations.id` to match `^[A-Za-z0-9_-]{1,64}$`. Reject upserts that don't. Migrate existing rows: scan `conversations`, identify violations, log them (don't auto-rename — too risky).

**Open question**: are there any existing rows with violations? `SELECT id FROM conversations WHERE id !~ '^[A-Za-z0-9_-]{1,64}$';` — run this first; if zero, we're safe to enforce immediately. If non-zero, we either back-fill a column (`stable_id TEXT UNIQUE`) or rename.

### 1b. New table for atomic UUID claim

```sql
CREATE TABLE IF NOT EXISTS conversation_claude_session (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    claude_session_uuid TEXT NOT NULL,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

The PK ensures only one row per conversation. The claim path uses Postgres native upsert semantics.

### 1c. The claim flow (every deep-mode turn)

Phase 0 ruled out pre-allocated UUIDs. We need an advisory lock around the first-turn critical section. After that, no lock — subsequent turns just SELECT and pass `resume`.

```python
# Pseudocode for the turn entrypoint (service.py)
ws_name = f"deep-conv-{conversation_id}"  # already sanitized per 1a

with conn.transaction():
    # 1. Fast path: row already exists, no lock needed (PG SELECT is concurrent-safe).
    row = conn.execute(
        "SELECT claude_session_uuid FROM conversation_claude_session "
        "WHERE conversation_id = %s",
        (conversation_id,)
    ).fetchone()

    if row:
        # Subsequent turn: just resume.
        claude_uuid = row["claude_session_uuid"]
        return run_turn(workspace=ws_name, resume=claude_uuid)

    # 2. Slow path: no row → first-turn establishment. Take advisory lock so
    #    only one concurrent caller does the establishment work.
    lock_key = hashtext(conversation_id)  # stable bigint hash
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))

    # 3. Re-check inside the lock (another caller may have just established it).
    row = conn.execute(
        "SELECT claude_session_uuid FROM conversation_claude_session "
        "WHERE conversation_id = %s",
        (conversation_id,)
    ).fetchone()
    if row:
        return run_turn(workspace=ws_name, resume=row["claude_session_uuid"])

    # 4. We hold the lock and there's no row. Run first turn without resume.
    response = run_turn(workspace=ws_name, resume=None)
    claude_uuid = response["sessionId"]  # NB: camelCase per Phase 0 finding

    # 5. Persist (within the same txn that holds the advisory lock).
    conn.execute(
        "INSERT INTO conversation_claude_session "
        "(conversation_id, claude_session_uuid) VALUES (%s, %s)",
        (conversation_id, claude_uuid),
    )
    return response
# Lock released at txn commit/rollback.
```

**Properties this gives us**:
- Only one first-turn ever runs per conversation, even under heavy concurrency. The advisory lock serializes them; second waiter wakes up after the first commits and takes the fast path.
- Subsequent turns never block (no advisory lock taken on the fast path).
- If the first turn fails (claudebox 5xx, network blip), we ROLLBACK and the lock + row are released — next caller retries cleanly with no orphan UUID.
- The transaction scope means the INSERT is atomic with the lock release: no window where the lock is held but the UUID is missing.

**One known race we accept**: between the advisory lock release and the second waiter's re-check, the row IS visible to a different connection. If a third caller comes in *during* the first turn, it'll either (a) see no row + acquire the same lock + wait, or (b) see the row and resume — both correct. The advisory lock is per-`hashtext(conversation_id)`, so different conversations don't block each other.

**Hash collision risk**: `hashtext()` returns `int4`. Two conversation_ids hashing to the same key would block each other unnecessarily. Acceptable — the only cost is extra serialization for one pair of unrelated conversations during their first turn. Not a correctness bug.

### 1d. Code changes

| File | Change |
|------|--------|
| `src/gmail_search/store/pg_schema.sql` | Add `conversation_claude_session` table + `TABLE_DOCS` entry. |
| `src/gmail_search/agents/service.py:252-255` | Rename to `_claudebox_workspace_for(conversation_id)`; return `f"deep-conv-{conversation_id}"`. Update callers. |
| `src/gmail_search/agents/runtime_claude.py` (`_build_request_body`, ~line 215) | Add `resume` arg; pass `body["resume"] = claude_session_uuid` when present. |
| `src/gmail_search/agents/service.py` (turn entrypoint) | Implement claim flow 1c. Pre-allocate UUID at conversation create if Phase 0 allows; otherwise lookup-then-INSERT-on-conflict. |
| `src/gmail_search/server.py:1176-1217` | Enforce conversation_id regex at upsert. |

---

## Phase 2 — Make MCP side-channel persistence reliable

**Status quo** (codex `mcp_tools_server.py:150-157, 225-234`):
- `_record_call` appends to `_SESSION_CALLS` (in-memory dict).
- Past `_MAX_CALLS_PER_SESSION` (1000), calls are silently dropped.
- Drain happens AFTER the run completes via `_resolve_tool_calls()` → `_fetch_structured_tool_calls()` (`runtime_claude.py:143-169, 789-799`).
- `unregister_session` clears the in-memory log unconditionally.
- Crash between run and drain → log lost.

**Fix**: write each call to `agent_events` *immediately* in `_record_call`, instead of buffering for end-of-turn drain.

**Code changes**:

| File | Change |
|------|--------|
| `src/gmail_search/agents/mcp_tools_server.py:_record_call` | Open a per-session write connection lazily (or use the existing `ctx.get_db_conn()`). On each call, append to `agent_events` with `session_id`, the next `seq`, `agent_name='mcp'`, `kind='tool_call'`, payload = `{name, args, response, ts}`. |
| Same file, `_MAX_CALLS_PER_SESSION` | Remove the cap, OR keep it solely as a write-throttle alarm (log a warning when crossed; never drop). The DB write is the source of truth now. |
| `runtime_claude.py:_fetch_structured_tool_calls` | Either delete (DB is source of truth) or keep as a no-op fallback for legacy code paths. |
| `runtime_claude.py:_resolve_tool_calls` callers | Read from `agent_events` instead of from the side channel. The shape they expect (`{name, args, response, ts}`) maps 1:1 to the `payload`. |

**Cost**: one extra DB INSERT per tool call. At ~50 tool calls per turn, that's 50 trivial inserts on a DB doing far heavier writes. Negligible.

**Win**: orchestrator crashes don't lose call history; the cap stops being a silent data-loss bug; debug pane reads from one source instead of stitching two.

---

## Phase 2.5 — Concurrent runs (verification only)

After Phase 1 + 2:
- Distinct conversations → distinct workspaces → claudebox `busy_workspaces` lock allows them to run concurrently with no orchestrator-side serialization.
- Same conversation hitting itself → 409 from claudebox (correct semantics: a conversation is one turn at a time).

**Verification test**: `tests/integration/test_concurrent_conversations.py` — fire two `/api/analyze` requests for two different conversations, assert both complete and their `agent_events` are well-ordered (no cross-contamination).

**UI 409 handling**: the chat UI must disable input on a conversation while a turn is mid-flight. If the user somehow sends anyway and gets a 409, surface as "this conversation is mid-turn — wait for it to finish" not a generic error. This is real UI work — flag it.

---

## Phase 3 — Runtime robustness + GC sweeper

### 3a. Fix the fan-out idle-watchdog false-positive

**Symptom** (observed in session `93df9ec6f3f74920`, 2026-04-27): a deep-mode turn fanned out into 8 parallel sub-agents via the Task tool. The orchestrator's JSONL tailer watches the **parent** session's JSONL for `tool_use` blocks. While sub-agents are running, the parent is idle (waiting for Task results); sub-agent activity goes to separate JSONLs under `<parent_uuid>/subagents/agent-<id>.jsonl`. After 304 s of parent-side silence the idle-timeout watchdog fired and aborted the run client-side. Server-side run may have continued.

**Fix** (two prongs, both):
1. **Tail sub-agent JSONLs too.** When the parent JSONL contains an open `Task` `tool_use` block (no matching `tool_result` yet), the tailer also watches `<parent_uuid>/subagents/*.jsonl` and counts ANY tool_use line there as "the run is making progress." Resets the idle clock.
2. **Bump idle threshold while a Task is in flight.** Independent of #1: when an open Task `tool_use` is detected, multiply the idle threshold (currently 5 min) by some factor (say 4× → 20 min) until that Task resolves. Belt-and-suspenders for cases where sub-agent JSONLs are slow to surface.

**Code touch points**:
- The watchdog lives in `runtime_claude.py` near `_make_progress_handler` / `_start_progress_tailer` (~line 600). The tailer in `jsonl_tail.py` watches one file today; needs an extension to watch a directory of subagent JSONLs and fold their events into the same handler.
- A small parser to detect "open Task tool_use" — track tool_use_id of `name == "Task"` blocks, mark resolved when matching `tool_result` arrives.

### 3c. Auto-sync host Claude credentials into the claudebox mount

**Symptom**: deep-mode turns started failing with `"Failed to authenticate. API Error: 401"` inside model output, ~daily. Root cause: host's `~/.claude/.credentials.json` is refreshed by Claude Code (~24h OAuth refresh cadence). The claudebox container's bind-mount at `deploy/claudebox/claude-config/.credentials.json` is a separate copy that lags. Manual `cp` was the workaround until now.

**Fix** (new module `src/gmail_search/claudebox_creds.py:sync_credentials_if_stale`):
1. **Web server startup hook** (`server.py` `@app.on_event("startup")`): always run a sync. Catches the most common case (host rotated overnight, server restarts in the morning).
2. **Per-deep-turn check** (`service.py:_real_run`): cheap mtime comparison; copies only if drifted. Catches the case where a long-running web server falls behind because the host rotated mid-uptime.

**Behavior**:
- No-op when mtimes match (microsecond `os.stat` cost).
- Backs up the stale mount file to `*.stale-bak-<ts>` before overwrite.
- Mode 0600 enforced on the new file.
- Mtime mirrored from source so the next compare matches.
- Failures log a warning and continue — never raises (the deep turn proceeds and claudebox surfaces its own 401 if creds are still bad).

### 3b. GC sweeper for per-conversation workspaces

**Why now**: workspaces accumulate forever otherwise (`gc.py` only handles `data/agent_scratch/` and `agent_artifacts`).

**The footgun the original plan missed**: GC by workspace mtime alone deletes long-idle but DB-active conversations. The next turn tries to `--resume` against missing state with no recovery path.

**Fix**: drive GC from the DB, not the filesystem.

```
def prune_conversation_workspaces(retention_days: int = 30):
    # Find conversations whose latest activity is older than retention
    stale_ids = SELECT c.id FROM conversations c
                LEFT JOIN agent_sessions s ON s.conversation_id = c.id
                GROUP BY c.id
                HAVING MAX(COALESCE(s.finished_at, c.updated_at)) < NOW() - INTERVAL ...
    for cid in stale_ids:
        delete deploy/claudebox/workspaces/deep-conv-{cid}/
        delete row from conversation_claude_session
        # leave conversations row + agent_sessions intact for analytics
```

**Recovery path** (also missed in v1): the `--resume` flow must catch "session UUID exists but JSONL/workspace is missing" and gracefully fall back to a fresh turn (delete the stale `conversation_claude_session` row, run without `--resume`). That gives idempotent recovery if GC ran while a conversation was idle.

**Code changes**:

| File | Change |
|------|--------|
| `src/gmail_search/agents/gc.py` | Add `prune_conversation_workspaces` per above. |
| `src/gmail_search/agents/service.py` (turn entrypoint) | Stat the workspace dir before `--resume`; if missing, drop the mapping row + run fresh. |

---

## Phase 4 — Sidebar with live state

**Backend** — new `GET /api/conversations/live`:
```json
{
  "conversations": [
    {
      "id": "...",
      "title": "...",
      "updated_at": "...",
      "claude_session_uuid": "...",
      "latest_session": {
        "id": "...",
        "status": "running" | "done" | "error",
        "started_at": "...",
        "finished_at": null,
        "tool_call_count": 23,
        "last_event_at": "..."
      }
    }
  ]
}
```

Joins `conversations` ⨝ `conversation_claude_session` ⨝ latest `agent_sessions` ⨝ count of `agent_events` per session.

**SSE** — `/api/conversations/<conversation_id>/events`:
- Today's SSE is session-scoped (`runtime_claude.py:401-406`). Now needs to be conversation-scoped — multiple sub-agent sessions per conversation, multiplexed onto one stream keyed by `conversation_id`.
- Implementation: subscriber holds a Postgres `LISTEN` on a `conversation_events_<id>` channel; producers (`_record_call` from Phase 2 + sub-agent stage events) `NOTIFY` after writing to `agent_events`.

**UI**:
- Sidebar component lists all conversations from `/api/conversations/live`, polls every few seconds OR opens a single SSE that streams summary updates.
- Each row: title + status chip + tool-count + relative timestamp.
- Click → existing chat view + new debug tab.

---

## Phase 5 — Per-conversation debug pane

For each conversation the pane shows:

| Section | Source |
|---------|--------|
| Tool-call timeline | `agent_events` joined on `agent_sessions.conversation_id`, ordered by `(session_id, seq)`. After Phase 2 this is comprehensive — every MCP call is here. |
| Artifacts | `agent_artifacts` joined the same way. |
| Workspace tree | `deploy/claudebox/workspaces/deep-conv-<id>/` listing — read via a new `GET /api/conversations/<id>/workspace/tree` endpoint. |
| Raw JSONL | `claude-config/projects/-workspaces-deep-conv-<id>/<claude_session_uuid>.jsonl` — served via `GET /api/conversations/<id>/jsonl` (streamed; potentially large). |

If Phase 0 said `--resume` creates new files per turn, the JSONL section becomes a list of files (one per turn) keyed by `agent_sessions.id`. Either way the user sees everything, just different layout.

---

## Non-deep turns

The chat UI also has a non-deep "fast" path that doesn't go through claudebox. For these:
- No workspace, no JSONL.
- They DO write to `agent_events` if we route their tool calls through the same MCP server (do they today? need to verify).
- Sidebar status: show them as "fast" turns with no debug pane (or a minimal one with just the chat history).

**Action**: explicitly document this in the UI so users know why the debug pane is empty for some turns.

---

## Honest effort estimate

| Phase | Effort | Notes |
|-------|--------|-------|
| 0: `--resume` test | ½ day | Gates everything. Don't skip. |
| 1: stable workspace + UUID claim + sanitization migration | 1 day | Includes the schema add, the migration audit, the conversation upsert tightening. |
| 2: side-channel → `agent_events` rewrite | ½ day | Real risk: change touches both MCP and orchestrator drain logic; needs careful test. |
| 2.5: concurrency verification + UI 409 plumbing | ½ day | Trivial backend, real UI work. |
| 3: GC sweeper + recovery path | ½ day | |
| 4: sidebar API + SSE channel | 1 day | LISTEN/NOTIFY plumbing is new ground in this codebase. |
| 5: debug pane (UI + new API endpoints) | 1.5 days | UI-heavy. |

**Total: ~5 days focused work.** With unknowns from Phase 0 and some inevitable yak-shaving, plan for 7. Codex's "fantasy" callout was fair — 3 days was wishful.

---

## Risks I'm leaving on the table

- **`--resume` context-window drift**: a long-running conversation's `--resume` will replay its full prior context. After dozens of turns, that prompt becomes huge and slow. Mitigation = periodic compaction (drop a `--resume` and start fresh, taking the loss). Not in this plan; should be a follow-up once we have data.
- **Concurrent reads of the workspace dir** (e.g., debug pane is open in the UI while a turn writes there): probably fine because reads are non-locking, but a streamed JSONL serve could observe partial writes. Acceptable — frontend re-polls.
- **Conversation deletion**: not specified. If the user deletes a conversation, we should cascade-delete the workspace dir + the `conversation_claude_session` row. Probably want a UI confirmation.

---

## Investigation notes

(Pre-revision findings, kept for reference.)

- `claudebox/api_server.py:200-244`: `RunRequest.resume: Optional[str]` → `--resume <id>`. Path stable.
- `claudebox/api_server.py:140, 272`: `busy_workspaces` is the per-workspace lock.
- `agents/runtime_claude.py:215` (`_build_request_body`): does NOT include `resume` today.
- `agents/service.py:252-255`: workspace is per-turn today.
- `agents/gc.py`: prunes `data/agent_scratch/` + `agent_artifacts` rows. Never touches `deploy/claudebox/workspaces/`.
- `agents/mcp_tools_server.py:166, 232-234`: `unregister_session` → `clear_session_calls` drops in-memory side channel. Drain order issue caught in review.

---

## Review history

### Self-adversarial review (Claude, hostile-reviewer mode)

10 pushbacks, of which the most material were:
- The "drains side channel before unregister" claim was load-bearing and unverified.
- `--resume` semantics asserted as fact, never tested.
- 3-day estimate ignored UI work.
- conversation_id format collision risk.
- 409-on-second-turn is a UI dragon, not a one-liner.
- GC by mtime can nuke active conversations.

### Codex review (read-only audit, hostile mode)

6 findings with file:line proof:
1. First-turn UUID race — needs DB-enforced claim, not "look up then maybe persist".
2. Side-channel drain happens AFTER run completes, not before unregister; plus `_MAX_CALLS_PER_SESSION` silently drops calls.
3. `--resume` JSONL-append is asserted as fact with zero test coverage; tailer follows newest `.jsonl` so a new file would silently bifurcate the debug pane.
4. conversation_id sanitization is a migration problem (sandbox.py:297-305 vs server.py:1176-1217), not an open question.
5. GC by mtime nukes idle-but-active conversations.
6. 3-day estimate is fantasy given everything actually being added.

### Resolution

This v2 folds in all of the above:
- Phase 0 added (gates everything on `--resume` empirical test).
- Phase 1 splits into 1a (sanitization migration) + 1b (new table) + 1c (claim flow with pre-allocated UUID as preferred path).
- Phase 2 added: `_record_call` writes directly to `agent_events`, removing the drain race AND the cap-drop bug.
- Phase 3 (GC) revised to query DB for activity, plus recovery path for missing-workspace.
- Phase 4 SSE explicitly conversation-scoped (not session-scoped).
- Phase 5 has a fallback layout if Phase 0 says JSONL doesn't append.
- Effort revised to 5 days (plan for 7).
- "Non-deep turns" called out as an open behavior question.
