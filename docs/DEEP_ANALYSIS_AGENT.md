# Deep Analysis Agent

**Status:** In-progress (Phase 1). User sign-off on shape: (c) local Docker
sandbox, yes critic agent, explicit `mode: "deep"` toggle.

**Goal.** An optional "deep analysis" path for the chat UI that uses the
Google Agent Development Kit (ADK) with multiple coordinated sub-agents
and a real Python code sandbox. Wins for the class of questions the
existing single-agent flow can't express well: cross-message
aggregation, trend analysis, clustering, charts, multi-step numerical
reasoning, contradiction-finding across threads.

The regular "chat" mode stays as-is. Deep mode is opt-in per turn via
a toggle next to the model picker.

## Scope boundary

- Kept local: Gemini API (same key), our Postgres, Docker sandbox on
  the same host. No new vendors.
- Not in scope: Vertex AI Agent Engine, persistent long-running
  sessions across turns, multi-user state. Each deep turn is its own
  ephemeral session.

## Architecture

```
Browser ─────────────► Next.js /api/chat ─────► (mode=chat)  existing agent
                                            └─► (mode=deep)  proxy to Python
                                                                     │
                              ┌──────────────────────────────────────┘
                              ▼
                      Python /api/agent/analyze  (FastAPI, new service)
                              │
                              ▼
                        ADK root agent
                              │
         ┌───────────┬────────┼────────┬───────────┐
         ▼           ▼        ▼        ▼           ▼
      Planner   Retriever  Analyst   Writer     Critic
                    │         │
                    ▼         ▼
             our /api/*    Docker sandbox
             (read-only)   (pandas + psycopg read-only, 60s budget)
```

## Sub-agents

1. **Planner** — reads the user question, outputs a JSON plan:
   `{retrieval_steps, analysis_steps, expected_output_shape}`. One
   LLM call, no tools.

2. **Retriever** — uses ADK `FunctionTool` wrappers over our existing
   endpoints: `search_emails`, `query_emails`, `sql_query`, `get_thread`,
   `get_attachment`. Emits structured `EvidenceBundle` rows with
   `cite_ref` on every item.

3. **Analyst** — the new one. Gets a Python sandbox pre-seeded with:
   - `evidence`: pandas DataFrame from the retrieval bundles
   - `db`: a read-only psycopg connection (role `gmail_analyst`)
   - `plt`, `pd`, `np`, `sklearn` — standard analysis stack
   - `save_artifact(name, obj)` — saves PNG / HTML / CSV to
     `agent_artifacts` table, returns an id.

   Analyst writes Python, we execute it, it sees stdout + any raised
   exception, can iterate up to N times.

4. **Writer** — composes the final markdown answer grounded ONLY in
   Retriever evidence + Analyst output. Handles citations
   (`[ref:<cite>]` for threads, `[att:<id>]` for attachments, and a
   new `[art:<id>]` for analyst artifacts that the UI renders
   inline).

5. **Critic** — adversarial pass. For each claim in the Writer's
   draft: is it grounded in a named evidence row or artifact? Any
   citation ids that didn't appear in this turn? Any numerical
   contradiction with the retrieved data? On any violation, sends
   notes back to Writer for ONE revision. Hard cap of two critic
   rounds.

## Sandbox

Docker container `gmail-search-analyst`, built from a pinned Python
image with:

- psycopg[binary] + pandas + matplotlib + numpy + scikit-learn +
  seaborn (all stable, pinned versions in `sandbox/requirements.txt`)
- No network access (`--network=none`) except for the postgres socket,
  which we expose via a mounted UNIX socket to avoid needing network
  ACLs
- Read-only root FS + writable `/work` tmpfs for scratch
- 512 MB RAM cap, 30 CPU-second budget per execute call, 60s wall
  clock
- Read-only Postgres role `gmail_analyst` with SELECT-only on:
  `messages`, `attachments`, `message_summaries`, `thread_summary`,
  `embeddings` (id + vector cols only), `topics`, `message_topics`,
  `contact_frequency`. No grants on `costs`, `sync_state`,
  `conversations`, `conversation_messages`, `agent_*`.

Execute flow:
1. Analyst emits `{code, inputs?}`.
2. Python service writes the code to `/work/run.py`, drops a
   `/work/inputs.pkl` with the retrieval DataFrame, spawns the
   sandbox container.
3. Container runs `python /work/run.py`, captures stdout / stderr,
   sweeps `/work/artifacts/*` for saved outputs, writes metadata
   back to `agent_artifacts` table.
4. Python service returns `{stdout, stderr, artifact_ids, wall_ms}`.
5. Analyst sees the result, decides to iterate or hand off.

## New schema (next migration)

```sql
CREATE TABLE agent_sessions (
  id           TEXT PRIMARY KEY,
  conversation_id TEXT,
  mode         TEXT NOT NULL,    -- 'deep' for now; room for more modes later
  question     TEXT NOT NULL,
  plan         JSONB,
  final_answer TEXT,
  started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  finished_at  TIMESTAMPTZ,
  status       TEXT NOT NULL DEFAULT 'running'   -- running | done | error
);

CREATE TABLE agent_events (
  id           BIGSERIAL PRIMARY KEY,
  session_id   TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
  seq          INT NOT NULL,
  agent_name   TEXT NOT NULL,    -- 'planner' | 'retriever' | 'analyst' | ...
  kind         TEXT NOT NULL,    -- 'plan' | 'tool_call' | 'tool_result'
                                 -- | 'code_run' | 'code_result' | 'draft'
                                 -- | 'critique' | 'revision' | 'final'
  payload      JSONB NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (session_id, seq)
);

CREATE TABLE agent_artifacts (
  id           BIGSERIAL PRIMARY KEY,
  session_id   TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
  name         TEXT NOT NULL,       -- human-readable
  mime_type    TEXT NOT NULL,       -- image/png | text/html | text/csv | ...
  data         BYTEA NOT NULL,      -- artifact bytes; capped at 10 MB
  meta         JSONB,               -- { rows, cols, summary_stats, ... }
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

## HTTP surface (new)

- `POST /api/agent/analyze` — body `{conversation_id, question,
  model}`. Returns SSE stream of events. Next.js `/api/chat` with
  `mode: "deep"` proxies this.
- `GET /api/artifact/<id>` — returns the artifact bytes with its
  mime type. UI renders PNG / HTML inline, offers CSV as a download.

## UI treatment

- New "Deep" toggle pill next to the model picker. Sticky per
  conversation (saved in `conversations.settings`).
- Deep-mode assistant messages get a left-border accent color and an
  "Analysis" badge so user knows the turn was expensive.
- Panel under the answer shows collapsible: **Plan** → **Retrieved**
  (N threads, N attachments) → **Analyst runs** (code + output +
  artifacts) → **Critic notes** → **Final**.
- Artifacts render inline: PNGs as images, HTML as sandboxed iframes,
  CSVs as a small table with "Download" link.

## Phases (build order)

- [ ] **Phase 1 — Foundation** *(this session)*
  - Design doc (this file)
  - Docker sandbox image + `sandbox/requirements.txt`
  - Read-only DB role + migration
  - Schema tables (agent_sessions, agent_events, agent_artifacts)
  - Python service skeleton (FastAPI) with `POST /api/agent/analyze`
    returning a hardcoded fake event stream
  - Next.js `mode: "deep"` flag + SSE proxy — end-to-end smoke test
- [ ] **Phase 2 — Sandbox executor**
  - `agents/sandbox.py`: `execute(code, inputs) -> ExecResult`
  - Container lifecycle, resource caps, artifact sweep
  - Unit tests for execute paths: happy, timeout, OOM, bad code
- [ ] **Phase 3 — Single Analyst agent (no others yet)**
  - Pre-seed evidence DataFrame from a one-shot search_emails call
  - Iteration loop capped at 3
  - Artifacts saved + returned in the event stream
- [ ] **Phase 4 — Planner + Retriever + Writer**
  - ADK FunctionTool wrappers for our existing endpoints
  - Root agent orchestration
- [ ] **Phase 5 — Critic**
  - Post-Writer adversarial pass + one revision loop
- [ ] **Phase 6 — UI**
  - Deep toggle, progress panel, inline artifact rendering
- [ ] **Phase 7 — Hardening**
  - E2E tests with mocked Gemini
  - Artifact GC (drop > 30 days after session finished)
  - Cost logging (ADK usage goes through `costs` table)

## Non-goals (v1)

- Multi-turn analyst sessions (each turn is its own session)
- User-editable Python in the UI
- Agent-suggested follow-up questions (agent only answers what's asked)
- Vertex AI hosting (all local)

## Open questions to revisit

- Should the analyst be allowed to write back to Postgres (e.g., tag
  messages)? V1 says no; write-path belongs on the user's explicit
  command.
- Artifact retention — the spec above is 30 days; may need to be
  longer for citation stability across conversations.
- Whether the sandbox should cache pip installs between runs (faster,
  more attack surface) — starting pinned-and-fresh for simplicity.
