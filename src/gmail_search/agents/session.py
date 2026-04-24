"""Session + event persistence for the deep-analysis flow.

Each POST to /api/agent/analyze creates one `agent_sessions` row and
streams a sequence of `agent_events` rows as the planner / retriever /
analyst / writer / critic agents do their work. The event table is
both the transcript the UI renders and the durable record we can
replay if the connection drops mid-turn.

Kept deliberately small and synchronous — no ORM, no event bus. The
HTTP layer writes events one at a time via `append_event()`; readers
(Next.js proxy) poll via SSE by `seq` > last-seen-seq.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class SessionEvent:
    """One row of the agent transcript. `seq` is monotonic within a
    session — SSE clients resume from `?after=<seq>`."""

    session_id: str
    seq: int
    agent_name: str
    kind: str
    payload: dict[str, Any]
    created_at: str


def new_session_id() -> str:
    """Short 16-hex-char id so URLs + log filenames stay readable.
    Probability of collision with any existing session is negligible
    at our volume (deep turns are human-initiated, seconds-apart)."""
    return uuid.uuid4().hex[:16]


def create_session(
    conn,
    *,
    session_id: str,
    conversation_id: str | None,
    mode: str,
    question: str,
) -> None:
    """Insert the row that anchors this turn's events. Status stays
    `running` until `finalize_session` flips it to `done` or `error`."""
    conn.execute(
        """INSERT INTO agent_sessions (id, conversation_id, mode, question, status)
           VALUES (%s, %s, %s, %s, 'running')""",
        (session_id, conversation_id, mode, question),
    )
    conn.commit()


def append_event(
    conn,
    *,
    session_id: str,
    agent_name: str,
    kind: str,
    payload: dict[str, Any],
) -> int:
    """Append the next event to the session. Returns the assigned
    `seq` so the caller can emit it to the SSE stream. We compute seq
    from MAX(seq)+1 inside a single INSERT — the session_id+seq unique
    constraint guards against any race, and the INSERT will fail loud
    if two writers try the same seq."""
    row = conn.execute(
        """INSERT INTO agent_events (session_id, seq, agent_name, kind, payload)
           VALUES (
             %s,
             COALESCE((SELECT MAX(seq) FROM agent_events WHERE session_id = %s), 0) + 1,
             %s, %s, %s::jsonb
           )
           RETURNING seq""",
        (session_id, session_id, agent_name, kind, json.dumps(payload)),
    ).fetchone()
    conn.commit()
    return int(row["seq"])


def fetch_events_after(
    conn,
    session_id: str,
    *,
    after_seq: int = 0,
    limit: int = 500,
) -> Iterator[SessionEvent]:
    """Pull events newer than `after_seq`. Used by the SSE reader to
    resume after a reconnect."""
    rows = conn.execute(
        """SELECT session_id, seq, agent_name, kind, payload, created_at
           FROM agent_events
           WHERE session_id = %s AND seq > %s
           ORDER BY seq ASC
           LIMIT %s""",
        (session_id, after_seq, limit),
    ).fetchall()
    for r in rows:
        payload = r["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        yield SessionEvent(
            session_id=r["session_id"],
            seq=int(r["seq"]),
            agent_name=r["agent_name"],
            kind=r["kind"],
            payload=payload,
            created_at=str(r["created_at"]),
        )


def finalize_session(
    conn,
    session_id: str,
    *,
    status: str,
    final_answer: str | None = None,
) -> None:
    """Close out the session row. `status` is 'done' for a normal
    finish, 'error' when the root agent raised. `finished_at` is
    stamped server-side so it matches the DB clock."""
    conn.execute(
        """UPDATE agent_sessions
           SET status = %s, final_answer = %s, finished_at = NOW()
           WHERE id = %s""",
        (status, final_answer, session_id),
    )
    conn.commit()


def save_artifact(
    conn,
    *,
    session_id: str,
    name: str,
    mime_type: str,
    data: bytes,
    meta: dict[str, Any] | None = None,
) -> int:
    """Persist an analyst-produced artifact (plot, CSV, etc.) and
    return the id the Writer cites as [art:<id>]. 10 MB cap is checked
    by the caller; the DB accepts any BYTEA and we don't want to
    surface a specific error here that'd duplicate the sandbox's own
    byte-budget policing."""
    row = conn.execute(
        """INSERT INTO agent_artifacts (session_id, name, mime_type, data, meta)
           VALUES (%s, %s, %s, %s, %s::jsonb)
           RETURNING id""",
        (session_id, name, mime_type, data, json.dumps(meta or {})),
    ).fetchone()
    conn.commit()
    return int(row["id"])


def get_artifact(conn, artifact_id: int) -> tuple[str, str, bytes] | None:
    """Fetch (name, mime_type, data) for the artifact id, or None. The
    /api/artifact/<id> endpoint returns the bytes directly; this helper
    is the only read path (intentional — we don't list all artifacts
    for a session, it's always by id)."""
    row = conn.execute(
        """SELECT name, mime_type, data FROM agent_artifacts WHERE id = %s""",
        (artifact_id,),
    ).fetchone()
    if row is None:
        return None
    data = row["data"]
    return row["name"], row["mime_type"], bytes(data) if isinstance(data, memoryview) else data


def session_log_path(data_dir: Path, session_id: str) -> Path:
    """Where to tail the Analyst's raw stdout/stderr during a run.
    Separate from the event table because the DB payload is JSON and
    we don't want a misbehaving snippet flooding it. Events store a
    path pointer; operators tail the file directly."""
    logs_dir = data_dir / "agent_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{session_id}.log"
