"""Runtime preamble the Analyst's Python snippet runs inside.

Responsibilities:
- Load the retrieval-evidence DataFrame the caller dropped at
  `/work/inputs.json` into the global name `evidence`.
- Open a read-only psycopg connection bound to the `gmail_analyst`
  role, exposed as `db`. Connection is returned autocommit=True +
  default_transaction_read_only=on so even a buggy snippet can't
  accidentally mutate state.
- Import the standard analysis stack (pd, np, plt, sns, sklearn) so
  snippets can just `evidence.groupby("from_addr").size()` without
  boilerplate.
- Provide `save_artifact(name, obj, mime_type=None)` — serialises obj
  to /work/artifacts/<name>.<ext> based on type. The orchestrator
  sweeps that dir after the run and persists what's there.

Convention: every top-level print() is captured as stdout and shown
to the Analyst in the next LLM turn, so the snippet should print
whatever it wants the model to see. Return values are NOT captured
(the snippet is executed via exec, not a REPL).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # belt + suspenders, in case MPLBACKEND env is clobbered

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import psycopg  # noqa: E402

# seaborn + sklearn are imported lazily since many snippets won't need
# them — saves ~150 ms of startup.


_WORK = Path("/work")
_ARTIFACTS = _WORK / "artifacts"
_ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _load_evidence() -> pd.DataFrame:
    """Read /work/inputs.json (written by the orchestrator) and
    materialise `evidence` as a DataFrame. We intentionally DO NOT
    use pickle here — the evidence bytes originate from email bodies
    and other user-influenced data, so pickle.load would be a remote
    code execution gadget the moment real retrieval records get
    plumbed in. JSON is safe and covers the columnar/record shapes
    the orchestrator produces.

    Accepted shapes (must match `sandbox._serialize_evidence_for_sandbox`):
      - {"evidence": null}              → empty DataFrame
      - {"evidence": [...rows...]}      → DataFrame from records
      - {"evidence": {...columns...}}   → DataFrame from columns
      - malformed or missing file       → empty DataFrame (best effort)
    """
    inputs = _WORK / "inputs.json"
    if not inputs.exists():
        return pd.DataFrame()
    try:
        with inputs.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        # A corrupt/missing file shouldn't crash the snippet at import
        # time — give it an empty frame and let the snippet's own
        # `evidence.empty` check drive the error message.
        return pd.DataFrame()
    if not isinstance(payload, dict):
        return pd.DataFrame()
    evidence = payload.get("evidence")
    if evidence is None:
        return pd.DataFrame()
    if isinstance(evidence, list):
        return pd.DataFrame(evidence)
    if isinstance(evidence, dict):
        return pd.DataFrame(evidence)
    return pd.DataFrame()


def _open_db():
    """Open a read-only, TENANT-SCOPED PG connection using the DSN the
    caller passed via env. Returns None if the DSN is absent so the
    snippet at least gets `evidence` without crashing at import time.

    Tenant scoping is the whole point: model-authored Python runs SQL
    like `db.execute("SELECT * FROM messages")` with no WHERE user_id,
    and the DSN is the superuser (BYPASSRLS). Before 2026-07-08 this
    opened that superuser handle with only read-only set — so a snippet
    during user A's turn could read EVERY tenant's mail. We now do what
    /api/sql does: drop to the non-superuser `gmail_search_reader` role
    and bind `app.user_id`, so the same Row-Level-Security policy that
    protects /api/sql applies here too.

    `SET ROLE` is not parameterizable, but the role name is a fixed
    literal (never user input). `app.user_id` IS bound as a parameter
    via set_config. A missing/empty ANALYST_USER_ID leaves app.user_id
    NULL → the RLS policy matches zero rows (fail-closed), so a
    misconfigured run sees nothing rather than everything.

    RESIDUAL LIMITATION (why this is defense-in-depth, not a full
    sandbox for hostile SQL): RLS keys on the `app.user_id` GUC, which
    is SESSION-SETTABLE. Model code holding this connection can run
    `set_config('app.user_id', '<victim>', false)` and read that
    victim's rows — the reader role has SELECT on every tenant's rows,
    gated only by the GUC. So this scoping stops ACCIDENTAL unscoped
    reads and blocks a superuser DSN, but does NOT contain deliberately
    hostile SQL. That is acceptable only because the handle is disabled
    in production: the sole caller passes db_dsn=None, so `db` is None.
    Do NOT wire a live DSN here without a non-GUC isolation mechanism
    (e.g. per-tenant DB roles keyed on current_user, or a
    security-definer view), or the tenant switch above becomes live."""
    dsn = os.environ.get("ANALYST_DB_DSN")
    if not dsn:
        return None
    user_id = os.environ.get("ANALYST_USER_ID") or ""
    conn = psycopg.connect(dsn, autocommit=True)
    with conn.cursor() as cur:
        # `SET ROLE` is REVERSIBLE: hostile code can `RESET ROLE` back to the
        # LOGIN role (session_user), or `SET ROLE` to any role the login is a
        # MEMBER of. So scoping only holds if the login role — and every role
        # it can reach by membership — is non-superuser and non-BYPASSRLS.
        # pg_has_role(session_user, oid, 'MEMBER') covers direct + inherited
        # membership. If ANY reachable role is privileged, REFUSE the handle
        # (fail closed: `db` is None) rather than expose an escapable one.
        # Enabling the sandbox `db` requires a dedicated
        # NOSUPERUSER/NOBYPASSRLS login role with no privileged memberships.
        row = cur.execute(
            "SELECT bool_or(r.rolsuper OR r.rolbypassrls) "
            "FROM pg_roles r WHERE pg_has_role(session_user, r.oid, 'MEMBER')"
        ).fetchone()
        # Fail CLOSED on a NULL/absent result too (should be impossible, but a
        # missing verdict must never be read as "safe").
        if row is None or row[0] is None or row[0]:
            conn.close()
            return None
        cur.execute("SET default_transaction_read_only = on")
        # Bind the tenant BEFORE dropping privilege (either role can set a
        # namespaced GUC; doing it first keeps intent obvious).
        cur.execute("SELECT set_config('app.user_id', %s, false)", (user_id,))
        # Narrow to the reader role's grants (SELECT on LLM-facing tables
        # only). RLS's tenant_isolation policy applies because the login
        # role — verified above — has no BYPASSRLS to reset back to.
        cur.execute("SET ROLE gmail_search_reader")
    return conn


evidence: pd.DataFrame = _load_evidence()
db = _open_db()


_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".html": "text/html",
    ".csv": "text/csv",
    ".json": "application/json",
    ".txt": "text/plain",
}


def _infer_ext_and_mime(obj, name: str) -> tuple[str, str]:
    """Pick a file extension + mime based on object type, with name hint
    as override. Returning `.png` for matplotlib figures is the common
    path; DataFrames → csv; strings → txt."""
    suffix = Path(name).suffix.lower()
    if suffix:
        return suffix, _MIME_BY_EXT.get(suffix, "application/octet-stream")
    if isinstance(obj, plt.Figure):
        return ".png", "image/png"
    if isinstance(obj, pd.DataFrame):
        return ".csv", "text/csv"
    if isinstance(obj, (bytes, bytearray)):
        return ".bin", "application/octet-stream"
    return ".txt", "text/plain"


def save_artifact(name: str, obj, mime_type: str | None = None) -> str:
    """Persist an analysis artifact to /work/artifacts/. The orchestrator
    sweeps that dir post-run, uploads bytes to the agent_artifacts
    table, and returns ids the Writer can cite as [art:<id>].

    Supported obj types: matplotlib Figure (saved as PNG), pandas
    DataFrame (CSV), str (text), bytes (raw). Explicit mime_type wins
    if provided.
    """
    ext, mime = _infer_ext_and_mime(obj, name)
    if mime_type:
        mime = mime_type
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    if not safe_name.endswith(ext):
        safe_name += ext
    dest = _ARTIFACTS / safe_name
    if isinstance(obj, plt.Figure):
        obj.savefig(dest, bbox_inches="tight")
        plt.close(obj)
    elif isinstance(obj, pd.DataFrame):
        obj.to_csv(dest, index=False)
    elif isinstance(obj, (bytes, bytearray)):
        dest.write_bytes(bytes(obj))
    else:
        dest.write_text(str(obj), encoding="utf-8")
    # Drop a manifest line the caller will slurp to build the event.
    with (_ARTIFACTS / "_manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(f'{{"name": "{safe_name}", "mime_type": "{mime}"}}\n')
    return safe_name
