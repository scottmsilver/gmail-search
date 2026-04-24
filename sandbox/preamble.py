"""Runtime preamble the Analyst's Python snippet runs inside.

Responsibilities:
- Load the retrieval-evidence DataFrame the caller dropped at
  `/work/inputs.pkl` into the global name `evidence`.
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

import os
import pickle
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
    inputs = _WORK / "inputs.pkl"
    if not inputs.exists():
        return pd.DataFrame()
    with inputs.open("rb") as fh:
        payload = pickle.load(fh)
    # Caller drops either a DataFrame or a dict with an "evidence" key.
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, dict) and "evidence" in payload:
        return pd.DataFrame(payload["evidence"])
    return pd.DataFrame()


def _open_db():
    """Open a read-only PG connection using the DSN the caller passed
    via env. Returns None if the DSN is absent so the snippet at least
    gets `evidence` without crashing at import time."""
    dsn = os.environ.get("ANALYST_DB_DSN")
    if not dsn:
        return None
    conn = psycopg.connect(dsn, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
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
