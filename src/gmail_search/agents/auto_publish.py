"""End-of-turn auto-publish sweep for deep-mode runs.

Belt-and-suspenders for the case where Claude wrote a file (via Bash,
external tool, sandbox, anything) but forgot to call
`publish_artifact`. Without this sweep, those files sit on the host
filesystem invisible to the user.

The sweep runs AFTER the LLM finishes a turn:
  1. enumerate files under `deploy/claudebox/workspaces/<workspace>/`
     and `data/agent_scratch/<conversation_id>/` whose mtime is at or
     after `turn_started_at`,
  2. drop the ones that were already published by name (dedup query
     against `agent_artifacts` for this `session_id`),
  3. drop dotfiles, runner scaffolding (`run.py`, `inputs.json`,
     `_manifest.jsonl`, etc.), files under hidden dirs, files outside
     the size band,
  4. insert each survivor via `save_artifact` and return a list of
     `{id, name, mime_type, size}` dicts so the caller can append
     `[art:<id>] <name>` chips to its final answer text or emit an
     `auto_published` event.

Failures inside the loop are logged + skipped — the sweep is a safety
net, never a hard dependency.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

from gmail_search.agents.session import save_artifact

logger = logging.getLogger(__name__)


_PUBLISH_WORKSPACE_ROOT = "deploy/claudebox/workspaces"
_PUBLISH_SCRATCH_ROOT = "data/agent_scratch"

# Filenames the sandbox + claudebox runner emit. None of these are
# user-visible artifacts; they are runner scaffolding that lives
# alongside the model's real outputs and would clutter the answer
# chip-list if we surfaced them.
_SCAFFOLDING_NAMES = frozenset(
    {
        "run.py",
        "inputs.json",
        "_manifest.jsonl",
    }
)


def _publish_roots(workspace: str | None, conversation_id: str | None) -> list[Path]:
    """Return the directories the sweep should walk. Either or both
    may be absent on a given turn (a turn could be workspace-only or
    scratch-only). Missing dirs are filtered later by `is_dir()`."""
    roots: list[Path] = []
    if workspace:
        roots.append(Path(_PUBLISH_WORKSPACE_ROOT) / workspace)
    if conversation_id:
        roots.append(Path(_PUBLISH_SCRATCH_ROOT) / conversation_id)
    return roots


def _has_hidden_segment(rel: Path) -> bool:
    """True if any segment of `rel` starts with a dot (`.git/foo`,
    `.mpl/cache`, etc.). The file itself starting with a dot is also
    caught here since `rel.parts[-1]` is its name."""
    for part in rel.parts:
        if part.startswith("."):
            return True
    return False


def _is_under_pycache(rel: Path) -> bool:
    """`__pycache__/` byte-compiled cruft never belongs in the answer.
    Matches both `_pycache_` (per the spec, defensive) and the actual
    Python convention `__pycache__`."""
    for part in rel.parts:
        if part in ("__pycache__", "_pycache_"):
            return True
    return False


def _is_artifacts_manifest(rel: Path) -> bool:
    """The sandbox writes `artifacts/_manifest.jsonl`; treat it as
    runner scaffolding regardless of where it sits in the tree."""
    parts = rel.parts
    if len(parts) >= 2 and parts[-2] == "artifacts" and parts[-1] == "_manifest.jsonl":
        return True
    return False


def _should_skip_path(rel: Path) -> bool:
    """Apply the always-skip filters: hidden segments, pycache,
    runner scaffolding files, manifests."""
    if _has_hidden_segment(rel):
        return True
    if _is_under_pycache(rel):
        return True
    if rel.name in _SCAFFOLDING_NAMES:
        return True
    if _is_artifacts_manifest(rel):
        return True
    return False


def _file_is_in_size_band(
    size: int,
    *,
    min_bytes_per_file: int,
    max_bytes_per_file: int,
    rel: Path,
) -> bool:
    """Reject files outside the publishable size band. Empty / tiny
    files are usually scratch placeholders; oversized files exceed the
    BYTEA column cap and would just error inside `save_artifact`. Both
    paths log so an operator can see why something was skipped."""
    if size < min_bytes_per_file:
        logger.info("auto_publish: skipping %s (size %d < min %d)", rel, size, min_bytes_per_file)
        return False
    if size > max_bytes_per_file:
        logger.warning(
            "auto_publish: skipping %s (size %d > max %d) — model should save a smaller version",
            rel,
            size,
            max_bytes_per_file,
        )
        return False
    return True


def _fetch_existing_names(conn, session_id: str) -> set[str]:
    """Names already published for this `session_id`. Used to dedup
    against explicit `publish_artifact` calls so the sweep doesn't
    create a second copy of a file Claude already cited.

    Failures are non-fatal: if the query blows up we return an empty
    set and let the loop run unfiltered. Worst case we publish a
    duplicate; better than silently swallowing the whole sweep."""
    try:
        rows = conn.execute(
            "SELECT name FROM agent_artifacts WHERE session_id = %s",
            (session_id,),
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("auto_publish: dedup query failed for session %s: %s", session_id, exc)
        return set()
    if not rows:
        return set()
    names: set[str] = set()
    for r in rows:
        # Support dict-row (psycopg with dict_row factory) and tuple-row.
        try:
            name = r["name"]
        except (KeyError, TypeError):
            try:
                name = r[0]
            except (IndexError, TypeError):
                continue
        if isinstance(name, str):
            names.add(name)
    return names


def _walk_candidate_files(root: Path, *, turn_started_at: float) -> list[tuple[Path, Path]]:
    """Yield `(absolute_path, path_relative_to_root)` pairs for files
    under `root` whose mtime is at or after `turn_started_at`. Returns
    an empty list when `root` is missing — a sweep over an
    inactive root is silent, not an error."""
    if not root.exists() or not root.is_dir():
        return []
    out: list[tuple[Path, Path]] = []
    try:
        for p in root.rglob("*"):
            try:
                if not p.is_file():
                    continue
                if p.stat().st_mtime < turn_started_at:
                    continue
                rel = p.relative_to(root)
                out.append((p, rel))
            except OSError as exc:
                logger.warning("auto_publish: stat/relpath failed for %s: %s", p, exc)
                continue
    except OSError as exc:
        logger.warning("auto_publish: walking %s failed: %s", root, exc)
    return out


def _sniff_mime_for(name: str) -> str:
    """`mimetypes.guess_type` with the project's standard fallback."""
    guessed, _ = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"


def _read_file_bytes(path: Path) -> bytes | None:
    """Read a file's bytes, or return None if read fails. Read errors
    don't fail the sweep — log and skip."""
    try:
        return path.read_bytes()
    except OSError as exc:
        logger.warning("auto_publish: read failed for %s: %s", path, exc)
        return None


def _publish_one(
    conn,
    *,
    session_id: str,
    abs_path: Path,
    rel_path: Path,
) -> dict[str, Any] | None:
    """Insert a single candidate into agent_artifacts. Returns the
    `{id, name, mime_type, size}` dict on success, None on failure
    (logged but never raised)."""
    data = _read_file_bytes(abs_path)
    if data is None:
        return None
    name = abs_path.name
    mime_type = _sniff_mime_for(name)
    try:
        art_id = save_artifact(
            conn,
            session_id=session_id,
            name=name,
            mime_type=mime_type,
            data=data,
            meta={"auto_published": True, "source_path": str(rel_path)},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "auto_publish: save_artifact failed for %s (session=%s): %s",
            rel_path,
            session_id,
            exc,
        )
        return None
    return {
        "id": art_id,
        "name": name,
        "mime_type": mime_type,
        "size": len(data),
    }


def auto_publish_unpublished_files(
    conn,
    *,
    session_id: str,
    workspace: str | None,
    conversation_id: str | None,
    turn_started_at: float,
    max_files: int = 12,
    max_bytes_per_file: int = 10 * 1024 * 1024,
    min_bytes_per_file: int = 64,
) -> list[dict[str, Any]]:
    """Sweep workspace + scratch dirs for unpublished files written
    during this turn and insert them as artifacts.

    See module docstring for the full filter chain. Returns the list
    of newly-published `{id, name, mime_type, size}` dicts in publish
    order. Always returns a list — empty when there's nothing to do."""
    roots = _publish_roots(workspace, conversation_id)
    if not roots:
        return []

    existing_names = _fetch_existing_names(conn, session_id)
    published: list[dict[str, Any]] = []

    for root in roots:
        if len(published) >= max_files:
            break
        for abs_path, rel_path in _walk_candidate_files(root, turn_started_at=turn_started_at):
            if len(published) >= max_files:
                logger.info(
                    "auto_publish: reached max_files=%d; halting sweep for session %s",
                    max_files,
                    session_id,
                )
                break
            if _should_skip_path(rel_path):
                continue
            try:
                size = abs_path.stat().st_size
            except OSError as exc:
                logger.warning("auto_publish: stat failed for %s: %s", abs_path, exc)
                continue
            if not _file_is_in_size_band(
                size,
                min_bytes_per_file=min_bytes_per_file,
                max_bytes_per_file=max_bytes_per_file,
                rel=rel_path,
            ):
                continue
            if abs_path.name in existing_names:
                # Already published explicitly — don't double-insert.
                continue
            row = _publish_one(
                conn,
                session_id=session_id,
                abs_path=abs_path,
                rel_path=rel_path,
            )
            if row is None:
                continue
            published.append(row)
            existing_names.add(row["name"])
            logger.info(
                "auto_publish: inserted artifact id=%d name=%s size=%d (session=%s)",
                row["id"],
                row["name"],
                row["size"],
                session_id,
            )

    return published


def build_auto_publish_footer(published: list[dict[str, Any]]) -> str:
    """Format the chip list appended to a native-mode final answer.
    Empty list → empty string (caller can unconditionally concat).
    The leading separator + intro line is intentional so the chips
    stand visually apart from the model's prose."""
    if not published:
        return ""
    lines = [
        "",
        "",
        "---",
        "",
        "_The following files were produced during this analysis but not "
        "explicitly cited above. They're available for you to download:_",
        "",
    ]
    for row in published:
        lines.append(f"- [art:{row['id']}] **{row['name']}**")
    lines.append("")
    return "\n".join(lines)
