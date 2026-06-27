import functools
import logging
import os
from pathlib import Path

import click

from gmail_search.config import load_config
from gmail_search.store.cost import check_budget, get_spend_breakdown, get_total_spend
from gmail_search.store.db import get_connection, init_db

logger = logging.getLogger(__name__)


def _setup_context(ctx, data_dir, config_path, verbose):
    """Shared initialisation — called by the group or lazily by subcommands."""
    # Re-init if subcommand provides any explicit override
    has_override = data_dir or (config_path and config_path != "config.yaml") or verbose
    if ctx.obj and ctx.obj.get("_initialised") and not has_override:
        return
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    data_path = Path(data_dir) if data_dir else Path.cwd() / "data"
    cfg = load_config(
        config_path=Path(config_path) if config_path else None,
        data_dir=data_path,
    )
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["data_dir"] = data_path
    ctx.obj["db_path"] = data_path / "gmail_search.db"
    ctx.obj["_initialised"] = True

    data_path.mkdir(parents=True, exist_ok=True)
    init_db(ctx.obj["db_path"])


def _message_for_invite_guard(row):
    """Build a lightweight Message from a catch-up DB row so the
    invitation guard can read sender/subject/body. Only the fields the
    guard touches are populated; the rest get harmless placeholders."""
    import json
    from datetime import datetime, timezone

    from gmail_search.store.models import Message

    labels = row["labels"]
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except (ValueError, TypeError):
            labels = []
    return Message(
        id=row["id"],
        thread_id=row["id"],
        from_addr=row["from_addr"] or "",
        to_addr="",
        subject=row["subject"] or "",
        body_text=row["body_text"] or "",
        body_html="",
        date=datetime(1970, 1, 1, tzinfo=timezone.utc),
        labels=labels if isinstance(labels, list) else [],
        history_id=0,
        raw_json="{}",
    )


def _att_metas_for_message(conn, message_id):
    """Fetch attachment mime/filename pairs for the calendar auto-skip
    check in the catch-up scan. Returns the att_metas shape the guard
    expects (only the keys it reads)."""
    rows = conn.execute(
        "SELECT mime_type, filename FROM attachments WHERE message_id = %s",
        (message_id,),
    ).fetchall()
    return [{"mime_type": r["mime_type"], "filename": r["filename"]} for r in rows]


def common_options(f):
    """Decorator that adds --data-dir, --config, and --verbose to a subcommand."""

    @click.option("--data-dir", type=click.Path(), default=None, help="Data directory path")
    @click.option("--config", "config_path", type=click.Path(), default=None, help="Config file path")
    @click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging")
    @functools.wraps(f)
    def wrapper(data_dir, config_path, verbose, *args, **kwargs):
        ctx = click.get_current_context()
        _setup_context(ctx, data_dir, config_path or "config.yaml", verbose)
        return ctx.invoke(f, *args, **kwargs)

    return wrapper


@click.group(help="Gmail Search — local semantic search over your Gmail")
@click.option("--data-dir", type=click.Path(), default=None, help="Data directory path")
@click.option("--config", "config_path", type=click.Path(), default="config.yaml", help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx, data_dir, config_path, verbose):
    _setup_context(ctx, data_dir, config_path, verbose)


@main.command(help="Download messages from Gmail")
@click.option("--max-messages", type=int, default=None, help="Max messages to download")
@common_options
@click.pass_context
def download(ctx, max_messages):
    from gmail_search.gmail.auth import build_gmail_service
    from gmail_search.gmail.client import download_messages

    cfg = ctx.obj["config"]
    service = build_gmail_service(ctx.obj["data_dir"])
    max_msg = max_messages or cfg["download"].get("max_messages")
    count = download_messages(
        service=service,
        db_path=ctx.obj["db_path"],
        data_dir=ctx.obj["data_dir"],
        batch_size=cfg["download"]["batch_size"],
        max_messages=max_msg,
        max_attachment_size=cfg["attachments"]["max_file_size_mb"] * 1024 * 1024,
    )
    click.echo(f"Downloaded {count} new messages.")


@main.command(help="Sync new messages since last download")
@common_options
@click.pass_context
def sync(ctx):
    from gmail_search.gmail.auth import build_gmail_service
    from gmail_search.gmail.client import sync_new_messages

    cfg = ctx.obj["config"]
    service = build_gmail_service(ctx.obj["data_dir"])
    count = sync_new_messages(
        service=service,
        db_path=ctx.obj["db_path"],
        data_dir=ctx.obj["data_dir"],
        max_attachment_size=cfg["attachments"]["max_file_size_mb"] * 1024 * 1024,
    )
    click.echo(f"Synced {count} new messages.")


@main.command(help="Extract text and images from downloaded attachments (and Drive docs linked from bodies)")
@common_options
@click.pass_context
def extract(ctx):
    """Unified extract step:

    1. Scans every message body for Drive URLs and upserts stub
       attachment rows (idempotent; cheap regex). Download-time
       already does this for new messages — this call catches up
       older messages that were downloaded before the Drive path
       existed.
    2. Iterates every attachment without `extracted_text`/`image_path`.
       - Rows with a raw_path: local dispatch via `extract.dispatch`.
       - Rows with mime `application/vnd.google-apps.*` (no raw_path):
         fetched via Drive API.

    Drive content lands in the attachments table, which means the
    embedding pipeline indexes it and the summarizer sees it — no
    separate enrichment job, no drift between "knows about" and
    "has fetched".
    """
    from tqdm import tqdm

    from gmail_search.extract import dispatch
    from gmail_search.gmail.drive import (
        build_drive_service,
        drive_id_from_stub_filename,
        drive_mime_for_kind,
        extract_drive_ids,
        fetch_doc_text,
    )
    from gmail_search.store.queries import get_attachments_for_message, upsert_drive_stub

    cfg = ctx.obj["config"]
    att_config = cfg.get("attachments", {})
    conn = get_connection(ctx.obj["db_path"])

    # ── 1. Catch-up: seed Drive stubs for any message whose body
    # we have but hasn't been scanned yet. Cheap regex per body.
    # Layering: regex + mime mapping live in gmail/drive (pure API
    # vocabulary), the INSERT lives in store/queries (schema owner).
    try:
        msg_rows = conn.execute("SELECT id, body_text FROM messages WHERE length(body_text) >= 50").fetchall()
        new_stubs = 0
        for r in tqdm(msg_rows, desc="Scanning bodies for Drive links"):
            for drive_id, kind in extract_drive_ids(r["body_text"] or ""):
                new_stubs += upsert_drive_stub(
                    conn,
                    message_id=r["id"],
                    drive_id=drive_id,
                    mime_type=drive_mime_for_kind(kind),
                )
        conn.commit()
        if new_stubs:
            click.echo(f"  inserted {new_stubs} new Drive stubs")
    except Exception as e:
        logger.warning(f"drive-stub scan failed: {e}")

    # ── 1b. URL stub catch-up: same pattern, but for plain URLs.
    # The crawl-urls command fetches the page text and fills
    # extracted_text; here we just seed the stubs so older messages
    # downloaded before the URL path existed get caught up.
    try:
        from gmail_search.gmail.invite_guard import skip_link_crawl_cached
        from gmail_search.gmail.url_extract import extract_crawlable_urls
        from gmail_search.store.queries import upsert_url_stub

        msg_rows = conn.execute(
            "SELECT id, from_addr, subject, body_text, labels FROM messages WHERE length(body_text) >= 50"
        ).fetchall()
        new_url_stubs = 0
        for r in tqdm(msg_rows, desc="Scanning bodies for URLs"):
            # Same invitation guard as the ingest path: skip ALL links
            # for actionable invitations so a GET can't accept an old
            # invite. att_metas (mime/filename) come from the
            # attachments table for the calendar auto-skip case.
            msg = _message_for_invite_guard(r)
            att_metas = _att_metas_for_message(conn, r["id"])
            if skip_link_crawl_cached(conn, msg, att_metas):
                continue
            for url in extract_crawlable_urls(r["body_text"] or "", labels=r["labels"]):
                new_url_stubs += upsert_url_stub(conn, message_id=r["id"], url=url)
        conn.commit()
        if new_url_stubs:
            click.echo(f"  inserted {new_url_stubs} new URL stubs")
    except Exception as e:
        logger.warning(f"url-stub scan failed: {e}")

    # ── 2. Drive service is optional — if token lacks drive.readonly
    # we gracefully skip Drive fetches while local extract continues.
    drive_service = None
    try:
        drive_service = build_drive_service(ctx.obj["data_dir"])
    except Exception as e:
        click.echo(
            f"Drive service unavailable ({e}); skipping Drive fetches. "
            "Delete data/token.json and re-run any command to re-auth with drive.readonly."
        )

    # ── 3. Dispatch loop.
    try:
        rows = conn.execute("SELECT id FROM messages").fetchall()
        updated = 0
        drive_fetched = 0
        drive_failed = 0

        for row in tqdm(rows, desc="Extracting attachments"):
            attachments = get_attachments_for_message(conn, row["id"])
            for att in attachments:
                if att.extracted_text or att.image_path:
                    continue

                # Drive stub path: no raw_path, vnd.google-apps.* mime.
                # All SQL goes through fill_drive_attachment — the cli
                # layer orchestrates, it doesn't write SQL directly.
                if not att.raw_path and att.mime_type.startswith("application/vnd.google-apps."):
                    if drive_service is None:
                        continue
                    drive_id = drive_id_from_stub_filename(att.filename)
                    if not drive_id:
                        continue
                    result = fetch_doc_text(drive_service, drive_id)
                    if result is None:
                        drive_failed += 1
                        continue
                    title, text = result
                    try:
                        fill_drive_attachment(conn, attachment_id=att.id, title=title, text=text, drive_id=drive_id)
                        conn.commit()
                        drive_fetched += 1
                    except Exception as e:
                        logger.warning(f"drive row update failed for {drive_id}: {e}")
                        drive_failed += 1
                    continue

                # Local file path.
                if not att.raw_path or not Path(att.raw_path).exists():
                    continue

                try:
                    result = dispatch(att.mime_type, Path(att.raw_path), att_config)
                except Exception as e:
                    logger.warning(f"Failed to extract {att.filename}: {e}")
                    continue
                if result is None:
                    continue

                updates = {}
                if result.text:
                    updates["extracted_text"] = result.text
                if result.images:
                    updates["image_path"] = str(result.images[0].parent if len(result.images) > 1 else result.images[0])

                if updates:
                    set_clause = ", ".join(f"{k} = %s" for k in updates)
                    conn.execute(
                        f"UPDATE attachments SET {set_clause} WHERE id = %s",
                        (*updates.values(), att.id),
                    )
                    conn.commit()
                    updated += 1
    finally:
        conn.close()
    click.echo(f"Local: {updated} attachments extracted. Drive: {drive_fetched} fetched, {drive_failed} failed.")


@main.command(help="Embed all unembedded messages and attachments")
@click.option("--model", default=None, help="Override embedding model")
@click.option("--budget", type=float, default=None, help="Override budget limit")
@click.option("--force", is_flag=True, help="Re-embed all messages (clears existing message embeddings)")
@click.option("--batch-api", is_flag=True, help="Use Gemini Batch API (50% cheaper, async with polling)")
@common_options
@click.pass_context
def embed(ctx, model, budget, force, batch_api):
    from gmail_search.embed.pipeline import run_embedding_pipeline

    cfg = ctx.obj["config"]
    if model:
        cfg["embedding"]["model"] = model
    if budget:
        cfg["budget"]["max_usd"] = budget

    if force:
        from gmail_search.store.db import clear_query_cache

        emb_model = cfg["embedding"]["model"]
        conn = get_connection(ctx.obj["db_path"])
        deleted = conn.execute(
            "DELETE FROM embeddings WHERE chunk_type = 'message' AND model = %s", (emb_model,)
        ).rowcount
        conn.commit()
        conn.close()
        cleared = clear_query_cache(ctx.obj["db_path"])
        click.echo(f"Cleared {deleted} message embeddings + {cleared} cached queries for re-embedding.")

    embedder = None
    if batch_api:
        from gmail_search.embed.client import BatchGeminiEmbedder

        embedder = BatchGeminiEmbedder(cfg)
        click.echo("Using Batch API (50% cheaper, polling for results)")

    conn = get_connection(ctx.obj["db_path"])
    ok, spent, remaining = check_budget(conn, cfg["budget"]["max_usd"])
    conn.close()
    click.echo(f"Budget: ${cfg['budget']['max_usd']:.2f} | Spent: ${spent:.2f} | Remaining: ${remaining:.2f}")

    count = run_embedding_pipeline(ctx.obj["db_path"], cfg, embedder=embedder)
    click.echo(f"Embedded {count} new chunks.")


@main.command(help="Rebuild the ScaNN search index and all downstream surfaces")
@common_options
@click.option(
    "--loop",
    is_flag=True,
    help="Run as a daemon: reindex (light) whenever a quantum of new embeddings accumulates. "
    "Decoupled from the embed loop, which no longer reindexes per cycle.",
)
@click.option("--quantum", default=60, help="Seconds between work checks in --loop mode.")
@click.option("--min-new", default=2000, help="Reindex once this many new embeddings exist.")
@click.option("--max-age", default=600, help="...or when any new exist and the index is this stale (s).")
@click.option(
    "--email",
    type=str,
    default=None,
    help="Gmail account whose index to rebuild (--loop). Defaults to GMS_BOOTSTRAP_EMAIL. "
    "The supervisor spawns one per-user reindex daemon with --email set.",
)
@click.pass_context
def reindex(ctx, loop, quantum, min_new, max_age, email):
    import os as _os

    from gmail_search.pipeline import reindex as _reindex

    db_path = ctx.obj["db_path"]
    data_dir = ctx.obj["data_dir"]
    cfg = ctx.obj["config"]
    if email:
        _os.environ["GMS_BOOTSTRAP_EMAIL"] = email

    if not loop:
        _reindex(db_path=db_path, data_dir=data_dir, cfg=cfg, light=False)
        click.echo("Index rebuilt (ScaNN + FTS + thread summary + contacts + spell + topics + aliases).")
        return

    # Daemon mode: the embed loop no longer reindexes; this rebuilds the
    # ScaNN/FTS surfaces on a quantum when there's enough new work.
    import time as _time

    from gmail_search.auth.write_user import resolve_write_user_id
    from gmail_search.pipeline import reindex_if_needed
    from gmail_search.store.db import JobProgress, get_connection

    _conn = get_connection(db_path)
    try:
        uid = resolve_write_user_id(_conn)
    finally:
        _conn.close()
    job_id = f"reindex:{uid}"
    progress = JobProgress(db_path, job_id, start_completed=0)
    click.echo(f"reindex daemon for {uid}: every {quantum}s (min_new={min_new}, max_age={max_age}s)")
    while True:
        try:
            did = reindex_if_needed(db_path, data_dir, cfg, user_id=uid, min_new=min_new, max_age_s=max_age)
            progress.update("reindex" if did else "idle", 0, 0, "rebuilt" if did else "no new work")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"reindex loop error: {e}")
            progress.heartbeat()
        _time.sleep(quantum)


@main.command(help="Crawl pending URL stubs (fast/slow lane). --loop runs as a daemon.")
@common_options
@click.option("--loop", is_flag=True, help="Run continuously as a daemon.")
@click.option("--interval", default=15, help="Seconds to sleep between crawl batches in --loop mode.")
@click.option("--concurrency", default=10, help="Concurrent headless-Chromium fetches.")
@click.option("--limit", default=300, help="URL stubs per batch.")
@click.pass_context
def crawl(ctx, loop, interval, concurrency, limit):
    """Decoupled URL crawler. Pulls pending stubs (newest never-tried first;
    failed ones back off and are abandoned after a cap — see pending_url_stubs)
    and fills attachments.extracted_text. Runs as its own daemon so it can't be
    starved by the embed loop (which used to run it inline, after an ~80-min
    attachment-embed pass that left it never executing)."""
    import asyncio as _asyncio

    from gmail_search.gmail.url_fetcher import run as _crawl_run
    from gmail_search.gmail.url_fetcher import run_continuous as _crawl_run_continuous  # noqa: F401 (used in loop)

    db_path = ctx.obj["db_path"]

    if not loop:
        r = _asyncio.run(_crawl_run(db_path, concurrency=concurrency, limit=limit))
        click.echo(f"Crawled {r['done']}/{r['total']} URL stubs ({r['failed']} failed).")
        return

    import time as _time

    from gmail_search.store.db import JobProgress

    def _mem_aware_browser_cap(ceiling: int = 6) -> int:
        """Scale the BROWSER pool to free RAM each batch. Only Chromium tabs
        (~0.7 GB each) cost memory; the HTTP pool is I/O-bound and ~free, so
        the memory bound applies to the browser pool ALONE now (post split-
        pool refactor in url_fetcher). Reserves a 4 GB buffer for serve's
        ScaNN index + spikes; floors at 1, ceils at `ceiling`."""
        try:
            avail_kb = next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable"))
            avail_gb = avail_kb / 1024 / 1024
        except Exception:
            return ceiling
        return max(1, min(ceiling, int((avail_gb - 4.0) / 0.7)))

    progress = JobProgress(db_path, "crawl", start_completed=0)
    click.echo(
        f"crawl daemon: batches of {limit}, HTTP concurrency {concurrency}, memory-aware browser pool, every {interval}s"
    )
    while True:
        try:
            # Continuous worker-pool (run_continuous): keeps the HTTP pool
            # saturated across the slow browser tail instead of barriering on
            # each batch. HTTP runs at the full ceiling (no memory cost); the
            # memory-heavy browser pool is throttled to free RAM each call.
            browser_cap = _mem_aware_browser_cap()
            r = _asyncio.run(
                _crawl_run_continuous(db_path, http_concurrency=concurrency, browser_cap=browser_cap, target=limit)
            )
            progress.update(
                "crawl" if r["total"] else "idle",
                r["done"],
                r["total"],
                f"{r['done']}/{r['total']} fetched, {r['failed']} failed",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"crawl loop error: {e}")
            progress.heartbeat()
        _time.sleep(interval)


def _extract_pending_attachments(conn, att_config: dict, *, user_id: str | None = None) -> int:
    """Run the extractor against every attachment still awaiting text —
    in frontfill-first order so today's arrivals drain before the
    multi-year backfill. Returns the number of rows updated.

    Split out of the `update` command loop so it's unit-testable and so
    the ordering guarantee (fresh < 24h first, then date DESC) lives
    next to the query that enforces it (`get_pending_extraction_message_ids`).
    """
    from gmail_search.extract import dispatch
    from gmail_search.store.queries import get_attachments_for_message, get_pending_extraction_message_ids

    message_ids = get_pending_extraction_message_ids(conn, user_id=user_id)
    extracted = 0
    for message_id in message_ids:
        for att in get_attachments_for_message(conn, message_id):
            if att.extracted_text or att.image_path:
                continue
            if not att.raw_path or not Path(att.raw_path).exists():
                continue
            try:
                result = dispatch(att.mime_type, Path(att.raw_path), att_config)
            except Exception as e:
                logger.warning(f"Failed to extract {att.filename}: {e}")
                continue
            if result is None:
                continue
            updates: dict = {}
            if result.text:
                # PG TEXT columns reject NUL bytes — some PDFs / docs
                # extract control bytes that would otherwise crash the
                # whole cycle on UPDATE. Strip them so one bad PDF
                # doesn't take down the daemon.
                updates["extracted_text"] = result.text.replace("\x00", "")
            if result.images:
                updates["image_path"] = str(result.images[0].parent if len(result.images) > 1 else result.images[0])
            if updates:
                set_clause = ", ".join(f"{k} = %s" for k in updates)
                conn.execute(
                    f"UPDATE attachments SET {set_clause} WHERE id = %s",
                    (*updates.values(), att.id),
                )
                conn.commit()
                extracted += 1
    return extracted


@main.command(help="Run full pipeline in rolling batches: download → extract → embed → reindex")
@click.option("--max-messages", type=int, default=None, help="Max messages to download")
@click.option("--budget", type=float, default=None, help="Override budget limit")
@click.option("--batch-size", type=int, default=500, help="Messages per batch before extract+embed")
@click.option(
    "--min-free-gb",
    type=float,
    default=None,
    help="Stop between batches if free disk drops below this many GiB",
)
@click.option(
    "--loop",
    is_flag=True,
    default=False,
    help="Never exit: after catching up (or crashing), sleep and retry forever. "
    "Use this for the long-running backfill — same pattern as `watch`.",
)
@click.option(
    "--loop-sleep",
    type=int,
    default=300,
    help="Seconds to sleep between loop iterations when caught up. Default 5 min.",
)
@click.option(
    "--email",
    type=str,
    default=None,
    help="Gmail account to sync. Defaults to GMS_BOOTSTRAP_EMAIL env. "
    "In multi-tenant mode the API spawns one daemon per user with --email set.",
)
@common_options
@click.pass_context
def update(ctx, max_messages, budget, batch_size, min_free_gb, loop, loop_sleep, email):
    import time as _time

    from gmail_search.embed.pipeline import run_embedding_pipeline
    from gmail_search.gmail.auth import build_gmail_service
    from gmail_search.gmail.client import download_messages
    from gmail_search.locks import write_lock
    from gmail_search.store.db import JobProgress

    cfg = ctx.obj["config"]
    db_path = ctx.obj["db_path"]
    data_dir = ctx.obj["data_dir"]
    att_config = cfg.get("attachments", {})

    if budget:
        cfg["budget"]["max_usd"] = budget

    # Surface the email to downstream code (gmail/auth.py reads this
    # env to pick the broker entry; the API spawns subprocesses with
    # --email and we mirror it into the env so all the per-user
    # write_user lookups (resolve_write_user_id, etc.) line up too.
    import os as _os

    if email:
        _os.environ["GMS_BOOTSTRAP_EMAIL"] = email
    service = build_gmail_service(data_dir, email=email)
    max_msg = max_messages or cfg["download"].get("max_messages")
    index_dir = data_dir / "scann_index"  # noqa: F841

    # Resolve the active user_id once. With GMS_BOOTSTRAP_EMAIL set
    # above, this picks up `email` (or the env default for legacy
    # single-pool installs). Used for per-user job_progress keys + to
    # scope the message COUNTs we report to the UI so silvershabbat's
    # progress bar doesn't include scottmsilver's messages.
    from gmail_search.auth.write_user import resolve_write_user_id as _resolve_uid

    _conn = get_connection(db_path)
    try:
        active_user_id = _resolve_uid(_conn)
    finally:
        _conn.close()
    job_id = f"update:{active_user_id}"

    def _one_cycle() -> None:
        """Run the full download→extract→embed→reindex loop once, then
        return. When --loop is set, the outer driver calls this in a
        crash-tolerant while-True around a sleep, so the job never dies
        permanently — matches the `watch` daemon's behaviour.
        """
        # Re-fetch a fresh Gmail token from the broker EACH cycle. The broker
        # hands a short-lived access token with no refresh fields, so a service
        # built once before the loop goes stale after ~1h and every later cycle
        # dies with a RefreshError (the bug that froze sync + the embed drain).
        nonlocal service
        try:
            service = build_gmail_service(data_dir, email=email)
        except Exception as e:
            logger.warning(f"couldn't refresh Gmail credentials this cycle: {e}")

        # Re-read the message count each cycle. Progress math is
        # relative to this cycle's start, not the first one, so the
        # UI rate/ETA reflects what's happening right now.
        conn = get_connection(db_path)
        start_count = conn.execute("SELECT COUNT(*) FROM messages WHERE user_id = %s", (active_user_id,)).fetchone()[0]
        conn.close()
        progress = JobProgress(db_path, job_id, start_completed=start_count)

        total_downloaded = 0
        total_extracted = 0
        total_embedded = 0
        batch_num = 0

        # Real denominator for the progress bar: the user's actual Gmail
        # message count (from users.getProfile). Falls back to 0 if the
        # call fails; the UI shows `—` rather than a misleading fraction.
        account_total = 0
        try:
            profile = service.users().getProfile(userId="me").execute()
            account_total = int(profile.get("messagesTotal", 0))
        except Exception as e:
            logger.warning(f"couldn't fetch account message total: {e}")
        progress_total = max_msg or account_total or 0

        while True:
            if min_free_gb is not None:
                import shutil as _shutil

                free_gb = _shutil.disk_usage(str(data_dir)).free / (1024**3)
                if free_gb < min_free_gb:
                    click.echo(f"Free disk {free_gb:.2f} GiB < min {min_free_gb} GiB — pausing.")
                    break

            # Serialise with the watch daemon (SQLite is single-writer).
            # Held for one batch; watch preempts between batches.
            with write_lock(data_dir):
                batch_num += 1
                current_limit = start_count + total_downloaded + batch_size
                if max_msg:
                    current_limit = min(current_limit, max_msg)

                click.echo(f"\n{'=' * 50}")
                click.echo(f"Batch {batch_num}: downloading up to {current_limit} total messages")
                click.echo(f"{'=' * 50}")

                progress.update("download", start_count + total_downloaded, progress_total, f"batch {batch_num}")

                dl_count = download_messages(
                    service=service,
                    db_path=db_path,
                    data_dir=data_dir,
                    batch_size=cfg["download"]["batch_size"],
                    max_messages=current_limit,
                    max_attachment_size=cfg["attachments"]["max_file_size_mb"] * 1024 * 1024,
                )
                total_downloaded += dl_count

                # Even when this cycle's download yielded zero new rows,
                # the `watch` daemon runs in parallel and may have
                # written fresh message+attachment rows straight to the
                # DB between our cycles. Don't short-circuit extract/
                # embed/reindex on dl_count==0 — the pending queues
                # might be non-empty from that out-of-band ingest.
                # (We still break out of the inner batching loop at the
                # end of this iteration so we don't spin forever.)
                drain_only = dl_count == 0
                if drain_only:
                    click.echo(
                        "No new messages to download this cycle — draining any pending "
                        "extract/embed/reindex from out-of-band (watch daemon) ingest."
                    )
                else:
                    click.echo(f"Downloaded {dl_count} messages.")

                # Extract pending attachments. Ordered so fresh arrivals
                # from the `watch` daemon (< 24h old) are drained before
                # the multi-year backfill — otherwise today's PDFs sit
                # unextracted for days while older rows grind through.

                progress.update("extract", start_count + total_downloaded, progress_total, f"+{dl_count} downloaded")
                click.echo("Extracting attachments...")
                conn = get_connection(db_path)
                extracted = _extract_pending_attachments(conn, att_config, user_id=active_user_id)
                conn.close()
                total_extracted += extracted
                click.echo(f"Extracted {extracted} attachments.")

                # Embed FIRST — this is the essential step. The URL crawl used
                # to run before embed (so embeds could include crawled page
                # text same-cycle), but a slow/failing crawl (dead links + the
                # Chromium memory footprint) starved embedding: the cycle
                # stalled or crashed mid-crawl and got respawned before embed
                # ran, leaving messages unembedded (the ~220k backlog). Embed
                # first; crawl best-effort after. Crawled page text is folded
                # in by the NEXT cycle's embed pass (one-cycle lag, acceptable).
                progress.update("embed", start_count + total_downloaded, progress_total, f"+{extracted} extracted")
                click.echo("Embedding...")
                conn = get_connection(db_path)
                ok, spent, remaining = check_budget(conn, cfg["budget"]["max_usd"], user_id=active_user_id)
                conn.close()
                if not ok:
                    click.echo(f"Budget exhausted (${spent:.2f} spent). Stopping embedding.")
                    break
                emb_count = run_embedding_pipeline(db_path, cfg, user_id=active_user_id, limit=5000)
                total_embedded += emb_count
                click.echo(f"Embedded {emb_count} chunks.")

                # NOTE: URL crawling and ScaNN reindexing used to run inline
                # here, which serialized the pipeline and let the slow steps
                # starve each other (embed's attachment pass starved crawl;
                # reindex ran every batch). They're now independent supervised
                # daemons — `crawl --loop` (global) and `reindex --loop`
                # (per-user, rebuilds on a quantum of new embeddings). This
                # cycle is just download → extract → embed.

                conn = get_connection(db_path)
                msg_count = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE user_id = %s", (active_user_id,)
                ).fetchone()[0]
                emb_total = conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE user_id = %s", (active_user_id,)
                ).fetchone()[0]
                cost = get_total_spend(conn, user_id=active_user_id)
                conn.close()
                click.echo(f"Status: {msg_count:,} messages | {emb_total:,} embeddings | ${cost:.2f} spent")

                if max_msg and msg_count >= max_msg:
                    click.echo("Reached max message limit.")
                    break

                # Single-pass drain when nothing new arrived from our
                # download. We already ran extract/embed/reindex against
                # whatever `watch` had written; no reason to spin
                # through another iteration of the inner loop.
                if drain_only:
                    break

        progress.finish(
            "done",
            f"+{total_downloaded} downloaded, +{total_extracted} extracted, +{total_embedded} embedded",
        )

        click.echo(f"\n{'=' * 50}")
        click.echo(
            f"Cycle done! +{total_downloaded} downloaded, +{total_extracted} extracted, +{total_embedded} embedded"
        )
        conn = get_connection(db_path)
        msg_count = conn.execute("SELECT COUNT(*) FROM messages WHERE user_id = %s", (active_user_id,)).fetchone()[0]
        emb_total = conn.execute("SELECT COUNT(*) FROM embeddings WHERE user_id = %s", (active_user_id,)).fetchone()[0]
        total_cost = get_total_spend(conn, user_id=active_user_id)
        conn.close()
        click.echo(f"Total: {msg_count:,} messages | {emb_total:,} embeddings | ${total_cost:.2f} spent")
        return total_downloaded + total_embedded

    # ── outer driver ──────────────────────────────────────────────────
    if not loop:
        _one_cycle()
        return

    cycle = 0
    while True:
        cycle += 1
        click.echo(f"\n{'#' * 50}\n# backfill loop cycle {cycle}\n{'#' * 50}")
        did_work = 0
        try:
            did_work = _one_cycle() or 0
        except KeyboardInterrupt:
            click.echo("interrupted — exiting loop.")
            raise
        except Exception as e:
            # Don't let one bad cycle kill the daemon. Log + surface to
            # job_progress so the Settings UI shows the failure reason,
            # then sleep and try again. Matches how `watch` survives
            # transient Gmail / network hiccups.
            logger.exception("update cycle crashed: %s", e)
            try:
                from gmail_search.store.db import JobProgress as _JP

                _JP(db_path, job_id, start_completed=0).update(
                    "error", 0, 0, f"crash: {type(e).__name__}: {str(e)[:120]}"
                )
            except Exception:
                pass
        # Adaptive sleep: while there's still a backlog to drain (a cycle that
        # downloaded and/or embedded something), loop quickly to keep the
        # workers busy; once caught up (a no-op cycle) back off to the full
        # loop_sleep so we don't poll Gmail every few seconds when idle.
        effective_sleep = 15 if did_work else loop_sleep
        click.echo(f"\n[backfill] sleeping {effective_sleep}s before next cycle...")
        # Hold the `update` heartbeat warm during the inter-cycle sleep
        # so the supervisor / HTTP status don't see the daemon as dead
        # while it's waiting to start the next cycle. We flip status back
        # to 'running' with stage=idle so /api/status reflects reality.
        try:
            from gmail_search.store.db import JobProgress as _JP

            idle = _JP(db_path, job_id, start_completed=0)
            idle.update("idle", 0, 0, "waiting for next cycle")
        except Exception:
            idle = None
        for i in range(effective_sleep):
            if idle is not None and i % 30 == 0:
                try:
                    idle.heartbeat()
                except Exception:
                    pass
            _time.sleep(1)


def _pid_file(data_dir: Path) -> Path:
    return data_dir / "watch.pid"


def _is_watch_running(data_dir: Path) -> tuple[bool, int | None]:
    """Check if a watch process is running. Returns (running, pid)."""
    import subprocess

    pid_path = _pid_file(data_dir)
    if not pid_path.exists():
        return False, None
    try:
        pid = int(pid_path.read_text().strip())
        # Check if the PID exists AND is actually a gmail-search watch process
        os.kill(pid, 0)
        cmdline = subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True).stdout
        if "gmail-search" not in cmdline and "gmail_search" not in cmdline:
            pid_path.unlink(missing_ok=True)
            return False, None
        return True, pid
    except (ValueError, ProcessLookupError, PermissionError):
        pid_path.unlink(missing_ok=True)
        return False, None


@main.command(help="Start watching for new emails in the background")
@click.option("--interval", type=int, default=120, help="Seconds between sync checks")
@click.option("--budget", type=float, default=None, help="Override budget limit")
@common_options
@click.pass_context
def start(ctx, interval, budget):
    """Start the watch daemon in the background."""
    import subprocess
    import sys

    data_dir = ctx.obj["data_dir"]
    running, pid = _is_watch_running(data_dir)
    if running:
        click.echo(f"Already running (PID {pid}). Use 'gmail-search stop' first.")
        return

    # Build the watch command
    import shutil

    gmail_search_bin = shutil.which("gmail-search") or f"{sys.executable} -m gmail_search.cli"
    cmd = [gmail_search_bin, "watch", "--interval", str(interval), "--data-dir", str(data_dir)]
    if budget:
        cmd.extend(["--budget", str(budget)])

    # Start as a detached subprocess
    log_path = data_dir / "watch.log"
    log_file = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)

    _pid_file(data_dir).write_text(str(proc.pid))
    click.echo(f"Started watch daemon (PID {proc.pid})")
    click.echo(f"  Log: {log_path}")
    click.echo(f"  Interval: {interval}s")
    click.echo("  Stop: gmail-search stop")


@main.command(help="Stop the background watch daemon")
@common_options
@click.pass_context
def stop(ctx):
    """Stop the watch daemon."""

    data_dir = ctx.obj["data_dir"]
    running, pid = _is_watch_running(data_dir)
    if not running:
        click.echo("No watch daemon running.")
        return

    os.kill(pid, 15)  # SIGTERM — triggers graceful shutdown
    _pid_file(data_dir).unlink(missing_ok=True)
    click.echo(f"Stopped watch daemon (PID {pid})")


def _set_gmail_health(db_path, user_id: str, status: str, reason) -> None:
    """Best-effort persist of Gmail credential health; never breaks the cycle."""
    try:
        from gmail_search.gmail.auth import record_credential_health
        from gmail_search.store.db import get_connection

        conn = get_connection(db_path)
        try:
            record_credential_health(conn, user_id, status, reason)
        finally:
            conn.close()
    except Exception:
        logger.debug("could not record gmail health", exc_info=True)


@main.command(help="Watch for new emails and process them continuously")
@click.option("--interval", type=int, default=120, help="Seconds between sync checks")
@click.option("--budget", type=float, default=None, help="Override budget limit")
@click.option("--max-cycles", type=int, default=None, help="Exit after N cycles (for one-shot 'frontfill' runs)")
@click.option(
    "--email",
    type=str,
    default=None,
    help="Gmail account to watch. Defaults to GMS_BOOTSTRAP_EMAIL env. "
    "API spawns one daemon per user with --email set.",
)
@common_options
@click.pass_context
def watch(ctx, interval, budget, max_cycles, email):
    """Poll for new emails, extract, embed, and reindex continuously."""
    import os as _os
    import signal
    import time as _time

    from gmail_search.embed.pipeline import run_embedding_pipeline
    from gmail_search.extract import dispatch
    from gmail_search.gmail.auth import build_gmail_service
    from gmail_search.gmail.client import sync_new_messages
    from gmail_search.store.db import JobProgress
    from gmail_search.store.queries import get_attachments_for_message

    cfg = ctx.obj["config"]
    db_path = ctx.obj["db_path"]
    data_dir = ctx.obj["data_dir"]
    att_config = cfg.get("attachments", {})
    index_dir = data_dir / "scann_index"

    if budget:
        cfg["budget"]["max_usd"] = budget

    if email:
        _os.environ["GMS_BOOTSTRAP_EMAIL"] = email
    service = build_gmail_service(data_dir, email=email)

    # Per-user job_progress key so two users' watch daemons don't
    # collide on a single global "watch" row.
    from gmail_search.auth.write_user import resolve_write_user_id as _resolve_uid

    _conn = get_connection(db_path)
    try:
        active_user_id = _resolve_uid(_conn)
    finally:
        _conn.close()
    job_id = f"watch:{active_user_id}"
    progress = JobProgress(db_path, job_id)
    running = True

    def handle_stop(sig, frame):
        nonlocal running
        click.echo("\nStopping watch...")
        running = False

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    if max_cycles:
        click.echo(f"Watching for new emails — will exit after {max_cycles} cycle(s).")
    else:
        click.echo(f"Watching for new emails every {interval}s. Ctrl+C to stop.")
    cycle = 0

    from gmail_search.locks import write_lock

    while running:
        cycle += 1
        if max_cycles is not None and cycle > max_cycles:
            break

        # Hold the write lock for the whole cycle body so a concurrent
        # `update` backfill can't race our upserts/reindex. Released
        # before the sleep — backfill preempts between cycles.
        with write_lock(data_dir):
            progress.update("sync", cycle, 0, "checking for new emails")

            try:
                # Fresh broker token each cycle — the handed access token has
                # no refresh fields and expires in ~1h, so a service built once
                # before the loop starts failing with RefreshError.
                service = build_gmail_service(data_dir, email=email)
                count = sync_new_messages(
                    service=service,
                    db_path=db_path,
                    data_dir=data_dir,
                    max_attachment_size=cfg["attachments"]["max_file_size_mb"] * 1024 * 1024,
                )
                _set_gmail_health(db_path, active_user_id, "healthy", None)
            except Exception as e:
                from gmail_search.gmail.auth import CREDENTIAL_HINTS, classify_credential_error

                reason = classify_credential_error(e)
                if reason:
                    # A credential/scope failure must NOT look like "no new mail":
                    # record health + a loud, actionable status (visible in the
                    # admin console) so a dead/scope-stripped token is caught fast.
                    _set_gmail_health(db_path, active_user_id, "unhealthy", reason)
                    hint = CREDENTIAL_HINTS.get(reason, "reconnect Gmail")
                    logger.error("Gmail sync blocked — %s: %s", reason, e)
                    progress.update("credentials", cycle, 0, f"⚠ CREDENTIALS: {hint}")
                else:
                    logger.warning(f"Sync failed: {e}")
                count = 0

            if count > 0:
                click.echo(f"[cycle {cycle}] Synced {count} new messages")

                # Extract attachments
                progress.update("extract", cycle, 0, f"+{count} synced")
                conn = get_connection(db_path)
                try:
                    # Filter to the active user — without this the watch
                    # daemon also extracts other users' pending
                    # attachments, which can OOM on a large peer backlog.
                    rows = conn.execute(
                        "SELECT DISTINCT m.id FROM messages m "
                        "JOIN attachments a ON a.message_id = m.id "
                        "WHERE a.extracted_text IS NULL AND a.image_path IS NULL "
                        "AND a.raw_path IS NOT NULL AND m.user_id = %s",
                        (active_user_id,),
                    ).fetchall()
                    for row in rows:
                        attachments = get_attachments_for_message(conn, row["id"])
                        for att in attachments:
                            if att.extracted_text or att.image_path or not att.raw_path:
                                continue
                            if not Path(att.raw_path).exists():
                                continue
                            try:
                                result = dispatch(att.mime_type, Path(att.raw_path), att_config)
                            except Exception:
                                continue
                            if result is None:
                                continue
                            updates = {}
                            if result.text:
                                # Strip NUL bytes — PG TEXT can't store
                                # them and one bad PDF would otherwise
                                # crash the whole watch cycle.
                                updates["extracted_text"] = result.text.replace("\x00", "")
                            if result.images:
                                updates["image_path"] = str(
                                    result.images[0].parent if len(result.images) > 1 else result.images[0]
                                )
                            if updates:
                                set_clause = ", ".join(f"{k} = %s" for k in updates)
                                conn.execute(
                                    f"UPDATE attachments SET {set_clause} WHERE id = %s",
                                    (*updates.values(), att.id),
                                )
                                conn.commit()
                finally:
                    conn.close()

                # Embed
                progress.update("embed", cycle, 0, "embedding new messages")
                conn = get_connection(db_path)
                ok, spent, remaining = check_budget(conn, cfg["budget"]["max_usd"], user_id=active_user_id)
                conn.close()
                if ok:
                    emb_count = run_embedding_pipeline(db_path, cfg, user_id=active_user_id, limit=5000)
                    if emb_count:
                        click.echo(f"[cycle {cycle}] Embedded {emb_count} chunks")
                else:
                    click.echo(f"[cycle {cycle}] Budget exhausted (${spent:.2f})")

                # Indexing is decoupled: the `reindex --loop` daemon rebuilds
                # the ScaNN/FTS surfaces on a quantum of new embeddings, so the
                # frontfill cycle no longer reindexes inline.

                conn = get_connection(db_path)
                msg_count = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE user_id = %s", (active_user_id,)
                ).fetchone()[0]
                emb_total = conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE user_id = %s", (active_user_id,)
                ).fetchone()[0]
                conn.close()
                click.echo(f"[cycle {cycle}] {msg_count:,} messages | {emb_total:,} embeddings")
            else:
                progress.update("idle", cycle, 0, "no new messages")

        # Sleep in small increments so Ctrl+C is responsive. Heartbeat
        # every ~30s so an idle watch (no new mail = no .update() call
        # this cycle) stays well under the supervisor's 90s stale
        # threshold without hammering the DB.
        for i in range(interval):
            if not running:
                break
            if i % 30 == 0:
                try:
                    progress.heartbeat()
                except Exception as _e:
                    logger.warning(f"heartbeat failed: {_e}")
            _time.sleep(1)

    progress.finish("stopped", f"ran {cycle} cycles")
    click.echo(f"Watch stopped after {cycle} cycles.")
    # ScaNN / embedding pools spawn non-daemon threads that keep the
    # interpreter alive after click returns. Force exit so systemd sees
    # the unit terminate cleanly instead of leaving a zombie.
    import sys as _sys

    _sys.exit(0)


@main.command(help="Clean up zombie running jobs (stale job_progress rows and their orphan processes)")
@click.option(
    "--staleness-seconds", type=int, default=600, help="Treat jobs idle longer than this as zombies (default 600)"
)
@click.option("--kill-timeout-sec", type=float, default=5.0, help="Seconds to wait after SIGTERM before SIGKILL")
@common_options
@click.pass_context
def reap(ctx, staleness_seconds, kill_timeout_sec):
    from gmail_search.reap import reap_zombie_jobs

    report = reap_zombie_jobs(
        ctx.obj["db_path"],
        staleness_seconds=staleness_seconds,
        kill_timeout_sec=kill_timeout_sec,
    )
    click.echo(f"Reaped {report.rows_reaped} stale row(s), killed {report.processes_killed} process(es).")
    for line in report.details:
        click.echo(f"  {line}")


@main.command(help="Reconcile thread_summary with messages (drift detector)")
@click.option(
    "--batch",
    type=int,
    default=2000,
    help="Max threads recomputed per pass. Keeps each pass bounded so a huge " "drift event doesn't stall the daemon.",
)
@click.option("--loop", is_flag=True, default=False, help="Keep running. Sleeps between passes.")
@click.option(
    "--loop-sleep",
    type=float,
    default=30.0,
    help="Seconds to sleep between passes when --loop is set (default 30).",
)
@common_options
@click.pass_context
def reconcile(ctx, batch, loop, loop_sleep):
    """Scan `messages` for thread drift and refresh `thread_summary`.

    Belt-and-suspenders for the inline `_touch_thread_summary` call in
    `upsert_message`: if any writer ever forgets to call it (new
    ingest path, a bulk import, a migration that bypasses the helper),
    this daemon catches up within one sleep interval.

    Watermark: `sync_state['last_reconciled_history_id']`. Each pass
    grabs threads whose messages have `history_id > watermark` and
    recomputes their summary rows, then advances the watermark.

    Also writes a heartbeat via JobProgress so the supervisor can see
    it's alive and respawn it if it dies.
    """
    import logging as _logging
    import time as _time

    from gmail_search.store.db import JobProgress, get_connection
    from gmail_search.store.queries import get_sync_state as _get_state
    from gmail_search.store.queries import recompute_thread_summary
    from gmail_search.store.queries import set_sync_state as _set_state

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = _logging.getLogger("reconcile")

    db_path = ctx.obj["db_path"]
    progress = JobProgress(db_path, "reconcile", start_completed=0)

    def _one_pass() -> tuple[int, int]:
        """Returns (threads_recomputed, new_watermark_bumped_by)."""
        conn = get_connection(db_path)
        try:
            watermark = int(_get_state(conn, "last_reconciled_history_id") or 0)
            rows = conn.execute(
                """SELECT DISTINCT thread_id
                     FROM messages
                    WHERE history_id > %s
                    ORDER BY thread_id
                    LIMIT %s""",
                (watermark, batch),
            ).fetchall()
            if not rows:
                # Also sweep threads that are in messages but missing
                # from thread_summary — catches historical data that
                # predated the inline touch + a zero watermark.
                missing = conn.execute(
                    """SELECT DISTINCT m.thread_id
                         FROM messages m
                         LEFT JOIN thread_summary ts USING (thread_id)
                        WHERE ts.thread_id IS NULL
                        LIMIT %s""",
                    (batch,),
                ).fetchall()
                rows = missing

            count = 0
            for r in rows:
                if recompute_thread_summary(conn, r["thread_id"]):
                    count += 1

            # Advance watermark to the max history_id seen so far, so
            # next pass starts from there. We query globally (not just
            # on the rows we touched) to avoid getting stuck if the
            # batch happened to land on old messages.
            max_row = conn.execute("SELECT max(history_id) AS m FROM messages").fetchone()
            new_watermark = int(max_row["m"] or 0) if max_row else watermark
            if new_watermark > watermark:
                _set_state(conn, "last_reconciled_history_id", str(new_watermark))
            conn.commit()
            return count, new_watermark - watermark
        finally:
            conn.close()

    pass_no = 0
    while True:
        pass_no += 1
        start = _time.time()
        try:
            recomputed, bump = _one_pass()
        except Exception as e:
            log.exception(f"pass crashed: {e}")
            progress.update("error", pass_no, 0, f"{type(e).__name__}: {str(e)[:100]}")
            if not loop:
                raise
            _time.sleep(loop_sleep)
            continue
        elapsed = _time.time() - start
        detail = f"pass {pass_no}: {recomputed} recomputed, watermark +{bump} in {elapsed:.2f}s"
        progress.update("reconciling", pass_no, 0, detail)
        log.info(detail)
        if not loop:
            break
        _time.sleep(loop_sleep)


@main.command(help="Extract atomic facts (propositions) from newly-ingested mail")
@click.option("--batch", type=int, default=500, help="Max messages propositionized per pass.")
@click.option("--loop", is_flag=True, default=False, help="Keep running. Sleeps between passes.")
@click.option(
    "--loop-sleep",
    type=float,
    default=30.0,
    help="Seconds to sleep between passes when --loop is set (default 30).",
)
@click.option(
    "--email",
    type=str,
    default=None,
    help="Gmail account to propositionize. Defaults to GMS_BOOTSTRAP_EMAIL env. "
    "The supervisor spawns one daemon per user with --email set.",
)
@common_options
@click.pass_context
def propositionize(ctx, batch, loop, loop_sleep, email):
    """Decoupled per-user daemon: turn newly-ingested messages into atomic facts
    via OpenRouter nova-lite, embed them, and store them idempotently
    (prop_processed marker). Deliberately OFF the latency-sensitive ingest path —
    the same decoupling pattern as summarize/reindex. Requires OPENROUTER_KEY.
    """
    import logging as _logging
    import os as _os
    import time as _time

    import httpx as _httpx

    from gmail_search import propositions as _P
    from gmail_search.auth.write_user import resolve_write_user_id as _resolve_uid
    from gmail_search.embed.client import GeminiEmbedder
    from gmail_search.llm.openrouter import OpenRouterBackend
    from gmail_search.store.db import JobProgress, get_connection

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = _logging.getLogger("propositionize")

    cfg = ctx.obj["config"]
    db_path = ctx.obj["db_path"]

    if email:
        _os.environ["GMS_BOOTSTRAP_EMAIL"] = email

    # Per-user job_progress key so two users' daemons don't collide.
    _conn = get_connection(db_path)
    try:
        active_user_id = _resolve_uid(_conn)
    finally:
        _conn.close()

    job_id = f"propositionize:{active_user_id}"
    progress = JobProgress(db_path, job_id, start_completed=0)

    # Pin extraction to nova via OpenRouter regardless of LLM_BACKEND.
    backend = OpenRouterBackend()
    embedder = GeminiEmbedder(cfg)

    conn0 = get_connection(db_path)
    try:
        _P.ensure_table(conn0)
        _P.ensure_bm25_index(conn0)
        _P.ensure_processed_table(conn0)
        owner = _P.owner_string_for_user(conn0, active_user_id)
    finally:
        conn0.close()

    def _one_pass() -> dict:
        conn = get_connection(db_path)
        try:
            with _httpx.Client() as client:
                return _P.propositionize_pending(
                    conn, client, backend, embedder, user_id=active_user_id, owner=owner, batch=batch
                )
        finally:
            conn.close()

    pass_no = 0
    while True:
        pass_no += 1
        start = _time.time()
        try:
            stats = _one_pass()
        except Exception as e:
            log.exception(f"pass crashed: {e}")
            progress.update("error", pass_no, 0, f"{type(e).__name__}: {str(e)[:100]}")
            if not loop:
                raise
            _time.sleep(loop_sleep)
            continue
        elapsed = _time.time() - start
        detail = (
            f"pass {pass_no}: {stats['messages']} msgs -> {stats['facts']} facts, "
            f"{stats['errors']} err in {elapsed:.2f}s"
        )
        progress.update("propositionizing", pass_no, 0, detail)
        log.info(detail)
        if not loop:
            break
        _time.sleep(loop_sleep)


@main.command(help="Watchdog that keeps each enrolled user's watch/update/summarize alive")
@click.option("--interval", type=int, default=30, help="Seconds between heartbeat checks")
@click.option(
    "--restart-delay",
    type=int,
    default=15,
    help="Seconds to wait before considering a dead daemon for respawn",
)
@common_options
@click.pass_context
def supervise(ctx, interval, restart_delay):
    """Multi-user watchdog. Walks `users` every `interval` seconds and
    ensures three daemons are running per `sync_enabled` user:
        watch:<user_id>      — frontfill (poll Gmail for new mail)
        update:<user_id>     — backfill (older mail)
        summarize:<user_id>  — local-LLM per-message summary
    Plus one global `reconcile` daemon for thread_summary drift (no
    per-user variant — scans all users).

    Each spawned daemon gets `--email <user-email>` so it picks the
    right broker tokens and writes to that user's `user_id`. The
    job_progress row key is `<subcommand>:<user_id>` so multiple users
    coexist without collision. Stale heartbeat + dead process →
    respawn; stale heartbeat + live process → assume blocked, leave
    alone (avoids the 10× duplicate summarize fight on a flaky vLLM).
    """
    import logging as _logging
    import signal as _signal
    import time as _time
    from datetime import datetime, timezone

    from gmail_search.jobs import gmail_search_command, is_daemon_running, spawn_detached
    from gmail_search.store.db import JobProgress

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _log = _logging.getLogger("supervisor")

    data_dir = ctx.obj["data_dir"]
    db_path = ctx.obj["db_path"]

    # Per-user daemons we expect to be alive for every sync_enabled user.
    # `argv_template` is appended with `--email <email>` per user. The
    # log path uses the user_id prefix so per-user crashes don't pollute
    # each other's tail-able log.
    PER_USER_DAEMONS = [
        {"key": "watch", "argv": ["watch", "--interval", "120"], "log_suffix": "watch.log"},
        # Backfill: embeds the ~220k downloaded-but-unembedded messages and
        # keeps catching up. Now safe to supervise — the cycle embeds BEFORE
        # the (best-effort, short-timeout) URL crawl, so a slow/failing crawl
        # can no longer starve embedding (the old crawl-first order left it
        # embedding 0 and being respawned).
        {"key": "update", "argv": ["update", "--loop", "--min-free-gb", "5"], "log_suffix": "backfill.log"},
        {
            "key": "summarize",
            "argv": ["summarize", "--concurrency", "12", "--batch-size", "1", "--loop"],
            "log_suffix": "summarize.log",
        },
        # Reindex is decoupled from the embed loop: this rebuilds the ScaNN/FTS
        # surfaces on a quantum of new embeddings (or every --max-age for
        # freshness on new mail) instead of every embed batch.
        {
            "key": "reindex",
            "argv": ["reindex", "--loop", "--quantum", "60", "--min-new", "10000", "--max-age", "900"],
            "log_suffix": "reindex.log",
        },
        # Proposition (atomic-fact) extraction for new mail — decoupled like
        # summarize/reindex. Skipped automatically per-user if OPENROUTER_KEY is
        # unset (the daemon would just crash-loop otherwise).
        {
            "key": "propositionize",
            "argv": ["propositionize", "--loop", "--loop-sleep", "30"],
            "log_suffix": "propositionize.log",
        },
    ]

    # Proposition extraction needs an OpenRouter key; without it the daemon would
    # crash-loop, so drop it from the supervised set rather than respawn forever.
    if not (os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        PER_USER_DAEMONS = [d for d in PER_USER_DAEMONS if d["key"] != "propositionize"]
        _log.info("OPENROUTER_KEY unset — propositionize daemon disabled")

    # Global daemons — no user_id needed (they scan across all users).
    GLOBAL_DAEMONS = [
        {
            "key": "reconcile",
            "argv": ["reconcile", "--loop"],
            "log": data_dir / "reconcile.log",
        },
        # URL crawler. Action-link denylist (RSVP/accept/approve/yes-no — see
        # url_extract._DENY_PATH_CONTAINS) prevents GET-ing non-idempotent
        # links. `--concurrency 16` sizes the HTTP pool (I/O-bound, ~no memory;
        # measured ~2x throughput vs 8 with flat latency, plateauing past 16).
        # The memory-heavy BROWSER pool is bounded SEPARATELY and memory-aware
        # (see crawl loop). The daemon uses run_continuous (worker-pool, no
        # per-batch barrier on the browser tail), so `--limit 500` is the soft
        # per-call target — large is fine since the only barrier (final render
        # drain) is amortised over it. Short --interval: just a breath between
        # continuous passes for progress/shutdown checks.
        {
            "key": "crawl",
            "argv": ["crawl", "--loop", "--concurrency", "16", "--limit", "500", "--interval", "2"],
            "log": data_dir / "crawl.log",
        },
    ]

    # Per-(job_id) last spawn timestamp so a freshly-spawned-but-not-yet-
    # heartbeating daemon doesn't get re-spawned 10x in 30s.
    last_spawn_attempt: dict[str, float] = {}

    running = True

    def _on_signal(sig, frame):
        nonlocal running
        _log.info(f"received signal {sig} — stopping supervisor (supervised daemons stay alive)")
        running = False

    _signal.signal(_signal.SIGINT, _on_signal)
    _signal.signal(_signal.SIGTERM, _on_signal)

    supervisor_progress = JobProgress(db_path, "supervisor", start_completed=0)
    supervisor_progress.update("starting", 0, 0, "supervisor online (multi-user)")

    _STALE_SECONDS = 90

    def _row_age_seconds(row: dict | None) -> float | None:
        if not row:
            return None
        try:
            updated = datetime.fromisoformat(row["updated_at"])
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - updated).total_seconds()
        except (ValueError, TypeError):
            return None

    def _is_alive(row: dict | None) -> bool:
        age = _row_age_seconds(row)
        if age is None:
            return False
        if row.get("status") != "running":
            return False
        return age < _STALE_SECONDS

    def _should_respawn(job_id: str) -> bool:
        now = _time.time()
        last = last_spawn_attempt.get(job_id, 0.0)
        return (now - last) >= restart_delay

    def _spawn(job_id: str, argv_tail: list[str], log_path: Path) -> None:
        argv = gmail_search_command() + argv_tail + ["--data-dir", str(data_dir)]
        pid = spawn_detached(argv, log_path)
        last_spawn_attempt[job_id] = _time.time()
        _log.info(f"supervisor: spawned {job_id} pid={pid}")

    def _stop_unwanted_daemon(job_id: str, email: str | None) -> None:
        """Converge *down*: SIGTERM a per-user daemon whose user is no
        longer `sync_enabled`. Without this the toggle would only stop
        *respawning* — already-running daemons would keep syncing until
        they crashed on their own, so "Sync off" wouldn't actually
        pause an account. Reads the pid from the daemon's job_progress
        row and only signals if the OS process is genuinely alive."""
        import os as _os  # inline: formatter strips top-level unused imports

        row = JobProgress.get(db_path, job_id)
        pid = row.get("pid") if row else None
        if not pid or not is_daemon_running(job_id, email=email):
            return
        # PID-reuse guard: verify this exact PID's cmdline is really this
        # daemon (subcommand + email) before signalling — a stale row could
        # point at a PID since recycled by an unrelated process.
        subcommand = job_id.split(":", 1)[0]
        try:
            with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError):
            return
        if subcommand not in cmdline or (email and email not in cmdline):
            _log.warning(f"converge-down: pid {pid} cmdline no longer matches {job_id} — not killing")
            return
        try:
            _os.kill(int(pid), _signal.SIGTERM)
            _log.info(f"supervisor: stopped {job_id} pid={pid} (sync disabled)")
        except (ProcessLookupError, PermissionError, ValueError):
            pass

    def _enrolled_users() -> list[dict]:
        """Read `users` for the current sync-enabled set. Re-queried
        every cycle so newly-invited users get picked up automatically
        without a supervisor restart. Same for `sync_enabled=false`
        flipping a user off — next cycle they're skipped.
        """
        conn = get_connection(db_path)
        try:
            rows = conn.execute("SELECT id, email FROM users WHERE sync_enabled = TRUE ORDER BY email").fetchall()
            return [{"id": r["id"], "email": r["email"]} for r in rows]
        finally:
            conn.close()

    def _disabled_users() -> list[dict]:
        """Users with `sync_enabled = FALSE` — their per-user daemons
        should be converged *down* (stopped) if any are still running."""
        conn = get_connection(db_path)
        try:
            rows = conn.execute("SELECT id, email FROM users WHERE sync_enabled = FALSE ORDER BY email").fetchall()
            return [{"id": r["id"], "email": r["email"]} for r in rows]
        finally:
            conn.close()

    click.echo(
        f"Supervisor online (multi-user). interval={interval}s "
        f"stale>{_STALE_SECONDS}s restart-delay={restart_delay}s"
    )

    while running:
        try:
            users = _enrolled_users()
            # Build the full {job_id → spec} map for this cycle.
            # Tuple shape: (job_id, argv_tail, log_path, email_or_None).
            # The trailing email is for is_daemon_running()'s per-user
            # disambiguation — without it, scott's `gmail-search watch`
            # would mark silvershabbat's `watch:<uid>` as already-running.
            wanted: list[tuple[str, list[str], Path, str | None]] = []
            for u in users:
                for spec in PER_USER_DAEMONS:
                    job_id = f"{spec['key']}:{u['id']}"
                    argv_tail = spec["argv"] + ["--email", u["email"]]
                    log_path = data_dir / f"{u['id']}-{spec['log_suffix']}"
                    wanted.append((job_id, argv_tail, log_path, u["email"]))
            for spec in GLOBAL_DAEMONS:
                wanted.append((spec["key"], spec["argv"], spec["log"], None))

            # Converge down: any per-user daemon for a now-disabled user
            # is stopped. Runs before the respawn pass so we don't fight
            # ourselves on a user toggled off mid-cycle.
            for u in _disabled_users():
                for spec in PER_USER_DAEMONS:
                    _stop_unwanted_daemon(f"{spec['key']}:{u['id']}", u["email"])

            alive = 0
            for job_id, argv_tail, log_path, email in wanted:
                row = JobProgress.get(db_path, job_id)
                if _is_alive(row):
                    alive += 1
                    continue
                if not _should_respawn(job_id):
                    continue
                # Stale heartbeat + OS process alive = blocked daemon, not
                # dead. Don't pile up duplicates fighting the same backend.
                if is_daemon_running(job_id, email=email):
                    _log.warning(f"{job_id} heartbeat stale but process still alive — skipping respawn")
                    alive += 1
                    continue
                prev_detail = (row.get("detail") if row else "") or "(no prior row)"
                age = _row_age_seconds(row)
                _log.info(
                    f"{job_id} stale "
                    f"(age={age!r}s, status={row.get('status') if row else 'MISSING'}, "
                    f"prev_detail={prev_detail!r}) — respawning"
                )
                try:
                    _spawn(job_id, argv_tail, log_path)
                except Exception as e:
                    _log.exception(f"failed to spawn {job_id}: {e}")

            supervisor_progress.update(
                "watching",
                alive,
                len(wanted),
                f"{alive}/{len(wanted)} alive across {len(users)} user(s)",
            )
        except Exception as e:
            # A transient failure (e.g. Postgres dropping the connection
            # under memory pressure) must NOT kill the supervisor — log it
            # and retry next interval. An unhandled OperationalError here
            # is exactly what silently crash-killed the loop before.
            _log.exception(f"supervisor cycle failed — retrying in {interval}s: {e}")

        for _ in range(interval):
            if not running:
                break
            _time.sleep(1)

    supervisor_progress.finish("stopped", "supervisor exited cleanly")
    click.echo("Supervisor stopped.")


@main.command(help="Show progress of running jobs and daemon status")
@common_options
@click.pass_context
def progress(ctx):
    from gmail_search.store.db import JobProgress

    data_dir = ctx.obj["data_dir"]

    # Daemon status
    running, pid = _is_watch_running(data_dir)
    if running:
        click.echo(f"Watch daemon: running (PID {pid})")
    else:
        click.echo("Watch daemon: not running")

    # DB stats
    conn = get_connection(ctx.obj["db_path"])
    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    cost = get_total_spend(conn)
    dates = conn.execute("SELECT MIN(date) as oldest, MAX(date) as newest FROM messages").fetchone()
    conn.close()
    click.echo(f"Messages: {msg_count:,} | Embeddings: {emb_count:,} | Cost: ${cost:.2f}")
    if dates["oldest"]:
        click.echo(f"Date range: {dates['oldest'][:10]} to {dates['newest'][:10]}")

    # Job history
    jobs = JobProgress.get(ctx.obj["db_path"])
    if jobs:
        click.echo("\nRecent jobs:")
        for j in jobs:
            from datetime import datetime

            pct = f"{j['completed'] * 100 // j['total']}%" if j["total"] > 0 else ""
            try:
                start = datetime.fromisoformat(j["started_at"])
                updated = datetime.fromisoformat(j["updated_at"])
                elapsed = f"{(updated - start).total_seconds():.0f}s"
            except Exception:
                elapsed = "?"
            status_icon = {"running": ">", "done": "+", "stopped": "-", "error": "!"}.get(j["status"], "?")
            click.echo(
                f"  {status_icon} {j['job_id']:8s} {j['status']:8s} {j['stage']:10s} {pct:>5s} {elapsed:>6s}  {j['detail'][:50]}"
            )

    # Log tail
    log_path = data_dir / "watch.log"
    if log_path.exists():
        lines = log_path.read_text().strip().split("\n")
        recent = [l for l in lines[-5:] if l.strip()]
        if recent:
            click.echo(f"\nRecent log ({log_path}):")
            for line in recent:
                click.echo(f"  {line[:100]}")


@main.command(help="Tail the watch daemon log")
@click.option("-n", "--lines", type=int, default=20, help="Number of lines")
@click.option("-f", "--follow", is_flag=True, help="Follow log output")
@common_options
@click.pass_context
def logs(ctx, lines, follow):
    """View the watch daemon log."""
    import subprocess

    log_path = ctx.obj["data_dir"] / "watch.log"
    if not log_path.exists():
        click.echo("No log file found. Start the daemon first: gmail-search start")
        return

    if follow:
        click.echo(f"Following {log_path} (Ctrl+C to stop):")
        subprocess.run(["tail", "-f", "-n", str(lines), str(log_path)])
    else:
        text = log_path.read_text().strip().split("\n")
        for line in text[-lines:]:
            click.echo(line)


@main.command(help="Search your email")
@click.argument("query")
@click.option("-k", "--top-k", type=int, default=None, help="Number of results")
@common_options
@click.pass_context
def search(ctx, query, top_k):
    from gmail_search.search.engine import SearchEngine

    cfg = ctx.obj["config"]
    k = top_k or cfg["search"]["default_top_k"]
    index_dir = ctx.obj["data_dir"] / "scann_index"

    engine = SearchEngine(ctx.obj["db_path"], index_dir, cfg)
    results = engine.search(query, top_k=k)

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        badge = f"[{r.match_type}]"
        att = f" ({r.attachment_filename})" if r.attachment_filename else ""
        click.echo(f"\n{i}. {r.subject} {badge}{att}")
        click.echo(f"   From: {r.from_addr} | Date: {r.date} | Score: {r.score:.3f}")
        click.echo(f"   {r.snippet}")


@main.command(help="Show cost information")
@click.option("--breakdown", is_flag=True, help="Show breakdown by operation")
@common_options
@click.pass_context
def cost(ctx, breakdown):
    conn = get_connection(ctx.obj["db_path"])
    total = get_total_spend(conn)
    click.echo(f"Total spend: ${total:.2f}")

    if breakdown:
        bd = get_spend_breakdown(conn)
        for op, amount in sorted(bd.items()):
            click.echo(f"  {op}: ${amount:.2f}")

    cfg = ctx.obj["config"]
    ok, spent, remaining = check_budget(conn, cfg["budget"]["max_usd"])
    click.echo(f"Budget: ${cfg['budget']['max_usd']:.2f} | Remaining: ${remaining:.2f}")
    conn.close()


@main.command(help="Show system status")
@common_options
@click.pass_context
def status(ctx):
    conn = get_connection(ctx.obj["db_path"])

    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    att_count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
    emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    total_cost = get_total_spend(conn)

    from gmail_search.store.queries import get_sync_state

    last_sync = get_sync_state(conn, "last_history_id")

    click.echo(f"Messages: {msg_count}")
    click.echo(f"Attachments: {att_count}")
    click.echo(f"Embeddings: {emb_count}")
    click.echo(f"Total cost: ${total_cost:.2f}")
    click.echo(f"Last history ID: {last_sync or 'never synced'}")
    conn.close()


@main.command(help="Generate per-message summaries via the configured local LLM backend")
@click.option(
    "--concurrency",
    type=int,
    default=12,
    help="Parallel LLM requests (default 12).",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Emails per LLM call (default 1). Batching regresses on small GPUs — leave at 1 unless benched.",
)
@click.option("--limit", type=int, default=None, help="Max messages per backfill pass (default: all pending)")
@click.option(
    "--loop",
    is_flag=True,
    default=False,
    help="Keep running — after each pass, re-query pending and start again. "
    "Combined with a small --limit this makes newly-arrived mail jump to "
    "the front (pending is ORDER BY date DESC) so the summarizer tracks "
    "the live inbox instead of grinding through a stale snapshot.",
)
@click.option(
    "--loop-batch",
    type=int,
    default=500,
    help="When --loop is set and --limit is not, cap each pass at this many messages (default 500).",
)
@click.option(
    "--loop-sleep",
    type=float,
    default=5.0,
    help="Seconds to sleep between passes when --loop is set and the last pass found nothing (default 5).",
)
@click.option(
    "--email",
    type=str,
    default=None,
    help="Gmail account to summarize. Defaults to GMS_BOOTSTRAP_EMAIL env. "
    "API spawns one daemon per user with --email set.",
)
@common_options
@click.pass_context
def summarize(ctx, concurrency, batch_size, limit, loop, loop_batch, loop_sleep, email):
    import logging
    import os as _os
    import time

    from gmail_search.llm import get_backend
    from gmail_search.summarize import backfill

    if email:
        _os.environ["GMS_BOOTSTRAP_EMAIL"] = email

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    backend = get_backend()
    pass_limit = limit if limit is not None else (loop_batch if loop else None)
    click.echo(
        f"Summarizing via {type(backend).__name__} (model={backend.model_id}, "
        f"concurrency={concurrency}, batch_size={batch_size}, "
        f"limit={pass_limit}, loop={loop})"
    )

    # Per-user job_progress so two users' summarize daemons coexist.
    from gmail_search.auth.write_user import resolve_write_user_id as _resolve_uid

    _conn = get_connection(ctx.obj["db_path"])
    try:
        active_user_id = _resolve_uid(_conn)
    finally:
        _conn.close()
    job_id = f"summarize:{active_user_id}"

    # In loop mode we keep an `summarize` job_progress row warm between
    # passes so the supervisor / /api/jobs/running see the daemon as
    # alive even when there's nothing to summarize. Created lazily —
    # no-op for the one-shot path.
    from gmail_search.store.db import JobProgress as _JP

    idle_progress = _JP(ctx.obj["db_path"], job_id, start_completed=0) if loop else None

    while True:
        if idle_progress is not None:
            try:
                idle_progress.update("pass", 0, 0, "checking for pending summaries")
            except Exception:
                pass
        result = backfill(
            ctx.obj["db_path"],
            concurrency=concurrency,
            batch_size=batch_size,
            limit=pass_limit,
        )
        click.echo(
            f"Pass: {result['done']}/{result['total']} summarized "
            f"({result['failed']} failed, {result['auto_classified']} auto) in {result['seconds']}s"
        )
        if not loop:
            break
        # Idle sleep when nothing to do; otherwise loop straight into the
        # next pass so newly-arrived mail gets picked up within one
        # batch's latency. Heartbeat during the idle sleep so the
        # supervisor doesn't consider us dead.
        if result["total"] == 0:
            if idle_progress is not None:
                try:
                    idle_progress.update("idle", 0, 0, "waiting for new pending messages")
                except Exception:
                    pass
            slept = 0
            while slept < loop_sleep:
                if idle_progress is not None and slept % 30 == 0:
                    try:
                        idle_progress.heartbeat()
                    except Exception:
                        pass
                chunk = min(1.0, loop_sleep - slept)
                time.sleep(chunk)
                slept += 1


@main.command(help="Crawl URLs linked in emails and store the page text as attachments.extracted_text")
@click.option(
    "--concurrency",
    type=int,
    default=10,
    help="Parallel crawl4ai fetches. Default 10 is sized for a ~20-core / 32GB box. "
    "Each Chromium tab is ~100-200MB + a CPU burst for JS, so bump to 15-20 on bigger "
    "hardware or drop to 3-5 if RAM is tight.",
)
@click.option("--limit", type=int, default=None, help="Max URLs per pass")
@click.option(
    "--loop",
    is_flag=True,
    default=False,
    help="Keep running — after each pass, re-query pending and start again.",
)
@click.option(
    "--loop-batch",
    type=int,
    default=100,
    help="When --loop is set and --limit is not, cap each pass at this many URLs (default 100).",
)
@click.option(
    "--loop-sleep",
    type=float,
    default=10.0,
    help="Seconds to sleep between passes when --loop is set and the last pass found nothing.",
)
@common_options
@click.pass_context
def crawl_urls(ctx, concurrency, limit, loop, loop_batch, loop_sleep):
    import asyncio
    import time

    from gmail_search.gmail import url_fetcher

    db_path = ctx.obj["db_path"]
    pass_limit = limit if limit is not None else (loop_batch if loop else None)
    click.echo(f"Crawling URLs (concurrency={concurrency}, limit={pass_limit}, loop={loop})")

    while True:
        result = asyncio.run(url_fetcher.run(db_path, concurrency=concurrency, limit=pass_limit))
        click.echo(f"Pass: {result['done']} ok / {result['failed']} failed of {result['total']} pending")
        if not loop:
            break
        if result["total"] == 0:
            time.sleep(loop_sleep)


@main.command(help="Start the web UI")
@click.option("--host", default=None)
@click.option("--port", type=int, default=None)
@common_options
@click.pass_context
def serve(ctx, host, port):
    import uvicorn

    from gmail_search.server import create_app

    cfg = ctx.obj["config"]
    h = host or cfg["server"]["host"]
    p = port or cfg["server"]["port"]

    app = create_app(ctx.obj["db_path"], ctx.obj["data_dir"], cfg)
    click.echo(f"Starting server at http://{h}:{p}")
    uvicorn.run(app, host=h, port=p)


@main.command(help="Delete deep-analysis artifacts older than the retention window")
@click.option(
    "--retention-days",
    type=int,
    default=30,
    help="Delete artifacts for sessions finished more than this many days ago. Default 30.",
)
@common_options
@click.pass_context
def prune_artifacts(ctx, retention_days):
    """Nightly GC for `agent_artifacts`. Intended to be run from a
    systemd timer or cron. Session + event rows are kept regardless
    — only the artifact bytes get dropped.
    """
    from gmail_search.agents.gc import prune_artifacts as _prune
    from gmail_search.agents.gc import prune_scratch_dirs as _prune_scratch

    conn = get_connection(ctx.obj["db_path"])
    try:
        result = _prune(conn, retention_days=retention_days)
    finally:
        conn.close()
    click.echo(
        f"pruned {result.artifacts_deleted} artifacts "
        f"(~{result.bytes_freed_estimate:,} bytes) across {result.sessions_considered} sessions "
        f"older than {retention_days}d"
    )
    scratch = _prune_scratch(retention_days=retention_days)
    click.echo(
        f"pruned {scratch.dirs_deleted} scratch dir(s) "
        f"(~{scratch.bytes_freed:,} bytes) older than {retention_days}d"
    )


# ─────────────────────────────────────────────────────────────────────
# Multi-tenant identity (Phase 1 of PER_USER_LOGIN_2026-04-27.md)
# ─────────────────────────────────────────────────────────────────────
# Op-only commands. They edit `users` / `invited_emails` regardless of
# whether GMAIL_MULTI_TENANT is set — the env flag gates the *runtime*
# auth wall, not the schema. This means a pre-flag invite call still
# works, so the bootstrap order is: invite → flip flag → sign in.


# Reuse the auth module's helper so the CLI and the runtime auth path
# stay aligned. If `normalize_email` ever grows (e.g. unicode NFKC for
# homograph defense), the CLI inherits it automatically — otherwise a
# `gmail-search invite` could store an email the callback's allowlist
# check then fails to recognize.
from gmail_search.auth.session import normalize_email as _normalize_email  # noqa: E402


@main.command(help="Invite an email address to sign in (multi-tenant Phase 1)")
@click.argument("email")
@click.option("--note", default=None, help="Free-text note recorded with the invite.")
@common_options
@click.pass_context
def invite(ctx, email, note):
    normalized = _normalize_email(email)
    conn = get_connection(ctx.obj["db_path"])
    try:
        # ON CONFLICT DO NOTHING so re-running is idempotent — convenient
        # for bootstrap scripts. We report whether the row was inserted
        # vs already-present so the operator knows which case happened.
        row = conn.execute(
            "INSERT INTO invited_emails (email, note) VALUES (%s, %s) "
            "ON CONFLICT (email) DO NOTHING RETURNING email",
            (normalized, note),
        ).fetchone()
        conn.commit()
    finally:
        conn.close()
    if row:
        click.echo(f"invited: {normalized}")
    else:
        click.echo(f"already invited: {normalized}")


@main.command("list-users", help="List invited emails and registered users (Phase 1)")
@common_options
@click.pass_context
def list_users(ctx):
    conn = get_connection(ctx.obj["db_path"])
    try:
        invites = conn.execute("SELECT email, invited_at, note FROM invited_emails ORDER BY invited_at").fetchall()
        users = conn.execute(
            "SELECT id, email, name, last_login_at, created_at FROM users ORDER BY created_at"
        ).fetchall()
    finally:
        conn.close()
    click.echo(f"invited_emails: {len(invites)}")
    for inv in invites:
        suffix = f" ({inv['note']})" if inv["note"] else ""
        click.echo(f"  {inv['email']}  invited_at={inv['invited_at']}{suffix}")
    click.echo(f"users: {len(users)}")
    for u in users:
        click.echo(
            f"  {u['id']}  {u['email']}  name={u['name'] or '-'}  "
            f"last_login={u['last_login_at'] or 'never'}  created={u['created_at']}"
        )


@main.command("delete-user", help="Hard-delete a user by email (Phase 1)")
@click.argument("email")
@click.option("--also-uninvite", is_flag=True, help="Also remove from invited_emails.")
@common_options
@click.pass_context
def delete_user(ctx, email, also_uninvite):
    normalized = _normalize_email(email)
    conn = get_connection(ctx.obj["db_path"])
    try:
        deleted = conn.execute("DELETE FROM users WHERE email = %s RETURNING id", (normalized,)).fetchone()
        uninvited = None
        if also_uninvite:
            uninvited = conn.execute(
                "DELETE FROM invited_emails WHERE email = %s RETURNING email",
                (normalized,),
            ).fetchone()
        conn.commit()
    finally:
        conn.close()
    if deleted:
        click.echo(f"deleted user: {normalized} (id={deleted['id']})")
    else:
        click.echo(f"no user found for: {normalized}")
    if also_uninvite:
        if uninvited:
            click.echo(f"uninvited: {normalized}")
        else:
            click.echo(f"no invite found for: {normalized}")
