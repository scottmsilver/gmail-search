import functools
import logging
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


@main.command(help="Run OAuth flow to authenticate with Gmail / Drive")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Delete existing token.json and re-prompt the browser consent screen. "
    "Required after adding a new scope (e.g. drive.readonly).",
)
@common_options
@click.pass_context
def auth(ctx, force):
    from gmail_search.gmail.auth import get_credentials

    data_dir = ctx.obj["data_dir"]
    if force:
        token_path = data_dir / "token.json"
        if token_path.exists():
            token_path.unlink()
            click.echo(f"Removed {token_path} — next step will re-prompt consent.")
    get_credentials(data_dir)
    click.echo("Authentication successful. Token saved.")


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
        from gmail_search.gmail.url_extract import extract_crawlable_urls
        from gmail_search.store.queries import upsert_url_stub

        msg_rows = conn.execute("SELECT id, body_text, labels FROM messages WHERE length(body_text) >= 50").fetchall()
        new_url_stubs = 0
        for r in tqdm(msg_rows, desc="Scanning bodies for URLs"):
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
@click.pass_context
def reindex(ctx):
    from gmail_search.pipeline import reindex as _reindex

    _reindex(
        db_path=ctx.obj["db_path"],
        data_dir=ctx.obj["data_dir"],
        cfg=ctx.obj["config"],
        light=False,
    )
    click.echo("Index rebuilt (ScaNN + FTS + thread summary + contacts + spell + topics + aliases).")


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
@common_options
@click.pass_context
def update(ctx, max_messages, budget, batch_size, min_free_gb, loop, loop_sleep):
    import time as _time

    from gmail_search.embed.pipeline import run_embedding_pipeline
    from gmail_search.extract import dispatch
    from gmail_search.gmail.auth import build_gmail_service
    from gmail_search.gmail.client import download_messages
    from gmail_search.locks import write_lock
    from gmail_search.pipeline import reindex as _reindex
    from gmail_search.store.db import JobProgress
    from gmail_search.store.queries import get_attachments_for_message

    cfg = ctx.obj["config"]
    db_path = ctx.obj["db_path"]
    data_dir = ctx.obj["data_dir"]
    att_config = cfg.get("attachments", {})

    if budget:
        cfg["budget"]["max_usd"] = budget

    service = build_gmail_service(data_dir)
    max_msg = max_messages or cfg["download"].get("max_messages")
    index_dir = data_dir / "scann_index"  # noqa: F841

    def _one_cycle() -> None:
        """Run the full download→extract→embed→reindex loop once, then
        return. When --loop is set, the outer driver calls this in a
        crash-tolerant while-True around a sleep, so the job never dies
        permanently — matches the `watch` daemon's behaviour.
        """
        # Re-read the message count each cycle. Progress math is
        # relative to this cycle's start, not the first one, so the
        # UI rate/ETA reflects what's happening right now.
        conn = get_connection(db_path)
        start_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        conn.close()
        progress = JobProgress(db_path, "update", start_completed=start_count)

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

                if dl_count == 0:
                    click.echo("No new messages to download this cycle.")
                    break

                click.echo(f"Downloaded {dl_count} messages.")

                # Extract new attachments
                progress.update("extract", start_count + total_downloaded, progress_total, f"+{dl_count} downloaded")
                click.echo("Extracting attachments...")
                conn = get_connection(db_path)
                rows = conn.execute(
                    "SELECT DISTINCT m.id FROM messages m "
                    "JOIN attachments a ON a.message_id = m.id "
                    "WHERE a.extracted_text IS NULL AND a.image_path IS NULL AND a.raw_path IS NOT NULL"
                ).fetchall()
                extracted = 0
                for row in rows:
                    attachments = get_attachments_for_message(conn, row["id"])
                    for att in attachments:
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
                        updates = {}
                        if result.text:
                            updates["extracted_text"] = result.text
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
                            extracted += 1
                conn.close()
                total_extracted += extracted
                click.echo(f"Extracted {extracted} attachments.")

                # Crawl pending URL stubs written at ingest. Folded into
                # this loop (instead of a separate --loop daemon) so new
                # mail's URLs get fetched in the same cycle that
                # downloaded the mail, and the embedding step below sees
                # the crawled content on its first pass.
                try:
                    import asyncio as _asyncio

                    from gmail_search.gmail.url_fetcher import run as _crawl_urls_run

                    # Sized for a ~62GB / 20-core box: ~10 concurrent
                    # Chromium tabs ≈ 1-2GB RAM, well under what's free.
                    # Halves the inline crawl latency per update cycle
                    # vs the old `concurrency=3` default.
                    crawl_result = _asyncio.run(_crawl_urls_run(db_path, concurrency=10, limit=200))
                    if crawl_result["total"]:
                        click.echo(
                            f"Crawled {crawl_result['done']}/{crawl_result['total']} URL stubs "
                            f"({crawl_result['failed']} failed)."
                        )
                except Exception as e:
                    # Don't let a crawl failure kill the whole update cycle —
                    # next cycle retries.
                    logger.warning(f"URL crawl step failed: {e}")

                # Embed new messages + attachments
                progress.update("embed", start_count + total_downloaded, progress_total, f"+{extracted} extracted")
                click.echo("Embedding...")
                conn = get_connection(db_path)
                ok, spent, remaining = check_budget(conn, cfg["budget"]["max_usd"])
                conn.close()
                if not ok:
                    click.echo(f"Budget exhausted (${spent:.2f} spent). Stopping embedding.")
                    break
                emb_count = run_embedding_pipeline(db_path, cfg)
                total_embedded += emb_count
                click.echo(f"Embedded {emb_count} chunks.")

                # Reindex so search is live. light=True because heavy
                # rebuilds (aliases, query cache wipe) run once at the
                # end of the full update, not per-batch.
                progress.update("reindex", start_count + total_downloaded, progress_total, f"+{emb_count} embedded")
                click.echo("Reindexing...")
                _reindex(db_path=db_path, data_dir=data_dir, cfg=cfg, light=True)

                conn = get_connection(db_path)
                msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                emb_total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                cost = get_total_spend(conn)
                conn.close()
                click.echo(f"Status: {msg_count:,} messages | {emb_total:,} embeddings | ${cost:.2f} spent")

                if max_msg and msg_count >= max_msg:
                    click.echo("Reached max message limit.")
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
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        emb_total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        total_cost = get_total_spend(conn)
        conn.close()
        click.echo(f"Total: {msg_count:,} messages | {emb_total:,} embeddings | ${total_cost:.2f} spent")

    # ── outer driver ──────────────────────────────────────────────────
    if not loop:
        _one_cycle()
        return

    cycle = 0
    while True:
        cycle += 1
        click.echo(f"\n{'#' * 50}\n# backfill loop cycle {cycle}\n{'#' * 50}")
        try:
            _one_cycle()
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

                _JP(db_path, "update", start_completed=0).update(
                    "error", 0, 0, f"crash: {type(e).__name__}: {str(e)[:120]}"
                )
            except Exception:
                pass
        click.echo(f"\n[backfill] sleeping {loop_sleep}s before next cycle...")
        # Hold the `update` heartbeat warm during the inter-cycle sleep
        # so the supervisor / HTTP status don't see the daemon as dead
        # while it's waiting to start the next cycle. We flip status back
        # to 'running' with stage=idle so /api/status reflects reality.
        try:
            from gmail_search.store.db import JobProgress as _JP

            idle = _JP(db_path, "update", start_completed=0)
            idle.update("idle", 0, 0, "waiting for next cycle")
        except Exception:
            idle = None
        for i in range(loop_sleep):
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
    import os
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
    import os

    data_dir = ctx.obj["data_dir"]
    running, pid = _is_watch_running(data_dir)
    if not running:
        click.echo("No watch daemon running.")
        return

    os.kill(pid, 15)  # SIGTERM — triggers graceful shutdown
    _pid_file(data_dir).unlink(missing_ok=True)
    click.echo(f"Stopped watch daemon (PID {pid})")


@main.command(help="Watch for new emails and process them continuously")
@click.option("--interval", type=int, default=120, help="Seconds between sync checks")
@click.option("--budget", type=float, default=None, help="Override budget limit")
@click.option("--max-cycles", type=int, default=None, help="Exit after N cycles (for one-shot 'frontfill' runs)")
@common_options
@click.pass_context
def watch(ctx, interval, budget, max_cycles):
    """Poll for new emails, extract, embed, and reindex continuously."""
    import signal
    import time as _time

    from gmail_search.embed.pipeline import run_embedding_pipeline
    from gmail_search.extract import dispatch
    from gmail_search.gmail.auth import build_gmail_service
    from gmail_search.gmail.client import sync_new_messages
    from gmail_search.pipeline import reindex as _reindex
    from gmail_search.store.db import JobProgress
    from gmail_search.store.queries import get_attachments_for_message

    cfg = ctx.obj["config"]
    db_path = ctx.obj["db_path"]
    data_dir = ctx.obj["data_dir"]
    att_config = cfg.get("attachments", {})
    index_dir = data_dir / "scann_index"

    if budget:
        cfg["budget"]["max_usd"] = budget

    service = build_gmail_service(data_dir)
    progress = JobProgress(db_path, "watch")
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
                count = sync_new_messages(
                    service=service,
                    db_path=db_path,
                    data_dir=data_dir,
                    max_attachment_size=cfg["attachments"]["max_file_size_mb"] * 1024 * 1024,
                )
            except Exception as e:
                logger.warning(f"Sync failed: {e}")
                count = 0

            if count > 0:
                click.echo(f"[cycle {cycle}] Synced {count} new messages")

                # Extract attachments
                progress.update("extract", cycle, 0, f"+{count} synced")
                conn = get_connection(db_path)
                try:
                    rows = conn.execute(
                        "SELECT DISTINCT m.id FROM messages m "
                        "JOIN attachments a ON a.message_id = m.id "
                        "WHERE a.extracted_text IS NULL AND a.image_path IS NULL AND a.raw_path IS NOT NULL"
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
                                updates["extracted_text"] = result.text
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
                ok, spent, remaining = check_budget(conn, cfg["budget"]["max_usd"])
                conn.close()
                if ok:
                    emb_count = run_embedding_pipeline(db_path, cfg)
                    if emb_count:
                        click.echo(f"[cycle {cycle}] Embedded {emb_count} chunks")
                else:
                    click.echo(f"[cycle {cycle}] Budget exhausted (${spent:.2f})")

                # Reindex — light=True keeps the hot path fast.
                progress.update("reindex", cycle, 0, "updating indexes")
                _reindex(db_path=db_path, data_dir=data_dir, cfg=cfg, light=True)

                conn = get_connection(db_path)
                msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                emb_total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
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


@main.command(help="Watchdog that keeps watch/update/summarize/reconcile alive")
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
    """Keep the three long-lived daemons alive.

    Polls `job_progress.updated_at` for each of `watch`, `update`, and
    `summarize`. If a row is stale for more than 90s (and we haven't
    just tried to restart it) we respawn the daemon detached. The
    supervisor itself also writes a `supervisor` row so you can see
    it's alive.

    Idempotent — running this under systemd-user or a tmux is fine;
    if a daemon was already alive the 'already running' check in each
    branch prevents a double-spawn.
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

    # Supervisor-side view of the supervised daemons. "job_id" is the
    # job_progress row key (same as the CLI subcommand), "argv" is what
    # we respawn if the daemon is stale, and "log" is the file stdout
    # + stderr land in.
    daemons = [
        {
            "job_id": "watch",
            "argv": ["watch", "--interval", "120"],
            "log": data_dir / "watch.log",
        },
        {
            "job_id": "update",
            "argv": ["update", "--loop"],
            "log": data_dir / "backfill.log",
        },
        {
            "job_id": "summarize",
            "argv": ["summarize", "--concurrency", "12", "--batch-size", "1", "--loop"],
            "log": data_dir / "summarize.log",
        },
        {
            "job_id": "reconcile",
            # Drift detector: scans for threads whose `thread_summary`
            # disagrees with `messages`, rebuilds them. Belt-and-
            # suspenders behind the inline `_touch_thread_summary`
            # call in upsert_message.
            "argv": ["reconcile", "--loop"],
            "log": data_dir / "reconcile.log",
        },
    ]

    # Track when we last attempted to spawn each daemon so we don't
    # thrash on a process that dies immediately after start.
    last_spawn_attempt: dict[str, float] = {}

    running = True

    def _on_signal(sig, frame):
        nonlocal running
        _log.info(f"received signal {sig} — stopping supervisor (supervised daemons stay alive)")
        running = False

    _signal.signal(_signal.SIGINT, _on_signal)
    _signal.signal(_signal.SIGTERM, _on_signal)

    # Self-heartbeat so operators can see the supervisor is alive.
    supervisor_progress = JobProgress(db_path, "supervisor", start_completed=0)
    supervisor_progress.update("starting", 0, len(daemons), "supervisor online")

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

    def _spawn(daemon: dict) -> None:
        argv = gmail_search_command() + daemon["argv"] + ["--data-dir", str(data_dir)]
        pid = spawn_detached(argv, daemon["log"])
        last_spawn_attempt[daemon["job_id"]] = _time.time()
        _log.info(f"supervisor: spawned {daemon['job_id']} pid={pid}")

    click.echo(
        f"Supervisor online: watching {[d['job_id'] for d in daemons]} "
        f"every {interval}s (stale > {_STALE_SECONDS}s, restart-delay {restart_delay}s)"
    )

    while running:
        alive = 0
        for daemon in daemons:
            job_id = daemon["job_id"]
            row = JobProgress.get(db_path, job_id)
            if _is_alive(row):
                alive += 1
                continue
            if not _should_respawn(job_id):
                _log.debug(f"{job_id} stale but in restart-delay window — skipping")
                continue
            # Stale heartbeat + OS process still alive = daemon blocked
            # (long HTTP call, flaky backend). Don't spawn a duplicate.
            # Before this guard, we'd pile up 10+ summarize daemons all
            # fighting for vLLM and reducing throughput to a crawl.
            if is_daemon_running(job_id):
                _log.warning(
                    f"{job_id} heartbeat stale but process still alive — "
                    "skipping respawn (probably blocked on backend)"
                )
                alive += 1
                continue
            prev_detail = (row.get("detail") if row else "") or "(no prior row)"
            age = _row_age_seconds(row)
            _log.info(
                f"{job_id} is stale "
                f"(age={age!r}s, status={row.get('status') if row else 'MISSING'}, "
                f"prev_detail={prev_detail!r}) — respawning"
            )
            try:
                _spawn(daemon)
            except Exception as e:
                _log.exception(f"failed to spawn {job_id}: {e}")

        try:
            supervisor_progress.update(
                "watching",
                alive,
                len(daemons),
                f"{alive}/{len(daemons)} alive",
            )
        except Exception:
            pass

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
@common_options
@click.pass_context
def summarize(ctx, concurrency, batch_size, limit, loop, loop_batch, loop_sleep):
    import logging
    import time

    from gmail_search.llm import get_backend
    from gmail_search.summarize import backfill

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    backend = get_backend()
    pass_limit = limit if limit is not None else (loop_batch if loop else None)
    click.echo(
        f"Summarizing via {type(backend).__name__} (model={backend.model_id}, "
        f"concurrency={concurrency}, batch_size={batch_size}, "
        f"limit={pass_limit}, loop={loop})"
    )

    # In loop mode we keep an `summarize` job_progress row warm between
    # passes so the supervisor / /api/jobs/running see the daemon as
    # alive even when there's nothing to summarize. Created lazily —
    # no-op for the one-shot path.
    from gmail_search.store.db import JobProgress as _JP

    idle_progress = _JP(ctx.obj["db_path"], "summarize", start_completed=0) if loop else None

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
