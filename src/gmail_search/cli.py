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


@main.command(help="Run OAuth flow to authenticate with Gmail")
@common_options
@click.pass_context
def auth(ctx):
    from gmail_search.gmail.auth import get_credentials

    data_dir = ctx.obj["data_dir"]
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


@main.command(help="Extract text and images from downloaded attachments")
@common_options
@click.pass_context
def extract(ctx):
    from tqdm import tqdm

    from gmail_search.extract import dispatch
    from gmail_search.store.queries import get_attachments_for_message

    cfg = ctx.obj["config"]
    att_config = cfg.get("attachments", {})
    conn = get_connection(ctx.obj["db_path"])
    try:
        rows = conn.execute("SELECT id FROM messages").fetchall()
        updated = 0

        for row in tqdm(rows, desc="Extracting attachments"):
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
                    updates["image_path"] = str(result.images[0].parent if len(result.images) > 1 else result.images[0])

                if updates:
                    set_clause = ", ".join(f"{k} = ?" for k in updates)
                    conn.execute(
                        f"UPDATE attachments SET {set_clause} WHERE id = ?",
                        (*updates.values(), att.id),
                    )
                    conn.commit()
                    updated += 1
    finally:
        conn.close()
    click.echo(f"Extracted content from {updated} attachments.")


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
            "DELETE FROM embeddings WHERE chunk_type = 'message' AND model = ?", (emb_model,)
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


@main.command(help="Rebuild the ScaNN search index and FTS index")
@common_options
@click.pass_context
def reindex(ctx):
    from gmail_search.index.builder import build_index
    from gmail_search.store.db import (
        clear_query_cache,
        rebuild_contact_frequency,
        rebuild_fts,
        rebuild_spell_dictionary,
        rebuild_term_aliases,
        rebuild_thread_summary,
        rebuild_topics,
    )

    cfg = ctx.obj["config"]
    db = ctx.obj["db_path"]
    index_dir = ctx.obj["data_dir"] / "scann_index"
    cleared = clear_query_cache(db)
    if cleared:
        click.echo(f"Cleared {cleared} cached query embeddings.")
    build_index(
        db_path=db, index_dir=index_dir, model=cfg["embedding"]["model"], dimensions=cfg["embedding"]["dimensions"]
    )
    fts_count = rebuild_fts(db)
    thread_count = rebuild_thread_summary(db)
    contact_count = rebuild_contact_frequency(db)
    word_count = rebuild_spell_dictionary(db, ctx.obj["data_dir"])
    topic_count = rebuild_topics(db)
    alias_count = rebuild_term_aliases(db)
    click.echo(
        f"Index rebuilt. ScaNN + {fts_count} FTS + {thread_count} threads + {contact_count} contacts + {word_count} words + {topic_count} topics + {alias_count} aliases."
    )


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
@common_options
@click.pass_context
def update(ctx, max_messages, budget, batch_size, min_free_gb):
    from gmail_search.embed.pipeline import run_embedding_pipeline
    from gmail_search.extract import dispatch
    from gmail_search.gmail.auth import build_gmail_service
    from gmail_search.gmail.client import download_messages
    from gmail_search.index.builder import build_index
    from gmail_search.store.db import (  # noqa: E501
        JobProgress,
        rebuild_contact_frequency,
        rebuild_fts,
        rebuild_spell_dictionary,
        rebuild_thread_summary,
        rebuild_topics,
    )
    from gmail_search.store.queries import get_attachments_for_message

    cfg = ctx.obj["config"]
    db_path = ctx.obj["db_path"]
    data_dir = ctx.obj["data_dir"]
    att_config = cfg.get("attachments", {})

    if budget:
        cfg["budget"]["max_usd"] = budget

    service = build_gmail_service(data_dir)
    max_msg = max_messages or cfg["download"].get("max_messages")
    index_dir = data_dir / "scann_index"

    progress = JobProgress(db_path, "update")

    total_downloaded = 0
    total_extracted = 0
    total_embedded = 0
    batch_num = 0

    # Get starting message count
    conn = get_connection(db_path)
    start_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    conn.close()

    while True:
        if min_free_gb is not None:
            import shutil as _shutil

            free_gb = _shutil.disk_usage(str(data_dir)).free / (1024**3)
            if free_gb < min_free_gb:
                click.echo(f"Free disk {free_gb:.2f} GiB < min {min_free_gb} GiB — stopping backfill.")
                break
        batch_num += 1
        # How many to download this round — cap at batch_size new messages
        current_limit = start_count + total_downloaded + batch_size
        if max_msg:
            current_limit = min(current_limit, max_msg)

        click.echo(f"\n{'='*50}")
        click.echo(f"Batch {batch_num}: downloading up to {current_limit} total messages")
        click.echo(f"{'='*50}")

        progress.update("download", total_downloaded, max_msg or current_limit, f"batch {batch_num}")

        # Download one batch
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
            click.echo("No new messages to download.")
            break

        click.echo(f"Downloaded {dl_count} messages.")

        # Extract new attachments
        progress.update("extract", total_downloaded, max_msg or current_limit, f"+{dl_count} downloaded")
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
                    updates["image_path"] = str(result.images[0].parent if len(result.images) > 1 else result.images[0])
                if updates:
                    set_clause = ", ".join(f"{k} = ?" for k in updates)
                    conn.execute(f"UPDATE attachments SET {set_clause} WHERE id = ?", (*updates.values(), att.id))
                    conn.commit()
                    extracted += 1
        conn.close()
        total_extracted += extracted
        click.echo(f"Extracted {extracted} attachments.")

        # Embed new messages + attachments
        progress.update("embed", total_downloaded, max_msg or current_limit, f"+{extracted} extracted")
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

        # Reindex so search is live
        progress.update("reindex", total_downloaded, max_msg or current_limit, f"+{emb_count} embedded")
        click.echo("Reindexing...")
        build_index(
            db_path=db_path,
            index_dir=index_dir,
            model=cfg["embedding"]["model"],
            dimensions=cfg["embedding"]["dimensions"],
        )
        rebuild_fts(db_path)
        rebuild_thread_summary(db_path)
        rebuild_contact_frequency(db_path)
        rebuild_spell_dictionary(db_path, data_dir)
        rebuild_topics(db_path)

        conn = get_connection(db_path)
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        emb_total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        cost = get_total_spend(conn)
        conn.close()
        click.echo(f"Status: {msg_count:,} messages | {emb_total:,} embeddings | ${cost:.2f} spent")

        # Check if we've hit the limit
        if max_msg and msg_count >= max_msg:
            click.echo("Reached max message limit.")
            break

    progress.finish("done", f"+{total_downloaded} downloaded, +{total_extracted} extracted, +{total_embedded} embedded")

    click.echo(f"\n{'='*50}")
    click.echo(f"Done! +{total_downloaded} downloaded, +{total_extracted} extracted, +{total_embedded} embedded")
    conn = get_connection(db_path)
    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    emb_total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    total_cost = get_total_spend(conn)
    conn.close()
    click.echo(f"Total: {msg_count:,} messages | {emb_total:,} embeddings | ${total_cost:.2f} spent")


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
    from gmail_search.index.builder import build_index
    from gmail_search.store.db import JobProgress, rebuild_fts, rebuild_thread_summary
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

    while running:
        cycle += 1
        if max_cycles is not None and cycle > max_cycles:
            break
        progress.update("sync", cycle, 0, "checking for new emails")

        # Sync new messages
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
                            set_clause = ", ".join(f"{k} = ?" for k in updates)
                            conn.execute(
                                f"UPDATE attachments SET {set_clause} WHERE id = ?", (*updates.values(), att.id)
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

            # Reindex (lightweight — just ScaNN, FTS, thread summary)
            progress.update("reindex", cycle, 0, "updating indexes")
            build_index(
                db_path=db_path,
                index_dir=index_dir,
                model=cfg["embedding"]["model"],
                dimensions=cfg["embedding"]["dimensions"],
            )
            rebuild_fts(db_path)
            rebuild_thread_summary(db_path)

            conn = get_connection(db_path)
            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            emb_total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            conn.close()
            click.echo(f"[cycle {cycle}] {msg_count:,} messages | {emb_total:,} embeddings")
        else:
            progress.update("idle", cycle, 0, "no new messages")

        # Sleep in small increments so Ctrl+C is responsive
        for _ in range(interval):
            if not running:
                break
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


@main.command(help="Generate per-message summaries via a local Ollama model")
@click.option("--model", default=None, help="Ollama model tag (default: qwen2.5:7b)")
@click.option("--concurrency", type=int, default=4, help="Parallel requests (default 4)")
@click.option("--limit", type=int, default=None, help="Max messages to process this run")
@common_options
@click.pass_context
def summarize(ctx, model, concurrency, limit):
    import logging

    from gmail_search.summarize import DEFAULT_MODEL, backfill

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chosen = model or DEFAULT_MODEL
    click.echo(f"Summarizing with {chosen} (concurrency={concurrency}, limit={limit})")
    result = backfill(
        ctx.obj["db_path"],
        model=chosen,
        concurrency=concurrency,
        limit=limit,
    )
    click.echo(
        f"Done: {result['done']}/{result['total']} summarized " f"({result['failed']} failed) in {result['seconds']}s"
    )


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
