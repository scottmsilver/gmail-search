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
@common_options
@click.pass_context
def update(ctx, max_messages, budget, batch_size):
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


@main.command(help="Watch for new emails and process them continuously")
@click.option("--interval", type=int, default=120, help="Seconds between sync checks")
@click.option("--budget", type=float, default=None, help="Override budget limit")
@common_options
@click.pass_context
def watch(ctx, interval, budget):
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

    click.echo(f"Watching for new emails every {interval}s. Ctrl+C to stop.")
    cycle = 0

    while running:
        cycle += 1
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


@main.command(help="Show progress of running jobs")
@common_options
@click.pass_context
def progress(ctx):
    from gmail_search.store.db import JobProgress

    jobs = JobProgress.get(ctx.obj["db_path"])
    if not jobs:
        click.echo("No jobs found.")
        return
    for j in jobs:
        pct = f"{j['completed']*100//j['total']}%" if j["total"] > 0 else "?"
        elapsed = ""
        try:
            from datetime import datetime

            start = datetime.fromisoformat(j["started_at"])
            updated = datetime.fromisoformat(j["updated_at"])
            elapsed = f" ({(updated - start).total_seconds():.0f}s elapsed)"
        except Exception:
            pass
        click.echo(f"{j['job_id']}: {j['status']} | {j['stage']} | {j['completed']}/{j['total']} ({pct}){elapsed}")
        if j["detail"]:
            click.echo(f"  {j['detail']}")


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
