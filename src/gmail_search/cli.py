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
    if ctx.obj and ctx.obj.get("_initialised"):
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
    conn = get_connection(ctx.obj["db_path"])
    att_config = cfg.get("attachments", {})

    rows = conn.execute("SELECT id FROM messages").fetchall()
    updated = 0

    for row in tqdm(rows, desc="Extracting attachments"):
        attachments = get_attachments_for_message(conn, row["id"])
        for att in attachments:
            if att.extracted_text or att.image_path:
                continue
            if not att.raw_path or not Path(att.raw_path).exists():
                continue

            result = dispatch(att.mime_type, Path(att.raw_path), att_config)
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

    conn.close()
    click.echo(f"Extracted content from {updated} attachments.")


@main.command(help="Embed all unembedded messages and attachments")
@click.option("--model", default=None, help="Override embedding model")
@click.option("--budget", type=float, default=None, help="Override budget limit")
@common_options
@click.pass_context
def embed(ctx, model, budget):
    from gmail_search.embed.pipeline import run_embedding_pipeline

    cfg = ctx.obj["config"]
    if model:
        cfg["embedding"]["model"] = model
    if budget:
        cfg["budget"]["max_usd"] = budget

    conn = get_connection(ctx.obj["db_path"])
    ok, spent, remaining = check_budget(conn, cfg["budget"]["max_usd"])
    conn.close()
    click.echo(f"Budget: ${cfg['budget']['max_usd']:.2f} | Spent: ${spent:.2f} | Remaining: ${remaining:.2f}")

    count = run_embedding_pipeline(ctx.obj["db_path"], cfg)
    click.echo(f"Embedded {count} new chunks.")


@main.command(help="Rebuild the ScaNN search index")
@common_options
@click.pass_context
def reindex(ctx):
    from gmail_search.index.builder import build_index

    cfg = ctx.obj["config"]
    index_dir = ctx.obj["data_dir"] / "scann_index"
    build_index(
        db_path=ctx.obj["db_path"],
        index_dir=index_dir,
        model=cfg["embedding"]["model"],
        dimensions=cfg["embedding"]["dimensions"],
    )
    click.echo("Index rebuilt.")


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
