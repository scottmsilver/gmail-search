# Gmail Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local tool that downloads Gmail, embeds messages+attachments with Gemini, and searches via ScaNN.

**Architecture:** Pipeline with independent stages (download → extract → embed → index → search) sharing a SQLite data layer. Each stage is idempotent and resumable. CLI for all operations, FastAPI web UI for browsing.

**Tech Stack:** Python, Gmail API, google-genai (Gemini embedding-2-preview), ScaNN, SQLite, pymupdf, FastAPI, Click

**Spec:** `docs/superpowers/specs/2026-04-14-gmail-search-design.md`

---

## File Map

```
gmail-search/
  src/gmail_search/
    __init__.py              — Package init, version
    config.py                — Config loading from config.yaml + defaults
    store/
      __init__.py
      db.py                  — SQLite connection, schema creation
      models.py              — Dataclasses: Message, Attachment, EmbeddingRecord, CostRecord
      queries.py             — CRUD for messages, attachments, embeddings, sync_state
      cost.py                — Cost tracking, budget check, spend reporting
    gmail/
      __init__.py
      auth.py                — OAuth2 flow, token load/save/refresh
      client.py              — Batch download, incremental sync
      parser.py              — Gmail API response → Message + Attachment objects
    extract/
      __init__.py            — Dispatcher: dispatch(mime_type, path, config) → ExtractResult
      pdf.py                 — PDF text + page images via pymupdf
      image.py               — Passthrough for image attachments
    embed/
      __init__.py
      client.py              — Gemini embed_text / embed_image wrapper
      pipeline.py            — Orchestrate: find unembedded → batch embed → store
    index/
      __init__.py
      builder.py             — Load embeddings → build ScaNN index → save
      searcher.py            — Load ScaNN index → query → return (ids, scores)
    search/
      __init__.py
      engine.py              — Full search: embed query → ScaNN → fetch → dedupe → rank
    cli.py                   — Click CLI group with all commands
    server.py                — FastAPI app + API endpoints
  templates/
    index.html               — Search web UI
  tests/
    conftest.py              — Shared fixtures (tmp db, config, sample data)
    test_config.py
    test_store_db.py
    test_store_queries.py
    test_store_cost.py
    test_gmail_parser.py
    test_extract_dispatch.py
    test_extract_pdf.py
    test_embed_client.py
    test_embed_pipeline.py
    test_index_builder.py
    test_index_searcher.py
    test_search_engine.py
    test_cli.py
  config.yaml                — Default config
  pyproject.toml
  .gitignore
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `config.yaml`
- Create: `src/gmail_search/__init__.py`
- Create: `src/gmail_search/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "gmail-search"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "google-api-python-client>=2.100.0",
    "google-auth-oauthlib>=1.2.0",
    "google-genai>=1.0.0",
    "scann>=1.3.0",
    "pymupdf>=1.24.0",
    "numpy>=1.26.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.29.0",
    "click>=8.1.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0",
]

[project.scripts]
gmail-search = "gmail_search.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create .gitignore**

```
data/
*.pyc
__pycache__/
*.egg-info/
dist/
build/
.venv/
```

- [ ] **Step 3: Create config.yaml**

```yaml
budget:
  max_usd: 5.00

embedding:
  model: "gemini-embedding-2-preview"
  dimensions: 3072
  task_type_document: "RETRIEVAL_DOCUMENT"
  task_type_query: "RETRIEVAL_QUERY"

attachments:
  max_file_size_mb: 10
  max_pdf_pages: 20
  max_images_per_message: 10
  max_attachment_text_tokens: 50000

download:
  batch_size: 100
  max_messages: null

search:
  default_top_k: 20

server:
  host: "127.0.0.1"
  port: 8080
```

- [ ] **Step 4: Create src/gmail_search/__init__.py**

```python
__version__ = "0.1.0"
```

- [ ] **Step 5: Write failing test for config loading**

Create `tests/test_config.py`:

```python
from gmail_search.config import load_config


def test_load_config_defaults(tmp_path):
    """Config loads with all defaults when no file exists."""
    cfg = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert cfg["budget"]["max_usd"] == 5.00
    assert cfg["embedding"]["model"] == "gemini-embedding-2-preview"
    assert cfg["embedding"]["dimensions"] == 3072
    assert cfg["attachments"]["max_file_size_mb"] == 10
    assert cfg["download"]["batch_size"] == 100
    assert cfg["search"]["default_top_k"] == 20
    assert cfg["server"]["port"] == 8080


def test_load_config_from_file(tmp_path):
    """Config loads from YAML and overrides defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("budget:\n  max_usd: 20.00\n")
    cfg = load_config(config_path=config_file)
    assert cfg["budget"]["max_usd"] == 20.00
    # Other defaults still present
    assert cfg["embedding"]["model"] == "gemini-embedding-2-preview"


def test_load_config_data_dir(tmp_path):
    """Config resolves data_dir relative to project root."""
    cfg = load_config(config_path=tmp_path / "nonexistent.yaml", data_dir=tmp_path / "data")
    assert cfg["data_dir"] == str(tmp_path / "data")
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /home/ssilver/development/gmail-search && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gmail_search.config'`

- [ ] **Step 7: Implement config.py**

Create `src/gmail_search/config.py`:

```python
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "budget": {
        "max_usd": 5.00,
    },
    "embedding": {
        "model": "gemini-embedding-2-preview",
        "dimensions": 3072,
        "task_type_document": "RETRIEVAL_DOCUMENT",
        "task_type_query": "RETRIEVAL_QUERY",
    },
    "attachments": {
        "max_file_size_mb": 10,
        "max_pdf_pages": 20,
        "max_images_per_message": 10,
        "max_attachment_text_tokens": 50000,
    },
    "download": {
        "batch_size": 100,
        "max_messages": None,
    },
    "search": {
        "default_top_k": 20,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8080,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Path | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = DEFAULTS.copy()
    cfg = _deep_merge(DEFAULTS, {})

    if config_path and config_path.exists():
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, file_cfg)

    if data_dir is None:
        data_dir = Path.cwd() / "data"
    cfg["data_dir"] = str(data_dir)

    return cfg
```

- [ ] **Step 8: Create tests/conftest.py with shared fixtures**

```python
from pathlib import Path

import pytest

from gmail_search.config import load_config


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def test_config(tmp_path, data_dir):
    return load_config(config_path=tmp_path / "nonexistent.yaml", data_dir=data_dir)
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `cd /home/ssilver/development/gmail-search && python -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 10: Install the project in dev mode**

Run: `cd /home/ssilver/development/gmail-search && pip install -e ".[dev]" 2>&1 | tail -5`

- [ ] **Step 11: Commit**

```bash
git init
git add pyproject.toml .gitignore config.yaml src/ tests/
git commit -m "feat: project scaffolding with config loading"
```

---

### Task 2: Data Models

**Files:**
- Create: `src/gmail_search/store/__init__.py`
- Create: `src/gmail_search/store/models.py`

- [ ] **Step 1: Create store/__init__.py**

```python
```

(Empty init file.)

- [ ] **Step 2: Create store/models.py with dataclasses**

```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    id: str
    thread_id: str
    from_addr: str
    to_addr: str
    subject: str
    body_text: str
    body_html: str
    date: datetime
    labels: list[str]
    history_id: int
    raw_json: str


@dataclass
class Attachment:
    id: int | None
    message_id: str
    filename: str
    mime_type: str
    size_bytes: int
    extracted_text: str | None = None
    image_path: str | None = None
    raw_path: str | None = None


@dataclass
class EmbeddingRecord:
    id: int | None
    message_id: str
    attachment_id: int | None
    chunk_type: str  # "message", "attachment_text", "attachment_image"
    chunk_text: str | None
    embedding: bytes  # raw float32 blob
    model: str


@dataclass
class CostRecord:
    id: int | None
    timestamp: datetime
    operation: str  # "embed_text", "embed_image"
    model: str
    input_tokens: int
    image_count: int
    estimated_cost_usd: float
    message_id: str
```

- [ ] **Step 3: Verify import works**

Run: `cd /home/ssilver/development/gmail-search && python -c "from gmail_search.store.models import Message, Attachment, EmbeddingRecord, CostRecord; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/gmail_search/store/
git commit -m "feat: add data model dataclasses"
```

---

### Task 3: SQLite Schema + Connection

**Files:**
- Create: `src/gmail_search/store/db.py`
- Create: `tests/test_store_db.py`

- [ ] **Step 1: Write failing test for DB init**

Create `tests/test_store_db.py`:

```python
import sqlite3

from gmail_search.store.db import init_db, get_connection


def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    assert "messages" in tables
    assert "attachments" in tables
    assert "embeddings" in tables
    assert "costs" in tables
    assert "sync_state" in tables


def test_init_db_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    init_db(db_path)  # Should not raise
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    assert "messages" in tables


def test_get_connection(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    assert conn is not None
    # WAL mode enabled
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_store_db.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement db.py**

Create `src/gmail_search/store/db.py`:

```python
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    from_addr TEXT NOT NULL,
    to_addr TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    body_text TEXT NOT NULL DEFAULT '',
    body_html TEXT NOT NULL DEFAULT '',
    date TEXT NOT NULL,
    labels TEXT NOT NULL DEFAULT '[]',
    history_id INTEGER NOT NULL DEFAULT 0,
    raw_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES messages(id),
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    extracted_text TEXT,
    image_path TEXT,
    raw_path TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES messages(id),
    attachment_id INTEGER REFERENCES attachments(id),
    chunk_type TEXT NOT NULL,
    chunk_text TEXT,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    image_count INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
    message_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_attachments_message_id ON attachments(message_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_message_id ON embeddings(message_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_lookup ON embeddings(message_id, attachment_id, chunk_type, model);
"""


def init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store_db.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/gmail_search/store/db.py tests/test_store_db.py
git commit -m "feat: SQLite schema and connection management"
```

---

### Task 4: Store Queries (CRUD)

**Files:**
- Create: `src/gmail_search/store/queries.py`
- Create: `tests/test_store_queries.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_store_queries.py`:

```python
import json
from datetime import datetime

from gmail_search.store.db import init_db, get_connection
from gmail_search.store.models import Message, Attachment, EmbeddingRecord
from gmail_search.store.queries import (
    upsert_message,
    get_message,
    get_messages_without_embeddings,
    upsert_attachment,
    get_attachments_for_message,
    insert_embedding,
    embedding_exists,
    load_all_embeddings,
    set_sync_state,
    get_sync_state,
)


def _make_message(id="msg1"):
    return Message(
        id=id,
        thread_id="thread1",
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        subject="Test subject",
        body_text="Hello world",
        body_html="<p>Hello world</p>",
        date=datetime(2025, 6, 15, 10, 30),
        labels=["INBOX", "IMPORTANT"],
        history_id=12345,
        raw_json='{"id": "msg1"}',
    )


def test_upsert_and_get_message(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    msg = _make_message()
    upsert_message(conn, msg)
    result = get_message(conn, "msg1")
    assert result is not None
    assert result.subject == "Test subject"
    assert result.from_addr == "alice@example.com"
    conn.close()


def test_upsert_message_updates_existing(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    msg = _make_message()
    upsert_message(conn, msg)
    msg.subject = "Updated subject"
    upsert_message(conn, msg)
    result = get_message(conn, "msg1")
    assert result.subject == "Updated subject"
    conn.close()


def test_get_messages_without_embeddings(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_message("msg1"))
    upsert_message(conn, _make_message("msg2"))
    # Insert embedding for msg1
    insert_embedding(conn, EmbeddingRecord(
        id=None, message_id="msg1", attachment_id=None,
        chunk_type="message", chunk_text="test",
        embedding=b"\x00" * 12, model="test-model",
    ))
    result = get_messages_without_embeddings(conn, model="test-model")
    assert len(result) == 1
    assert result[0].id == "msg2"
    conn.close()


def test_attachment_crud(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_message())
    att = Attachment(
        id=None, message_id="msg1", filename="doc.pdf",
        mime_type="application/pdf", size_bytes=1024,
    )
    att_id = upsert_attachment(conn, att)
    assert att_id is not None
    atts = get_attachments_for_message(conn, "msg1")
    assert len(atts) == 1
    assert atts[0].filename == "doc.pdf"
    conn.close()


def test_embedding_exists(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_message())
    assert not embedding_exists(conn, "msg1", None, "message", "test-model")
    insert_embedding(conn, EmbeddingRecord(
        id=None, message_id="msg1", attachment_id=None,
        chunk_type="message", chunk_text="test",
        embedding=b"\x00" * 12, model="test-model",
    ))
    assert embedding_exists(conn, "msg1", None, "message", "test-model")
    conn.close()


def test_load_all_embeddings(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_message())
    insert_embedding(conn, EmbeddingRecord(
        id=None, message_id="msg1", attachment_id=None,
        chunk_type="message", chunk_text="test",
        embedding=b"\x00" * 12, model="test-model",
    ))
    ids, blobs = load_all_embeddings(conn, model="test-model")
    assert len(ids) == 1
    assert len(blobs) == 1
    conn.close()


def test_sync_state(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    assert get_sync_state(conn, "last_history_id") is None
    set_sync_state(conn, "last_history_id", "99999")
    assert get_sync_state(conn, "last_history_id") == "99999"
    set_sync_state(conn, "last_history_id", "100000")
    assert get_sync_state(conn, "last_history_id") == "100000"
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_store_queries.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement queries.py**

Create `src/gmail_search/store/queries.py`:

```python
import json
import sqlite3
from datetime import datetime

from gmail_search.store.models import Message, Attachment, EmbeddingRecord


def upsert_message(conn: sqlite3.Connection, msg: Message) -> None:
    conn.execute(
        """INSERT INTO messages (id, thread_id, from_addr, to_addr, subject,
           body_text, body_html, date, labels, history_id, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             thread_id=excluded.thread_id, from_addr=excluded.from_addr,
             to_addr=excluded.to_addr, subject=excluded.subject,
             body_text=excluded.body_text, body_html=excluded.body_html,
             date=excluded.date, labels=excluded.labels,
             history_id=excluded.history_id, raw_json=excluded.raw_json""",
        (msg.id, msg.thread_id, msg.from_addr, msg.to_addr, msg.subject,
         msg.body_text, msg.body_html, msg.date.isoformat(),
         json.dumps(msg.labels), msg.history_id, msg.raw_json),
    )
    conn.commit()


def get_message(conn: sqlite3.Connection, message_id: str) -> Message | None:
    row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    if row is None:
        return None
    return Message(
        id=row["id"], thread_id=row["thread_id"],
        from_addr=row["from_addr"], to_addr=row["to_addr"],
        subject=row["subject"], body_text=row["body_text"],
        body_html=row["body_html"],
        date=datetime.fromisoformat(row["date"]),
        labels=json.loads(row["labels"]),
        history_id=row["history_id"], raw_json=row["raw_json"],
    )


def get_messages_without_embeddings(
    conn: sqlite3.Connection, model: str
) -> list[Message]:
    rows = conn.execute(
        """SELECT m.* FROM messages m
           WHERE m.id NOT IN (
             SELECT DISTINCT message_id FROM embeddings
             WHERE chunk_type = 'message' AND model = ?
           )""",
        (model,),
    ).fetchall()
    return [
        Message(
            id=r["id"], thread_id=r["thread_id"],
            from_addr=r["from_addr"], to_addr=r["to_addr"],
            subject=r["subject"], body_text=r["body_text"],
            body_html=r["body_html"],
            date=datetime.fromisoformat(r["date"]),
            labels=json.loads(r["labels"]),
            history_id=r["history_id"], raw_json=r["raw_json"],
        )
        for r in rows
    ]


def upsert_attachment(conn: sqlite3.Connection, att: Attachment) -> int:
    cursor = conn.execute(
        """INSERT INTO attachments (message_id, filename, mime_type, size_bytes,
           extracted_text, image_path, raw_path)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (att.message_id, att.filename, att.mime_type, att.size_bytes,
         att.extracted_text, att.image_path, att.raw_path),
    )
    conn.commit()
    return cursor.lastrowid


def get_attachments_for_message(
    conn: sqlite3.Connection, message_id: str
) -> list[Attachment]:
    rows = conn.execute(
        "SELECT * FROM attachments WHERE message_id = ?", (message_id,)
    ).fetchall()
    return [
        Attachment(
            id=r["id"], message_id=r["message_id"],
            filename=r["filename"], mime_type=r["mime_type"],
            size_bytes=r["size_bytes"], extracted_text=r["extracted_text"],
            image_path=r["image_path"], raw_path=r["raw_path"],
        )
        for r in rows
    ]


def insert_embedding(conn: sqlite3.Connection, rec: EmbeddingRecord) -> int:
    cursor = conn.execute(
        """INSERT INTO embeddings (message_id, attachment_id, chunk_type,
           chunk_text, embedding, model)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (rec.message_id, rec.attachment_id, rec.chunk_type,
         rec.chunk_text, rec.embedding, rec.model),
    )
    conn.commit()
    return cursor.lastrowid


def embedding_exists(
    conn: sqlite3.Connection,
    message_id: str,
    attachment_id: int | None,
    chunk_type: str,
    model: str,
) -> bool:
    if attachment_id is None:
        row = conn.execute(
            """SELECT 1 FROM embeddings
               WHERE message_id = ? AND attachment_id IS NULL
               AND chunk_type = ? AND model = ?""",
            (message_id, chunk_type, model),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT 1 FROM embeddings
               WHERE message_id = ? AND attachment_id = ?
               AND chunk_type = ? AND model = ?""",
            (message_id, attachment_id, chunk_type, model),
        ).fetchone()
    return row is not None


def load_all_embeddings(
    conn: sqlite3.Connection, model: str
) -> tuple[list[int], list[bytes]]:
    rows = conn.execute(
        "SELECT id, embedding FROM embeddings WHERE model = ?", (model,)
    ).fetchall()
    ids = [r["id"] for r in rows]
    blobs = [r["embedding"] for r in rows]
    return ids, blobs


def set_sync_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO sync_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def get_sync_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store_queries.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/gmail_search/store/queries.py tests/test_store_queries.py
git commit -m "feat: CRUD queries for messages, attachments, embeddings, sync state"
```

---

### Task 5: Cost Tracking + Budget

**Files:**
- Create: `src/gmail_search/store/cost.py`
- Create: `tests/test_store_cost.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_store_cost.py`:

```python
from datetime import datetime

from gmail_search.store.db import init_db, get_connection
from gmail_search.store.cost import (
    record_cost,
    get_total_spend,
    get_spend_breakdown,
    check_budget,
    estimate_cost,
)


def test_record_and_get_total_spend(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    record_cost(conn, operation="embed_text", model="test", input_tokens=1000,
                image_count=0, estimated_cost_usd=0.50, message_id="msg1")
    record_cost(conn, operation="embed_text", model="test", input_tokens=2000,
                image_count=0, estimated_cost_usd=0.75, message_id="msg2")
    total = get_total_spend(conn)
    assert abs(total - 1.25) < 0.001
    conn.close()


def test_get_spend_breakdown(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    record_cost(conn, operation="embed_text", model="test", input_tokens=1000,
                image_count=0, estimated_cost_usd=0.50, message_id="msg1")
    record_cost(conn, operation="embed_image", model="test", input_tokens=0,
                image_count=5, estimated_cost_usd=0.10, message_id="msg1")
    breakdown = get_spend_breakdown(conn)
    assert breakdown["embed_text"] == 0.50
    assert breakdown["embed_image"] == 0.10
    conn.close()


def test_check_budget_under(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    record_cost(conn, operation="embed_text", model="test", input_tokens=1000,
                image_count=0, estimated_cost_usd=1.00, message_id="msg1")
    ok, spent, remaining = check_budget(conn, max_budget_usd=5.00)
    assert ok is True
    assert abs(spent - 1.00) < 0.001
    assert abs(remaining - 4.00) < 0.001
    conn.close()


def test_check_budget_over(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    record_cost(conn, operation="embed_text", model="test", input_tokens=1000,
                image_count=0, estimated_cost_usd=5.50, message_id="msg1")
    ok, spent, remaining = check_budget(conn, max_budget_usd=5.00)
    assert ok is False
    assert remaining < 0
    conn.close()


def test_estimate_cost_text():
    cost = estimate_cost(input_tokens=1_000_000, image_count=0)
    assert abs(cost - 0.20) < 0.001


def test_estimate_cost_images():
    cost = estimate_cost(input_tokens=0, image_count=1000)
    assert abs(cost - 0.10) < 0.001


def test_estimate_cost_mixed():
    cost = estimate_cost(input_tokens=500_000, image_count=500)
    expected = 0.10 + 0.05  # $0.10 text + $0.05 images
    assert abs(cost - expected) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_store_cost.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement cost.py**

Create `src/gmail_search/store/cost.py`:

```python
import sqlite3
from datetime import datetime, timezone

# Gemini embedding-2-preview pricing
TEXT_COST_PER_MILLION_TOKENS = 0.20
IMAGE_COST_PER_IMAGE = 0.0001


def estimate_cost(input_tokens: int = 0, image_count: int = 0) -> float:
    text_cost = (input_tokens / 1_000_000) * TEXT_COST_PER_MILLION_TOKENS
    image_cost = image_count * IMAGE_COST_PER_IMAGE
    return text_cost + image_cost


def record_cost(
    conn: sqlite3.Connection,
    operation: str,
    model: str,
    input_tokens: int,
    image_count: int,
    estimated_cost_usd: float,
    message_id: str,
) -> None:
    conn.execute(
        """INSERT INTO costs (timestamp, operation, model, input_tokens,
           image_count, estimated_cost_usd, message_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(timezone.utc).isoformat(), operation, model,
         input_tokens, image_count, estimated_cost_usd, message_id),
    )
    conn.commit()


def get_total_spend(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM costs").fetchone()
    return row[0]


def get_spend_breakdown(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute(
        "SELECT operation, SUM(estimated_cost_usd) as total FROM costs GROUP BY operation"
    ).fetchall()
    return {r["operation"]: r["total"] for r in rows}


def check_budget(
    conn: sqlite3.Connection, max_budget_usd: float
) -> tuple[bool, float, float]:
    spent = get_total_spend(conn)
    remaining = max_budget_usd - spent
    return remaining >= 0, spent, remaining
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store_cost.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/gmail_search/store/cost.py tests/test_store_cost.py
git commit -m "feat: cost tracking with budget enforcement"
```

---

### Task 6: Gmail Auth

**Files:**
- Create: `src/gmail_search/gmail/__init__.py`
- Create: `src/gmail_search/gmail/auth.py`

- [ ] **Step 1: Create gmail/__init__.py**

```python
```

- [ ] **Step 2: Implement auth.py**

This module wraps Google's OAuth flow. It's hard to unit test without mocking the entire Google auth library, so we'll write it directly and test via the CLI integration later.

Create `src/gmail_search/gmail/auth.py`:

```python
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_credentials(data_dir: Path) -> Credentials:
    creds_path = data_dir / "credentials.json"
    token_path = data_dir / "token.json"

    if not creds_path.exists():
        raise FileNotFoundError(
            f"Missing {creds_path}. Download OAuth client credentials from "
            "Google Cloud Console and save as data/credentials.json"
        )

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    return creds


def build_gmail_service(data_dir: Path):
    creds = get_credentials(data_dir)
    return build("gmail", "v1", credentials=creds)
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from gmail_search.gmail.auth import get_credentials, build_gmail_service; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/gmail_search/gmail/
git commit -m "feat: Gmail OAuth2 authentication"
```

---

### Task 7: Gmail Parser

**Files:**
- Create: `src/gmail_search/gmail/parser.py`
- Create: `tests/test_gmail_parser.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_gmail_parser.py`:

```python
import json

from gmail_search.gmail.parser import parse_message


SAMPLE_API_RESPONSE = {
    "id": "18f1234abcd",
    "threadId": "18f1234abcd",
    "historyId": "999999",
    "labelIds": ["INBOX", "UNREAD"],
    "payload": {
        "headers": [
            {"name": "From", "value": "Alice <alice@example.com>"},
            {"name": "To", "value": "Bob <bob@example.com>"},
            {"name": "Subject", "value": "Meeting tomorrow"},
            {"name": "Date", "value": "Mon, 15 Jun 2025 10:30:00 -0700"},
        ],
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "text/plain",
                "body": {"data": "SGVsbG8gV29ybGQ="},  # base64 "Hello World"
            },
            {
                "mimeType": "text/html",
                "body": {"data": "PHA-SGVsbG8gV29ybGQ8L3A-"},  # base64 "<p>Hello World</p>"
            },
            {
                "mimeType": "application/pdf",
                "filename": "report.pdf",
                "body": {"attachmentId": "ANGjdJ8abc", "size": 50000},
            },
        ],
    },
}


def test_parse_message_basic():
    msg, att_metas = parse_message(SAMPLE_API_RESPONSE)
    assert msg.id == "18f1234abcd"
    assert msg.thread_id == "18f1234abcd"
    assert msg.from_addr == "Alice <alice@example.com>"
    assert msg.to_addr == "Bob <bob@example.com>"
    assert msg.subject == "Meeting tomorrow"
    assert "Hello World" in msg.body_text
    assert msg.labels == ["INBOX", "UNREAD"]
    assert msg.history_id == 999999


def test_parse_message_attachments():
    msg, att_metas = parse_message(SAMPLE_API_RESPONSE)
    assert len(att_metas) == 1
    assert att_metas[0]["filename"] == "report.pdf"
    assert att_metas[0]["mime_type"] == "application/pdf"
    assert att_metas[0]["attachment_id"] == "ANGjdJ8abc"
    assert att_metas[0]["size"] == 50000


SIMPLE_API_RESPONSE = {
    "id": "simple1",
    "threadId": "simple1",
    "historyId": "100",
    "labelIds": [],
    "payload": {
        "headers": [
            {"name": "From", "value": "test@test.com"},
            {"name": "To", "value": "me@test.com"},
            {"name": "Subject", "value": "Simple"},
            {"name": "Date", "value": "Tue, 1 Jan 2025 00:00:00 +0000"},
        ],
        "mimeType": "text/plain",
        "body": {"data": "SnVzdCB0ZXh0"},  # base64 "Just text"
    },
}


def test_parse_simple_message():
    msg, att_metas = parse_message(SIMPLE_API_RESPONSE)
    assert msg.id == "simple1"
    assert "Just text" in msg.body_text
    assert len(att_metas) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gmail_parser.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement parser.py**

Create `src/gmail_search/gmail/parser.py`:

```python
import base64
import json
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

from gmail_search.store.models import Message


def _get_header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _decode_body(data: str) -> str:
    # Gmail uses URL-safe base64
    padded = data + "=" * (4 - len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _extract_parts(
    payload: dict,
) -> tuple[str, str, list[dict]]:
    """Recursively extract text, html, and attachment metadata from payload."""
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[dict] = []

    mime_type = payload.get("mimeType", "")

    # Leaf node with body data
    if "parts" not in payload:
        body = payload.get("body", {})
        if body.get("attachmentId"):
            attachments.append({
                "filename": payload.get("filename", ""),
                "mime_type": mime_type,
                "attachment_id": body["attachmentId"],
                "size": body.get("size", 0),
            })
        elif "data" in body:
            decoded = _decode_body(body["data"])
            if mime_type == "text/plain":
                text_parts.append(decoded)
            elif mime_type == "text/html":
                html_parts.append(decoded)
        return "\n".join(text_parts), "\n".join(html_parts), attachments

    # Recurse into parts
    for part in payload.get("parts", []):
        t, h, a = _extract_parts(part)
        if t:
            text_parts.append(t)
        if h:
            html_parts.append(h)
        attachments.extend(a)

    return "\n".join(text_parts), "\n".join(html_parts), attachments


def parse_message(raw: dict[str, Any]) -> tuple[Message, list[dict]]:
    payload = raw["payload"]
    headers = payload.get("headers", [])

    date_str = _get_header(headers, "Date")
    try:
        date = parsedate_to_datetime(date_str)
    except (ValueError, TypeError):
        date = datetime(1970, 1, 1)

    body_text, body_html, att_metas = _extract_parts(payload)

    msg = Message(
        id=raw["id"],
        thread_id=raw.get("threadId", raw["id"]),
        from_addr=_get_header(headers, "From"),
        to_addr=_get_header(headers, "To"),
        subject=_get_header(headers, "Subject"),
        body_text=body_text,
        body_html=body_html,
        date=date,
        labels=raw.get("labelIds", []),
        history_id=int(raw.get("historyId", 0)),
        raw_json=json.dumps(raw),
    )

    return msg, att_metas
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gmail_parser.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/gmail_search/gmail/parser.py tests/test_gmail_parser.py
git commit -m "feat: Gmail API response parser"
```

---

### Task 8: Gmail Download Client

**Files:**
- Create: `src/gmail_search/gmail/client.py`

- [ ] **Step 1: Implement client.py**

This module calls the Gmail API directly. Testing it requires real credentials or heavy mocking. We test it end-to-end via the CLI. The parser (already tested) handles all the data transformation.

Create `src/gmail_search/gmail/client.py`:

```python
import logging
import time
from pathlib import Path
from typing import Any

from googleapiclient.discovery import Resource
from googleapiclient.http import BatchHttpRequest
from tqdm import tqdm

from gmail_search.gmail.parser import parse_message
from gmail_search.store.db import get_connection
from gmail_search.store.models import Attachment
from gmail_search.store.queries import (
    upsert_message,
    upsert_attachment,
    get_sync_state,
    set_sync_state,
)

logger = logging.getLogger(__name__)


def _download_attachment_data(service: Resource, message_id: str, attachment_id: str) -> bytes:
    import base64
    result = service.users().messages().attachments().get(
        userId="me", messageId=message_id, id=attachment_id
    ).execute()
    data = result.get("data", "")
    padded = data + "=" * (4 - len(data) % 4)
    return base64.urlsafe_b64decode(padded)


def download_messages(
    service: Resource,
    db_path: Path,
    data_dir: Path,
    batch_size: int = 100,
    max_messages: int | None = None,
    max_attachment_size: int = 10 * 1024 * 1024,
) -> int:
    conn = get_connection(db_path)
    attachments_dir = data_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    # Fetch all message IDs
    message_ids: list[str] = []
    page_token = None
    logger.info("Fetching message IDs...")

    while True:
        kwargs: dict[str, Any] = {"userId": "me", "maxResults": 500}
        if page_token:
            kwargs["pageToken"] = page_token
        result = service.users().messages().list(**kwargs).execute()
        messages = result.get("messages", [])
        message_ids.extend(m["id"] for m in messages)

        if max_messages and len(message_ids) >= max_messages:
            message_ids = message_ids[:max_messages]
            break

        page_token = result.get("nextPageToken")
        if not page_token:
            break

    logger.info(f"Found {len(message_ids)} messages to download")

    # Skip already-downloaded messages
    existing = set()
    for row in conn.execute("SELECT id FROM messages").fetchall():
        existing.add(row["id"])
    to_download = [mid for mid in message_ids if mid not in existing]
    logger.info(f"{len(existing)} already downloaded, {len(to_download)} remaining")

    if not to_download:
        conn.close()
        return 0

    max_history_id = 0
    downloaded = 0
    progress = tqdm(total=len(to_download), desc="Downloading messages")

    for i in range(0, len(to_download), batch_size):
        batch_ids = to_download[i : i + batch_size]
        batch_results: list[dict] = []

        def _callback(request_id, response, exception):
            if exception:
                logger.warning(f"Error fetching {request_id}: {exception}")
            else:
                batch_results.append(response)

        batch = service.new_batch_http_request(callback=_callback)
        for mid in batch_ids:
            batch.add(
                service.users().messages().get(userId="me", id=mid, format="full"),
                request_id=mid,
            )

        retries = 0
        while retries < 5:
            try:
                batch.execute()
                break
            except Exception as e:
                if "429" in str(e) or "rateLimitExceeded" in str(e):
                    wait = 2 ** retries
                    logger.warning(f"Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    retries += 1
                else:
                    raise

        for raw in batch_results:
            msg, att_metas = parse_message(raw)
            upsert_message(conn, msg)

            if msg.history_id > max_history_id:
                max_history_id = msg.history_id

            # Download attachments
            for att_meta in att_metas:
                if att_meta["size"] > max_attachment_size:
                    logger.warning(
                        f"Skipping attachment {att_meta['filename']} "
                        f"({att_meta['size']} bytes) — exceeds size limit"
                    )
                    continue

                msg_att_dir = attachments_dir / msg.id
                msg_att_dir.mkdir(exist_ok=True)
                raw_path = msg_att_dir / att_meta["filename"]

                try:
                    data = _download_attachment_data(
                        service, msg.id, att_meta["attachment_id"]
                    )
                    raw_path.write_bytes(data)
                except Exception as e:
                    logger.warning(f"Failed to download attachment: {e}")
                    continue

                att = Attachment(
                    id=None,
                    message_id=msg.id,
                    filename=att_meta["filename"],
                    mime_type=att_meta["mime_type"],
                    size_bytes=len(data),
                    raw_path=str(raw_path),
                )
                upsert_attachment(conn, att)

            downloaded += 1

        progress.update(len(batch_results))

    progress.close()

    if max_history_id > 0:
        set_sync_state(conn, "last_history_id", str(max_history_id))

    conn.close()
    return downloaded


def sync_new_messages(
    service: Resource,
    db_path: Path,
    data_dir: Path,
    max_attachment_size: int = 10 * 1024 * 1024,
) -> int:
    conn = get_connection(db_path)
    last_history_id = get_sync_state(conn, "last_history_id")
    conn.close()

    if not last_history_id:
        logger.warning("No last_history_id found. Run full download first.")
        return 0

    new_message_ids: list[str] = []
    page_token = None

    try:
        while True:
            kwargs: dict[str, Any] = {
                "userId": "me",
                "startHistoryId": last_history_id,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            result = service.users().history().list(**kwargs).execute()

            for record in result.get("history", []):
                for added in record.get("messagesAdded", []):
                    new_message_ids.append(added["message"]["id"])

            page_token = result.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        if "404" in str(e):
            logger.warning("History expired. Run full download to re-sync.")
            return 0
        raise

    if not new_message_ids:
        logger.info("No new messages since last sync")
        return 0

    # Deduplicate
    new_message_ids = list(set(new_message_ids))
    logger.info(f"Found {len(new_message_ids)} new messages")

    # Fetch and store each one
    conn = get_connection(db_path)
    attachments_dir = data_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    max_history_id = int(last_history_id)
    count = 0

    for mid in tqdm(new_message_ids, desc="Syncing new messages"):
        try:
            raw = service.users().messages().get(
                userId="me", id=mid, format="full"
            ).execute()
        except Exception as e:
            logger.warning(f"Failed to fetch {mid}: {e}")
            continue

        msg, att_metas = parse_message(raw)
        upsert_message(conn, msg)

        if msg.history_id > max_history_id:
            max_history_id = msg.history_id

        for att_meta in att_metas:
            if att_meta["size"] > max_attachment_size:
                continue
            msg_att_dir = attachments_dir / msg.id
            msg_att_dir.mkdir(exist_ok=True)
            raw_path = msg_att_dir / att_meta["filename"]
            try:
                data = _download_attachment_data(service, msg.id, att_meta["attachment_id"])
                raw_path.write_bytes(data)
            except Exception as e:
                logger.warning(f"Failed to download attachment: {e}")
                continue
            att = Attachment(
                id=None, message_id=msg.id, filename=att_meta["filename"],
                mime_type=att_meta["mime_type"], size_bytes=len(data),
                raw_path=str(raw_path),
            )
            upsert_attachment(conn, att)

        count += 1

    set_sync_state(conn, "last_history_id", str(max_history_id))
    conn.close()
    return count
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from gmail_search.gmail.client import download_messages, sync_new_messages; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/gmail_search/gmail/client.py
git commit -m "feat: Gmail batch download and incremental sync"
```

---

### Task 9: Attachment Extraction

**Files:**
- Create: `src/gmail_search/extract/__init__.py`
- Create: `src/gmail_search/extract/pdf.py`
- Create: `src/gmail_search/extract/image.py`
- Create: `tests/test_extract_dispatch.py`
- Create: `tests/test_extract_pdf.py`

- [ ] **Step 1: Write failing tests for dispatcher**

Create `tests/test_extract_dispatch.py`:

```python
from pathlib import Path

from gmail_search.extract import dispatch, ExtractResult


def test_dispatch_unknown_mime_returns_none():
    result = dispatch("application/zip", Path("/fake/path.zip"), {})
    assert result is None


def test_dispatch_image_passthrough(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
    result = dispatch("image/jpeg", img, {})
    assert result is not None
    assert result.text is None
    assert len(result.images) == 1
    assert result.images[0] == img


def test_dispatch_png_passthrough(tmp_path):
    img = tmp_path / "diagram.png"
    img.write_bytes(b"\x89PNG")
    result = dispatch("image/png", img, {})
    assert result is not None
    assert len(result.images) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extract_dispatch.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement extract/__init__.py (dispatcher) and image.py**

Create `src/gmail_search/extract/__init__.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractResult:
    text: str | None = None
    images: list[Path] = field(default_factory=list)


def dispatch(mime_type: str, file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    from gmail_search.extract.image import extract_image
    from gmail_search.extract.pdf import extract_pdf

    extractors = {
        "application/pdf": extract_pdf,
        "image/jpeg": extract_image,
        "image/jpg": extract_image,
        "image/png": extract_image,
        "image/gif": extract_image,
    }

    extractor = extractors.get(mime_type)
    if extractor is None:
        return None

    return extractor(file_path, config)
```

Create `src/gmail_search/extract/image.py`:

```python
from pathlib import Path
from typing import Any

from gmail_search.extract import ExtractResult


def extract_image(file_path: Path, config: dict[str, Any]) -> ExtractResult:
    return ExtractResult(text=None, images=[file_path])
```

- [ ] **Step 4: Run dispatcher tests to verify they pass**

Run: `python -m pytest tests/test_extract_dispatch.py -v`
Expected: 3 passed

- [ ] **Step 5: Write failing test for PDF extraction**

Create `tests/test_extract_pdf.py`:

```python
from pathlib import Path

import fitz  # pymupdf

from gmail_search.extract.pdf import extract_pdf


def _create_test_pdf(path: Path, pages: int = 3) -> None:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} content here")
    doc.save(str(path))
    doc.close()


def test_extract_pdf_text(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _create_test_pdf(pdf_path, pages=2)
    result = extract_pdf(pdf_path, {"max_pdf_pages": 20})
    assert result.text is not None
    assert "Page 1" in result.text
    assert "Page 2" in result.text


def test_extract_pdf_images(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _create_test_pdf(pdf_path, pages=3)
    result = extract_pdf(pdf_path, {"max_pdf_pages": 20})
    assert len(result.images) == 3
    for img_path in result.images:
        assert img_path.exists()
        assert img_path.suffix == ".png"


def test_extract_pdf_respects_page_limit(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _create_test_pdf(pdf_path, pages=10)
    result = extract_pdf(pdf_path, {"max_pdf_pages": 3})
    assert len(result.images) == 3
    # Text still extracted from all pages
    assert "Page 10" in result.text
```

- [ ] **Step 6: Run PDF tests to verify they fail**

Run: `python -m pytest tests/test_extract_pdf.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 7: Implement pdf.py**

Create `src/gmail_search/extract/pdf.py`:

```python
import logging
from pathlib import Path
from typing import Any

import fitz  # pymupdf

from gmail_search.extract import ExtractResult

logger = logging.getLogger(__name__)


def extract_pdf(file_path: Path, config: dict[str, Any]) -> ExtractResult:
    max_pages = config.get("max_pdf_pages", 20)

    doc = fitz.open(str(file_path))

    # Extract text from all pages
    text_parts: list[str] = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            text_parts.append(text.strip())

    full_text = "\n\n".join(text_parts) if text_parts else None

    # Render pages as images (up to limit)
    images: list[Path] = []
    images_dir = file_path.parent / f"{file_path.stem}_pages"
    images_dir.mkdir(exist_ok=True)

    for i, page in enumerate(doc):
        if i >= max_pages:
            logger.info(f"Reached page limit ({max_pages}), skipping remaining pages for image rendering")
            break
        mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
        pix = page.get_pixmap(matrix=mat)
        img_path = images_dir / f"page_{i + 1:04d}.png"
        pix.save(str(img_path))
        images.append(img_path)

    doc.close()
    return ExtractResult(text=full_text, images=images)
```

- [ ] **Step 8: Run PDF tests to verify they pass**

Run: `python -m pytest tests/test_extract_pdf.py -v`
Expected: 3 passed

- [ ] **Step 9: Commit**

```bash
git add src/gmail_search/extract/ tests/test_extract_dispatch.py tests/test_extract_pdf.py
git commit -m "feat: attachment extraction with PDF and image support"
```

---

### Task 10: Gemini Embedding Client

**Files:**
- Create: `src/gmail_search/embed/__init__.py`
- Create: `src/gmail_search/embed/client.py`
- Create: `tests/test_embed_client.py`

- [ ] **Step 1: Create embed/__init__.py**

```python
```

- [ ] **Step 2: Write test for token estimation and text formatting**

We can test the helper functions without calling the Gemini API.

Create `tests/test_embed_client.py`:

```python
from gmail_search.embed.client import estimate_tokens, truncate_to_token_limit, format_message_text


def test_estimate_tokens():
    # Rough estimate: ~4 chars per token
    text = "Hello world this is a test"
    tokens = estimate_tokens(text)
    assert 4 <= tokens <= 10


def test_truncate_to_token_limit():
    short = "Hello world"
    assert truncate_to_token_limit(short, 8192) == short

    long_text = "word " * 10000  # ~10000 tokens
    truncated = truncate_to_token_limit(long_text, 100)
    tokens = estimate_tokens(truncated)
    assert tokens <= 110  # some slack is ok


def test_format_message_text():
    text = format_message_text(
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        date="2025-06-15",
        subject="Hello",
        body="Body text here",
    )
    assert "From: alice@example.com" in text
    assert "To: bob@example.com" in text
    assert "Subject: Hello" in text
    assert "Body text here" in text
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_embed_client.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 4: Implement client.py**

Create `src/gmail_search/embed/client.py`:

```python
import logging
import struct
import time
from pathlib import Path
from typing import Any

from google import genai

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    # Truncate by characters (4 chars ≈ 1 token)
    max_chars = max_tokens * 4
    return text[:max_chars]


def format_message_text(
    from_addr: str, to_addr: str, date: str, subject: str, body: str
) -> str:
    return f"From: {from_addr} | To: {to_addr} | Date: {date} | Subject: {subject} | {body}"


def format_attachment_text(filename: str, subject: str, extracted_text: str) -> str:
    return f"Attachment: {filename} | From email: {subject} | {extracted_text}"


def embedding_to_blob(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def blob_to_embedding(blob: bytes, dimensions: int) -> list[float]:
    return list(struct.unpack(f"{dimensions}f", blob))


class GeminiEmbedder:
    def __init__(self, config: dict[str, Any]):
        self.model = config["embedding"]["model"]
        self.dimensions = config["embedding"]["dimensions"]
        self.task_type_document = config["embedding"]["task_type_document"]
        self.task_type_query = config["embedding"]["task_type_query"]
        self.client = genai.Client()

    def embed_text(self, text: str, task_type: str | None = None) -> list[float]:
        if task_type is None:
            task_type = self.task_type_document
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config={
                "task_type": task_type,
                "output_dimensionality": self.dimensions,
            },
        )
        return result.embeddings[0].values

    def embed_texts_batch(
        self, texts: list[str], task_type: str | None = None
    ) -> list[list[float]]:
        if task_type is None:
            task_type = self.task_type_document
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config={
                "task_type": task_type,
                "output_dimensionality": self.dimensions,
            },
        )
        return [e.values for e in result.embeddings]

    def embed_image(self, image_path: Path, task_type: str | None = None) -> list[float]:
        if task_type is None:
            task_type = self.task_type_document
        image_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
        mime_type = mime_map.get(suffix, "image/png")
        result = self.client.models.embed_content(
            model=self.model,
            contents=genai.types.Content(
                parts=[genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes))]
            ),
            config={
                "task_type": task_type,
                "output_dimensionality": self.dimensions,
            },
        )
        return result.embeddings[0].values

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query, task_type=self.task_type_query)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_embed_client.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/gmail_search/embed/ tests/test_embed_client.py
git commit -m "feat: Gemini embedding client with text and multimodal support"
```

---

### Task 11: Embedding Pipeline

**Files:**
- Create: `src/gmail_search/embed/pipeline.py`
- Create: `tests/test_embed_pipeline.py`

- [ ] **Step 1: Write failing test for pipeline logic**

We'll test the pipeline orchestration by mocking the embedder. The pipeline's job is: find unembedded messages/attachments → call embedder → store results → track cost.

Create `tests/test_embed_pipeline.py`:

```python
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

from gmail_search.store.db import init_db, get_connection
from gmail_search.store.models import Message, Attachment
from gmail_search.store.queries import upsert_message, upsert_attachment, embedding_exists
from gmail_search.store.cost import get_total_spend
from gmail_search.embed.pipeline import run_embedding_pipeline
from gmail_search.config import load_config


def _fake_vector(dims=3072):
    return [0.1] * dims


def _make_msg(id="msg1"):
    from datetime import datetime
    return Message(
        id=id, thread_id="t1", from_addr="a@b.com", to_addr="c@d.com",
        subject="Test", body_text="Hello world", body_html="",
        date=datetime(2025, 1, 1), labels=[], history_id=1, raw_json="{}",
    )


def test_pipeline_embeds_messages(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_msg("msg1"))
    upsert_message(conn, _make_msg("msg2"))
    conn.close()

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"

    mock_embedder = MagicMock()
    mock_embedder.model = "test-model"
    mock_embedder.dimensions = 3072
    mock_embedder.embed_texts_batch.return_value = [_fake_vector(), _fake_vector()]

    count = run_embedding_pipeline(db_path, cfg, embedder=mock_embedder)
    assert count == 2

    conn = get_connection(db_path)
    assert embedding_exists(conn, "msg1", None, "message", "test-model")
    assert embedding_exists(conn, "msg2", None, "message", "test-model")
    conn.close()


def test_pipeline_skips_already_embedded(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_msg("msg1"))
    conn.close()

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"

    mock_embedder = MagicMock()
    mock_embedder.model = "test-model"
    mock_embedder.dimensions = 3072
    mock_embedder.embed_texts_batch.return_value = [_fake_vector()]

    # Run twice
    run_embedding_pipeline(db_path, cfg, embedder=mock_embedder)
    mock_embedder.reset_mock()
    count = run_embedding_pipeline(db_path, cfg, embedder=mock_embedder)
    assert count == 0
    mock_embedder.embed_texts_batch.assert_not_called()


def test_pipeline_tracks_cost(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_msg("msg1"))
    conn.close()

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"

    mock_embedder = MagicMock()
    mock_embedder.model = "test-model"
    mock_embedder.dimensions = 3072
    mock_embedder.embed_texts_batch.return_value = [_fake_vector()]

    run_embedding_pipeline(db_path, cfg, embedder=mock_embedder)

    conn = get_connection(db_path)
    total = get_total_spend(conn)
    assert total > 0
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_embed_pipeline.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement pipeline.py**

Create `src/gmail_search/embed/pipeline.py`:

```python
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from gmail_search.embed.client import (
    GeminiEmbedder,
    embedding_to_blob,
    estimate_tokens,
    format_attachment_text,
    format_message_text,
    truncate_to_token_limit,
)
from gmail_search.store.cost import check_budget, estimate_cost, record_cost
from gmail_search.store.db import get_connection
from gmail_search.store.models import EmbeddingRecord
from gmail_search.store.queries import (
    embedding_exists,
    get_attachments_for_message,
    get_messages_without_embeddings,
    insert_embedding,
)

logger = logging.getLogger(__name__)

TEXT_BATCH_SIZE = 100
MAX_INPUT_TOKENS = 8192


def run_embedding_pipeline(
    db_path: Path,
    config: dict[str, Any],
    embedder: GeminiEmbedder | Any = None,
) -> int:
    conn = get_connection(db_path)
    model = config["embedding"]["model"]
    max_budget = config["budget"]["max_usd"]
    att_config = config.get("attachments", {})

    if embedder is None:
        embedder = GeminiEmbedder(config)

    # Phase 1: Embed messages
    messages = get_messages_without_embeddings(conn, model=model)
    logger.info(f"{len(messages)} messages to embed")

    total_embedded = 0

    # Batch message embeddings
    for i in range(0, len(messages), TEXT_BATCH_SIZE):
        ok, spent, remaining = check_budget(conn, max_budget)
        if not ok:
            logger.warning(f"Budget limit ${max_budget:.2f} reached (${spent:.2f} spent). Stopping.")
            break

        batch = messages[i : i + TEXT_BATCH_SIZE]
        texts = []
        batch_msgs = []

        for msg in batch:
            text = format_message_text(
                from_addr=msg.from_addr,
                to_addr=msg.to_addr,
                date=msg.date.strftime("%Y-%m-%d"),
                subject=msg.subject,
                body=msg.body_text,
            )
            text = truncate_to_token_limit(text, MAX_INPUT_TOKENS)
            texts.append(text)
            batch_msgs.append(msg)

        vectors = embedder.embed_texts_batch(texts)

        total_tokens = sum(estimate_tokens(t) for t in texts)
        cost = estimate_cost(input_tokens=total_tokens)
        record_cost(
            conn, operation="embed_text", model=model,
            input_tokens=total_tokens, image_count=0,
            estimated_cost_usd=cost, message_id=f"batch_{i}",
        )

        for msg, text, vector in zip(batch_msgs, texts, vectors):
            insert_embedding(conn, EmbeddingRecord(
                id=None, message_id=msg.id, attachment_id=None,
                chunk_type="message", chunk_text=text[:500],
                embedding=embedding_to_blob(vector), model=model,
            ))
            total_embedded += 1

    # Phase 2: Embed attachments
    all_messages = conn.execute("SELECT id, subject FROM messages").fetchall()
    max_images_per_msg = att_config.get("max_images_per_message", 10)

    for row in tqdm(all_messages, desc="Processing attachments"):
        msg_id = row["id"]
        msg_subject = row["subject"]
        attachments = get_attachments_for_message(conn, msg_id)

        for att in attachments:
            # Text embedding for attachment
            if att.extracted_text and not embedding_exists(conn, msg_id, att.id, "attachment_text", model):
                ok, spent, remaining = check_budget(conn, max_budget)
                if not ok:
                    logger.warning(f"Budget limit reached. Stopping.")
                    conn.close()
                    return total_embedded

                text = format_attachment_text(att.filename, msg_subject, att.extracted_text)
                text = truncate_to_token_limit(text, MAX_INPUT_TOKENS)
                vector = embedder.embed_texts_batch([text])[0]

                tokens = estimate_tokens(text)
                cost = estimate_cost(input_tokens=tokens)
                record_cost(
                    conn, operation="embed_text", model=model,
                    input_tokens=tokens, image_count=0,
                    estimated_cost_usd=cost, message_id=msg_id,
                )

                insert_embedding(conn, EmbeddingRecord(
                    id=None, message_id=msg_id, attachment_id=att.id,
                    chunk_type="attachment_text",
                    chunk_text=text[:500],
                    embedding=embedding_to_blob(vector), model=model,
                ))
                total_embedded += 1

            # Image embeddings for attachment
            if att.image_path:
                image_path = Path(att.image_path)
                if image_path.is_dir():
                    # Directory of page images (from PDF extraction)
                    image_files = sorted(image_path.glob("*.png"))[:max_images_per_msg]
                elif image_path.is_file():
                    image_files = [image_path]
                else:
                    image_files = []

                for img_file in image_files:
                    if embedding_exists(conn, msg_id, att.id, "attachment_image", model):
                        continue
                    ok, spent, remaining = check_budget(conn, max_budget)
                    if not ok:
                        logger.warning(f"Budget limit reached. Stopping.")
                        conn.close()
                        return total_embedded

                    try:
                        vector = embedder.embed_image(img_file)
                    except Exception as e:
                        logger.warning(f"Failed to embed image {img_file}: {e}")
                        continue

                    cost = estimate_cost(image_count=1)
                    record_cost(
                        conn, operation="embed_image", model=model,
                        input_tokens=0, image_count=1,
                        estimated_cost_usd=cost, message_id=msg_id,
                    )

                    insert_embedding(conn, EmbeddingRecord(
                        id=None, message_id=msg_id, attachment_id=att.id,
                        chunk_type="attachment_image",
                        chunk_text=f"[Image: {img_file.name} from {att.filename}]",
                        embedding=embedding_to_blob(vector), model=model,
                    ))
                    total_embedded += 1

    conn.close()
    logger.info(f"Embedded {total_embedded} chunks total")
    return total_embedded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_embed_pipeline.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/gmail_search/embed/pipeline.py tests/test_embed_pipeline.py
git commit -m "feat: embedding pipeline with batching, budget enforcement, and idempotency"
```

---

### Task 12: ScaNN Index Builder

**Files:**
- Create: `src/gmail_search/index/__init__.py`
- Create: `src/gmail_search/index/builder.py`
- Create: `tests/test_index_builder.py`

- [ ] **Step 1: Create index/__init__.py**

```python
```

- [ ] **Step 2: Write failing test**

Create `tests/test_index_builder.py`:

```python
import struct
import numpy as np

from gmail_search.store.db import init_db, get_connection
from gmail_search.store.models import Message, EmbeddingRecord
from gmail_search.store.queries import upsert_message, insert_embedding
from gmail_search.index.builder import build_index, load_index_metadata
from datetime import datetime


def _make_embedding(dims=16):
    """Use small dims for fast tests."""
    vec = [float(i) / dims for i in range(dims)]
    return struct.pack(f"{dims}f", *vec)


def test_build_index(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    # Insert 50 messages with embeddings (ScaNN needs enough data)
    for i in range(50):
        msg = Message(
            id=f"msg{i}", thread_id="t1", from_addr="a@b.com", to_addr="c@d.com",
            subject="Test", body_text="Hello", body_html="",
            date=datetime(2025, 1, 1), labels=[], history_id=1, raw_json="{}",
        )
        upsert_message(conn, msg)
        insert_embedding(conn, EmbeddingRecord(
            id=None, message_id=f"msg{i}", attachment_id=None,
            chunk_type="message", chunk_text="test",
            embedding=_make_embedding(16), model="test-model",
        ))
    conn.close()

    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=16)
    assert index_dir.exists()

    ids = load_index_metadata(index_dir)
    assert len(ids) == 50


def test_build_index_empty_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=16)
    # Should not crash, just log warning
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_index_builder.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 4: Implement builder.py**

Create `src/gmail_search/index/builder.py`:

```python
import json
import logging
import math
import struct
from pathlib import Path

import numpy as np
import scann

from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)


def build_index(
    db_path: Path,
    index_dir: Path,
    model: str,
    dimensions: int,
) -> None:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, embedding FROM embeddings WHERE model = ?", (model,)
    ).fetchall()
    conn.close()

    if not rows:
        logger.warning("No embeddings found. Skipping index build.")
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "ids.json").write_text("[]")
        return

    ids = [r["id"] for r in rows]
    vectors = np.array(
        [list(struct.unpack(f"{dimensions}f", r["embedding"])) for r in rows],
        dtype=np.float32,
    )

    logger.info(f"Building ScaNN index with {len(ids)} vectors, {dimensions} dims")

    num_leaves = max(int(math.sqrt(len(ids))), 1)
    num_leaves = min(num_leaves, len(ids))

    builder = scann.scann_ops_pybind.builder(vectors, 10, "dot_product")

    if len(ids) >= 100:
        builder = builder.tree(
            num_leaves=num_leaves,
            num_leaves_to_search=min(num_leaves, 100),
            training_sample_size=min(len(ids), 250000),
        )
        builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
        builder = builder.reorder(100)
    else:
        builder = builder.score_brute_force()

    searcher = builder.build()

    index_dir.mkdir(parents=True, exist_ok=True)
    searcher.serialize(str(index_dir))
    (index_dir / "ids.json").write_text(json.dumps(ids))

    logger.info(f"Index built and saved to {index_dir}")


def load_index_metadata(index_dir: Path) -> list[int]:
    ids_file = index_dir / "ids.json"
    if not ids_file.exists():
        return []
    return json.loads(ids_file.read_text())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_index_builder.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/gmail_search/index/ tests/test_index_builder.py
git commit -m "feat: ScaNN index builder with adaptive configuration"
```

---

### Task 13: ScaNN Searcher

**Files:**
- Create: `src/gmail_search/index/searcher.py`
- Create: `tests/test_index_searcher.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_index_searcher.py`:

```python
import struct
import numpy as np
from datetime import datetime

from gmail_search.store.db import init_db, get_connection
from gmail_search.store.models import Message, EmbeddingRecord
from gmail_search.store.queries import upsert_message, insert_embedding
from gmail_search.index.builder import build_index
from gmail_search.index.searcher import ScannSearcher


def _make_vec(dims, value):
    """Create a vector where all values are `value`."""
    return struct.pack(f"{dims}f", *([value] * dims))


def test_searcher_returns_nearest(tmp_path):
    dims = 16
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    # Insert 50 messages with varying embeddings
    for i in range(50):
        msg = Message(
            id=f"msg{i}", thread_id="t1", from_addr="a@b.com", to_addr="c@d.com",
            subject=f"Message {i}", body_text="test", body_html="",
            date=datetime(2025, 1, 1), labels=[], history_id=1, raw_json="{}",
        )
        upsert_message(conn, msg)
        # Embedding value scales with i so we can predict nearest neighbors
        val = float(i) / 50
        insert_embedding(conn, EmbeddingRecord(
            id=None, message_id=f"msg{i}", attachment_id=None,
            chunk_type="message", chunk_text=f"Message {i}",
            embedding=_make_vec(dims, val), model="test-model",
        ))
    conn.close()

    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=dims)

    searcher = ScannSearcher(index_dir, dimensions=dims)

    # Query with a vector close to msg49 (value=0.98)
    query = np.array([0.98] * dims, dtype=np.float32)
    embedding_ids, scores = searcher.search(query, top_k=5)

    assert len(embedding_ids) == 5
    assert len(scores) == 5
    # Scores should be in descending order
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_index_searcher.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement searcher.py**

Create `src/gmail_search/index/searcher.py`:

```python
import json
import logging
from pathlib import Path

import numpy as np
import scann

logger = logging.getLogger(__name__)


class ScannSearcher:
    def __init__(self, index_dir: Path, dimensions: int):
        self.index_dir = index_dir
        self.dimensions = dimensions

        ids_file = index_dir / "ids.json"
        if not ids_file.exists():
            raise FileNotFoundError(f"Index not found at {index_dir}. Run 'gmail-search reindex' first.")

        self.embedding_ids: list[int] = json.loads(ids_file.read_text())

        if not self.embedding_ids:
            self.searcher = None
            logger.warning("Empty index loaded")
            return

        self.searcher = scann.scann_ops_pybind.load_searcher(str(index_dir))
        logger.info(f"Loaded ScaNN index with {len(self.embedding_ids)} vectors")

    def search(
        self, query_vector: np.ndarray, top_k: int = 20
    ) -> tuple[list[int], list[float]]:
        if self.searcher is None or not self.embedding_ids:
            return [], []

        neighbors, distances = self.searcher.search(query_vector, final_num_neighbors=top_k)

        embedding_ids = [self.embedding_ids[i] for i in neighbors]
        scores = distances.tolist()

        return embedding_ids, scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_index_searcher.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/gmail_search/index/searcher.py tests/test_index_searcher.py
git commit -m "feat: ScaNN searcher with index loading and query"
```

---

### Task 14: Search Engine

**Files:**
- Create: `src/gmail_search/search/__init__.py`
- Create: `src/gmail_search/search/engine.py`
- Create: `tests/test_search_engine.py`

- [ ] **Step 1: Create search/__init__.py**

```python
```

- [ ] **Step 2: Write failing test**

Create `tests/test_search_engine.py`:

```python
import struct
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np

from gmail_search.config import load_config
from gmail_search.search.engine import SearchEngine, SearchResult
from gmail_search.store.db import init_db, get_connection
from gmail_search.store.models import Message, EmbeddingRecord
from gmail_search.store.queries import upsert_message, insert_embedding
from gmail_search.index.builder import build_index


def _make_vec(dims, value):
    return struct.pack(f"{dims}f", *([value] * dims))


def _setup_db_with_index(tmp_path, dims=16, n=50):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    for i in range(n):
        msg = Message(
            id=f"msg{i}", thread_id=f"t{i}", from_addr=f"user{i}@test.com",
            to_addr="me@test.com", subject=f"Subject {i}",
            body_text=f"Body of message {i}", body_html="",
            date=datetime(2025, 1, 1 + i % 28), labels=["INBOX"],
            history_id=i, raw_json="{}",
        )
        upsert_message(conn, msg)
        val = float(i) / n
        insert_embedding(conn, EmbeddingRecord(
            id=None, message_id=f"msg{i}", attachment_id=None,
            chunk_type="message", chunk_text=f"Body of message {i}",
            embedding=_make_vec(dims, val), model="test-model",
        ))
    conn.close()

    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=dims)
    return db_path, index_dir


def test_search_engine_returns_results(tmp_path):
    dims = 16
    db_path, index_dir = _setup_db_with_index(tmp_path, dims=dims)

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.9] * dims
    mock_embedder.model = "test-model"
    mock_embedder.dimensions = dims

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"
    cfg["embedding"]["dimensions"] = dims

    engine = SearchEngine(db_path, index_dir, cfg, embedder=mock_embedder)
    results = engine.search("test query", top_k=5)

    assert len(results) == 5
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.message_id.startswith("msg") for r in results)
    assert all(r.subject.startswith("Subject") for r in results)
    # Scores descending
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_search_engine_deduplicates_by_message(tmp_path):
    dims = 16
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    msg = Message(
        id="msg1", thread_id="t1", from_addr="a@b.com", to_addr="c@d.com",
        subject="Duped", body_text="Body", body_html="",
        date=datetime(2025, 1, 1), labels=[], history_id=1, raw_json="{}",
    )
    upsert_message(conn, msg)

    # Two embeddings for same message (message + attachment)
    vec = _make_vec(dims, 0.9)
    insert_embedding(conn, EmbeddingRecord(
        id=None, message_id="msg1", attachment_id=None,
        chunk_type="message", chunk_text="Body",
        embedding=vec, model="test-model",
    ))
    insert_embedding(conn, EmbeddingRecord(
        id=None, message_id="msg1", attachment_id=1,
        chunk_type="attachment_text", chunk_text="Attachment",
        embedding=vec, model="test-model",
    ))

    # Need enough total vectors for ScaNN
    for i in range(48):
        m = Message(
            id=f"pad{i}", thread_id="t", from_addr="x@x.com", to_addr="y@y.com",
            subject="Pad", body_text="pad", body_html="",
            date=datetime(2025, 1, 1), labels=[], history_id=1, raw_json="{}",
        )
        upsert_message(conn, m)
        insert_embedding(conn, EmbeddingRecord(
            id=None, message_id=f"pad{i}", attachment_id=None,
            chunk_type="message", chunk_text="pad",
            embedding=_make_vec(dims, 0.01), model="test-model",
        ))
    conn.close()

    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=dims)

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.9] * dims
    mock_embedder.model = "test-model"
    mock_embedder.dimensions = dims

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"
    cfg["embedding"]["dimensions"] = dims

    engine = SearchEngine(db_path, index_dir, cfg, embedder=mock_embedder)
    results = engine.search("test", top_k=10)

    # msg1 should appear only once despite two embedding hits
    msg1_results = [r for r in results if r.message_id == "msg1"]
    assert len(msg1_results) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_search_engine.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 4: Implement engine.py**

Create `src/gmail_search/search/engine.py`:

```python
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gmail_search.embed.client import GeminiEmbedder
from gmail_search.index.searcher import ScannSearcher
from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    score: float
    message_id: str
    subject: str
    from_addr: str
    date: str
    snippet: str
    match_type: str  # "message", "attachment_text", "attachment_image"
    attachment_filename: str | None = None


class SearchEngine:
    def __init__(
        self,
        db_path: Path,
        index_dir: Path,
        config: dict[str, Any],
        embedder: GeminiEmbedder | Any = None,
    ):
        self.db_path = db_path
        self.config = config
        dims = config["embedding"]["dimensions"]
        self.searcher = ScannSearcher(index_dir, dimensions=dims)
        self.embedder = embedder or GeminiEmbedder(config)

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        query_vector = np.array(self.embedder.embed_query(query), dtype=np.float32)

        # Fetch more than top_k to allow for deduplication
        fetch_k = min(top_k * 3, len(self.searcher.embedding_ids))
        embedding_ids, scores = self.searcher.search(query_vector, top_k=fetch_k)

        if not embedding_ids:
            return []

        conn = get_connection(self.db_path)

        # Fetch embedding metadata
        placeholders = ",".join("?" * len(embedding_ids))
        rows = conn.execute(
            f"""SELECT e.id, e.message_id, e.attachment_id, e.chunk_type, e.chunk_text,
                       m.subject, m.from_addr, m.date,
                       a.filename as att_filename
                FROM embeddings e
                JOIN messages m ON e.message_id = m.id
                LEFT JOIN attachments a ON e.attachment_id = a.id
                WHERE e.id IN ({placeholders})""",
            embedding_ids,
        ).fetchall()
        conn.close()

        # Map embedding_id → row
        row_map = {r["id"]: r for r in rows}

        # Build results with dedup by message_id
        seen_messages: dict[str, SearchResult] = {}

        for emb_id, score in zip(embedding_ids, scores):
            row = row_map.get(emb_id)
            if row is None:
                continue

            msg_id = row["message_id"]
            if msg_id in seen_messages:
                # Keep the higher score
                if score > seen_messages[msg_id].score:
                    seen_messages[msg_id].score = score
                    seen_messages[msg_id].match_type = row["chunk_type"]
                    seen_messages[msg_id].snippet = (row["chunk_text"] or "")[:200]
                continue

            seen_messages[msg_id] = SearchResult(
                score=score,
                message_id=msg_id,
                subject=row["subject"],
                from_addr=row["from_addr"],
                date=row["date"],
                snippet=(row["chunk_text"] or "")[:200],
                match_type=row["chunk_type"],
                attachment_filename=row["att_filename"],
            )

        results = sorted(seen_messages.values(), key=lambda r: r.score, reverse=True)
        return results[:top_k]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_search_engine.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/gmail_search/search/ tests/test_search_engine.py
git commit -m "feat: search engine with query embedding, ScaNN lookup, and deduplication"
```

---

### Task 15: CLI

**Files:**
- Create: `src/gmail_search/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test for CLI smoke test**

Create `tests/test_cli.py`:

```python
from click.testing import CliRunner

from gmail_search.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Gmail Search" in result.output


def test_cli_status(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Messages:" in result.output


def test_cli_cost_empty(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["cost", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "$0.00" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement cli.py**

Create `src/gmail_search/cli.py`:

```python
import logging
from pathlib import Path

import click

from gmail_search.config import load_config
from gmail_search.store.db import init_db, get_connection
from gmail_search.store.cost import get_total_spend, get_spend_breakdown, check_budget

logger = logging.getLogger(__name__)


@click.group(help="Gmail Search — local semantic search over your Gmail")
@click.option("--data-dir", type=click.Path(), default=None, help="Data directory path")
@click.option("--config", "config_path", type=click.Path(), default="config.yaml", help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx, data_dir, config_path, verbose):
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

    data_path.mkdir(parents=True, exist_ok=True)
    init_db(ctx.obj["db_path"])


@main.command(help="Run OAuth flow to authenticate with Gmail")
@click.pass_context
def auth(ctx):
    from gmail_search.gmail.auth import get_credentials
    data_dir = ctx.obj["data_dir"]
    get_credentials(data_dir)
    click.echo("Authentication successful. Token saved.")


@main.command(help="Download messages from Gmail")
@click.option("--max-messages", type=int, default=None, help="Max messages to download")
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
@click.pass_context
def extract(ctx):
    from gmail_search.extract import dispatch
    from gmail_search.store.queries import get_attachments_for_message
    from tqdm import tqdm

    cfg = ctx.obj["config"]
    conn = get_connection(ctx.obj["db_path"])
    att_config = cfg.get("attachments", {})

    rows = conn.execute("SELECT id FROM messages").fetchall()
    updated = 0

    for row in tqdm(rows, desc="Extracting attachments"):
        attachments = get_attachments_for_message(conn, row["id"])
        for att in attachments:
            if att.extracted_text or att.image_path:
                continue  # Already extracted
            if not att.raw_path or not Path(att.raw_path).exists():
                continue

            result = dispatch(att.mime_type, Path(att.raw_path), att_config)
            if result is None:
                continue

            updates = {}
            if result.text:
                updates["extracted_text"] = result.text
            if result.images:
                # Store the directory containing page images, or the single image path
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
@click.pass_context
def embed(ctx, model, budget):
    from gmail_search.embed.pipeline import run_embedding_pipeline
    from gmail_search.store.cost import estimate_cost

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/gmail_search/cli.py tests/test_cli.py
git commit -m "feat: Click CLI with all commands"
```

---

### Task 16: Web UI Server

**Files:**
- Create: `src/gmail_search/server.py`
- Create: `templates/index.html`

- [ ] **Step 1: Implement server.py**

Create `src/gmail_search/server.py`:

```python
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from gmail_search.config import load_config
from gmail_search.search.engine import SearchEngine
from gmail_search.store.cost import check_budget, get_total_spend
from gmail_search.store.db import get_connection
from gmail_search.store.queries import get_message, get_attachments_for_message


def create_app(
    db_path: Path,
    data_dir: Path,
    config: dict[str, Any],
) -> FastAPI:
    app = FastAPI(title="Gmail Search")

    templates_dir = Path(__file__).parent.parent.parent / "templates"
    index_dir = data_dir / "scann_index"

    _engine: SearchEngine | None = None

    def get_engine() -> SearchEngine:
        nonlocal _engine
        if _engine is None:
            _engine = SearchEngine(db_path, index_dir, config)
        return _engine

    @app.get("/", response_class=HTMLResponse)
    async def home():
        html_file = templates_dir / "index.html"
        return html_file.read_text()

    @app.get("/api/search")
    async def api_search(q: str = Query(...), k: int = Query(20)):
        engine = get_engine()
        results = engine.search(q, top_k=k)
        return [
            {
                "score": r.score,
                "message_id": r.message_id,
                "subject": r.subject,
                "from_addr": r.from_addr,
                "date": r.date,
                "snippet": r.snippet,
                "match_type": r.match_type,
                "attachment_filename": r.attachment_filename,
            }
            for r in results
        ]

    @app.get("/api/message/{message_id}")
    async def api_message(message_id: str):
        conn = get_connection(db_path)
        msg = get_message(conn, message_id)
        if msg is None:
            conn.close()
            return {"error": "Message not found"}
        attachments = get_attachments_for_message(conn, message_id)
        conn.close()
        return {
            "id": msg.id,
            "thread_id": msg.thread_id,
            "from_addr": msg.from_addr,
            "to_addr": msg.to_addr,
            "subject": msg.subject,
            "body_text": msg.body_text,
            "body_html": msg.body_html,
            "date": msg.date.isoformat(),
            "labels": msg.labels,
            "attachments": [
                {
                    "id": a.id,
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                    "size_bytes": a.size_bytes,
                }
                for a in attachments
            ],
        }

    @app.get("/api/attachment/{attachment_id}")
    async def api_attachment(attachment_id: int):
        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT raw_path, mime_type, filename FROM attachments WHERE id = ?",
            (attachment_id,),
        ).fetchone()
        conn.close()
        if row is None or not row["raw_path"]:
            return {"error": "Attachment not found"}
        return FileResponse(
            row["raw_path"],
            media_type=row["mime_type"],
            filename=row["filename"],
        )

    @app.get("/api/status")
    async def api_status():
        conn = get_connection(db_path)
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        total_cost = get_total_spend(conn)
        ok, spent, remaining = check_budget(conn, config["budget"]["max_usd"])
        conn.close()
        return {
            "messages": msg_count,
            "embeddings": emb_count,
            "total_cost_usd": round(total_cost, 4),
            "budget_remaining_usd": round(remaining, 4),
        }

    return app
```

- [ ] **Step 2: Create templates/index.html**

Create `templates/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gmail Search</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
  h1 { margin-bottom: 20px; color: #333; }
  .search-box { display: flex; gap: 8px; margin-bottom: 20px; }
  .search-box input { flex: 1; padding: 12px 16px; font-size: 16px; border: 2px solid #ddd; border-radius: 8px; outline: none; }
  .search-box input:focus { border-color: #4285f4; }
  .search-box button { padding: 12px 24px; font-size: 16px; background: #4285f4; color: white; border: none; border-radius: 8px; cursor: pointer; }
  .search-box button:hover { background: #3367d6; }
  .result { background: white; padding: 16px; margin-bottom: 12px; border-radius: 8px; border: 1px solid #e0e0e0; cursor: pointer; }
  .result:hover { border-color: #4285f4; }
  .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .result-subject { font-weight: 600; color: #1a1a1a; }
  .result-score { font-size: 12px; color: #999; }
  .result-meta { font-size: 13px; color: #666; margin-bottom: 8px; }
  .result-snippet { font-size: 14px; color: #444; line-height: 1.4; }
  .badge { display: inline-block; font-size: 11px; padding: 2px 6px; border-radius: 4px; background: #e8f0fe; color: #4285f4; margin-left: 8px; }
  .detail-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 100; }
  .detail-panel { position: fixed; top: 0; right: 0; width: 600px; height: 100vh; background: white; overflow-y: auto; padding: 24px; z-index: 101; box-shadow: -2px 0 10px rgba(0,0,0,0.2); }
  .detail-panel h2 { margin-bottom: 12px; }
  .detail-panel .meta { color: #666; margin-bottom: 16px; font-size: 14px; }
  .detail-panel .body { white-space: pre-wrap; line-height: 1.6; font-size: 14px; }
  .detail-panel .close { float: right; cursor: pointer; font-size: 24px; color: #666; }
  .status-bar { font-size: 12px; color: #999; margin-bottom: 16px; }
  .empty { text-align: center; padding: 40px; color: #999; }
</style>
</head>
<body>
<h1>Gmail Search</h1>
<div class="status-bar" id="status"></div>
<div class="search-box">
  <input type="text" id="query" placeholder="Search your email..." autofocus>
  <button onclick="doSearch()">Search</button>
</div>
<div id="results"></div>
<div class="detail-overlay" id="overlay" onclick="closeDetail()"></div>
<div class="detail-panel" id="detail" style="display:none">
  <span class="close" onclick="closeDetail()">&times;</span>
  <div id="detail-content"></div>
</div>
<script>
const query = document.getElementById('query');
const results = document.getElementById('results');

query.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

async function doSearch() {
  const q = query.value.trim();
  if (!q) return;
  results.innerHTML = '<div class="empty">Searching...</div>';
  const res = await fetch('/api/search?q=' + encodeURIComponent(q) + '&k=20');
  const data = await res.json();
  if (!data.length) { results.innerHTML = '<div class="empty">No results found.</div>'; return; }
  results.innerHTML = data.map((r, i) => `
    <div class="result" onclick="showDetail('${r.message_id}')">
      <div class="result-header">
        <span class="result-subject">${esc(r.subject)}</span>
        <span class="result-score">${r.score.toFixed(3)}</span>
      </div>
      <div class="result-meta">
        ${esc(r.from_addr)} &middot; ${r.date ? r.date.split('T')[0] : ''}
        <span class="badge">${r.match_type}</span>
        ${r.attachment_filename ? '<span class="badge">' + esc(r.attachment_filename) + '</span>' : ''}
      </div>
      <div class="result-snippet">${esc(r.snippet)}</div>
    </div>
  `).join('');
}

async function showDetail(id) {
  const res = await fetch('/api/message/' + id);
  const m = await res.json();
  document.getElementById('detail-content').innerHTML = `
    <h2>${esc(m.subject)}</h2>
    <div class="meta">
      <strong>From:</strong> ${esc(m.from_addr)}<br>
      <strong>To:</strong> ${esc(m.to_addr)}<br>
      <strong>Date:</strong> ${m.date}<br>
      <strong>Labels:</strong> ${(m.labels||[]).join(', ')}
    </div>
    <div class="body">${esc(m.body_text)}</div>
    ${m.attachments && m.attachments.length ? '<h3 style="margin-top:16px">Attachments</h3>' + m.attachments.map(a =>
      '<div style="margin:4px 0"><a href="/api/attachment/' + a.id + '" target="_blank">' + esc(a.filename) + '</a> (' + (a.size_bytes/1024).toFixed(1) + ' KB)</div>'
    ).join('') : ''}
  `;
  document.getElementById('detail').style.display = 'block';
  document.getElementById('overlay').style.display = 'block';
}

function closeDetail() {
  document.getElementById('detail').style.display = 'none';
  document.getElementById('overlay').style.display = 'none';
}

function esc(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

fetch('/api/status').then(r => r.json()).then(s => {
  document.getElementById('status').textContent =
    s.messages + ' messages | ' + s.embeddings + ' embeddings | $' + s.total_cost_usd.toFixed(2) + ' spent';
});
</script>
</body>
</html>
```

- [ ] **Step 3: Verify server import works**

Run: `python -c "from gmail_search.server import create_app; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/gmail_search/server.py templates/
git commit -m "feat: FastAPI web UI with search, detail view, and status"
```

---

### Task 17: Integration Smoke Test

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Verify CLI end-to-end (offline, no Gmail)**

Run: `cd /home/ssilver/development/gmail-search && gmail-search status --data-dir /tmp/gmail-test-data`
Expected: Shows `Messages: 0`, `Embeddings: 0`, etc.

Run: `gmail-search cost --data-dir /tmp/gmail-test-data`
Expected: Shows `$0.00`

- [ ] **Step 3: Commit any test fixes if needed**

```bash
git add -u
git commit -m "fix: integration test fixes"
```

---

## Execution Checklist

After all tasks are complete, the user should be able to:

1. `pip install -e .`
2. Place `credentials.json` in `data/`
3. `gmail-search auth` → browser opens, authenticate
4. `gmail-search download --max-messages 10000` → downloads 10k messages
5. `gmail-search extract` → extracts text/images from PDF attachments
6. `gmail-search embed` → embeds everything (~$2, shows progress)
7. `gmail-search reindex` → builds ScaNN index
8. `gmail-search search "contract from the accountant"` → results in terminal
9. `gmail-search serve` → browse results at localhost:8080
10. `gmail-search cost --breakdown` → see what you spent
