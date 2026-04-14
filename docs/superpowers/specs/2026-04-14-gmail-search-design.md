# Gmail Search — Local Semantic Search Over Gmail

**Date:** 2026-04-14
**Status:** Approved

## Overview

A local tool that downloads your entire Gmail mailbox, embeds messages and attachments using Gemini's multimodal embedding model, and makes everything searchable via ScaNN vector search. CLI + web UI interface.

**Primary use case:** Personal knowledge retrieval — "I know I got an email about X, find it."

## Tech Stack

- **Language:** Python
- **Gmail access:** Gmail API (OAuth2, `gmail.readonly` scope)
- **Embeddings:** `gemini-embedding-2-preview` (3072 dims, 8192 token input, multimodal)
- **Vector search:** ScaNN (in-memory, asymmetric hashing)
- **Storage:** SQLite (messages, attachments metadata, embeddings, cost tracking)
- **CLI:** Click
- **Web UI:** FastAPI + vanilla HTML/JS

## Data Model (SQLite)

### `messages`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT PK | Gmail message ID |
| thread_id | TEXT | Gmail thread ID |
| from_addr | TEXT | Sender |
| to_addr | TEXT | Recipients |
| subject | TEXT | |
| body_text | TEXT | Plain text body |
| body_html | TEXT | HTML body (kept for display) |
| date | DATETIME | |
| labels | TEXT | JSON array of Gmail labels |
| history_id | INTEGER | For incremental sync |
| raw_json | TEXT | Full API response |

### `attachments`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | Auto-increment |
| message_id | TEXT FK | → messages.id |
| filename | TEXT | |
| mime_type | TEXT | |
| size_bytes | INTEGER | |
| extracted_text | TEXT | From PDF/doc extraction |
| image_path | TEXT | Path to rendered image (for multimodal embedding) |
| raw_path | TEXT | Path to original file on disk |

### `embeddings`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | Auto-increment |
| message_id | TEXT FK | → messages.id |
| attachment_id | INTEGER FK | Nullable — NULL for message-level embeddings |
| chunk_type | TEXT | "message", "attachment_text", "attachment_image" |
| chunk_text | TEXT | The text that was embedded (for display in results) |
| embedding | BLOB | Raw float32 vector, 3072 dims = 12KB per row |
| model | TEXT | "gemini-embedding-2-preview" |

### `costs`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | Auto-increment |
| timestamp | DATETIME | |
| operation | TEXT | "embed_text", "embed_image", "download" |
| model | TEXT | "gemini-embedding-2-preview" |
| input_tokens | INTEGER | For text embeddings |
| image_count | INTEGER | For multimodal embeddings |
| estimated_cost_usd | FLOAT | Calculated from published pricing |
| message_id | TEXT | Which message triggered this |

### `sync_state`

| Column | Type | Notes |
|--------|------|-------|
| key | TEXT PK | e.g. "last_history_id", "last_sync_time" |
| value | TEXT | Stored as text, parsed by consumer |

## Gmail Download Pipeline

- OAuth2 with `gmail.readonly` scope. Credentials in `data/credentials.json`, token in `data/token.json`.
- First run opens browser for consent. Token auto-refreshes after that.
- `messages.list` with batching (pages of 500 IDs), then `messages.get` with `format=full` in batch HTTP requests (up to 100 per batch).
- Rate limit: ~50 messages/second (250 quota units/sec, 5 units per get).
- 10k messages: ~3-4 minutes. 50k: ~15-20 minutes.
- Attachments downloaded via `messages.attachments.get`, saved to `data/attachments/{message_id}/`.
- Checkpoint after each batch (SQLite commit). Crash-safe resume.
- Retry with exponential backoff on 429 errors.
- Incremental sync via `history.list` with `startHistoryId`. Handles `messagesAdded` events.

## Attachment Extraction

Dispatcher pattern: `mime_type → extractor function → ExtractResult(text, images)`.

### Prototype scope: PDF + images

- **PDF:** `pymupdf` for text extraction + page rendering (PNG at 150 DPI). Returns both text and page images.
- **Images (jpg/png/gif):** Passthrough — file path returned as-is for multimodal embedding.

### Limits

| Limit | Value |
|-------|-------|
| Max file size | 10MB |
| Max pages per PDF | 20 |
| Max images per message | 10 |
| Max attachment text | 50k tokens |

All configurable. Exceeding a limit = skip with warning log, not crash.

### Future extensibility

- Office docs: add `office.py`, register mime types in dispatcher.
- OCR: Tesseract/vision-model fallback when PDF text extraction returns empty.
- Zero changes to rest of pipeline.

## Embedding Pipeline

### Gemini client wrapper

- Two methods: `embed_text(text, task_type)` and `embed_image(image_path, task_type)`
- `task_type=RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search
- Returns `list[float]`, 3072 dimensions

### What gets embedded per message

| Chunk | Type | Content |
|-------|------|---------|
| Message | text | `From: {from} \| To: {to} \| Date: {date} \| Subject: {subject} \| {body_text}` |
| Attachment text | text | `Attachment: {filename} \| From email: {subject} \| {extracted_text}` |
| Attachment image | image | Raw PNG of each rendered page / image attachment |

### Batching

- Text: up to 100 per batch call. 10k messages → ~150 batch calls, a few minutes.
- Images: one per call currently.
- Rate limiting: respect 1500 req/min free tier with built-in backoff.

### Idempotency

- Check if embedding row exists for `(message_id, attachment_id, chunk_type, model)` before embedding.
- Re-running `embed` only processes the delta.
- Re-embed with new model: `--model=new-model-name`, old and new coexist.

### Token overflow

- Chunks exceeding 8192 tokens: truncate body from the end, preserving metadata prefix + subject.
- Multi-page PDF text: split into ~6000 token chunks, embed separately.

### Cost tracking and budget

- Every embedding call logged to `costs` table with tokens/images, estimated USD, and message_id.
- Default budget: $5.00 (configurable in `config.yaml`).
- Before each batch: check cumulative spend. Stop if next batch would exceed budget.
- CLI: `gmail-search cost` and `gmail-search cost --breakdown`.
- On startup: print estimated cost for remaining work.

### Estimated costs

| Scope | Text cost | Image cost | Total |
|-------|-----------|------------|-------|
| 10k messages (prototype) | $1.80 | $0.30 | ~$2.10 |
| 50k messages (scale-up) | $9.00 | $1.50 | ~$10.50 |

## ScaNN Index & Search

### Index build

1. Load all embeddings from SQLite into numpy array (N x 3072).
2. Build ScaNN index: `num_leaves=sqrt(N)`, asymmetric hashing with 2-byte quantization, top-100 reordering with exact distance.
3. Save to `data/scann_index/`.
4. Save parallel array mapping ScaNN row index → `embeddings.id`.

### Rebuild strategy

- `gmail-search reindex` — full rebuild. ~10 seconds for 50k vectors.
- No incremental update needed until 500k+.

### Search flow

```
User query
  → Gemini embed (task_type=RETRIEVAL_QUERY)
  → ScaNN query (top K=20 neighbors)
  → Map ScaNN indices → embedding IDs → message IDs
  → Fetch messages + attachments from SQLite
  → Deduplicate by message_id (keep highest score, note all matching chunks)
  → Return ranked results
```

### Result object

```json
{
  "score": 0.87,
  "message_id": "abc123",
  "subject": "Q3 contract renewal",
  "from": "john@example.com",
  "date": "2025-11-03",
  "snippet": "...relevant portion of body...",
  "match_type": "message | attachment_text | attachment_image",
  "attachment_filename": "contract.pdf"
}
```

### Snippet generation

- Text matches: first 200 chars of the embedded chunk.
- Image matches: message subject + attachment filename. Image displayed in web UI.

## CLI & Web Interface

### CLI (`gmail-search`)

```
gmail-search auth              — Run OAuth flow, save token
gmail-search download          — Initial download (or resume interrupted)
gmail-search sync              — Incremental sync (new messages only)
gmail-search extract           — Extract text/images from downloaded attachments
gmail-search embed             — Embed all unembedded messages + attachments
gmail-search reindex           — Rebuild ScaNN index
gmail-search search "query"    — Search, print top 10 results
gmail-search cost              — Show total spend
gmail-search cost --breakdown  — Spend by operation type
gmail-search status            — Message count, embedding count, index freshness, budget remaining
gmail-search serve             — Start web UI on localhost:8080
```

### Web UI (FastAPI + vanilla HTML/JS)

Single page: search box, results list, detail view.

Each result shows: score, from, date, subject, snippet, match type badge.
Detail view: full email body + attachment list. PDF page images inline for multimodal matches.

### API endpoints

```
GET  /                          — Search page
GET  /api/search?q=...&k=20    — JSON search results
GET  /api/message/{id}         — Full message detail
GET  /api/attachment/{id}      — Serve attachment file / image
GET  /api/status               — Stats + cost
```

## Project Structure

```
gmail-search/
  src/
    gmail_search/
      __init__.py
      config.py          — Config loading (config.yaml + CLI overrides)
      store/
        __init__.py
        db.py            — SQLite connection, schema init
        models.py        — Message, Attachment, Embedding dataclasses
        queries.py       — CRUD operations
        cost.py          — Cost tracking + budget enforcement
      gmail/
        __init__.py
        auth.py          — OAuth2 flow + token management
        client.py        — Download, sync, batch fetching
        parser.py        — Raw API response → Message/Attachment objects
      extract/
        __init__.py      — Dispatcher: mime_type → extractor
        pdf.py           — pymupdf: text + page images
        image.py         — Passthrough for jpg/png
      embed/
        __init__.py
        client.py        — Gemini API wrapper (text + multimodal)
        pipeline.py      — Orchestrates: load unembedded → batch embed → store
      index/
        __init__.py
        builder.py       — Load embeddings → build ScaNN → save
        searcher.py      — Load ScaNN → query → return ranked IDs
      search/
        __init__.py
        engine.py        — Orchestrates: embed query → ScaNN → fetch → dedupe → rank
      cli.py             — Click CLI entry points
      server.py          — FastAPI app
  templates/
    index.html           — Search UI
  data/                  — Gitignored. DB, index, attachments, tokens.
  config.yaml            — User config
  pyproject.toml
  README.md
```

## Dependencies

```
google-api-python-client   — Gmail API
google-auth-oauthlib       — OAuth2 flow
google-genai               — Gemini embedding API
scann                      — Vector search
pymupdf                    — PDF text + page rendering
numpy                      — Embedding array ops
fastapi                    — Web UI
uvicorn                    — ASGI server
click                      — CLI
tqdm                       — Progress bars
pyyaml                     — Config
```

## Configuration (`config.yaml`)

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
  max_messages: null  # null = all, set to 10000 for prototype

search:
  default_top_k: 20

server:
  host: "127.0.0.1"
  port: 8080
```

## Scale Path

- Prototype: 10k messages, in-memory ScaNN, ~$2 embedding cost.
- Scale-up: same architecture handles 50k-500k. ScaNN stays in-memory.
- Beyond 500k: evaluate disk-backed alternatives (FAISS with IVF, or Milvus).
