# Gmail Search

Local semantic search over your Gmail. Downloads your mailbox, embeds messages and attachments using Google's Gemini embedding model, and makes everything searchable via a web UI or CLI.

## How it works

1. **Download** — fetches messages via Gmail API, stores in SQLite
2. **Extract** — pulls text and page images from PDF/image attachments
3. **Embed** — sends text + images to Gemini for vector embeddings
4. **Index** — builds a ScaNN vector index + SQLite FTS5 keyword index
5. **Search** — hybrid ranking: semantic similarity + BM25 keyword match + recency + Gmail labels + thread engagement

## Quick start

### Prerequisites

- Python 3.11+
- A Google Cloud project with the Gmail API enabled
- A Gemini API key (set as `GEMINI_API_KEY` or `GOOGLE_API_KEY` env var)

### 1. Install

```bash
git clone <repo-url>
cd gmail-search
pip install -e .
```

### 2. Set up Google Cloud credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create or select a project
3. Enable the **Gmail API** (APIs & Services > Library)
4. Create **OAuth client ID** credentials (APIs & Services > Credentials)
   - Application type: Desktop app
   - Download the JSON file
5. Save it:

```bash
mkdir -p data
mv ~/Downloads/client_secret_*.json data/credentials.json
```

### 3. Set your Gemini API key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Get one at [ai.google.dev](https://ai.google.dev/gemini-api/docs/api-key).

### 4. Authenticate and download

```bash
# Authenticate with Gmail (opens browser)
gmail-search auth

# Download and process everything in rolling batches
gmail-search update --max-messages 10000
```

The `update` command runs the full pipeline in batches of 500 messages: download, extract attachments, embed, and rebuild the search index. Search gets better as it runs.

### 5. Search

```bash
# CLI search
gmail-search search "contract from the accountant"

# Web UI
gmail-search serve --port 8080
```

Open http://localhost:8080 in your browser.

## Commands

| Command | Description |
|---------|-------------|
| `gmail-search auth` | Run OAuth flow, save token |
| `gmail-search update` | Full pipeline: download, extract, embed, reindex (rolling batches) |
| `gmail-search download` | Download messages only |
| `gmail-search sync` | Incremental sync (new messages since last download) |
| `gmail-search extract` | Extract text/images from attachments |
| `gmail-search embed` | Embed unembedded messages and attachments |
| `gmail-search reindex` | Rebuild ScaNN + FTS indexes |
| `gmail-search search "query"` | Search from the command line |
| `gmail-search serve` | Start the web UI |
| `gmail-search status` | Show message count, embeddings, cost |
| `gmail-search cost --breakdown` | Show embedding spend by operation |

## Configuration

Default config is in `config.yaml`. Create `config.local.yaml` (gitignored) for personal overrides:

```yaml
budget:
  max_usd: 20.00

server:
  port: 8081

download:
  max_messages: 50000
```

### Key settings

| Setting | Default | Description |
|---------|---------|-------------|
| `budget.max_usd` | 5.00 | Embedding spend limit |
| `embedding.model` | gemini-embedding-2-preview | Gemini model for embeddings |
| `embedding.dimensions` | 3072 | Vector dimensions |
| `attachments.max_file_size_mb` | 10 | Skip attachments larger than this |
| `attachments.max_pdf_pages` | 20 | Max PDF pages to render as images |
| `download.batch_size` | 25 | Gmail API batch size |
| `search.default_top_k` | 20 | Default number of search results |
| `server.port` | 8080 | Web UI port |

## Search ranking

Results are ranked by a weighted blend of signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Semantic similarity | 40% | Gemini embedding cosine similarity |
| BM25 keyword match | 15% | SQLite FTS5 exact word overlap |
| Recency | 15% | Exponential decay, 60-day half-life |
| Gmail labels | 12% | IMPORTANT, PERSONAL boost; PROMOTIONS penalty |
| You replied | 8% | Threads you participated in |
| Match density | 6% | Fraction of thread that matched |
| Thread size | 4% | Multi-message threads preferred |

## Cost

Embedding uses Gemini's `gemini-embedding-2-preview` model:
- Text: $0.20 per 1M tokens
- Images: $0.0001 per image

Typical cost: ~$2.50 per 10k messages with attachments. The `--budget` flag sets a hard spending limit.

## Architecture

```
gmail-search/
  src/gmail_search/
    config.py        — Config loading with local overrides
    cli.py           — Click CLI
    server.py        — FastAPI web UI
    store/           — SQLite: schema, models, queries, cost tracking
    gmail/           — OAuth, download, message parsing
    extract/         — Attachment text/image extraction (PDF, images)
    embed/           — Gemini embedding client and pipeline
    index/           — ScaNN index build and search
    search/          — Hybrid search engine with multi-signal ranking
  data/              — Local data (gitignored): DB, credentials, attachments, indexes
```

## Tech stack

- **Gmail API** — message download with OAuth2
- **Gemini embedding-2-preview** — text + multimodal embeddings (3072 dims)
- **ScaNN** — vector similarity search
- **SQLite FTS5** — keyword search (BM25)
- **FastAPI** — web UI
- **pymupdf** — PDF text extraction + page rendering
