# Gmail Search

Local semantic search over your entire Gmail. Downloads your mailbox, embeds messages and attachments using Google's Gemini embedding model, and makes everything searchable through a Gmail-style web UI or CLI.

## Why this exists

Gmail search is keyword-only. It can't find "that email about the construction budget" unless those exact words appear. It can't search inside PDF attachments. It doesn't understand that your accountant's emails about "engagement letter" are related to your search for "tax documents."

Gmail Search fixes this with hybrid search: semantic understanding (what you mean) + keyword matching (what you said) + signals Gmail already computed (labels, importance) + your own engagement patterns (who you reply to). The result is search that actually finds what you're looking for, even with typos, even inside attachments, even when you can't remember the exact words.

**What makes it good:**

- **Hybrid ranking with 8 signals** — not just embeddings. Combines semantic similarity, BM25 keyword match, recency, Gmail labels (IMPORTANT/PERSONAL/PROMOTIONS), contact frequency, thread engagement, match density, and thread size. Each signal catches things the others miss.
- **Searches inside attachments** — PDFs get text extracted and page-rendered as images. Both text and images are embedded, so you can find a contract by describing what's in it.
- **Spell correction** — "draw requst" finds "draw request." Searches with both corrected and original query so nothing is lost.
- **Structured filters** — `from:landmarks draw request` or `after:march invoice` work like Gmail operators, applied as SQL filters before ranking.
- **Temporal awareness** — "recent invoice" automatically boosts recency. "that email from last week" does the right thing.
- **LLM reranker** — top 30 results are reranked by Gemini Flash Lite, catching relevance subtleties that embeddings and keywords miss.
- **Thread-grouped results** — shows conversations, not individual messages. Deduplicates repeat newsletters.
- **Rolling pipeline** — `update` processes in batches of 500: download, extract, embed, reindex. Search gets better continuously as it runs, not just at the end.
- **Cost-controlled** — hard budget limit, per-operation cost tracking, ~$5 per 20k messages.
- **Fully local** — your email never leaves your machine. The SQLite DB, attachments, and indexes are all in a gitignored `data/` directory. Only embedding vectors are sent to Gemini's API.

## Quick start

### Prerequisites

- Python 3.11+
- A Google Cloud project with the Gmail API enabled
- A Gemini API key (set as `GEMINI_API_KEY` or `GOOGLE_API_KEY` env var)

### 1. Install

```bash
git clone https://github.com/scottmsilver/gmail-search.git
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

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                  User Query                  │
                    └─────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────────────┐
                    │         Query Processing Layer               │
                    │  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
                    │  │  Spell   │ │  Parse   │ │  Temporal   │ │
                    │  │  Correct │ │  Filters │ │  Detection  │ │
                    │  │(Flash Lt)│ │(from: ..)│ │(recent,last)│ │
                    │  └──────────┘ └──────────┘ └─────────────┘ │
                    └─────────────┬───────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
   ┌──────────▼────────┐ ┌───────▼───────┐ ┌────────▼────────┐
   │  Vector Search     │ │ Keyword Search│ │ SQL Filters     │
   │  (ScaNN)           │ │ (FTS5 BM25)  │ │ (from/to/date)  │
   │                    │ │               │ │                  │
   │  Gemini embedding  │ │ Phrase match  │ │ Applied to       │
   │  → cosine sim      │ │ + individual  │ │ thread_summary   │
   │                    │ │ + both queries│ │                  │
   └──────────┬─────────┘ └───────┬───────┘ └────────┬────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                    ┌─────────────▼───────────────────────────────┐
                    │         Merge + Multi-Signal Ranking         │
                    │                                              │
                    │  Similarity (40%) + BM25 (15%) +             │
                    │  Recency (15%, dynamic) + Labels (12%) +     │
                    │  Contact Freq (8%) + Replied (8%) +          │
                    │  Match Density (6%) + Thread Size (4%)       │
                    │                                              │
                    │  + Temporal boost shifts weight to recency   │
                    │  + Structured filters remove non-matches     │
                    └─────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────────────┐
                    │         LLM Reranker (Flash Lite)            │
                    │  Top 30 candidates reranked for relevance    │
                    └─────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────────────┐
                    │         Post-Processing                      │
                    │  Thread grouping → Sender collapsing →       │
                    │  Top K results                                │
                    └─────────────────────────────────────────────┘
```

### Data pipeline

```
Gmail API ──► SQLite ──► Extract ──► Embed ──► Index
              (messages,   (PDF text,  (Gemini    (ScaNN vectors
               attachments  page imgs)  3072-dim)  + FTS5 keywords
               raw files)                          + thread summaries
                                                   + contact freq)
```

Each stage is idempotent. Crash at any point, re-run, and it picks up where it left off. The `update` command runs all stages in rolling batches of 500 messages, rebuilding indexes after each batch so search improves continuously.

### Precomputed tables (built at reindex time)

| Table | What | Why |
|-------|------|-----|
| `thread_summary` | Participants, labels, dates, message count per thread | Avoids N+1 queries during search (16x speedup) |
| `contact_frequency` | Log-scaled message count per sender | Boosts results from frequent correspondents |
| `messages_fts` | FTS5 full-text index over subjects + bodies | BM25 keyword matching |
| `attachments_fts` | FTS5 index over attachment text | Keyword search inside PDFs |

### Module map

```
src/gmail_search/
  config.py        — Config loading (config.yaml + config.local.yaml overrides)
  cli.py           — Click CLI: auth, download, sync, extract, embed, reindex,
                     search, cost, status, serve, update (rolling pipeline)
  server.py        — FastAPI: /api/search, /api/thread, /api/message,
                     /api/attachment, /api/status
  store/
    db.py          — SQLite schema, FTS5, thread_summary, contact_frequency
    models.py      — Dataclasses: Message, Attachment, EmbeddingRecord, CostRecord
    queries.py     — CRUD, FTS search (phrase + individual + dual-query)
    cost.py        — Per-operation cost tracking with budget enforcement
  gmail/
    auth.py        — OAuth2 (gmail.readonly scope, token with 0600 perms)
    client.py      — Batch download with rate limit retry, incremental sync
    parser.py      — Gmail API response → Message + attachment metadata
  extract/
    __init__.py    — Dispatcher: mime_type → extractor
    pdf.py         — pymupdf: text extraction + page rendering (150 DPI PNG)
    image.py       — Passthrough for jpg/png/gif attachments
  embed/
    client.py      — Gemini wrapper (text + multimodal), quote stripping
    pipeline.py    — Batched embedding with retry, budget checks, idempotency
  index/
    builder.py     — ScaNN index: adaptive config (brute force < 100, AH > 100)
    searcher.py    — ScaNN query → (embedding_ids, scores)
  search/
    engine.py      — Query processing, hybrid ranking, LLM reranker,
                     thread grouping, sender collapsing
```

## Commands

| Command | Description |
|---------|-------------|
| `gmail-search auth` | Run OAuth flow, save token |
| `gmail-search update` | Full pipeline in rolling batches: download, extract, embed, reindex |
| `gmail-search download` | Download messages only |
| `gmail-search sync` | Incremental sync (new messages since last download) |
| `gmail-search extract` | Extract text/images from attachments |
| `gmail-search embed` | Embed unembedded messages and attachments |
| `gmail-search embed --force` | Re-embed all messages (e.g., after changing text processing) |
| `gmail-search reindex` | Rebuild ScaNN + FTS + thread summary + contact frequency |
| `gmail-search search "query"` | Search from the command line |
| `gmail-search serve` | Start the web UI |
| `gmail-search status` | Show message count, embeddings, cost |
| `gmail-search cost --breakdown` | Show embedding spend by operation |

## Configuration

Default config is in `config.yaml`. Create `config.local.yaml` (gitignored) for personal overrides:

```yaml
budget:
  max_usd: 20.00

search:
  rerank: false  # disable LLM reranker for faster results

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
| `search.rerank` | true | Enable LLM reranker (adds ~1.3s latency, improves relevance) |
| `search.default_top_k` | 20 | Default number of search results |
| `attachments.max_file_size_mb` | 10 | Skip attachments larger than this |
| `attachments.max_pdf_pages` | 20 | Max PDF pages to render as images |
| `download.batch_size` | 25 | Gmail API batch size |
| `server.port` | 8080 | Web UI port |

## Search ranking

Results are ranked by a weighted blend of signals, with dynamic adjustment based on query intent:

| Signal | Weight | Description |
|--------|--------|-------------|
| Semantic similarity | 40% (dynamic) | Gemini embedding cosine similarity |
| BM25 keyword match | 15% | FTS5 phrase match (1.5x boost) + individual terms |
| Recency | 15% (dynamic) | Exponential decay, 60-day half-life |
| Gmail labels | 12% | IMPORTANT, PERSONAL boost; PROMOTIONS penalty |
| Contact frequency | 8% | Log-scaled message count per sender |
| You replied | 8% | Threads you participated in |
| Match density | 6% | Fraction of thread that matched |
| Thread size | 4% | Multi-message threads preferred |

"Dynamic" means the weight shifts based on temporal intent in the query. "recent invoice" moves up to 35% of the similarity weight into recency.

On top of the weighted blend, the LLM reranker reorders the top 30 candidates using Gemini Flash Lite, catching relevance nuances that statistical signals miss.

## Cost

Embedding uses Gemini's `gemini-embedding-2-preview` model:
- Text: $0.20 per 1M tokens
- Images: $0.0001 per image

| Scale | Embedding cost | Search cost (with reranker) |
|-------|---------------|---------------------------|
| 10k messages | ~$2.50 | ~$0.0003/query |
| 20k messages | ~$5.00 | ~$0.0003/query |
| 100k messages | ~$25.00 | ~$0.0003/query |

The `--budget` flag sets a hard spending limit. `gmail-search cost --breakdown` shows exactly where money went.

## Performance

Benchmarked on 20k messages / 30k embeddings:

| Metric | Value |
|--------|-------|
| Median search latency (no reranker) | 230ms |
| Median search latency (with reranker) | 1.6s |
| Index build time | 1.5s |
| Messages downloaded/sec | ~5 (rate limited by Gmail API) |
| Embeddings/sec | ~50 (batched) |

## Tech stack

- **Gmail API** — message download with OAuth2, batch requests, incremental sync
- **Gemini embedding-2-preview** — text + multimodal embeddings (3072 dims, 8192 token input)
- **Gemini Flash Lite** — spell correction + LLM reranker
- **ScaNN** — Google's vector similarity search (asymmetric hashing, sub-ms queries)
- **SQLite + FTS5** — storage, keyword search (BM25), precomputed tables
- **FastAPI** — web UI and API
- **pymupdf** — PDF text extraction + page rendering

## Privacy

Your email stays on your machine. The only data sent externally:
- **Embedding text** sent to Gemini API (for generating vector representations)
- **Search queries** sent to Gemini API (for query embedding, spell correction, reranking)

No email content is stored by Google's API (Gemini API data is not used for training). Credentials and tokens are stored with restricted file permissions (0600). The `data/` directory containing your email database, attachments, and indexes is gitignored.
