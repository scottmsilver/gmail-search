# Gmail Search

Local semantic search over your entire Gmail. Downloads your mailbox, embeds messages and attachments using Google's Gemini embedding model, and makes everything searchable through a Gmail-style web UI or CLI.

## Why this exists

Gmail search is keyword-only. It can't find "that email about the construction budget" unless those exact words appear. It can't search inside PDF attachments. It doesn't understand that your accountant's emails about "engagement letter" are related to your search for "tax documents."

Gmail Search fixes this with hybrid search: semantic understanding (what you mean) + keyword matching (what you said) + signals Gmail already computed (labels, importance) + your own engagement patterns (who you reply to). The result is search that actually finds what you're looking for, even with typos, even inside attachments, even when you can't remember the exact words.

**What makes it good:**

- **Hybrid ranking with 8 signals** — not just embeddings. Combines semantic similarity, BM25 keyword match, recency, Gmail labels (IMPORTANT/PERSONAL/PROMOTIONS), contact frequency, thread engagement, match density, and thread size. Each signal catches things the others miss.
- **Searches inside attachments** — PDFs get text extracted and page-rendered as images. Both text and images are embedded, so you can find a contract by describing what's in it.
- **Spell correction** — "draw requst" finds "draw request." Local SymSpell correction using a dictionary built from your own email corpus (0.1ms, no API call). Searches with both corrected and original query so nothing is lost.
- **Personal abbreviation expansion** — "KE board" finds "Kol Emeth board" because the system mines co-occurrence patterns from your emails at reindex time. Discovers abbreviations like KE=Kol Emeth, HOA=Colony, FRCO=Frank Rimerman automatically.
- **Structured filters** — `from:landmarks draw request` or `after:march invoice` work like Gmail operators, applied as SQL filters before ranking.
- **Temporal awareness** — "recent invoice" automatically boosts recency. "that email from last week" does the right thing.
- **Off-topic filtering** — drops clearly irrelevant results using an adaptive score gap threshold. Toggle in the UI.
- **LLM reranker** — conditionally reranks top 30 results using Gemini Flash Lite, but only when the top scores are tightly clustered (ambiguous). Clear winners skip the reranker (~250ms vs ~1.4s).
- **Hierarchical topic browsing** — emails are clustered into a semantic topic tree (built via recursive bisecting k-means on embeddings, auto-labeled by Gemini). Browse by topic in the sidebar: Personal > Family & Finance > Tax Documents.
- **Query embedding cache** — repeated queries are instant (62ms) via a persistent SQLite cache. No API call on cache hit.
- **Thread-grouped results** — shows conversations, not individual messages. Deduplicates repeat newsletters by sender+subject.
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
                    │                                              │
                    │  1. Spell correct (local SymSpell, 0.1ms)    │
                    │  2. Expand aliases (KE → kol emeth)          │
                    │  3. Parse filters (from:, after:, etc.)      │
                    │  4. Detect temporal intent (recent, last)     │
                    │  5. Check embedding cache (SQLite)            │
                    └─────────────┬───────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
   ┌──────────▼────────┐ ┌───────▼───────┐ ┌────────▼────────┐
   │  Vector Search     │ │ Keyword Search│ │ SQL Filters     │
   │  (ScaNN)           │ │ (FTS5 BM25)  │ │ (from/to/date)  │
   │                    │ │               │ │                  │
   │  Gemini embedding  │ │ Phrase 1.5x + │ │ Applied to       │
   │  (cached) → cosine │ │ individual +  │ │ thread_summary   │
   │                    │ │ both queries  │ │                  │
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
                    └─────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────────────┐
                    │         Conditional LLM Reranker             │
                    │  Only when top-5 score spread < 0.15         │
                    │  (ambiguous results need reranking)           │
                    └─────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────────────┐
                    │         Post-Processing                      │
                    │  Off-topic filter → Sender collapsing →      │
                    │  Thread grouping → Topic facets → Top K       │
                    └─────────────────────────────────────────────┘
```

### Data pipeline

```
Gmail API ──► SQLite ──► Extract ──► Embed ──► Index + Analyze
              (messages,   (PDF text,  (Gemini    (ScaNN vectors
               attachments  page imgs)  3072-dim)  + FTS5 keywords
               raw files)                          + thread summaries
                                                   + contact freq
                                                   + topic hierarchy
                                                   + term aliases
                                                   + spell dictionary
                                                   + query cache)
```

Each stage is idempotent. Crash at any point, re-run, and it picks up where it left off. The `update` command runs all stages in rolling batches of 500 messages, rebuilding indexes after each batch so search improves continuously.

### Precomputed tables (built at reindex time)

| Table | What | Why |
|-------|------|-----|
| `thread_summary` | Participants, labels, dates, message count per thread | Avoids N+1 queries during search (16x speedup) |
| `contact_frequency` | Log-scaled message count per sender | Boosts results from frequent correspondents |
| `topics` | Hierarchical topic tree (recursive bisecting k-means, auto-labeled) | Browse-by-topic sidebar, off-topic filtering |
| `message_topics` | Message → topic assignments (leaf + ancestors) | Client-side topic filtering |
| `term_aliases` | Abbreviation → expansion mappings mined from co-occurrence | "KE" → "kol emeth" query expansion |
| `query_cache` | Query text → embedding vector (persistent) | 200x speedup on repeated queries |
| `messages_fts` | FTS5 full-text index over subjects + bodies | BM25 keyword matching |
| `attachments_fts` | FTS5 index over attachment text | Keyword search inside PDFs |
| `spell_dictionary.txt` | Word frequency dictionary built from corpus | Local spell correction (SymSpell, 0.1ms) |

### Module map

```
src/gmail_search/
  config.py        — Config loading (config.yaml + config.local.yaml overrides)
  cli.py           — Click CLI: auth, download, sync, extract, embed, reindex,
                     search, cost, status, serve, update (rolling pipeline)
  server.py        — FastAPI: /api/search (with facets), /api/topics (tree),
                     /api/thread, /api/message, /api/attachment, /api/status
  store/
    db.py          — SQLite schema, FTS5 indexes, precomputed tables:
                     thread_summary, contact_frequency, topics (hierarchical),
                     term_aliases, query_cache, spell_dictionary
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
    client.py      — Gemini wrapper (text + multimodal), quote stripping,
                     BatchGeminiEmbedder for Batch API (50% cheaper)
    pipeline.py    — Batched embedding with retry, budget checks, idempotency
  index/
    builder.py     — ScaNN index: adaptive config (brute force < 100, AH > 100)
    searcher.py    — ScaNN query → (embedding_ids, scores)
  search/
    engine.py      — Full search pipeline: spell correct → alias expand →
                     parse filters → embed (cached) → ScaNN + BM25 →
                     merge → rank (8 signals) → conditional rerank →
                     off-topic filter → sender collapse → topic facets
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
| `gmail-search embed --batch-api` | Use Gemini Batch API (50% cheaper, for 100k+ messages) |
| `gmail-search reindex` | Rebuild all indexes: ScaNN, FTS, topics, aliases, contacts, spell dict |
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

Benchmarked on 20k messages / 32k embeddings:

| Metric | Value |
|--------|-------|
| Median search (clear results, cached query) | 62ms |
| Median search (clear results, new query) | 250ms |
| Median search (ambiguous, triggers reranker) | 1.4s |
| Index build time (ScaNN + FTS + topics + aliases) | ~90s |
| Messages downloaded/sec | ~5 (rate limited by Gmail API) |
| Embeddings/sec | ~50 (batched) |
| Spell correction | 0.1ms (local SymSpell) |
| Topic filter (client-side) | instant |

## Tech stack

- **Gmail API** — message download with OAuth2, batch requests, incremental sync
- **Gemini embedding-2-preview** — text + multimodal embeddings (3072 dims, 8192 token input)
- **Gemini Flash Lite** — conditional LLM reranker, topic auto-labeling
- **ScaNN** — Google's vector similarity search (asymmetric hashing, sub-ms queries)
- **SQLite + FTS5** — storage, keyword search (BM25), precomputed tables, query cache
- **SymSpell** — local corpus-trained spell correction (0.1ms per query)
- **FastAPI** — web UI and API
- **pymupdf** — PDF text extraction + page rendering

## Privacy

Your email stays on your machine. The only data sent externally:
- **Embedding text** sent to Gemini API (for generating vector representations)
- **Search queries** sent to Gemini API (for query embedding on cache miss, conditional reranking)
- **Topic summaries** sent to Gemini Flash Lite (one-time at reindex, for auto-labeling clusters)

Spell correction, abbreviation expansion, and query caching are fully local — no API calls.

No email content is stored by Google's API (Gemini API data is not used for training). Credentials and tokens are stored with restricted file permissions (0600). The `data/` directory containing your email database, attachments, and indexes is gitignored.
