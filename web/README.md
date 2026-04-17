# Gmail Search — Chat

Next.js chatbot that answers questions about your Gmail using the main
Python backend as a RAG + metadata query layer.

## What it does

An agent (Gemini 3.1 Flash Lite) with two tools:

- **`search_emails`** — semantic + keyword hybrid search (wraps `/api/search`).
- **`query_emails`** — structured metadata filter: sender, subject, date
  range, label, has_attachment (wraps `/api/query`).

Answers stream with inline `[ref:THREAD_ID]` citations that render as
chips linking back to the Python UI's thread view.

## Run

1. Start the Python backend (defaults to port 8080):

   ```bash
   cd ..
   gmail-search serve --port 8080
   ```

2. Configure env:

   ```bash
   cp .env.local.example .env.local
   # edit .env.local if your Python backend runs on a different port
   ```

3. Make sure `GEMINI_API_KEY` is exported (or add it to `.env.local`):

   ```bash
   export GEMINI_API_KEY="..."
   ```

4. Install and start:

   ```bash
   npm install
   npm run dev
   ```

   Open http://localhost:3000.

## Files

- `app/api/chat/route.ts` — streaming agent endpoint (NDJSON).
- `lib/agent.ts` — Gemini function-calling loop.
- `lib/tools.ts` — tool schemas + executors.
- `lib/backend.ts` — thin wrappers around the Python API.
- `components/Chat.tsx` — chat UI with streaming + citations.

## Env vars

| Var | Purpose |
| --- | --- |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Gemini SDK key. |
| `PYTHON_API_URL` | Where FastAPI is listening. Default `http://127.0.0.1:8080`. |
| `NEXT_PUBLIC_PYTHON_UI_URL` | URL that citation chips link to. |
