# Gmail Search — Deep analysis

Every chat request runs deep analysis. Choose Claude Code or Pi in the model picker. Pi offers Muse Spark 1.3,
Gemini 3.8 Flash through Google, and Claude Opus 5 through Anthropic. Muse Spark
1.3 currently uses OpenRouter until a direct Meta credential is configured.

Enable **Deep analysis battle** to compare two randomly selected backend/model
combinations. Both receive the same conversation context and run independently.
The UI shows progress, cost, and answers side by side; vote to reveal identities
and inspect their work. Both answers are saved for later follow-up questions.
Historical battles remain readable. Standalone Search, citation previews, and
settings remain available. Inbox and Priority pages have been removed. `/deep` redirects home.

## Run

Start the Python service with the desired deep backends configured. Copy
`.env.local.example` to `.env.local` and set `PYTHON_API_URL`, then run
`npm install` and `npm run dev`. Open http://localhost:3000.

Provider credentials belong to the Python service or the relevant sandbox;
Next.js does not call a model provider directly. See `../deploy/pi/README.md`
for OpenRouter configuration.

## Verify

```bash
node scripts/test-units.mjs
node --import tsx scripts/test-deep-battle.mjs
node --import tsx scripts/test-deep-route.mjs
npm run build
```

The battle tests use simulated upstream streams without paid model calls.

### Dedicated public frontend

Set `GMS_PUBLIC_ORIGIN=https://gms.oursilverfamily.com` only on the dedicated
public Next instance, with `PYTHON_API_URL` pointing at the isolated public
backend. Presence activates the boundary; an empty or invalid value fails
closed. The private instance must leave this variable unset.

Every public API handler enforces a route/method allowlist, exact Host/Origin,
backend session authentication, and a 256 KiB body limit counted from the stream
before parsing or logging. Middleware also applies host/header checks to pages.
The reverse proxy must remove incoming Next internal routing headers before
Next handles the request, especially `x-middleware-subrequest`. OAuth login and
callback GETs work without an existing session. Admin, jobs, logs, diagnostic,
battle, and direct agent APIs are unavailable publicly.

The public chat UI presents Gemini with Gmail retrieval tools. It ignores saved
private model settings, disables battles, and relies on the public backend to
force its isolated retrieval-only runtime. Public questions are limited to
16,000 characters and histories to 100 messages. Responses use private/no-store,
no-referrer, and nosniff headers.

Run the boundary regression tests with:

```sh
node --import tsx scripts/test-public-boundary.mjs
node --import tsx scripts/test-deep-route.mjs
node --import tsx scripts/test-deep-battle.mjs
npx tsc --noEmit
```
