# Deep analysis and deep battle implementation plan

**Goal:** Make all chat requests deep analysis; replace shallow battle with two isolated deep runs and remove obsolete shallow-chat code.

**Architecture:** Keep the existing deep stream for single runs. A separate deepBattle module invokes the Python analyze endpoint twice with independent sessions (no shared conversation ID), forwards the user's cookie, collects progress/cost/final/error events, and publishes a single persistent battle message. Both runs receive identical textual conversation context. Retain blind voting and legacy battle rendering. Each variant identifies its backend and explicit model. Settings no longer offer a deep toggle or ineffective thinking controls. Remove other standalone browsing pages if confirmed by the user, retaining settings and tool/data APIs needed by deep agents and citation drawers.

**Tech Stack:** Next.js/TypeScript, AI SDK UI streams, Python/FastAPI.

- [x] Add failing tests for deep variant selection, streaming outcomes, isolation, auth forwarding, errors, and conversation context.
- [x] Implement deepBattle helper and replace shallow chat/battle request branches. Await initial conversation persistence before running to prevent overwrite races.
- [x] Remove obsolete mode settings and controls; show battle progress and per-side cost; retain backend/model selection for single runs.
- [x] Remove unused shallow-chat helpers and requested standalone pages after checking references. Preserve prior Muse Spark changes and local secrets.
- [x] Verify unit tests, Python relevant tests, TypeScript and production build; review changes; reload services and check HTTP responses.

Validation: 81 Python tests and 49 web tests passed; Ruff, TypeScript via the production build, and `next build` passed. Battle integration tests use simulated backend streams, including failures and persistence.

Scope correction: standalone Search and its navigation, filters, results, and thread previews are restored at the user’s request. Chat remains deep-only with optional deep battles.

Backend simplification: new requests expose only Claude Code and Pi. Pi models are Muse Spark 1.3, Gemini 3.8 Flash, and Claude Opus 5, all via OpenRouter. Removed backend preferences migrate to Pi/Muse; historical battle labels remain readable. All three models passed live Pi smoke tests.
