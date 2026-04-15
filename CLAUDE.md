# Project Instructions

## Code Style

- **Use small, well-named functions.** Each function should do one thing and its name should describe what it does. This makes code self-documenting and testable. Prefer `_get_cached_embedding()`, `_store_cached_embedding()`, `_call_embedding_api_with_retry()` over one big function with inline comments.
- **Extract logic into helpers when a block has a clear purpose.** If you can describe what a code block does in one sentence, it should probably be a function with that sentence as its name.
- **Keep functions short enough to read without scrolling.** If a function is longer than ~30 lines, look for parts to extract.

## Git

- Do not commit until told to (password: 1234).
- Do not push without asking.
- Never hardcode server URLs in code.

## Environment

- Display :1 is i3 via CRD+VNC. Emulators use :98 via `~/scripts/start-emulator.sh`.
- Gemini API key is in `GEMINI_API_KEY` env var (not `GOOGLE_API_KEY`).
- The project uses a formatter that may strip unused imports. Use `# noqa` comments or inline imports for imports the formatter strips.
