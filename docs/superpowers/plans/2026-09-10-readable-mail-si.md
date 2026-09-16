# Readable mail and Gemini system instructions

Goal: keep Gemini 3.8 Flash as the default Pi model and reduce the avoidable
reading/orchestration overhead observed in the two live pilots.

Approved direction: improve SI and try email predigestion, including HTML to
Markdown. Start with deterministic conversion rather than paid summaries.

Design: convert at the agent tool boundary. Preserve stored originals and the
web mail API. Default thread reads provide readable Markdown in body_text,
with format/source metadata and cite_ref. Allow selecting message IDs and
explicit raw reads; report truncation and continuation offsets. Preserve dates,
amounts, table relationships, links and quoted correspondence. Strip scripts,
styles and layout markup. No network requests during conversion.

Replace the SI's mandatory phase plans, per-call narration, and substr advice
with question-scaled discovery, batched readable reads, accurate citations,
source verification, bounded error recovery, and stopping/coverage criteria.
Retain optional workflow capabilities; no required delegation. Gemini becomes
the default only where the application has not explicitly selected another model.

Implementation checklist:
- [x] Add failing converter tests (receipt tables, HTML-only mail, quoted
      corrections, plain fallback, malformed HTML and dangerous markup).
- [x] Implement converter in agents/mail_content.py; benchmark on private
      original receipts from the eval, recording size/time and key facts.
- [x] Add failing tool tests, then integrate markdown/raw formats, selected
      message reads, bounded continuation, citation and truncation metadata.
- [x] Update MCP schema/descriptions and forwarding tests.
- [x] Rewrite SI and default model with focused regression checks.
- [x] Review code and run relevant tests; integrate only owned changes into
      the original dirty workspace. Document activation and eval commands.

Validation: synthetic tests plus source-backed receipt checks, followed by
isolated end-to-end evaluation if feasible. Keep the prior pilot artifacts
immutable. No mailbox backfill or stored-content rewrite is required.
