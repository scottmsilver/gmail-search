---
name: timelines
description: Use when a Gmail question asks for chronology, changing plans, deadlines, or events whose dates differ from when the messages were sent.
---

# Timelines

Separate message sent time from event time. Resolve relative dates such as “next Tuesday” against the original author's dated message and timezone when available; forwarded text retains its original context. Label ambiguity instead of silently choosing a date.

For each useful event, retain the event date or range, message sent date, what changed, whether it was proposed/confirmed/cancelled/completed, and the supporting message ID. Deduplicate quoted copies. Sort by event time for an event chronology, or by sent time when explaining how a plan changed.

For example, a June 12 message saying “we shipped yesterday” supports a reported June 11 shipment, not a June 12 shipment. An earlier estimated ship date remains an estimate.

A compact table can help when several changes matter. Fetch surrounding messages or search a wider date range when a gap affects the answer. Direct tools, `mcpScript`, and optional delegated research are all valid; no separate planning stage is required.
