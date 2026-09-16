# Run metadata query service

`RunMetadataService` restores structured mail filtering through the existing
restricted `QueryGateway`. It needs no embedding provider, native search index,
new reader grant, privileged database connection, DDL or bootstrap identity.
HTTP and guest/MCP mounting are separate composition work.

## Trusted API and model contract

Construct `RunMetadataService(capabilities, gateway)` with the same query gateway
and shared `DataAdmission` used by SQL and retrieval. The service authorizes
`retrieval` / `query.emails`; the capability supplies the owner. Its public method
is `query_emails(token, *, sender='', subject_contains='', date_from='',
date_to='', label='', has_attachment=None, order_by='date_desc', limit=20)`.
`authorize(token)` is available for HTTP authentication before reading the body
and for the final transport publication check.

The service accepts only these named arguments. Sender and subject are limited
to 1,000 characters; label to 256. Empty strings mean no filter. Text must be
valid UTF-8 without NUL. Dates must be empty or valid `YYYY-MM-DD`, with the lower
bound no later than the upper. Attachment selection is exactly boolean or null;
ordering is `date_desc` or `date_asc`; limit is an integer from 1 to 100.

Each result preserves `thread_id`, `subject`, decoded `participants` string
list, `message_count`, `date_first`, `date_last`, `snippet` and
`cite_ref=thread_id`. Snippets contain at most 500 characters. Null message body
becomes an empty snippet. Other malformed result fields produce a fixed error;
reader diagnostics or raw SQL never become response fields.

## Selection and compatibility

A single fixed SQL statement groups matching messages by owner and thread, picks
`limit + 1` threads, and joins thread summaries and a deterministic latest-message
row. Its CTEs, correlated `EXISTS`, window function and joins all compile through
the unchanged restricted analytical compiler. All values use Psycopg literal
quoting before the compiler rebuilds the query with bound parameters. No input
is treated as SQL syntax. RLS restricts every parent relation; explicit owner
predicates and owner-qualified message/thread joins add a separate check.

Threads are ordered by the greatest date among **matching messages**. The snippet
comes from the latest message among **all messages in that selected thread**.
A thread with a recent unrelated message therefore does not outrank a newer
matching thread merely because its summary's `date_last` is recent. Ties use
thread ID; latest-message ties use message ID. Attachment predicates apply to
the matching message: `has_attachment=False` can match a thread that contains
another message with attachments. Presence means an attachment row exists,
regardless of fetch state. Composite owner/message joins prevent colliding IDs
from changing another owner's attachment predicate.

This restoration deliberately preserves the inherited server filter behavior:

- Sender and subject use case-sensitive PostgreSQL `LIKE`, including input `%`,
  `_` and backslash pattern semantics. They are not normalized `ILIKE` filters.
- Labels use the existing quoted-label `LIKE` pattern over stored JSON text;
  this does not introduce exact decoded JSON membership semantics.
- Dates compare stored text using `>= date_from` and
  `<= date_to + 'T23:59:59+00:00'`. These are not normalized timestamp comparisons;
  fractional seconds and noncanonical timezone text retain the existing edge
  behavior at the final-day boundary.

## Coverage, bounds and lifecycle

The response is `{'results': [...], 'coverage': {...}}`. Coverage contains
`selection_complete`, `returned_threads`, `limit`, and sorted closed `reasons`:
`result_limit`, `query_budget`, `missing_summary`, or `response_bytes`.
One extra selected thread establishes `result_limit`; exactly `limit` rows with
no extra row do not. Missing summaries are reported rather than silently
implying complete retrieval. No total mailbox count is inferred. If an incomplete
selection yields no usable rows, the service raises
`Metadata query could not complete within its limits.`

At most 101 selected rows leave SQL, before the gateway's independent row and
byte limits (2,000,000 bytes by default). `substr` bounds bodies in PostgreSQL.
A final 4 MiB cap includes coverage. Atomic rows are never clipped to fit; the
snippet is the established fixed preview, not a full-body completeness claim.

One service deadline covers 30 seconds; the gateway can enforce a stricter
query deadline. Token revocation cancels in-flight work. Repeated cancellation
waits for query and authorization-thread cleanup. All owned tasks finish before
a final fresh authorization/deadline check, with no subsequent await in the
service. HTTP composition must additionally use the reviewed transport helper's
fresh authorization callback after transport cleanup. Neither boundary makes
later ASGI/socket scheduling atomic with revocation or recalls already sent data.

## Verification

The focused suite has 30 tests, including two actual PostgreSQL owner cases.
The fixture creates and removes its own random `gms_gateway_test_*` database and
reader roles; it never modifies a production mailbox. Both owners have colliding
thread/message/attachment IDs. Tests distinguish matching-message order from
latest-all-thread snippets, prove per-message attachment-false behavior, verify
shared admission rejection, and cover legacy case/date behavior and citations.
Other tests cover malformed output, limit-plus-one, incomplete empty selections,
atomic output caps, revocation, repeated cancellation and authorization/deadline
checks after delayed watcher cleanup. Initial missing-module tests and nullable
snippet compatibility were verified failing before their fixes.

```bash
GMS_GATEWAY_TEST_DSN="$SYNTHETIC_GATEWAY_DSN" PYTHONPATH=src \
  /home/ssilver/development/gmail-search/.venv/bin/python -m pytest -q \
  tests/test_gateway_metadata_service.py
```
