# Pi citation normalization

Approved scope: resolve message IDs to thread IDs automatically and validate
email citations before displaying the final Pi answer.

`agents/citations.py` resolves `[ref:...]` identifiers against the session
owner's messages with parameterized, batched queries. Existing exact thread IDs
win over message IDs. Unknown IDs become `[source unavailable]`; a missing
session owner never triggers a mailbox query. Artifact citations are untouched.
Citations alone inside inline code are unwrapped so the UI can render them.

`runtime_pi._finish_ok` normalizes after artifact handling and before emitting
both draft/final events and saving `final_answer`. Original model transcripts
and historical persisted answers are not rewritten. This checks source existence
and ownership, not whether a source supports an adjacent factual claim.

Validation: six focused tests cover mapping, unknown/foreign IDs, missing owners,
inline code, no-query answers and identical normalized event/persisted text.
Together with runtime tests: 46 passed. Offline replay of the two prior eval
answers corrected all 10 message-ID purchase citations, preserved valid travel
citations, and left no unresolved refs. No new paid eval or deployment performed.

Model-based reading remains a proposal: an optional question-specific reader
receives a bounded batch of Markdown emails and returns attributable excerpts,
contradictions and gaps. Validate excerpt provenance against supplied messages;
retain full-message access. Measure total reader-plus-parent cost and latency
against direct Markdown reading before adopting it. This does not independently
solve retrieval coverage.
