# Pi workflow evaluation

The CLI compares the existing `baseline` and optional `workflow` profiles with
`openrouter/meta/muse-spark-1.3`, medium thinking, and a fixed reference date
(default `2026-09-10`). It does not assign workflows per question. Historical
reporting is read-only; live model calls require an explicit billing opt-in.

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_agent_workflows.py historical \
  --sessions-root deploy/pi/sessions \
  --output data/agent-evals/history
```

The default exports the latest five session questions and one simple lookup
control into private `cases.json`. Use repeated `--session-id` flags to fix the
five cases explicitly, and `--user-id` to restrict a tenant. Historical sessions
can contain different models or repeated questions: inspect the private manifest
before comparing. The cost ledger records the actual historical model.
Conversation transcripts are sliced to each DB turn's start/end timestamps.
For manually sliced, single-turn transcripts use `--transcript-dir`, naming each
file `<session-id>.jsonl`. Malformed JSONL fails explicitly.

All report destinations must be under ignored `data/agent-evals`. Files are
created exclusively with mode 0600; existing reports are not overwritten.
Prompts, answers, and case manifests must not be committed. Source documents
contain no personal case prompts. Runtime session/workspace artifacts also use
fresh IDs in the existing runtime's ignored storage directories.

To deliberately run a billed comparison against the configured MCP service:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_agent_workflows.py live \
  --cases data/agent-evals/history/cases.json \
  --user-id YOUR_USER_ID --repeats 3 --allow-paid \
  --output data/agent-evals/comparison
```

This runs six cases × three repetitions × two profiles: 36 parent runs, plus
whatever optional child calls the model chooses. `--profile baseline` or
`--profile workflow` limits the profiles. Model and thinking are fixed, and each
run gets a new session, conversation, and workspace. Profile order alternates
between repetitions to reduce systematic warm-cache bias. The live path calls
`pi_run(workflow_profile=...)`; it does not rebuild, restart, or deploy services.
Configure an isolated sandbox/MCP environment first if production corpus access
is undesirable. It records database sessions and uses the normal session-bound
MCP authentication. Run failures are saved and subsequent runs continue.

Reports include elapsed seconds, the union of tool intervals, call counts,
model turns, recursive tool-error detection (including JSON nested in batch
results), repeated identical reads, recorded result bytes, citations, usage,
and the cost ledger where available. Durable full MCP events are preferred over
clipped UI mirrors so calls/results are not double-counted. Pi orchestration
calls have a separate count. A duplicate read means the same read tool and exact
arguments; it does not prove redundant evidence or detect overlapping batches.

`tool_wall_source` identifies direct Pi timing versus DB event timestamps. The
latter includes recording latency. Missing/ambiguous start/end pairs produce
`null`, never a manufactured zero. Persisted Pi message JSONL alone cannot
supply exact execution intervals. Token/cache/cost fields remain `null` when
unavailable. Workflow telemetry supplies separate parent and child observed
usage; `observed_total` is their sum. `total` remains unknown unless a complete `workflow_trace_summary`
and usage for every observed model message are present. Events with the same
`event_id` are counted once. The raw live cost callbacks are retained
in `recorded_costs`; do not add them to transcript usage again.

Every result has an explicit pending-review rubric for correctness,
completeness, citation support, temporal grounding, and uncertainty. Review
answers against original message evidence, preferably blinded to profile;
there is no score based on verbosity, length, or previous unverified answers.
Report the quality review alongside distributions of duration, cost, turns,
errors, and payload volume across repetitions. Historical comparisons are
observational, especially when models/questions differ.

The CLI records the corpus maximum history ID/latest date and database
reachability; retrieval backend health is `not_probed` to avoid loading indexes.
A before/after watermark mismatch flags changing data, but an unchanged
watermark does not prove a stable corpus. Freeze the corpus and backend versions
for publishable comparisons. No paid evaluation is part of the unit tests.

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/test_workflow_eval.py
```

## Record a source-backed quality review

These modes are entirely offline and never call a model or open the database:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_agent_workflows.py review-template \
  --report data/agent-evals/comparison/comparison.json \
  --output data/agent-evals/review-draft
# Edit the private review-template.json after inspecting original source emails.
PYTHONPATH=src .venv/bin/python scripts/eval_agent_workflows.py review-import \
  --report data/agent-evals/comparison/comparison.json \
  --review data/agent-evals/review-draft/review-template.json \
  --output data/agent-evals/reviewed
```

For each run, identify the reviewer, change status to `reviewed`, record notes
for every rubric dimension, and list the checked claims. Each claim requires
`claim`, `cited_id`, `checked_excerpt`, and a `verdict` of `supported`,
`contradicted`, or `uncertain`. Add explanatory `notes` as needed. Use actual
source excerpts; preserve disagreements and uncertainty. Remove unfinished
runs from the imported review list; they remain pending in the output.
The importer validates structure and run identity, not the truth or provenance
of human-entered excerpts. Private historical reports now include the answer
so the template can show the exact text being evaluated.

The reviewed report retains per-run evidence checks and summarizes checked
claim/verdict counts by profile. It provides no automatic quality score. Compare
matched cases and rubric notes; claim counts alone are sensitive to which
claims a reviewer chose to inspect.

Redacted workflow telemetry may contain result byte counts without payloads
or argument values. Byte-count coverage and payload completeness are reported
separately. Unknown arguments cannot establish duplicate reads; check
`duplicate_read_calls_with_args`, `duplicate_read_calls_total`, and
`duplicate_reads_complete` before interpreting the duplicate count. Recursive
nested-error counts require payloads; top-level error coverage is recorded
separately. The fixed reference date applies to the user's question; relative
dates quoted in emails remain anchored to each source email's sent date.

Workflow billing uses per-turn root and child telemetry, including successful
compaction and branch-summary usage, instead of cumulative Pi session stats.
Pi session stats can include child usage projected into tool results; adding
those stats to child costs would count delegated work twice. Accounting runs in
the finalizer so cancellation still records observed parent and child usage.
Failed/cancelled compaction, or a summary without usage, produces an explicit
accounting gap and prevents claims of a complete total. Recorded usage on an
incomplete trace is a lower bound. Summary calls are counted separately from
ordinary assistant turns. Citation counts include the application's `[ref:ID]`
syntax as well as email/thread/art references.
