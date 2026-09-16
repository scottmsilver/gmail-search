# Pi workflow capabilities and eval

Approved scope: give Spark optional direct retrieval, mcpScript, task tracking,
and managed delegation; measure choices against the existing Pi baseline.
Do not run paid evaluations or alter the running production service this task.

## Design

Keep a baseline profile that preserves the existing prompt/tool configuration.
Add a workflow profile loading pinned pi-subagents, a persistent todo tool,
and focused Gmail skills. Pin parent and child model to the selected Spark
model during comparisons. The model chooses its own decomposition and tools.
Children must receive session-bound Gmail authentication, source citation rules,
and bounded execution. Track child lifecycle/usage and defer turn completion
until owned child work has settled and parent has consumed its results.

## Implementation

- [x] Inspect pinned extension APIs and select a compatible release.
- [x] Build eval CLI: historical trace report, fixed cases, opt-in live baseline
      and workflow runs with fresh sessions and JSON reports. Measure elapsed,
      tool wall time, sequential turns, errors, repeated reads, payload volume,
      parent/child tokens, citations. Quality is an explicit review rubric,
      never inferred from response length or existing unverified answers.
- [x] Add workflow extension/configuration, optional tasks and focused skills.
- [x] Add driver profile selection and lifecycle/event handling for background
      children, cancellation, token cleanup, cost and evidence aggregation.
- [x] Unit and integration checks with synthetic provider responses; no paid
      model calls. Test in an isolated container if needed.
- [x] Run historical eval against existing transcripts without modifying them.
- [x] Review and integrate only task changes into original dirty workspace.

## Eval controls

Use openrouter/meta/muse-spark-1.3, medium thinking, a fixed reference date,
separate conversation IDs/workspaces, identical questions and stable data when
comparing. Historical results are observational, not a controlled A/B baseline.
Freeze a corpus snapshot for publishable comparisons; record backend health and
corpus watermark for live runs. Include a simple lookup as a delegation control.
Save private prompts, answers and report artifacts only in ignored data paths.
Report all model costs, including children. No hardcoded workflow per question.

Validation: 98 focused Python tests, 12 Node tests, container image build, and
real synthetic completion/cancellation smoke passed. No paid eval or production
activation performed. Historical findings are in ignored
`data/agent-evals/historical-20260910-final/findings.md`.
