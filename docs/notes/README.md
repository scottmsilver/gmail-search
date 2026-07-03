# Design notes & post-mortems

Point-in-time documents — design plans, incident write-ups, and evaluation
notes. They record why things are the way they are; the current behavior is
documented in code and the top-level README. Several are referenced from
code comments, so keep filenames stable.

| Note | What it covers |
|------|----------------|
| [OOM_INCIDENT_2026-04-18](OOM_INCIDENT_2026-04-18.md) | Watch-loop memory leak post-mortem; why the index builder runs in a subprocess. |
| [LAYER_ANALYSIS_2026-04-19](LAYER_ANALYSIS_2026-04-19.md) | Layer-by-layer architecture analysis of the retrieval stack. |
| [PER_CONVERSATION_SESSIONS_2026-04-26](PER_CONVERSATION_SESSIONS_2026-04-26.md) | Plan: pinning one deep-mode session per chat conversation. |
| [PER_USER_LOGIN_2026-04-27](PER_USER_LOGIN_2026-04-27.md) | Plan: multi-tenant identity (users table, session cookies, RLS scoping). |
| [SCANN_COMPACTION_2026-04-27](SCANN_COMPACTION_2026-04-27.md) | ScaNN index compaction evaluation; source of the manual-rerank variant. |
| [PROPOSITIONS_PROTOTYPE_2026-06-18](PROPOSITIONS_PROTOTYPE_2026-06-18.md) | Findings from the fact-extraction (propositions) prototype behind `find_facts`. |
