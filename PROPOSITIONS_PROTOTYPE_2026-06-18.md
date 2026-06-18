# Proposition (fact) extraction prototype — findings

**Date:** 2026-06-18
**Status:** validated prototype, **isolated** from the live search/index path (own
`propositions` table, own brute-force retrieval, no wiring into `serve`/ScaNN).
**Code:** `src/gmail_search/propositions.py` · **frozen benchmark:** `prop_eval_set` table (800 msgs).

> Note: the eval set and extracted facts contain personal mailbox data and live
> ONLY in Postgres — they are never committed. This doc is scrubbed of personal
> specifics; examples below are generic.

## Why
Most real questions over the mailbox aren't keyword search — they're *enumerate /
aggregate / track* ("find **all** instances of X", "how much spent at vendor Y",
"latest on project Z"). The top-*k* semantic ranker fights these, so the deep agent
brute-forces it (one "find all X" turn fired ~25 query reformulations / 230 tool
calls). This prototype tests **Family A — proposition retrieval** (Dense X Retrieval,
arXiv 2312.06648, adapted for email): extract atomic, self-contained facts at ingest,
embed them, and look them up — turning enumeration into a DB recall problem instead of
an "LLM reads 414k emails" problem (impossible).

## Architecture (what we validated)
1. **Extract once, at ingest.** Per-message LLM call → atomic "facts" (propositions),
   each self-contained with owner-aware coreference ("my car" → "<owner>'s car").
   Prompt: `propositions._PROP_SYSTEM` (v3).
2. **Store** each fact + embedding + back-pointer to its source message (`propositions`).
3. **Retrieve = HYBRID** (semantic ∪ BM25 over fact text), fused by Reciprocal Rank
   Fusion. The keyword half is **load-bearing**: bare alphanumeric facts (plates, VINs,
   order #s) embed near-zero (a bare ID string scored cosine ~0.32 vs the query) and are
   only recoverable by keyword. `find_facts(exhaustive=True, hybrid=True)`.
4. **Dedup = cosine candidates + LLM verify.** Cosine alone *over-merges* (see below);
   an LLM adjudicates each candidate cluster before merging. `cluster_duplicates` + verify.

## Key results (frozen 800-message eval set, v3 + gemini-3.5-flash)
| metric | value | notes |
|---|---|---|
| facts/msg | **1.50** | representative 20-yr span |
| zero-fact messages | **49%** | **confirmed correct** by hand-judge — half the mailbox is promo/newsletter/transient |
| signal rate | **~90%** | hand-judged (Claude); noise is ~10%, consistently *transient status events* |
| extraction throughput | 272 msg/min | Flash, concurrency 24 |
| known-entity recall (ground truth) | extraction **4/4**; pure-semantic retrieval 2/4; **hybrid 4/4** | a 5-item enumeration target |
| safe dedup | **7%** | naive cosine@0.88 = 29% but mostly **false merges** (unsafe) |

### Whole-mailbox projection (414k messages)
- ~**620k facts** (~578k after dedup) — *not* the 2–5.6M earlier small-sample estimates.
- Index RAM **~1.8 GB @ 768-dim** (truncate; 3072-dim = ~7 GB).
- Embedding cost ~**$2**; Flash extraction ~**$60** one-time; backfill ~a day (concurrency).
- **49% zero-fact ⇒ a cheap sender/"has-durable-facts?" pre-filter ~halves cost & time** at near-zero recall loss (the empty-message hypothesis was confirmed safe).

## Decisions / lessons
- **Model: use a cloud model (gemini-3.5-flash), not the local 4B.** Local was 3.6× lower
  recall on a fixed A/B *and* 25 msg/min → **11.5 days** to backfill (non-starter). Flash:
  better recall, restraint on junk, 272 msg/min.
- **Hybrid retrieval is mandatory.** Embeddings cannot rank bare IDs; BM25 over fact text
  is required, not optional.
- **Naive embedding dedup is unsafe.** At 0.88–0.92 cosine it merged *distinct* facts with
  parallel sentence structure (e.g. two different people's emails; two different medications;
  two different months' meter readings). Cosine is a **candidate generator**; an LLM verify
  tier is required (it split 22/23 candidate clusters correctly). This is also the
  entity-resolution mechanism (a yes/no "same entity?" merge decision).
- **Methodology: a big *frozen* eval set is essential.** Small re-drawn samples produced a
  "+36% facts" mirage and a 3–9× over-estimate of scale; both dissolved on the frozen
  800-set. Automated LLM judges were unreliable (flash: format breakage + called order-IDs
  noise; 3.1-pro: 60% error rate + discarded dated plans) → **hand-judging by a different
  model family (Claude) is the trustworthy quality gauge.** Stop prompt-tuning at the point
  of diminishing returns to avoid overfitting to the dozen examples you keep looking at.

## Open items (before productionizing)
- Wire `find_facts` into deep mode as a tool (stop the agent's 230-call brute-forcing).
- Pre-filter the 49% fact-empty mail to halve backfill cost/time.
- Move retrieval off brute-force onto a separate (768-dim) ScaNN index at full scale.
- Entity resolution proper: merge mentions into entities to answer "how many of X" (counts),
  not just "find the matching strings".
- Fold the eval drivers (currently ad-hoc) into a reusable harness against `prop_eval_set`.
