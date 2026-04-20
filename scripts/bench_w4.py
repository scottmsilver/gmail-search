"""Wave 4 creative bench — explore beyond the prompt/context sweep
of wave 3. Assumes vLLM is already running at the appropriate config
for the run (caller manages restarts); this script is the runner and
knows how to dispatch each "pipeline" variant.

Pipelines implemented:

- **single**: the current v5-style one-shot prompt. The baseline.
- **cot_strip**: v6 prompt asks the model to list facts THEN write a
  "SUMMARY:" line. We post-strip everything before "SUMMARY:".
- **structured**: v7 prompt asks for JSON; we parse and render to
  prose via a template.
- **headline_detail**: v8 prompt; we keep the headline+detail block
  as-is (it's already two-line, useful in the UI).
- **critique_revise**: two LLM calls. First a normal v5 summary;
  then a second call that feeds the summary + body back and asks
  "what did this miss? rewrite with those facts added."
- **hierarchical**: split email body by reply markers (`>`, `On ...
  wrote:`, `----- Forwarded message -----`). Summarise each segment
  individually, then synthesise a "timeline" summary. Good on long
  threads.
- **few_shot**: retrieve 2 semantically similar messages from the DB
  that already have v5 summaries. Include them as (body → summary)
  examples in the system prompt.

All runs write to scripts/bench_out/W4_<label>.json using the same
schema as scripts/summary_quality_bench.py so one analyzer handles
everything.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from gmail_search import summarize as summarize_mod

sys.path.insert(0, str(Path(__file__).parent))
from bench_prompts import PROMPTS  # noqa: E402

DEFAULT_SAMPLE_PATH = Path("scripts/bench_sample.json")
OUT_DIR = Path("scripts/bench_out")
VLLM_URL = "http://127.0.0.1:8001"
MODEL = "ciocan/gemma-4-E4B-it-W4A16"


# ── transport ─────────────────────────────────────────────────────────────


def chat(
    client: httpx.Client,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float = 0.1,
    enable_thinking=None,
    json_format: bool = False,
) -> str:
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if json_format:
        body["response_format"] = {"type": "json_object"}
    r = client.post(f"{VLLM_URL}/v1/chat/completions", json=body, timeout=180.0)
    r.raise_for_status()
    choices = r.json().get("choices") or []
    return choices[0]["message"]["content"] if choices else ""


# ── utilities ─────────────────────────────────────────────────────────────


def build_prompt(msg: dict, max_body: int, tail: int) -> str:
    """Inline truncation so we don't mutate module globals across threads."""
    body = summarize_mod._clean_body(msg["body_text"] or "")
    if len(body) > max_body:
        head_chars = max_body - tail
        body = body[:head_chars] + "\n\n[...]\n\n" + body[-tail:]
    return f"From: {msg['from_addr']}\nSubject: {msg['subject']}\n\n{body}"


# ── pipelines ─────────────────────────────────────────────────────────────


def pipeline_single(client, msg, *, prompt_version, max_body, tail, max_tokens, temperature, enable_thinking):
    user = build_prompt(msg, max_body, tail)
    raw = chat(
        client,
        [{"role": "system", "content": PROMPTS[prompt_version]}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
    return summarize_mod._clean_llm_output(raw), len(raw)


_COT_SUMMARY = re.compile(r"summary\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def pipeline_cot_strip(client, msg, *, max_body, tail, max_tokens, temperature):
    user = build_prompt(msg, max_body, tail)
    raw = chat(
        client,
        [{"role": "system", "content": PROMPTS["v6"]}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    m = _COT_SUMMARY.search(raw)
    summary = m.group(1).strip() if m else raw.strip()
    return summarize_mod._clean_llm_output(summary), len(raw)


def pipeline_structured(client, msg, *, max_body, tail, max_tokens, temperature):
    user = build_prompt(msg, max_body, tail)
    raw = chat(
        client,
        [{"role": "system", "content": PROMPTS["v7"]}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
        json_format=True,
    )
    try:
        obj = json.loads(raw)
    except Exception:
        return "", len(raw)
    parts = []
    sender = obj.get("sender") or "Someone"
    topic = obj.get("topic") or ""
    facts = obj.get("facts") or []
    asks = obj.get("asks") or []
    if facts:
        parts.append(f"{sender}: {topic}." if topic else f"{sender} wrote:")
        parts.append("; ".join(str(f).strip(".") for f in facts[:8]) + ".")
    if asks:
        for a in asks[:3]:
            frm = a.get("from") or sender
            to = a.get("to") or "recipient"
            action = a.get("action") or ""
            if action:
                parts.append(f"{frm} asks {to} to {action}.")
    return " ".join(parts).strip(), len(raw)


def pipeline_headline(client, msg, *, max_body, tail, max_tokens, temperature):
    user = build_prompt(msg, max_body, tail)
    raw = chat(
        client,
        [{"role": "system", "content": PROMPTS["v8"]}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # Keep both HEADLINE and DETAIL lines as plain text for bench
    # scoring; in production we'd render them as two lines in the UI.
    return summarize_mod._clean_llm_output(raw), len(raw)


def pipeline_critique_revise(client, msg, *, max_body, tail, max_tokens, temperature):
    # Pass 1: normal v5 summary
    user = build_prompt(msg, max_body, tail)
    first = chat(
        client,
        [{"role": "system", "content": PROMPTS["v5"]}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    v1 = summarize_mod._clean_llm_output(first)
    # Pass 2: critique-and-revise
    revise_system = (
        "You are improving an email summary for a retrieval index. "
        "Given the original email and a draft summary, identify specific "
        "facts (names, amounts, dates, percentages, decisions, asks) that "
        "the draft missed, then output ONLY the revised summary (no "
        "critique, no preamble). Keep it under 600 characters and start "
        "with the sender's name."
    )
    revise_user = f"ORIGINAL EMAIL:\n{user}\n\nDRAFT SUMMARY:\n{v1}\n\nOutput only the revised summary."
    second = chat(
        client,
        [{"role": "system", "content": revise_system}, {"role": "user", "content": revise_user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return summarize_mod._clean_llm_output(second), len(first) + len(second)


_REPLY_MARK = re.compile(
    r"(^>.*(?:\n>.*)*)|(^On .+?wrote:)|(^-{5,}\s*Original Message\s*-{5,})|(^-{5,}\s*Forwarded message\s*-{5,})",
    re.MULTILINE,
)


def pipeline_hierarchical(client, msg, *, max_body, tail, max_tokens, temperature):
    """Split body on reply-thread markers; summarize each segment;
    synthesize a top-level timeline. If no markers found, fall through
    to single-shot v5.
    """
    body = summarize_mod._clean_body(msg["body_text"] or "")
    # Cap body if gigantic — applied BEFORE splitting so we don't burn
    # budget summarising 20 quoted hops.
    if len(body) > max_body:
        body = body[: max_body - tail] + "\n\n[...]\n\n" + body[-tail:]
    segments = _REPLY_MARK.split(body)
    segments = [s.strip() for s in segments if s and s.strip() and len(s.strip()) > 50]
    if len(segments) <= 1:
        # No real thread structure — just run v5 single-shot.
        return pipeline_single(
            client,
            msg,
            prompt_version="v5",
            max_body=max_body,
            tail=tail,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=None,
        )
    # Summarize each segment
    seg_summaries = []
    raw_total = 0
    for seg in segments[:8]:  # cap segments
        seg_prompt = f"From: {msg['from_addr']}\nSubject: {msg['subject']}\n\n{seg}"
        r = chat(
            client,
            [
                {
                    "role": "system",
                    "content": "Summarize this email fragment in ONE sentence (under 200 chars). Start with the apparent speaker if known.",
                },
                {"role": "user", "content": seg_prompt},
            ],
            max_tokens=120,
            temperature=temperature,
        )
        raw_total += len(r)
        seg_summaries.append(summarize_mod._clean_llm_output(r))
    # Compose final
    compose_sys = (
        "Given a list of bullet summaries from an email thread (oldest to newest), "
        "write a single 2-4 sentence summary (under 600 chars) that captures the "
        "thread's arc: who asked what, key facts exchanged, and the latest state. "
        "Start with the most recent sender's name."
    )
    bullets = "\n".join(f"- {s}" for s in seg_summaries)
    r2 = chat(
        client,
        [{"role": "system", "content": compose_sys}, {"role": "user", "content": bullets}],
        max_tokens=400,
        temperature=temperature,
    )
    raw_total += len(r2)
    return summarize_mod._clean_llm_output(r2), raw_total


def pipeline_few_shot(client, msg, *, max_body, tail, max_tokens, temperature, corpus_examples):
    """Pick 2 random-but-diverse corpus examples as few-shot. For speed
    we don't do real vector retrieval in the bench; a fixed diverse set
    is close enough to test whether in-context examples help at all.
    """
    shots = "\n\n".join(
        f"EMAIL:\nFrom: {e['from_addr']}\nSubject: {e['subject']}\n\n{e['body_text'][:800]}\n\nSUMMARY: {e['summary']}"
        for e in corpus_examples
    )
    sys_prompt = (
        "You summarize emails for a retrieval index, matching the style of the examples "
        "below. Start with the sender's name, capture every specific fact "
        "(amounts, dates, names, percentages, order numbers), close with any "
        "explicit ask. 2-4 sentences, under 600 characters.\n\n"
        "EXAMPLES:\n\n" + shots
    )
    user = build_prompt(msg, max_body, tail)
    r = chat(
        client,
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return summarize_mod._clean_llm_output(r), len(r)


# ── runner ─────────────────────────────────────────────────────────────────


def run_pipeline(
    label: str, pipeline_name: str, concurrency: int = 4, sample_path: Path = DEFAULT_SAMPLE_PATH, **kwargs
) -> None:
    sample = json.loads(sample_path.read_text())
    results: list[dict] = []

    # Warm the few-shot corpus if needed — pick 2 diverse v5 summaries.
    corpus_examples = []
    if pipeline_name == "few_shot":
        import sqlite3

        conn = sqlite3.connect("data/gmail_search.db")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT m.from_addr, m.subject, m.body_text, s.summary
            FROM message_summaries s
            JOIN messages m ON m.id = s.message_id
            WHERE s.model = 'ciocan/gemma-4-E4B-it-W4A16+v5'
              AND length(m.body_text) BETWEEN 500 AND 3000
              AND s.summary NOT LIKE 'Promotional%'
              AND s.summary NOT LIKE 'Social%'
              AND s.summary NOT LIKE 'Mailing-list%'
            ORDER BY RANDOM() LIMIT 2
            """
        ).fetchall()
        corpus_examples = [dict(r) for r in rows]
        conn.close()
        kwargs["corpus_examples"] = corpus_examples
        print(f"  few_shot corpus: {len(corpus_examples)} examples")

    pipeline = {
        "single": pipeline_single,
        "cot_strip": pipeline_cot_strip,
        "structured": pipeline_structured,
        "headline": pipeline_headline,
        "critique_revise": pipeline_critique_revise,
        "hierarchical": pipeline_hierarchical,
        "few_shot": pipeline_few_shot,
    }[pipeline_name]

    def _one(m: dict) -> dict:
        t0 = time.time()
        try:
            with httpx.Client() as client:
                summary, raw_len = pipeline(client, m, **kwargs)
            error = None
        except Exception as e:
            summary = ""
            raw_len = 0
            error = f"{type(e).__name__}: {e}"[:200]
        return {
            "id": m["id"],
            "bucket": m["bucket"],
            "subject": m["subject"],
            "from": m["from_addr"],
            "body_len": m["body_len"],
            "summary": summary,
            "raw_len": raw_len,
            "error": error,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    print(
        f"== {label}: pipeline={pipeline_name} kwargs={ {k: v for k, v in kwargs.items() if k != 'corpus_examples'} } =="
    )
    total_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(_one, m): m for m in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if not r["error"] and r["summary"] else "✗"
            print(f"  [{i:2d}/{len(sample)}] {mark} {r['bucket']:<9} {r['elapsed_ms']:>5}ms {r['subject'][:40]!r}")

    results.sort(key=lambda r: (r["bucket"], r["id"]))
    total = time.time() - total_start
    out = {
        "label": label,
        "pipeline": pipeline_name,
        "config": {k: v for k, v in kwargs.items() if k != "corpus_examples"},
        "totals": {
            "ok": sum(1 for r in results if r["summary"] and not r["error"]),
            "failed": sum(1 for r in results if r["error"] or not r["summary"]),
            "total_seconds": round(total, 1),
        },
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{label}.json").write_text(json.dumps(out, indent=2))
    print(
        f"  wrote {label}.json  ok={out['totals']['ok']}/{len(sample)}  failed={out['totals']['failed']}  {total:.1f}s\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument(
        "--pipeline",
        required=True,
        choices=["single", "cot_strip", "structured", "headline", "critique_revise", "hierarchical", "few_shot"],
    )
    ap.add_argument("--prompt-version", default="v5")
    ap.add_argument("--max-body", type=int, default=15000)
    ap.add_argument("--tail", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--enable-thinking", default=None, type=lambda v: None if v in (None, "none") else v == "1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATH), help="Path to sample JSON")
    args = ap.parse_args()

    kwargs: dict = {
        "max_body": args.max_body,
        "tail": args.tail,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.pipeline == "single":
        kwargs["prompt_version"] = args.prompt_version
        kwargs["enable_thinking"] = args.enable_thinking

    run_pipeline(args.label, args.pipeline, concurrency=args.concurrency, sample_path=Path(args.sample), **kwargs)


if __name__ == "__main__":
    main()
