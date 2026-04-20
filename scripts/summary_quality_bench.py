"""Run the summary-quality bench against a running vLLM at :8001.

Does NOT spawn vLLM — the caller is expected to launch it with the
desired `--max-model-len` / `--max-num-seqs` flags first. This script
is "dumb": it reads scripts/bench_sample.json, summarises each message
with given knobs, and writes results to scripts/bench_out/<label>.json.

Knobs:
  --prompt-version v1|v2|v3|v4   system prompt variant (see bench_prompts.py)
  --max-body-chars N             head cap before head+tail truncation
  --tail-chars N                 tail chars kept when body is truncated
  --max-tokens N                 LLM max_tokens for the response
  --enable-thinking 0|1|auto     chat_template_kwargs.enable_thinking
                                 ('auto' = omit the kwarg entirely)
  --concurrency N

Output schema:
    {
      "label": "...",
      "config": {...},
      "results": [...],
      "totals": {"ok": N, "failed": M, "total_seconds": ...}
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from gmail_search import summarize as summarize_mod

# bench_prompts lives next to this script.
sys.path.insert(0, str(Path(__file__).parent))
from bench_prompts import PROMPTS  # noqa: E402

SAMPLE_PATH = Path("scripts/bench_sample.json")
OUT_DIR = Path("scripts/bench_out")


class BenchBackend:
    """Minimal Backend-protocol-compatible shim that POSTs to :8001
    with optional chat_template_kwargs.enable_thinking passthrough.
    Surfaces HTTP errors instead of swallowing them — callers record
    them in the bench output.
    """

    URL = "http://127.0.0.1:8001"
    model_id = "ciocan/gemma-4-E4B-it-W4A16"

    def __init__(self, enable_thinking):
        # `None` means "don't send the kwarg at all" (use the template
        # default). `True`/`False` sends an explicit override.
        self.enable_thinking = enable_thinking

    def chat(self, client, messages, *, max_tokens, json_format=False):
        body = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if json_format:
            body["response_format"] = {"type": "json_object"}
        r = client.post(f"{self.URL}/v1/chat/completions", json=body, timeout=180.0)
        r.raise_for_status()
        choices = r.json().get("choices") or []
        return choices[0]["message"]["content"] if choices else ""


def _summarize_one(client, backend, msg, system_prompt: str, max_tokens: int) -> dict:
    prompt_user = summarize_mod._build_user_prompt(msg["from_addr"], msg["subject"], msg["body_text"])
    start = time.time()
    try:
        raw = backend.chat(
            client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user},
            ],
            max_tokens=max_tokens,
        )
        summary = summarize_mod._clean_llm_output(raw)
        error = None
    except Exception as e:
        raw = ""
        summary = ""
        error = f"{type(e).__name__}: {e}"
    return {
        "id": msg["id"],
        "bucket": msg["bucket"],
        "subject": msg["subject"],
        "from": msg["from_addr"],
        "body_len": msg["body_len"],
        "prompt_user_len": len(prompt_user),
        "raw_len": len(raw or ""),
        "summary": summary,
        "error": error,
        "elapsed_ms": int((time.time() - start) * 1000),
    }


def _parse_thinking(val: str):
    if val == "auto":
        return None
    if val in ("1", "true", "on"):
        return True
    if val in ("0", "false", "off"):
        return False
    raise ValueError(f"bad --enable-thinking: {val!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt-version", default="v1", choices=list(PROMPTS.keys()))
    ap.add_argument("--max-body-chars", type=int, required=True)
    ap.add_argument("--tail-chars", type=int, default=1200)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--enable-thinking", default="auto", help="0 | 1 | auto")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    summarize_mod.MAX_BODY_CHARS = args.max_body_chars
    summarize_mod.TAIL_CHARS = args.tail_chars
    system_prompt = PROMPTS[args.prompt_version]
    enable_thinking = _parse_thinking(args.enable_thinking)

    sample = json.loads(SAMPLE_PATH.read_text())
    backend = BenchBackend(enable_thinking=enable_thinking)
    results: list[dict] = []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"== {args.label}: prompt={args.prompt_version} "
        f"body={args.max_body_chars} tok={args.max_tokens} "
        f"think={args.enable_thinking} ==",
        flush=True,
    )
    total_start = time.time()

    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(_summarize_one, client, backend, m, system_prompt, args.max_tokens): m for m in sample}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                mark = "✓" if not r["error"] and r["summary"] else "✗"
                print(
                    f"  [{i:2d}/{len(sample)}] {mark} {r['bucket']:9s} "
                    f"{r['elapsed_ms']:5d}ms {r['subject'][:40]!r}",
                    flush=True,
                )

    results.sort(key=lambda r: (r["bucket"], r["id"]))
    total_seconds = time.time() - total_start
    out = {
        "label": args.label,
        "config": {
            "prompt_version": args.prompt_version,
            "max_body_chars": args.max_body_chars,
            "tail_chars": args.tail_chars,
            "max_tokens": args.max_tokens,
            "enable_thinking": args.enable_thinking,
            "concurrency": args.concurrency,
        },
        "totals": {
            "ok": sum(1 for r in results if not r["error"] and r["summary"]),
            "failed": sum(1 for r in results if r["error"] or not r["summary"]),
            "total_seconds": round(total_seconds, 1),
        },
        "results": results,
    }
    out_path = OUT_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"\n  wrote {out_path}  ok={out['totals']['ok']}/{len(results)}  "
        f"failed={out['totals']['failed']}  {total_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
