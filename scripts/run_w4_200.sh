#!/usr/bin/env bash
# Wave 4 de-risk bench on 200 messages. Assumes vLLM is at ctx=8192
# seqs=8 for configs 1-5 (same context); bounces to ctx=16384 seqs=1
# for ctx16k at the end. Writes results to scripts/bench_out/W4_200_*.json.
set -euo pipefail
cd /home/ssilver/development/gmail-search

SAMPLE="scripts/bench_sample_200.json"

echo "=== [1/5] baseline v5 ==="
python scripts/bench_w4.py --sample "$SAMPLE" --label W4_200_baseline --pipeline single --prompt-version v5 --max-body 15000 --tail 4000 --max-tokens 500 --concurrency 4

echo "=== [2/5] structured v7 ==="
python scripts/bench_w4.py --sample "$SAMPLE" --label W4_200_structured --pipeline structured --max-body 15000 --tail 4000 --max-tokens 700 --concurrency 4

echo "=== [3/5] critique_revise (2-pass) ==="
python scripts/bench_w4.py --sample "$SAMPLE" --label W4_200_critique --pipeline critique_revise --max-body 15000 --tail 4000 --max-tokens 500 --concurrency 4

echo "=== [4/5] headline v8 ==="
python scripts/bench_w4.py --sample "$SAMPLE" --label W4_200_headline --pipeline headline --max-body 15000 --tail 4000 --max-tokens 300 --concurrency 4

echo "=== [5/5] cot v6 ==="
python scripts/bench_w4.py --sample "$SAMPLE" --label W4_200_cot --pipeline cot_strip --max-body 15000 --tail 4000 --max-tokens 800 --concurrency 4

# vLLM restart for ctx=16384 run
echo "=== bounce vLLM to ctx=16384 seqs=4 ==="
kill "$(cat /tmp/vllm_bench.pid)" 2>/dev/null || true
sleep 8
./scripts/launch_vllm_bench.sh 16384 4

echo "=== [6/6] ctx=16384 ==="
python scripts/bench_w4.py --sample "$SAMPLE" --label W4_200_ctx16k --pipeline single --prompt-version v5 --max-body 25000 --tail 6000 --max-tokens 600 --concurrency 4

echo "=== DONE ==="
