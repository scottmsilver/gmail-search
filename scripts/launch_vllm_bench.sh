#!/usr/bin/env bash
# Spawn vLLM with a specific max-model-len / max-num-seqs combo and
# wait until /v1/models returns 200. Writes pid to $PIDFILE so the
# bench driver can kill it between configs.
#
# Usage: launch_vllm_bench.sh <max-model-len> <max-num-seqs>
set -euo pipefail

MAX_LEN="${1:?max-model-len required}"
MAX_SEQS="${2:?max-num-seqs required}"

VLLM="/home/ssilver/development/vllm-env/.venv/bin/vllm"
MODEL="ciocan/gemma-4-E4B-it-W4A16"
LOG="/tmp/vllm_bench.log"
PIDFILE="/tmp/vllm_bench.pid"

nohup "$VLLM" serve "$MODEL" \
  --quantization gptq_marlin \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.92 \
  --enforce-eager \
  --max-num-seqs "$MAX_SEQS" \
  --host 127.0.0.1 --port 8001 \
  >"$LOG" 2>&1 &

echo $! >"$PIDFILE"
echo "launched vllm (pid $(cat $PIDFILE)) — waiting for health..."

for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    echo "  healthy after ${i}s"
    exit 0
  fi
  sleep 2
done

echo "  TIMEOUT — last 30 log lines:" >&2
tail -30 "$LOG" >&2
kill "$(cat $PIDFILE)" 2>/dev/null || true
exit 1
