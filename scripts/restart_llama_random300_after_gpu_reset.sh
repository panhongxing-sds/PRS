#!/usr/bin/env bash
# Run after Autodl instance/GPU restart when GPUs show 0% util (not wedged at 100%/0MiB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="/root/autodl-tmp/panda-outputs/maintable_llama32_1b_deepscaler_random300/logs"
mkdir -p "$LOG"
export OUT_BASE="/root/autodl-tmp/panda-outputs/maintable_llama32_1b_deepscaler_random300"
export MODEL_PATH="/root/autodl-tmp/panda-models/TFB-Llama-3.2-1B-Instruct"
export PANDA_DYNAMIC_CLAIM=1
export GPU_MEMORY_UTIL=0.90
cd "$ROOT"
nohup bash scripts/run_random300_6gpu.sh phase_b >> "$LOG/llama_phase_b_nohup.log" 2>&1 &
echo "Llama phase_b PID=$!"
nohup env OUT_DIR="$OUT_BASE/seed41" K9_JSONL="$ROOT/experiments/spurious_consensus/data/samples/samples_llama32_1b_seed41_deepscaler_random300_k9.jsonl" OUT_BASE="$OUT_BASE" LOCK="$LOG/watch_random300_finish.lock" LOG="$LOG/watch_random300_finish.log" \
  bash scripts/watch_random300_finish.sh >> "$LOG/llama_watch_finish_nohup.log" 2>&1 &
echo "Llama watcher PID=$!"
