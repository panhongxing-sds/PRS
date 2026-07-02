#!/usr/bin/env bash
# End-to-end: SC@9 (GPU K=9) → PANDA 6-GPU → metrics watcher.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/env.sh"

OUT_BASE="${OUT_BASE:-/root/autodl-tmp/prs-outputs/maintable_qwen3_8b_deepscaler_random300}"
OUT_DIR="${OUT_DIR:-$OUT_BASE/seed41}"
LOG="$OUT_BASE/logs"
K9_JSONL="$ROOT/experiments/spurious_consensus/data/samples/samples_qwen3_8b_seed41_deepscaler_random300_k9.jsonl"
mkdir -p "$LOG"

echo "=== Qwen3-8B random300 full pipeline $(date -Iseconds) ===" | tee "$LOG/full_orchestrator.log"

bash "$ROOT/scripts/run_sc_k9_qwen3_8b_random300.sh" 2>&1 | tee -a "$LOG/full_orchestrator.log"

nohup env OUT_DIR="$OUT_DIR" OUT_BASE="$OUT_BASE" \
  K9_JSONL="$K9_JSONL" \
  LOG="$LOG/watch_random300_finish.log" \
  LOCK="$LOG/watch_random300_finish.lock" \
  bash "$ROOT/scripts/watch_random300_finish.sh" \
  >> "$LOG/watch_random300_finish.nohup.log" 2>&1 &
WATCH_PID=$!
echo "Watcher PID=$WATCH_PID" | tee -a "$LOG/full_orchestrator.log"

bash "$ROOT/scripts/run_random300_qwen3_8b_6gpu.sh" all 2>&1 | tee -a "$LOG/full_orchestrator.log"
