#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Queue ASE + TokUR experiments for additional TFB models (sequential, waits for GPUs).
#
# Models (after current Qwen2.5-3B jobs finish):
#   1. TFB-Llama-3.2-1B-Instruct  (TokUR paper aligned)
#   2. TFB-Llama-3.1-8B-Instruct
#   3. TFB-Qwen3-8B
#
# Usage:
#   nohup bash scripts/queue_ase_extra_models.sh > $PRS_OUTPUTS/ase_queue.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QUEUE_LOG="$PRS_OUTPUTS/ase_model_queue.log"
mkdir -p "$(dirname "$QUEUE_LOG")"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"; }

PAUSE_FLAG="$PRS_OUTPUTS/PAUSE_ASE"

wait_for_gpus() {
  log "Waiting for TokUR priority / ASE GPU jobs to finish..."
  while [[ -f "$PAUSE_FLAG" ]] \
    || pgrep -f "run_tokur_strict_baseline" >/dev/null 2>&1 \
    || pgrep -f "run_tokur_priority" >/dev/null 2>&1 \
    || pgrep -f "run_tokur_then_advw" >/dev/null 2>&1 \
    || pgrep -f "run_advw_ase_full" >/dev/null 2>&1 \
    || pgrep -f "greedy_unc_single_batch_refine" >/dev/null 2>&1 \
    || pgrep -f "python3 -m prs.ase.run_advw_ase_experiment" >/dev/null 2>&1 \
    || pgrep -f "python3 -m prs.ase.run_ase_experiment" >/dev/null 2>&1 \
    || pgrep -f "python3 -m prs.ase.score_tokur_baseline" >/dev/null 2>&1; do
    if [[ -f "$PAUSE_FLAG" ]]; then
      log "PAUSE_ASE set — TokUR priority in progress, sleeping..."
    fi
    sleep 120
  done
  log "GPUs free."
}

declare -a MODELS=(
  "llama32_1b|$PRS_MODELS/TFB-Llama-3.2-1B-Instruct"
  "llama31_8b|$PRS_MODELS/TFB-Llama-3.1-8B-Instruct"
  "qwen3_8b|$PRS_MODELS/TFB-Qwen3-8B"
)

export MAX_SAMPLES="${MAX_SAMPLES:-400}"
export MAX_SAMPLES_MINERVA="${MAX_SAMPLES_MINERVA:-272}"

raw_count() {
  local out=$1 ds=$2
  local dir="$out/$ds/raw_runs"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  find "$dir" -name '*.json' ! -name '*.error.json' ! -name '*.partial.json' 2>/dev/null | wc -l
}

targets_met() {
  local out=$1
  [[ $(raw_count "$out" minerva) -ge 267 ]] \
    && [[ $(raw_count "$out" math500) -ge 395 ]] \
    && [[ $(raw_count "$out" gsm8k) -ge 395 ]] \
    && [[ $(raw_count "$out" deepscaler) -ge 395 ]]
}

log "ASE extra-model queue started (llama32_1b first, then 8B/Qwen3)"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag path <<< "$entry"
  out="$PRS_OUTPUTS/ase_${tag}"

  if targets_met "$out"; then
    log "SKIP $tag (all datasets at target: minerva=$(raw_count "$out" minerva) math500=$(raw_count "$out" math500) gsm8k=$(raw_count "$out" gsm8k) deepscaler=$(raw_count "$out" deepscaler))"
    touch "$out/PIPELINE_DONE"
    continue
  fi

  export ASE_SKIP_GENERATE=0
  n_minerva=$(raw_count "$out" minerva)
  n_deepscaler=$(raw_count "$out" deepscaler)
  log "RESUME $tag: minerva=$n_minerva math500=$(raw_count "$out" math500) gsm8k=$(raw_count "$out" gsm8k) deepscaler=$n_deepscaler → target 272/400/400/400"

  wait_for_gpus
  log "START pipeline: $tag ($path) [ASE_FAST=1 parallel datasets]"
  if ASE_FAST=1 ASE_SKIP_GENERATE="$ASE_SKIP_GENERATE" bash "$ROOT/scripts/run_ase_model_pipeline.sh" "$tag" "$path" "$out" 2>&1 | tee -a "$QUEUE_LOG"; then
    touch "$out/PIPELINE_DONE"
    log "DONE $tag → $out"
  else
    log "FAILED $tag — stopping queue"
    exit 1
  fi
done

log "All extra models queued and completed."
