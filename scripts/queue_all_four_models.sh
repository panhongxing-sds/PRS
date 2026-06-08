#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Run ASE generate + metrics + tables for all four TFB models (no TokUR).
#
# Usage:
#   nohup bash scripts/queue_all_four_models.sh >> $PRS_OUTPUTS/ase_four_models.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/ase_gpu_lock.sh
source "$ROOT/scripts/ase_gpu_lock.sh"

QUEUE_LOG="$PRS_OUTPUTS/ase_four_models.log"
mkdir -p "$(dirname "$QUEUE_LOG")"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"; }

export ASE_FAST=1
export ASE_SKIP_TOKUR=1
# Official TokUR runs separately: bash scripts/queue_strict_tokur_four_models.sh
export MAX_SAMPLES="${MAX_SAMPLES:-400}"
export MAX_SAMPLES_MINERVA="${MAX_SAMPLES_MINERVA:-272}"

raw_count() {
  local out=$1 ds=$2
  local dir="$out/$ds/raw_runs"
  [[ -d "$dir" ]] || { echo 0; return; }
  find "$dir" -name '*.json' ! -name '*.error.json' ! -name '*.partial.json' 2>/dev/null | wc -l
}

need_for() {
  case "$1" in minerva) echo "${MAX_SAMPLES_MINERVA}" ;; *) echo "${MAX_SAMPLES}" ;; esac
}

targets_met() {
  local out=$1 ds need
  for ds in minerva math500 gsm8k deepscaler; do
    need=$(need_for "$ds")
    [[ $(raw_count "$out" "$ds") -ge $((need - 3)) ]] || return 1
  done
  return 0
}

wait_for_gpus() {
  log "Waiting for GPU lock + ASE/prefix processes..."
  wait_for_gpus_compatible
  log "GPUs free."
}

run_pipeline_gpu() {
  local tag=$1 path=$2 out=$3 skip_gen=$4
  wait_for_gpus
  wait_gpu_free
  local seq_flag=0
  case "$tag" in
    llama31_8b|qwen3_8b) seq_flag=1 ;;
  esac
  ASE_SKIP_TOKUR=1 ASE_SKIP_GENERATE="$skip_gen" ASE_FAST=1 ASE_8B_SEQUENTIAL="$seq_flag" \
    bash "$ROOT/scripts/run_ase_model_pipeline.sh" "$tag" "$path" "$out" >> "$QUEUE_LOG" 2>&1
}

declare -a MODELS=(
  "qwen25_3b|$PRS_MODELS/TFB-Qwen2.5-3B-Instruct|$PRS_OUTPUTS/ase_full"
  "llama32_1b|$PRS_MODELS/TFB-Llama-3.2-1B-Instruct|$PRS_OUTPUTS/ase_llama32_1b"
  "llama31_8b|$PRS_MODELS/TFB-Llama-3.1-8B-Instruct|$PRS_OUTPUTS/ase_llama31_8b"
  "qwen3_8b|$PRS_MODELS/TFB-Qwen3-8B|$PRS_OUTPUTS/ase_qwen3_8b"
)

log "Four-model ASE queue started (fast mode, no TokUR)"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag path out <<< "$entry"

  if targets_met "$out"; then
    log "SKIP $tag — raw complete (minerva=$(raw_count "$out" minerva) math500=$(raw_count "$out" math500) gsm8k=$(raw_count "$out" gsm8k) deepscaler=$(raw_count "$out" deepscaler))"
    export ASE_SKIP_GENERATE=1
    if run_pipeline_gpu "$tag" "$path" "$out" 1; then
      touch "$out/PIPELINE_DONE"
      log "DONE $tag (metrics/tables only)"
    else
      log "WARN $tag metrics/tables failed"
    fi
    continue
  fi

  export ASE_SKIP_GENERATE=0
  log "START $tag → $out"
  log "  progress: minerva=$(raw_count "$out" minerva)/$(need_for minerva) math500=$(raw_count "$out" math500)/$(need_for math500) gsm8k=$(raw_count "$out" gsm8k)/$(need_for gsm8k) deepscaler=$(raw_count "$out" deepscaler)/$(need_for deepscaler)"

  if run_pipeline_gpu "$tag" "$path" "$out" 0; then
    touch "$out/PIPELINE_DONE"
    log "DONE $tag → $out"
  else
    log "FAILED $tag — continuing to next model"
  fi
done

log "Four-model queue finished."
