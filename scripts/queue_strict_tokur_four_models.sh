#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Run official TokUR (vLLM + TFB greedy EU) for all four ASE model outputs.
# Waits for GPUs to be free of ASE generate, then runs one model at a time (8-GPU shards).
#
# Usage:
#   nohup bash scripts/queue_strict_tokur_four_models.sh >> $PRS_OUTPUTS/strict_tokur_four.log 2>&1 &
#
# Env:
#   DATASETS="minerva math500 gsm8k deepscaler"
#   FORCE_DATASETS="gsm8k deepscaler"   # only these (e.g. finish qwen partial)
#   SKIP_IF_OFFICIAL=1                 # skip dataset if tokur_baseline already official_vllm + row count OK
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/ase_gpu_lock.sh
source "$ROOT/scripts/ase_gpu_lock.sh"

LOG="${LOG:-$PRS_OUTPUTS/strict_tokur_four.log}"
mkdir -p "$(dirname "$LOG")"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

DATASETS="${DATASETS:-minerva math500 gsm8k deepscaler}"
FORCE_DATASETS="${FORCE_DATASETS:-}"
SKIP_IF_OFFICIAL="${SKIP_IF_OFFICIAL:-1}"
MIN_RAW="${MIN_RAW:-50}"
PARALLEL_SHARDS="${PARALLEL_SHARDS:-1}"
SEEDS="${SEEDS:-96}"

declare -A NEED=(
  [minerva]=272
  [math500]=400
  [gsm8k]=400
  [deepscaler]=400
)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

raw_count() {
  local out=$1 ds=$2
  find "$out/$ds/raw_runs" -name '*.json' ! -name '*.error.json' ! -name '*.partial.json' 2>/dev/null | wc -l
}

tokur_ok() {
  local out=$1 ds=$2
  local tb="$out/$ds/tokur_baseline.jsonl"
  [[ -f "$tb" ]] || return 1
  local n mode raw
  n=$(wc -l <"$tb" | tr -d ' ')
  raw=$(raw_count "$out" "$ds")
  mode=$(python3 -c "import json; print(json.loads(open('$tb').readline()).get('scoring_mode',''))" 2>/dev/null || echo "")
  # official 且覆盖当前全部 raw（partial 也算完成）
  [[ "$mode" == official_vllm* ]] && [[ "$n" -ge $((raw - 5)) ]]
}

ready_datasets() {
  local out=$1
  local -a ok=()
  for ds in $DATASETS; do
    if [[ -n "$FORCE_DATASETS" ]] && [[ " $FORCE_DATASETS " != *" $ds "* ]]; then
      continue
    fi
    if [[ "$SKIP_IF_OFFICIAL" == "1" ]] && tokur_ok "$out" "$ds"; then
      log "  skip $ds (official tokur complete)"
      continue
    fi
    local n=${NEED[$ds]}
    local r
    r=$(raw_count "$out" "$ds")
    if [[ "$r" -ge "$MIN_RAW" ]]; then
      ok+=("$ds")
    else
      log "  skip $ds raw=$r need>=$MIN_RAW"
    fi
  done
  echo "${ok[*]}"
}

run_strict() {
  local model_tag=$1 out_dir=$2
  local ds_list
  ds_list=$(ready_datasets "$out_dir")
  if [[ -z "$ds_list" ]]; then
    log "SKIP $model_tag — no dataset ready"
    return 0
  fi
  log "=== strict TokUR $model_tag OUT=$out_dir datasets: $ds_list ==="
  wait_for_gpus_compatible
  acquire_gpu tokur_strict "strict TokUR $model_tag"
  OUT_DIR="$out_dir" MODEL_TAG="$model_tag" DATASETS="$ds_list" SEEDS="$SEEDS" \
    PARALLEL_SHARDS="$PARALLEL_SHARDS" \
    bash "$ROOT/scripts/run_tokur_strict_baseline.sh" >>"$LOG" 2>&1
  release_gpu tokur_strict
  log "=== verify $model_tag ==="
  for ds in $ds_list; do
    if tokur_ok "$out_dir" "$ds"; then
      log "  OK $ds"
    else
      log "  WARN $ds incomplete or non-official"
    fi
  done
}

declare -a MODELS=(
  "qwen3b|$PRS_OUTPUTS/ase_full"
  "llama1b|$PRS_OUTPUTS/ase_llama32_1b"
  "llama8b|$PRS_OUTPUTS/ase_llama31_8b"
  "qwen8b|$PRS_OUTPUTS/ase_qwen3_8b"
)

log "queue_strict_tokur_four_models START SKIP_IF_OFFICIAL=$SKIP_IF_OFFICIAL"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag out <<< "$entry"
  run_strict "$tag" "$out" || log "FAIL $tag (continuing)"
done

log "Rebuilding paper tables (features-only) for models with tokur..."
MAX_JOBS=4 SKIP_RECOMPUTE=1 bash "$ROOT/scripts/fast_four_model_tables.sh" >>"$LOG" 2>&1 || true

log "queue_strict_tokur_four_models DONE"
