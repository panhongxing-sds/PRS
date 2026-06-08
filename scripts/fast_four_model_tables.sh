#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Fast path: CPU-parallel recompute + paper tables (AUROC / AUPRC / ACC*).
# Does NOT run model generate (use GPUs elsewhere). Skips models with insufficient raw.
#
# Usage:
#   bash scripts/fast_four_model_tables.sh
#   MAX_JOBS=16 SKIP_RECOMPUTE=1 bash scripts/fast_four_model_tables.sh   # tables only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
LOG="$PRS_OUTPUTS/ase_four_models_tables.log"
mkdir -p "$(dirname "$LOG")"

MAX_JOBS="${MAX_JOBS:-16}"
SKIP_RECOMPUTE="${SKIP_RECOMPUTE:-0}"
DATASETS=(minerva math500 gsm8k deepscaler)
declare -A NEED=(
  [minerva]=272
  [math500]=400
  [gsm8k]=400
  [deepscaler]=400
)

declare -a MODELS=(
  "qwen25_3b|$PRS_OUTPUTS/ase_full"
  "llama32_1b|$PRS_OUTPUTS/ase_llama32_1b"
  "llama31_8b|$PRS_OUTPUTS/ase_llama31_8b"
  "qwen3_8b|$PRS_OUTPUTS/ase_qwen3_8b"
)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

raw_count() {
  local out=$1 ds=$2
  find "$out/$ds/raw_runs" -name '*.json' ! -name '*.error.json' ! -name '*.partial.json' 2>/dev/null | wc -l
}

# Partial raw 视为可用（论文主表不再等满 400/272）
MIN_RAW="${MIN_RAW:-50}"

ready_datasets() {
  local out=$1
  local ok=()
  for ds in "${DATASETS[@]}"; do
    local n
    n=$(raw_count "$out" "$ds")
    if [[ "$n" -ge "$MIN_RAW" ]]; then
      ok+=("$ds")
    fi
  done
  (IFS=,; echo "${ok[*]}")
}

run_pool() {
  local -n _pids=$1
  while ((${#_pids[@]} >= MAX_JOBS)); do
    local np
    np=$(jobs -rp | head -1)
    if [[ -n "$np" ]]; then
      wait "$np" || true
    else
      sleep 1
    fi
    _pids=($(jobs -rp))
  done
}

log "=== fast_four_model_tables MAX_JOBS=$MAX_JOBS SKIP_RECOMPUTE=$SKIP_RECOMPUTE ==="

PIDS=()
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag out <<< "$entry"
  ds_csv=$(ready_datasets "$out")
  if [[ -z "$ds_csv" ]]; then
    log "SKIP $tag — insufficient raw"
    continue
  fi
  log "$tag ready datasets: $ds_csv (raw: minerva=$(raw_count "$out" minerva) math500=$(raw_count "$out" math500) gsm8k=$(raw_count "$out" gsm8k) deepscaler=$(raw_count "$out" deepscaler))"

  if [[ "$SKIP_RECOMPUTE" != "1" ]]; then
    IFS=',' read -ra DS_ARR <<< "$ds_csv"
    for ds in "${DS_ARR[@]}"; do
      run_pool PIDS
      (
        log "recompute $tag $ds"
        python3 -m prs.ase.recompute_metrics --out-dir "$out" --datasets "$ds"
      ) >>"$LOG" 2>&1 &
      PIDS+=($!)
    done
  fi

  run_pool PIDS
  (
    log "paper_tables $tag"
    python3 -m prs.ase.analyze_paper_tables \
      --out-dir "$out" \
      --paper-dir "$ROOT/paper/models/${tag}" \
      --datasets "$ds_csv" \
      --features-only
    python3 "$ROOT/scripts/aggregate_four_model_tables.py" --single "$tag" || true
  ) >>"$LOG" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=$((FAIL + 1))
done

python3 "$ROOT/scripts/aggregate_four_model_tables.py" --all || true

log "=== done fail=$FAIL ==="
exit "$FAIL"
