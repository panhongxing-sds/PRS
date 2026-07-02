!/usr/bin/env bash
# Minimal GPU plan for PANDA paper — only what CPU cannot replace.
#
# P0 (required): Official TokUR EU for main-table 4 models × 3 math × 3 seeds
# P1 (optional): skip unless reviewer asks — σ/rank sweep, Q/K ablation
#
# Prerequisites on this machine:
#   - PANDA_MODELS → /root/autodl-tmp/panda-models (4 TFB checkpoints)
#   - PANDA_OUTPUTS → /root/autodl-tmp/panda-outputs (maintable raw_runs exist)
#   - Phase A+B already done for math; this script only runs Phase C (TokUR)
#
# Usage:
#   source scripts/env.sh
#   export PANDA_OUTPUTS=/root/autodl-tmp/panda-outputs
#   export PANDA_MODELS=/root/autodl-tmp/panda-models
#   bash scripts/run_gpu_minimal_plan.sh           # setup + queue all TokUR
#   bash scripts/run_gpu_minimal_plan.sh --dry-run
#   bash scripts/run_gpu_minimal_plan.sh --setup-only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"

export PANDA_OUTPUTS="${PANDA_OUTPUTS:-/root/autodl-tmp/panda-outputs}"
export PANDA_MODELS="${PANDA_MODELS:-/root/autodl-tmp/panda-models}"
export CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

DRY_RUN=0
SETUP_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --setup-only) SETUP_ONLY=1 ;;
  esac
done

MODELS=(qwen25_3b llama32_1b llama31_8b qwen3_8b)
DATASETS=(minerva math500 gsm8k)
SEEDS=(41 42 43)

LOG_DIR="$PANDA_OUTPUTS/gpu_plan_logs"
mkdir -p "$LOG_DIR"

echo "=== PANDA minimal GPU plan ==="
echo "PANDA_OUTPUTS=$PANDA_OUTPUTS"
echo "PANDA_MODELS=$PANDA_MODELS"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo unknown)"
echo ""

# --- Step 0: TokUR venv (required for official EU baseline) ---
if [[ ! -x "${TOKUR_VENV:-$PANDA_ROOT/.tokur_venv}/bin/python" ]]; then
  echo "[setup] TokUR venv missing → running setup_after_clone.sh (TokUR part only may take ~10 min)"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY_RUN: would run bash scripts/setup_after_clone.sh"
  else
    bash "$ROOT/scripts/setup_after_clone.sh" 2>&1 | tee "$LOG_DIR/setup_after_clone.log"
  fi
else
  echo "[ok] TokUR venv: ${TOKUR_VENV:-$PANDA_ROOT/.tokur_venv}"
fi

if [[ "$SETUP_ONLY" == "1" ]]; then
  echo "Setup done (--setup-only)."
  exit 0
fi

# --- Step 1: CPU analyses (<60s, cache-only, no network) ---
echo ""
echo "[cpu] Running fast paper CPU analyses (cache + summary)..."
if [[ "$DRY_RUN" == "1" ]]; then
  echo "  DRY_RUN: python scripts/run_cpu_paper_analyses.py"
else
  time python3 "$ROOT/scripts/run_cpu_paper_analyses.py" \
    --outputs-root "$PANDA_OUTPUTS" \
    2>&1 | tee "$LOG_DIR/cpu_analyses.log"
fi

# --- Step 2: TokUR EU Phase C only (SKIP_VLLM + SKIP_HF) ---
echo ""
echo "[gpu] Queue TokUR EU (Phase C) — 4×3×3 = 36 jobs, resumable"
TOTAL=0
SKIP=0
RUN=0

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      OUT="$PANDA_OUTPUTS/maintable_${model}/seed${seed}"
      TOKUR_OUT="$OUT/${ds}/tokur_baseline.jsonl"
      TOTAL=$((TOTAL + 1))
      if [[ -s "$TOKUR_OUT" ]]; then
        echo "  [skip] $model seed$seed $ds (tokur exists)"
        SKIP=$((SKIP + 1))
        continue
      fi
      RAW_DIR="$OUT/${ds}/raw_runs"
      if [[ ! -d "$RAW_DIR" ]] || [[ -z "$(ls -A "$RAW_DIR"/*.json 2>/dev/null)" ]]; then
        echo "  [skip] $model seed$seed $ds (no raw_runs — run Phase A+B first)"
        SKIP=$((SKIP + 1))
        continue
      fi
      RUN=$((RUN + 1))
      CMD="OUT_DIR=$OUT PANDA_MODEL_TAG=$model DATASET=$ds TOKUR_SEED=$seed bash $ROOT/scripts/run_tokur_official_maintable.sh"
      echo "  [run] $model seed$seed $ds"
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "    $CMD"
      else
        (
          export OUT_DIR="$OUT"
          export PANDA_MODEL_TAG="$model"
          export DATASET="$ds"
          export TOKUR_SEED="$seed"
          export SKIP_VLLM=1
          export SKIP_HF=1
          bash "$ROOT/scripts/run_tokur_official_maintable.sh"
        ) >> "$LOG_DIR/tokur_${model}_s${seed}_${ds}.log" 2>&1 \
          || echo "WARN: TokUR failed $model seed$seed $ds (see log)"
      fi
    done
  done
done

echo ""
echo "TokUR jobs: total=$TOTAL skip=$SKIP run=$RUN"
echo "Logs: $LOG_DIR/"

# --- Step 3: Re-aggregate main table after TokUR ---
if [[ "$DRY_RUN" != "1" && "$RUN" -gt 0 ]]; then
  echo ""
  echo "[cpu] Re-aggregate main tables..."
  for model in "${MODELS[@]}"; do
    python3 "$ROOT/scripts/aggregate_panda_v2.py" --model "$model" 2>&1 | tee -a "$LOG_DIR/aggregate_${model}.log" || true
  done
fi

echo ""
echo "=== Done ==="
echo "CPU results: paper/analysis/cpu_results.md"
echo "Next: paste SC figures from experiments/spurious_consensus/figures/ into paper Section 4.2"
echo "Skipped GPU (not in minimal plan): σ/rank sweep, Q/K perturb ablation, open-ended tasks"
