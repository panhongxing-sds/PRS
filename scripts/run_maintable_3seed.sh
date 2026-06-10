#!/usr/bin/env bash
# 3-seed 主表实验：models × datasets × seeds → outputs/maintable_{tag}/seed{S}/
#
# Usage:
#   bash scripts/run_maintable_3seed.sh llama32_1b
#   DATASETS=leg_counting,zebra_puzzles,color_cube,humaneval SEEDS=41,42,43 bash scripts/run_maintable_3seed.sh llama32_1b
#   DRY_RUN=1 bash scripts/run_maintable_3seed.sh qwen25_3b
set -euo pipefail
source "$(dirname "$0")/env.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_TAG="${1:?model_tag required (qwen25_3b|llama32_1b|llama31_8b|qwen3_8b)}"
SEEDS="${SEEDS:-41,42,43}"
DATASETS="${DATASETS:-minerva,math500,gsm8k,leg_counting,zebra_puzzles,color_cube}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
# 公平采样预算 K=8：R=4 text rephrase + W=4 weight perturb；SE 取同等 8 个高温样本
N_REPHRASES="${N_REPHRASES:-4}"
WEIGHT_SEEDS="${WEIGHT_SEEDS:-42,43,44,45}"
SE_SAMPLES="${SE_SAMPLES:-8}"
ASE_FAST="${ASE_FAST:-1}"
DRY_RUN="${DRY_RUN:-0}"

declare -A MODEL_PATH=(
  [qwen25_3b]="${PRS_MODELS}/TFB-Qwen2.5-3B-Instruct"
  [llama32_1b]="${PRS_MODELS}/TFB-Llama-3.2-1B-Instruct"
  [llama31_8b]="${PRS_MODELS}/TFB-Llama-3.1-8B-Instruct"
  [qwen3_8b]="${PRS_MODELS}/TFB-Qwen3-8B"
)

MODEL_PATH="${MODEL_PATH[$MODEL_TAG]:-}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "Unknown MODEL_TAG=$MODEL_TAG" >&2
  exit 1
fi

OUT_BASE="${OUT_BASE:-$PRS_OUTPUTS/maintable_${MODEL_TAG}}"
LOG="$OUT_BASE/logs"
mkdir -p "$LOG"

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra DS_ARR <<< "$DATASETS"

echo "Maintable 3-seed run: model=$MODEL_TAG out=$OUT_BASE"
echo "  seeds=${SEEDS} datasets=${DATASETS}"

for seed in "${SEED_ARR[@]}"; do
  OUT="$OUT_BASE/seed${seed}"
  mkdir -p "$OUT/logs"
  for ds in "${DS_ARR[@]}"; do
    ds=$(echo "$ds" | tr '-' '_')
    ms="$MAX_SAMPLES"
    # minerva 全集仅 272；逻辑集取 min(MAX_SAMPLES, 仓库全量)，loader 会自动截断
    if [[ "$ds" == "minerva" ]]; then ms="${MAX_SAMPLES_MINERVA:-272}"; fi

    echo ""
    echo ">>> seed=$seed dataset=$ds max_samples=$ms"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY_RUN: skip GPU"
      continue
    fi

    # generate + metrics (single GPU; user can shard manually)
    python3 -m prs.ase.run_ase_experiment \
      --mode all \
      --dataset "$ds" \
      --max-samples "$ms" \
      --n-rephrases "$N_REPHRASES" \
      --weight-seeds "$WEIGHT_SEEDS" \
      --se-samples "$SE_SAMPLES" \
      --weight-sigma 0.03 \
      --weight-rank 4 \
      --max-new-tokens "${ASE_MAX_TOKENS:-2048}" \
      --topk-save 10 \
      --model-path "$MODEL_PATH" \
      --out-dir "$OUT" \
      --device "${CUDA_DEVICE:-cuda:0}" \
      --resume \
      ${ASE_FAST:+--fast} \
      > "$LOG/${ds}_seed${seed}.log" 2>&1 || echo "WARN: $ds seed$seed failed"

    # TokUR EU baseline (approx unless ASE_TOKUR_STRICT=1)
    if [[ "${ASE_SKIP_TOKUR:-0}" != "1" ]]; then
      python3 -m prs.ase.score_tokur_baseline \
        --out-dir "$OUT" \
        --dataset "$ds" \
        --model-path "$MODEL_PATH" \
        --device "${CUDA_DEVICE:-cuda:0}" \
        --resume \
        >> "$LOG/tokur_${ds}_seed${seed}.log" 2>&1 || true
    fi
  done
done

echo "Done → $OUT_BASE (aggregate with scripts/aggregate_maintable.sh)"
