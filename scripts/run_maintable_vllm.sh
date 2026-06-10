#!/usr/bin/env bash
# Hybrid vLLM + HF main-table pipeline (per model).
#
#   Phase A (vLLM, batched): clean greedy + R text rephrases + SE high-temp samples
#   Phase B (HF, per-question): weight-perturbation branch + metrics
#   Phase C (official TokUR venv): greedy_unc → tokur_baseline.jsonl
#
# Both phases are resumable; the whole thing can be run in batches (by dataset,
# by seed, or by shard) and re-invoked safely — completed work is skipped.
#
# Usage:
#   bash scripts/run_maintable_vllm.sh qwen25_3b
#   DATASETS=math500,gsm8k SEEDS=41 bash scripts/run_maintable_vllm.sh qwen25_3b
#   SKIP_VLLM=1 bash scripts/run_maintable_vllm.sh qwen25_3b   # only Phase B (weight) + metrics
#   SKIP_HF=1   bash scripts/run_maintable_vllm.sh qwen25_3b   # only Phase A (vLLM gen)
set -euo pipefail
source "$(dirname "$0")/env.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_TAG="${1:?model_tag required (qwen25_3b|llama32_1b|llama31_8b|qwen3_8b)}"
SEEDS="${SEEDS:-41,42,43}"
DATASETS="${DATASETS:-minerva,math500,gsm8k,leg_counting,zebra_puzzles,color_cube}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
# Fair sampling budget K=8: R=4 text + W=4 weight; SE=8 high-temp samples.
N_REPHRASES="${N_REPHRASES:-4}"
WEIGHT_SEEDS="${WEIGHT_SEEDS:-42,43,44,45}"
SE_SAMPLES="${SE_SAMPLES:-8}"
ASE_MAX_TOKENS="${ASE_MAX_TOKENS:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
SKIP_VLLM="${SKIP_VLLM:-0}"
SKIP_HF="${SKIP_HF:-0}"
DRY_RUN="${DRY_RUN:-0}"

declare -A MODEL_PATH=(
  [qwen25_3b]="${PRS_MODELS}/TFB-Qwen2.5-3B-Instruct"
  [llama32_1b]="${PRS_MODELS}/TFB-Llama-3.2-1B-Instruct"
  [llama31_8b]="${PRS_MODELS}/TFB-Llama-3.1-8B-Instruct"
  [qwen3_8b]="${PRS_MODELS}/TFB-Qwen3-8B"
)
MODEL_PATH="${MODEL_PATH[$MODEL_TAG]:-}"
[[ -z "$MODEL_PATH" ]] && { echo "Unknown MODEL_TAG=$MODEL_TAG" >&2; exit 1; }

OUT_BASE="${OUT_BASE:-$PRS_OUTPUTS/maintable_${MODEL_TAG}}"
LOG="$OUT_BASE/logs"; mkdir -p "$LOG"

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra DS_ARR <<< "$DATASETS"

echo "=== Hybrid vLLM pipeline: model=$MODEL_TAG out=$OUT_BASE ==="
echo "  seeds=$SEEDS datasets=$DATASETS K=8 (R=$N_REPHRASES + W=$WEIGHT_SEEDS) SE=$SE_SAMPLES"

for seed in "${SEED_ARR[@]}"; do
  OUT="$OUT_BASE/seed${seed}"; mkdir -p "$OUT/logs"
  for ds in "${DS_ARR[@]}"; do
    ds=$(echo "$ds" | tr '-' '_')
    ms="$MAX_SAMPLES"
    [[ "$ds" == "minerva" ]] && ms="${MAX_SAMPLES_MINERVA:-272}"

    echo ""; echo ">>> seed=$seed dataset=$ds max_samples=$ms"
    if [[ "$DRY_RUN" == "1" ]]; then echo "DRY_RUN: skip"; continue; fi

    # --- Phase A: vLLM clean + R + SE (resumable, chunk-checkpointed) ---
    if [[ "$SKIP_VLLM" != "1" ]]; then
      python3 -m prs.ase.run_vllm_phase \
        --dataset "$ds" \
        --max-samples "$ms" \
        --n-rephrases "$N_REPHRASES" \
        --se-samples "$SE_SAMPLES" \
        --weight-seeds "$WEIGHT_SEEDS" \
        --weight-sigma 0.03 --weight-rank 4 \
        --max-new-tokens "$ASE_MAX_TOKENS" \
        --topk-save 10 \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --model-path "$MODEL_PATH" \
        --out-dir "$OUT" \
        --resume \
        > "$LOG/vllm_${ds}_seed${seed}.log" 2>&1 || echo "WARN: vLLM phase A $ds seed$seed failed"
    fi

    # --- Phase B: HF weight branch + metrics (skips clean/R/SE already in partial) ---
    if [[ "$SKIP_HF" != "1" ]]; then
      python3 -m prs.ase.run_ase_experiment \
        --mode all \
        --dataset "$ds" \
        --max-samples "$ms" \
        --n-rephrases "$N_REPHRASES" \
        --weight-seeds "$WEIGHT_SEEDS" \
        --se-samples "$SE_SAMPLES" \
        --weight-sigma 0.03 --weight-rank 4 \
        --max-new-tokens "$ASE_MAX_TOKENS" \
        --topk-save 10 \
        --model-path "$MODEL_PATH" \
        --out-dir "$OUT" \
        --device "${CUDA_DEVICE:-cuda:0}" \
        --resume \
        ${ASE_FAST:+--fast} \
        > "$LOG/hf_${ds}_seed${seed}.log" 2>&1 || echo "WARN: HF phase B $ds seed$seed failed"

    fi

    # --- Phase C: official TokUR (third_party/TokUR vLLM + TFB greedy EU) ---
    if [[ "${ASE_SKIP_TOKUR:-0}" != "1" && "$SKIP_HF" != "1" ]]; then
      OUT_DIR="$OUT" PRS_MODEL_TAG="$MODEL_TAG" DATASET="$ds" TOKUR_SEED="$seed" \
        bash scripts/run_tokur_official_maintable.sh \
        >> "$LOG/tokur_${ds}_seed${seed}.log" 2>&1 || echo "WARN: official TokUR $ds seed$seed failed"
    fi
  done
done

echo ""; echo "Done → $OUT_BASE (aggregate with scripts/aggregate_maintable.sh $MODEL_TAG)"
