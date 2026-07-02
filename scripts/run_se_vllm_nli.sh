#!/usr/bin/env bash
# Official sample-UQ pipeline: vLLM high-temp backfill + NLI (SE + U_Ecc + U_Deg).
#
# Usage:
#   bash scripts/setup_vllm_panda_venv.sh          # once
#   bash scripts/run_se_vllm_nli.sh qwen25_3b    # all math seeds
#   MODELS="llama31_8b qwen3_8b" DATASETS=math500 SEEDS=41 bash scripts/run_se_vllm_nli.sh qwen25_3b
#
# Env:
#   PANDA_VLLM_VENV / VLLM_PY  — vLLM venv (default .vllm_venv)
#   NGPU                     — parallel vLLM shards (default 1; raise on 4090 multi-GPU)
#   SE_SAMPLES=8 SE_TEMP=0.5 SE_CHUNK=32
#   SKIP_VLLM=1              — NLI-only (high_temp already present)
#   SKIP_NLI=1               — vLLM-only backfill
#   FORCE_VLLM=1             — replace placeholder SE rows (answers-only → full_response)
#
# Note: aggregate_maintable.sh is pure CPU — run it in parallel while GPUs are busy elsewhere.
set -euo pipefail
source "$(dirname "$0")/env.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_TAG="${1:?model_tag (qwen25_3b|llama32_1b|llama31_8b|qwen3_8b)}"
SEEDS="${SEEDS:-41,42,43}"
DATASETS="${DATASETS:-minerva,math500,gsm8k}"
SE_SAMPLES="${SE_SAMPLES:-8}"
SE_TEMP="${SE_TEMP:-0.5}"
SE_TOP_P="${SE_TOP_P:-0.95}"
SE_CHUNK="${SE_CHUNK:-32}"
NGPU="${NGPU:-1}"
SKIP_VLLM="${SKIP_VLLM:-0}"
SKIP_NLI="${SKIP_NLI:-0}"
FORCE_VLLM="${FORCE_VLLM:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

declare -A MODEL_PATH=(
  [qwen25_3b]="${PANDA_MODELS}/TFB-Qwen2.5-3B-Instruct"
  [llama32_1b]="${PANDA_MODELS}/TFB-Llama-3.2-1B-Instruct"
  [llama31_8b]="${PANDA_MODELS}/TFB-Llama-3.1-8B-Instruct"
  [qwen3_8b]="${PANDA_MODELS}/TFB-Qwen3-8B"
)
MODEL_PATH="${MODEL_PATH[$MODEL_TAG]:-}"
[[ -z "$MODEL_PATH" ]] && { echo "Unknown MODEL_TAG=$MODEL_TAG" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }

PANDA_VLLM_VENV="${PANDA_VLLM_VENV:-$PANDA_ROOT/.vllm_venv}"
VLLM_PY="${VLLM_PY:-$PANDA_VLLM_VENV/bin/python}"
HF_PY="${HF_PY:-python3}"
OUT_BASE="${OUT_BASE:-$PANDA_OUTPUTS/maintable_${MODEL_TAG}}"
LOG="$OUT_BASE/logs"; mkdir -p "$LOG"

export PANDA_NLI_MODEL="${PANDA_NLI_MODEL:-$PANDA_MODELS/deberta-v2-xlarge-mnli}"
export PANDA_NLI_DEVICE="${PANDA_NLI_DEVICE:-cuda:0}"

max_tokens_for() { case "$1" in gsm8k) echo "${PANDA_MAX_TOKENS_GSM8K:-1024}";; *) echo "${PANDA_MAX_TOKENS:-2048}";; esac; }
max_samples_for() { case "$1" in minerva) echo "${MAX_SAMPLES_MINERVA:-272}";; *) echo "${MAX_SAMPLES:-300}";; esac; }

if [[ "$SKIP_VLLM" != "1" ]]; then
  [[ -x "$VLLM_PY" ]] || { echo "Run: bash scripts/setup_vllm_panda_venv.sh" >&2; exit 1; }
  "$VLLM_PY" -c "import vllm" 2>/dev/null || { echo "vLLM import failed in $VLLM_PY" >&2; exit 1; }
fi

if [[ "$SKIP_NLI" != "1" ]]; then
  if [[ ! -f "$PANDA_NLI_MODEL/config.json" ]]; then
    echo "Downloading NLI model → $PANDA_NLI_MODEL"
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" "$HF_PY" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("microsoft/deberta-v2-xlarge-mnli", local_dir="${PANDA_NLI_MODEL}")
PY
  fi
  "$HF_PY" -c "import sentencepiece" 2>/dev/null || "$HF_PY" -m pip install -q sentencepiece
fi

echo "=== sample-UQ (SE+U_Ecc+U_Deg): model=$MODEL_TAG out=$OUT_BASE SE=$SE_SAMPLES T=$SE_TEMP vLLM=$VLLM_PY NLI=$HF_PY ==="

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra DS_ARR <<< "$DATASETS"

for seed in "${SEED_ARR[@]}"; do
  OUT="$OUT_BASE/seed${seed}"
  [[ -d "$OUT" ]] || { echo "skip seed$seed: no $OUT"; continue; }
  for ds in "${DS_ARR[@]}"; do
    ds=$(echo "$ds" | tr '-' '_')
    ms=$(max_samples_for "$ds")
    mt=$(max_tokens_for "$ds")
    echo ""
    echo ">>> SE seed=$seed ds=$ds samples=$ms tok=$mt"

    if [[ "$SKIP_VLLM" != "1" ]]; then
      pids=()
      for shard in $(seq 0 $((NGPU - 1))); do
        gpu=$shard
        CUDA_VISIBLE_DEVICES=$gpu "$VLLM_PY" -u -m panda.core.run_vllm_se_backfill \
          --dataset "$ds" \
          --max-new-tokens "$mt" \
          --se-samples "$SE_SAMPLES" \
          --se-temperature "$SE_TEMP" \
          --se-top-p "$SE_TOP_P" \
          --chunk-size "$SE_CHUNK" \
          --gpu-memory-utilization "$GPU_MEM_UTIL" \
          --model-path "$MODEL_PATH" \
          --out-dir "$OUT" \
          --shard-id "$shard" \
          --num-shards "$NGPU" \
          --resume \
          ${FORCE_VLLM:+--force} \
          > "$LOG/se_vllm_${ds}_seed${seed}_shard${shard}.log" 2>&1 &
        pids+=($!)
      done
      fail=0
      for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
      [[ $fail -eq 0 ]] || echo "WARN: vLLM SE backfill had $fail shard failures ($ds seed$seed)"
    fi

    if [[ "$SKIP_NLI" != "1" ]]; then
      CUDA_VISIBLE_DEVICES="${PANDA_NLI_DEVICE##cuda:}" "$HF_PY" -u -m panda.core.recompute_metrics \
        --from-cache --recompute-se \
        --out-dir "$OUT" \
        --datasets "$ds" \
        > "$LOG/se_nli_${ds}_seed${seed}.log" 2>&1 \
        || echo "WARN: NLI recompute failed ($ds seed$seed)"
    fi
  done
done

echo ""
echo "Done. Optional CPU aggregate (while GPUs busy): bash scripts/aggregate_maintable.sh $MODEL_TAG"
