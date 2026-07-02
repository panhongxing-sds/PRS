#!/bin/bash
source "$(dirname "$0")/env.sh"
# Strict TokUR baseline for ASE paper tables (official vLLM + TFB, NOT post-hoc PANDA weight EU).
#
# Pipeline per dataset:
#   1. Export ASE raw_runs → TokUR/datasets/panda_{dataset}.jsonl
#   2. greedy_unc_single_batch_refine.py (temp=0, native EU during generation)
#   3. pkl → outputs/panda_full/{dataset}/tokur_baseline.jsonl
#
# Official settings (TokUR repo):
#   TFB-Qwen2.5-3B-Instruct: bayes_sigma=0.1, rank-8 basis on q_proj+v_proj, num_samples=5
#   greedy temperature=0, stop_token_ids [151645, 151643] for Qwen
#   multi-seed paper eval: 96 89 64 (default single seed 96 for PANDA jsonl)
#
# Usage:
#   bash scripts/run_tokur_strict_baseline.sh
#   DATASETS="math500 gsm8k" SEEDS="96" bash scripts/run_tokur_strict_baseline.sh
#   SKIP_GENERATE=1 bash scripts/run_tokur_strict_baseline.sh   # convert existing pkl only
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOKUR_ROOT="${TOKUR_ROOT:-$PANDA_ROOT/third_party/TokUR}"
OUT_DIR="${OUT_DIR:-$PANDA_OUTPUTS/panda_full}"
MODEL_TAG="${MODEL_TAG:-qwen3b}"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-$PANDA_MODELS}"
DATASETS="${DATASETS:-math500 gsm8k deepscaler minerva}"
SEEDS="${SEEDS:-96}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS=(${GPU_IDS:-0 1 2 3 4 5 6 7})
BATCH_SIZE="${BATCH_SIZE:-16}"
# Launch one vLLM shard at a time (safer on shared GPUs). Set PARALLEL_SHARDS=1 for all-at-once.
PARALLEL_SHARDS="${PARALLEL_SHARDS:-0}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKUR_VENV="${TOKUR_VENV:-${PANDA_ROOT}/.tokur_venv}"
export TOKUR_PY="${TOKUR_PY:-${TOKUR_VENV}/bin/python}"
export PATH="${TOKUR_VENV}/bin:${PATH}"
export TMPDIR="${TMPDIR:-/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMPDIR}/pip-cache}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TOKUR_MAX_MODEL_LEN="${TOKUR_MAX_MODEL_LEN:-4096}"
export TOKUR_MAX_NUM_BATCHED_TOKENS="${TOKUR_MAX_NUM_BATCHED_TOKENS:-4096}"

if [[ ! -x "$TOKUR_PY" ]]; then
  echo "Missing TokUR venv: $TOKUR_PY (run bash scripts/setup_after_clone.sh)" >&2
  exit 1
fi

resolve_model_path() {
  case "$MODEL_TAG" in
    qwen3b)
      for p in "$MODEL_BASE_DIR/TFB-Qwen2.5-3B-Instruct" "$PANDA_MODELS/TFB-Qwen2.5-3B-Instruct"; do
        [[ -d "$p" ]] && echo "$p" && return 0
      done ;;
    qwen8b)
      for p in "$MODEL_BASE_DIR/TFB-Qwen3-8B" "$PANDA_MODELS/TFB-Qwen3-8B"; do
        [[ -d "$p" ]] && echo "$p" && return 0
      done ;;
    llama8b)
      for p in "$MODEL_BASE_DIR/TFB-Meta-Llama-3.1-8B-Instruct" "$PANDA_MODELS/TFB-Llama-3.1-8B-Instruct"; do
        [[ -d "$p" ]] && echo "$p" && return 0
      done ;;
    llama1b)
      for p in \
        "$MODEL_BASE_DIR/TFB-Llama3.2-1B-Instruct" \
        "$PANDA_MODELS/TFB-Llama-3.2-1B-Instruct" \
        "$PANDA_MODELS/TFB-Llama3.2-1B-Instruct"; do
        [[ -d "$p" ]] && echo "$p" && return 0
      done ;;
    *)
      echo "Unknown MODEL_TAG=$MODEL_TAG" >&2
      return 1 ;;
  esac
  echo "TFB weights not found for $MODEL_TAG" >&2
  return 1
}

# Per-model TokUR dataset slug (jsonl + pkl subdir). Legacy qwen3b/panda_full keeps panda_{ds}.
tokur_dataset_slug() {
  local dataset=$1
  if [[ "${TOKUR_DS_LEGACY:-0}" == "1" ]]; then
    echo "ase_${dataset}"
  else
    echo "ase_${MODEL_TAG}_${dataset}"
  fi
}

MODEL_PATH="$(resolve_model_path)" || exit 1
# Legacy: existing qwen2.5 strict pkls live under results/qwen3b_results_vllm_pg/panda_{dataset}/
if [[ "$MODEL_TAG" == "qwen3b" && "$(basename "$OUT_DIR")" == "panda_full" ]]; then
  TOKUR_DS_LEGACY=1
else
  TOKUR_DS_LEGACY=0
fi
echo "TFB model: $MODEL_PATH (legacy_tokur_ds=$TOKUR_DS_LEGACY)"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Missing config.json under $MODEL_PATH" >&2
  exit 1
fi
"$TOKUR_PY" -c "import json; c=json.load(open('$MODEL_PATH/config.json')); print('bayes_sigma', c.get('bayes_sigma'), 'num_samples', c.get('num_samples'), 'basis_idx len', len(c.get('basis_idx',[])))"

cd "$ROOT"
PANDA_PY="${ROOT}/src"

for dataset in $DATASETS; do
  echo "======== strict TokUR: $dataset ========"
  TOKUR_DS="$(tokur_dataset_slug "$dataset")"
  PANDA_JSONL="$TOKUR_ROOT/datasets/${TOKUR_DS}.jsonl"

  if [[ "$SKIP_EXPORT" != "1" ]]; then
    PYTHONPATH="${PANDA_PY}:${PYTHONPATH:-}" python3 -m panda.core.export_tokur_jsonl \
      --out-dir "$OUT_DIR" \
      --dataset "$dataset" \
      --jsonl "$PANDA_JSONL" \
      --tokur-root "$TOKUR_ROOT"
  fi

  if [[ ! -f "$PANDA_JSONL" ]]; then
    echo "Missing $PANDA_JSONL (export failed?)" >&2
    exit 1
  fi
  TOTAL=$(wc -l < "$PANDA_JSONL" | tr -d ' ')
  echo "PANDA jsonl: $PANDA_JSONL ($TOTAL lines)"

  if [[ "$SKIP_GENERATE" != "1" ]]; then
    for seed in $SEEDS; do
      OUT_PKL="$TOKUR_ROOT/results/${MODEL_TAG}_results_vllm_pg/${TOKUR_DS}/seed${seed}/greedy_unc"
      mkdir -p "$OUT_PKL"
      chunk=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
      launched=0
      for ((i=0; i<NUM_GPUS; i++)); do
        gpu=${GPU_IDS[$i]:-$i}
        start=$(( i * chunk ))
        end=$(( start + chunk ))
        [[ $start -ge $TOTAL ]] && continue
        [[ $end -gt $TOTAL ]] && end=$TOTAL
        echo "  seed=$seed GPU=$gpu [$start,$end)"
        # Skip shard if a pkl already covers this index range (resume-friendly).
        if compgen -G "$OUT_PKL/batch_results_${start}_*.pkl" >/dev/null 2>&1; then
          echo "    SKIP (pkl exists for start=$start)"
          continue
        fi
        run_shard() {
          cd "$TOKUR_ROOT"
          env -u PYTHONPATH CUDA_VISIBLE_DEVICES=$gpu "$TOKUR_PY" run/greedy_unc_single_batch_refine.py \
            --dataset-path "$PANDA_JSONL" \
            --dataset-start "$start" \
            --dataset-end "$end" \
            --model-path "$MODEL_PATH" \
            --output-dir "$OUT_PKL" \
            --seed "$seed" \
            --batch-size "$BATCH_SIZE"
        }
        if [[ "$PARALLEL_SHARDS" == "1" ]]; then
          run_shard &
          launched=$((launched + 1))
        else
          run_shard
        fi
      done
      [[ $launched -gt 0 ]] && wait
    done
  fi

  MERGE_ARG=""
  if [[ "$(echo "$SEEDS" | wc -w)" -gt 1 ]]; then
    MERGE_ARG="--merge-seeds $(echo "$SEEDS" | tr ' ' ',')"
  fi
  LEGACY_FLAG=""
  [[ "$TOKUR_DS_LEGACY" == "1" ]] && LEGACY_FLAG="--legacy-tokur-ds"
  PYTHONPATH="${PANDA_PY}:${PYTHONPATH:-}" python3 -m panda.core.score_tokur_official \
    --out-dir "$OUT_DIR" \
    --dataset "$dataset" \
    --tokur-root "$TOKUR_ROOT" \
    --model-tag "$MODEL_TAG" \
    --seed $(echo "$SEEDS" | awk '{print $1}') \
    $LEGACY_FLAG \
    $MERGE_ARG

  n=$(wc -l < "$OUT_DIR/$dataset/tokur_baseline.jsonl" | tr -d ' ')
  echo "→ $OUT_DIR/$dataset/tokur_baseline.jsonl ($n rows)"
done

echo ""
echo "Done. Re-run ASE tables:"
echo "  python3 -m panda.core.recompute_metrics --out-dir $OUT_DIR --datasets $(echo $DATASETS | tr ' ' ',')"
echo "  python3 -m panda.core.analyze_main_tables --out-dir $OUT_DIR --datasets $(echo $DATASETS | tr ' ' ',')"
