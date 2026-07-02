#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Full ASE + TokUR pipeline for one TFB model.
#
# Fast mode (default for extra models via PANDA_FAST=1):
#   - Small models: 4 datasets in parallel on GPU subsets (2+2+2+2)
#   - 8B models (PANDA_8B_SEQUENTIAL=1): one dataset at a time, 8 GPUs × 1 shard
#   - --fast: sparse checkpoints, topk=10, sdpa
#   - SDPA attention, gsm8k max_new_tokens=1024
#   - llama32_1b: 16 shards (2 processes/GPU)
#
# Usage:
#   PANDA_FAST=1 bash scripts/run_panda_model_pipeline.sh llama32_1b $PANDA_MODELS/TFB-Llama-3.2-1B-Instruct
#   PANDA_8B_SEQUENTIAL=1 PANDA_FAST=1 bash scripts/run_panda_model_pipeline.sh qwen3_8b $PANDA_MODELS/TFB-Qwen3-8B
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/panda_gpu_lock.sh
source "$ROOT/scripts/panda_gpu_lock.sh"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
PANDA_PARALLEL_RECOMPUTE="${PANDA_PARALLEL_RECOMPUTE:-1}"
export TMPDIR="${TMPDIR:-/HDDDATA/phx/tmp}"
mkdir -p "$TMPDIR"

MODEL_TAG="${1:?model_tag required}"
MODEL_PATH="${2:?model_path required}"
OUT="${3:-$PANDA_OUTPUTS/ase_${MODEL_TAG}}"
LOG="$OUT/logs"
MAX_SAMPLES="${MAX_SAMPLES:-400}"
PANDA_FAST="${PANDA_FAST:-1}"
DATASETS=(minerva math500 gsm8k deepscaler)

max_samples_for() {
  case "$1" in
    minerva) echo "${MAX_SAMPLES_MINERVA:-272}" ;;
    *) echo "$MAX_SAMPLES" ;;
  esac
}

mkdir -p "$LOG" "$OUT"

# Model-specific parallelism
NUM_GPUS=8
SHARDS_PER_GPU=1
case "$MODEL_TAG" in
  llama32_1b)
    NUM_SHARDS=16
    SHARDS_PER_GPU=2
    ;;
  *)
    NUM_SHARDS=8
    ;;
esac

FAST_ARGS=()
if [[ "$PANDA_FAST" == "1" ]]; then
  FAST_ARGS+=(--fast --attn-implementation sdpa)
fi

max_tokens_for() {
  case "$1" in
    gsm8k) echo "${PANDA_MAX_TOKENS_GSM8K:-1024}" ;;
    *) echo "${PANDA_MAX_TOKENS:-2048}" ;;
  esac
}

gpu_for_shard() {
  local shard=$1
  echo $((shard / SHARDS_PER_GPU))
}

# 8B 单卡一分片：显存不足时把 shard 挪到其它空闲 GPU（避免 GPU1 被占时整片失败）
PANDA_GPU_MIN_FREE_MIB="${PANDA_GPU_MIN_FREE_MIB:-18000}"

gpu_free_mib() {
  local gpu=$1
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v g="$gpu" '$1 == g {print $2; exit}'
}

# 为 shard 选 GPU：优先 shard==gpu，否则任意未占用且空闲显存足够的卡
pick_gpu_for_shard() {
  local shard=$1
  local -n taken_ref=$2
  local g mib pref=$shard

  if [[ -z "${taken_ref[$pref]:-}" ]]; then
    mib=$(gpu_free_mib "$pref")
    if [[ "${mib:-0}" -ge "$PANDA_GPU_MIN_FREE_MIB" ]]; then
      echo "$pref"
      return 0
    fi
  fi
  for g in 0 1 2 3 4 5 6 7; do
    [[ -n "${taken_ref[$g]:-}" ]] && continue
    mib=$(gpu_free_mib "$g")
    if [[ "${mib:-0}" -ge "$PANDA_GPU_MIN_FREE_MIB" ]]; then
      echo "$g"
      return 0
    fi
  done
  return 1
}

# Parallel layout: minerva 2 GPUs, math500 3, gsm8k 3 (fast mode only)
dataset_gpu_plan() {
  case "$1" in
    minerva) echo "0,1" ;;
    math500) echo "2,3" ;;
    gsm8k) echo "4,5" ;;
    deepscaler) echo "6,7" ;;
  esac
}

launch_generate_shard() {
  local dataset=$1 shard=$2 num_shards=$3 gpu=$4
  local logfile="$LOG/${dataset}_shard${shard}.log"
  local max_tok
  max_tok=$(max_tokens_for "$dataset")
  local ms
  ms=$(max_samples_for "$dataset")
  CUDA_VISIBLE_DEVICES=$gpu env TMPDIR="$TMPDIR" python3 -u -m panda.core.run_panda_experiment \
    --mode generate \
    --dataset "$dataset" \
    --max-samples "$ms" \
    --n-rephrases 8 \
    --weight-sigma 0.03 \
    --weight-rank 4 \
    --max-new-tokens "$max_tok" \
    --topk-save 10 \
    --model-path "$MODEL_PATH" \
    --out-dir "$OUT" \
    --device cuda:0 \
    --shard-id "$shard" \
    --num-shards "$num_shards" \
    --resume \
    "${FAST_ARGS[@]}" \
    > "$logfile" 2>&1 &
}

launch_dataset_generate() {
  local dataset=$1
  local gpu_list=${2:-}
  local num_shards=${3:-$NUM_SHARDS}
  echo ""
  echo "========== [$MODEL_TAG] $dataset generate (shards=$num_shards) $(date) =========="

  local PIDS=() FAIL=0
  if [[ -n "$gpu_list" && "$PANDA_FAST" == "1" ]]; then
    IFS=',' read -ra GPUS <<< "$gpu_list"
    if [[ "$SHARDS_PER_GPU" -gt 1 ]]; then
      num_shards=$((${#GPUS[@]} * SHARDS_PER_GPU))
      local shard=0
      for gpu in "${GPUS[@]}"; do
        for _local in $(seq 0 $((SHARDS_PER_GPU - 1))); do
          echo "[gpu $gpu] shard $shard/$num_shards → $LOG/${dataset}_shard${shard}.log"
          launch_generate_shard "$dataset" "$shard" "$num_shards" "$gpu"
          PIDS+=($!)
          shard=$((shard + 1))
        done
      done
    else
      num_shards=${#GPUS[@]}
      local shard=0 gpu
      declare -A GPU_TAKEN=()
      for shard in $(seq 0 $((num_shards - 1))); do
        if ! gpu=$(pick_gpu_for_shard "$shard" GPU_TAKEN); then
          echo "ERROR: no GPU with >=${PANDA_GPU_MIN_FREE_MIB} MiB free for shard $shard" >&2
          FAIL=$((FAIL + 1))
          continue
        fi
        if [[ "$gpu" != "$shard" ]]; then
          echo "[gpu $gpu] shard $shard/$num_shards (remapped from cuda:$shard) → $LOG/${dataset}_shard${shard}.log"
        else
          echo "[gpu $gpu] shard $shard/$num_shards → $LOG/${dataset}_shard${shard}.log"
        fi
        GPU_TAKEN[$gpu]=1
        launch_generate_shard "$dataset" "$shard" "$num_shards" "$gpu"
        PIDS+=($!)
      done
    fi
  else
    for shard in $(seq 0 $((num_shards - 1))); do
      local gpu
      gpu=$(gpu_for_shard "$shard")
      echo "[gpu $gpu] shard $shard/$num_shards"
      launch_generate_shard "$dataset" "$shard" "$num_shards" "$gpu"
      PIDS+=($!)
    done
  fi

  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAIL=$((FAIL + 1)); fi
  done
  [[ $FAIL -gt 0 ]] && echo "WARNING: ${FAIL} generate shards failed for ${dataset}"
  python3 -m panda.core.recompute_metrics --out-dir "$OUT" --datasets "$dataset"
  local n
  n=$(find "$OUT/$dataset/raw_runs" -name '*.json' ! -name '*.error.json' 2>/dev/null | wc -l)
  echo "[$MODEL_TAG] $dataset generate done: $n raw files $(date)"
}

run_tokur_shard() {
  local dataset=$1 shard=$2 num_shards=$3 gpu=$4
  local logfile="$LOG/tokur_${dataset}_shard${shard}.log"
  CUDA_VISIBLE_DEVICES=$gpu env TMPDIR="$TMPDIR" python3 -u -m panda.core.score_tokur_baseline \
    --out-dir "$OUT" \
    --dataset "$dataset" \
    --model-path "$MODEL_PATH" \
    --device cuda:0 \
    --shard-id "$shard" \
    --num-shards "$num_shards" \
    --resume \
    > "$logfile" 2>&1 &
}

run_tokur_dataset() {
  local dataset=$1
  local gpu_list=${2:-}
  local num_shards=${3:-$NUM_SHARDS}
  echo ""
  echo "========== [$MODEL_TAG] $dataset TokUR (shards=$num_shards) $(date) =========="
  local PIDS=()
  if [[ -n "$gpu_list" && "$PANDA_FAST" == "1" ]]; then
    IFS=',' read -ra GPUS <<< "$gpu_list"
    if [[ "$SHARDS_PER_GPU" -gt 1 ]]; then
      num_shards=$((${#GPUS[@]} * SHARDS_PER_GPU))
      local shard=0
      for gpu in "${GPUS[@]}"; do
        for _local in $(seq 0 $((SHARDS_PER_GPU - 1))); do
          run_tokur_shard "$dataset" "$shard" "$num_shards" "$gpu"
          PIDS+=($!)
          shard=$((shard + 1))
        done
      done
    else
      num_shards=${#GPUS[@]}
      local shard=0
      for gpu in "${GPUS[@]}"; do
        run_tokur_shard "$dataset" "$shard" "$num_shards" "$gpu"
        PIDS+=($!)
        shard=$((shard + 1))
      done
    fi
  else
    for shard in $(seq 0 $((num_shards - 1))); do
      run_tokur_shard "$dataset" "$shard" "$num_shards" "$(gpu_for_shard "$shard")"
      PIDS+=($!)
    done
  fi
  for pid in "${PIDS[@]}"; do wait "$pid" || true; done
  python3 -m panda.core.score_tokur_baseline --out-dir "$OUT" --dataset "$dataset" --merge-only
}

raw_count() {
  local dataset=$1
  find "$OUT/$dataset/raw_runs" -name '*.json' ! -name '*.error.json' ! -name '*.partial.json' 2>/dev/null | wc -l
}

all_generate_done() {
  for ds in "${DATASETS[@]}"; do
    local need
    need=$(max_samples_for "$ds")
    [[ $(raw_count "$ds") -ge $((need - 5)) ]] || return 1
  done
  return 0
}

dataset_generate_done() {
  local ds=$1
  local need
  need=$(max_samples_for "$ds")
  [[ $(raw_count "$ds") -ge $((need - 5)) ]]
}

is_8b_model() {
  case "$MODEL_TAG" in
    llama31_8b|qwen3_8b) return 0 ;;
    *) return 1 ;;
  esac
}

# Order datasets by most remaining samples first (8B sequential mode).
# math500 first when incomplete — resume interrupted shards before larger gaps.
ordered_datasets_by_remaining() {
  local ds need have rem
  for ds in math500 deepscaler minerva gsm8k; do
    if dataset_generate_done "$ds"; then
      continue
    fi
    echo "$ds"
  done
}

ALL_GPUS="0,1,2,3,4,5,6,7"
PANDA_8B_SEQUENTIAL="${PANDA_8B_SEQUENTIAL:-0}"
if is_8b_model && [[ "$PANDA_8B_SEQUENTIAL" == "1" ]]; then
  USE_8B_SEQUENTIAL=1
else
  USE_8B_SEQUENTIAL=0
fi

echo "PANDA pipeline [$MODEL_TAG] started $(date)"
echo "  model: $MODEL_PATH"
echo "  out:   $OUT"
echo "  fast:  $PANDA_FAST  shards=$NUM_SHARDS  shards_per_gpu=$SHARDS_PER_GPU  8b_sequential=$USE_8B_SEQUENTIAL"

if [[ "${PANDA_SKIP_GENERATE:-0}" == "1" ]] || all_generate_done; then
  echo "Skip generate (raw already complete or PANDA_SKIP_GENERATE=1)"
else
  acquire_gpu ase_generate "pipeline $MODEL_TAG generate"
  if [[ "$USE_8B_SEQUENTIAL" == "1" ]]; then
    echo "8B sequential mode: one dataset at a time, 8 GPUs (0-7), ordered by remaining samples"
    mapfile -t PENDING_DS < <(ordered_datasets_by_remaining)
    if ((${#PENDING_DS[@]} == 0)); then
      echo "All datasets at target count"
    fi
    for ds in "${PENDING_DS[@]}"; do
      if dataset_generate_done "$ds"; then
        echo "Skip $ds (already at target: $(raw_count "$ds")/$(max_samples_for "$ds"))"
        continue
      fi
      echo ">>> [$MODEL_TAG] focus dataset $ds (have $(raw_count "$ds")/$(max_samples_for "$ds"))"
      launch_dataset_generate "$ds" "$ALL_GPUS" 8
    done
  elif [[ "$PANDA_FAST" == "1" ]]; then
    echo "Fast mode: 4 datasets in parallel (GPU 0-1 / 2-3 / 4-5 / 6-7)"
    PIDS=()
    for ds in "${DATASETS[@]}"; do
      gpus=$(dataset_gpu_plan "$ds")
      launch_dataset_generate "$ds" "$gpus" &
      PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid" || true; done
  else
    for ds in "${DATASETS[@]}"; do
      launch_dataset_generate "$ds"
    done
  fi
  release_gpu ase_generate
fi

if [[ "$PANDA_PARALLEL_RECOMPUTE" == "1" ]]; then
  bash "$ROOT/scripts/recompute_metrics_parallel.sh" "$OUT" "$(IFS=,; echo "${DATASETS[*]}")"
else
  python3 -m panda.core.recompute_metrics --out-dir "$OUT" --datasets "$(IFS=,; echo "${DATASETS[*]}")"
fi

if [[ "${PANDA_SKIP_TOKUR:-0}" != "1" ]]; then
  if [[ "${PANDA_TOKUR_STRICT:-0}" == "1" ]]; then
    echo "Official TokUR (vLLM) via run_tokur_strict_baseline.sh"
    case "$MODEL_TAG" in
      qwen25_3b) _TOKUR_TAG=qwen3b ;;
      llama32_1b) _TOKUR_TAG=llama1b ;;
      llama31_8b) _TOKUR_TAG=llama8b ;;
      qwen3_8b) _TOKUR_TAG=qwen8b ;;
      *) echo "Unknown MODEL_TAG for strict TokUR: $MODEL_TAG" >&2; exit 1 ;;
    esac
    acquire_gpu tokur_strict "pipeline $MODEL_TAG strict tokur"
    OUT_DIR="$OUT" MODEL_TAG="$_TOKUR_TAG" DATASETS="${DATASETS[*]}" \
      PARALLEL_SHARDS="${PANDA_TOKUR_PARALLEL_SHARDS:-1}" \
      bash "$ROOT/scripts/run_tokur_strict_baseline.sh"
    release_gpu tokur_strict
  else
    echo "WARN: approx TokUR (score_tokur_baseline). Set PANDA_TOKUR_STRICT=1 for official vLLM."
    acquire_gpu ase_generate "pipeline $MODEL_TAG tokur"
    if [[ "$PANDA_FAST" == "1" ]]; then
      PIDS=()
      for ds in "${DATASETS[@]}"; do
        gpus=$(dataset_gpu_plan "$ds")
        run_tokur_dataset "$ds" "$gpus" &
        PIDS+=($!)
      done
      for pid in "${PIDS[@]}"; do wait "$pid" || true; done
    else
      for ds in "${DATASETS[@]}"; do
        run_tokur_dataset "$ds"
      done
    fi
    release_gpu ase_generate
  fi
else
  echo "Skip TokUR (PANDA_SKIP_TOKUR=1)"
fi

PAPER_DIR="$ROOT/paper/models/${MODEL_TAG}"
mkdir -p "$PAPER_DIR/tables"
for ds in "${DATASETS[@]}"; do
  python3 -m panda.core.analyze_main_tables --out-dir "$OUT" --dataset "$ds"
done
python3 -m panda.core.analyze_paper_tables --out-dir "$OUT" --paper-dir "$PAPER_DIR" || true

echo "PANDA pipeline [$MODEL_TAG] DONE $(date) → $OUT"
