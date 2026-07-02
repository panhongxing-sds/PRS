#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Full ASE: raw_runs/{id}.json per sample (token traces, top-k, perturb config, semantic cache)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/HDDDATA/phx/tmp}"
mkdir -p "$TMPDIR"

OUT=$PANDA_OUTPUTS/panda_full
LOG=$OUT/logs
MODEL=$PANDA_MODELS/TFB-Qwen2.5-3B-Instruct
NUM_GPUS=8
MAX_SAMPLES=200
DATASETS=(minerva math500 gsm8k deepscaler)

mkdir -p "$LOG" "$OUT"

launch_shards() {
  local dataset=$1
  echo ""
  echo "========== $dataset (max_samples=$MAX_SAMPLES) =========="
  local PIDS=()
  for shard in $(seq 0 $((NUM_GPUS - 1))); do
    log="$LOG/${dataset}_shard${shard}.log"
    echo "[gpu $shard] → $log"
    CUDA_VISIBLE_DEVICES=$shard env TMPDIR="$TMPDIR" python3 -m panda.core.run_panda_experiment \
      --mode generate \
      --dataset "$dataset" \
      --max-samples "$MAX_SAMPLES" \
      --n-rephrases 8 \
      --weight-sigma 0.03 \
      --weight-rank 4 \
      --max-new-tokens 2048 \
      --topk-save 20 \
      --model-path "$MODEL" \
      --out-dir "$OUT" \
      --device cuda:0 \
      --shard-id "$shard" \
      --num-shards "$NUM_GPUS" \
      --resume \
      > "$log" 2>&1 &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || echo "WARN pid $pid failed"; done
  python3 -m panda.core.recompute_metrics --out-dir "$OUT" --datasets "$dataset"
  n=$(find "$OUT/$dataset/raw_runs" -name '*.json' ! -name '*.error.json' 2>/dev/null | wc -l)
  echo "$dataset done: $n raw files"
}

echo "PANDA FULL (rich raw_runs) started $(date)"
for dataset in "${DATASETS[@]}"; do
  launch_shards "$dataset"
done
python3 -m panda.core.recompute_metrics --out-dir "$OUT" --datasets "$(IFS=,; echo "${DATASETS[*]}")"
python3 -m panda.core.analyze_ase --out-dir "$OUT" --datasets "$(IFS=,; echo "${DATASETS[*]}")"
python3 -m panda.core.analyze_atu_extended --out-dir "$OUT" --datasets "$(IFS=,; echo "${DATASETS[*]}")"
echo "DONE $(date) → $OUT"
