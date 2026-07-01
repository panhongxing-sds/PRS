#!/usr/bin/env bash
# 单卡跑某模型的一个 shard（全 benchmark）。vLLM。
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
GPU="${GPU:?}"
MODEL_TAG="${MODEL_TAG:?}"
MODEL_PATH="${MODEL_PATH:?}"
SHARD_ID="${SHARD_ID:?}"
NUM_SHARDS="${NUM_SHARDS:?}"
K_CHUNK="${K_CHUNK:-64}"
PROMPT_BATCH="${PROMPT_BATCH:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/autodl-tmp/vllm_cache}"
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT"

CFG="$TMPDIR/sc_${MODEL_TAG}_sh${SHARD_ID}_$$.yaml"
$PY - <<PY
import os, yaml
c = yaml.safe_load(open("config.yaml"))
c["model"]["tag"] = os.environ["MODEL_TAG"]
c["model"]["path"] = os.environ["MODEL_PATH"]
c["sampling"]["k_chunk"] = int(os.environ["K_CHUNK"])
c["sampling"]["prompt_batch"] = int(os.environ["PROMPT_BATCH"])
yaml.dump(c, open("$CFG", "w"))
PY

OUT_SUFFIX="${OUT_SUFFIX:-}"
for b in deepscaler gpqa_diamond aime_2024; do
  echo "=== $MODEL_TAG / $b shard$SHARD_ID/$NUM_SHARDS suffix=$OUT_SUFFIX ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY sample_vllm.py --config "$CFG" --benchmark "$b" \
    --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" --out-suffix "$OUT_SUFFIX" --resume || {
    echo "[warn] $b shard$SHARD_ID 中断，60s 重试" >&2; sleep 60
    CUDA_VISIBLE_DEVICES=$GPU $PY sample_vllm.py --config "$CFG" --benchmark "$b" \
      --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" --out-suffix "$OUT_SUFFIX" --resume
  }
done
rm -f "$CFG"
