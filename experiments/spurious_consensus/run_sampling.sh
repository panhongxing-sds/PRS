#!/usr/bin/env bash
# 跨模型全量采样（默认 vLLM；Gemma-3 设 BACKEND=hf）
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
GPU=${GPU:-0}
: "${MODEL_TAG:?}"
: "${MODEL_PATH:?}"
BACKEND=${BACKEND:-vllm}
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 缓存/临时全部放数据盘，避免写满系统盘
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/autodl-tmp/vllm_cache}"
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT"

CFG="$TMPDIR/spurious_consensus_$$.yaml"
$PY - <<PY
import os, yaml
cfg = yaml.safe_load(open("config.yaml"))
cfg["model"]["tag"] = os.environ["MODEL_TAG"]
cfg["model"]["path"] = os.environ["MODEL_PATH"]
if os.environ.get("K_CHUNK"):
    cfg["sampling"]["k_chunk"] = int(os.environ["K_CHUNK"])
if os.environ.get("PROMPT_BATCH"):
    cfg["sampling"]["prompt_batch"] = int(os.environ["PROMPT_BATCH"])
yaml.dump(cfg, open("$CFG", "w"))
PY

SAMPLER=sample_vllm.py
if [[ "$BACKEND" == "hf" ]]; then
  SAMPLER=sample.py
fi

for b in deepscaler gpqa_diamond aime_2024; do
  echo "=== $MODEL_TAG / $b ($BACKEND) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY "$SAMPLER" --config "$CFG" --benchmark "$b" --resume || {
    echo "[warn] $b 中断，60s 后重试..." >&2
    sleep 60
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SAMPLER" --config "$CFG" --benchmark "$b" --resume
  }
done
rm -f "$CFG"
