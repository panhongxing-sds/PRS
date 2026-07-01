#!/usr/bin/env bash
# vLLM 全 benchmark 采样（Gemma-3 自动走 HF）
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
GPU=${GPU:-0}
: "${MODEL_TAG:?}"
: "${MODEL_PATH:?}"
BACKEND=${BACKEND:-vllm}

CFG=/tmp/spurious_consensus_$$.yaml
$PY - <<PY
import os, yaml
cfg = yaml.safe_load(open("config.yaml"))
cfg["model"]["tag"] = os.environ["MODEL_TAG"]
cfg["model"]["path"] = os.environ["MODEL_PATH"]
yaml.dump(cfg, open("$CFG", "w"))
PY

SAMPLER=sample_vllm.py
if [[ "$BACKEND" == "hf" ]]; then
  SAMPLER=sample.py
fi

for b in aime_2024 gpqa_diamond deepscaler; do
  echo "=== $MODEL_TAG / $b ($BACKEND) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY "$SAMPLER" --config "$CFG" --benchmark "$b" --resume
done
rm -f "$CFG"
