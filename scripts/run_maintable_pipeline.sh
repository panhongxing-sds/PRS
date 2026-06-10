#!/usr/bin/env bash
# 完整主表管线：可选 GPU 实验 + CPU 聚合
#
# Usage:
#   # 仅逻辑+代码表（Llama-3.2-1B，参考论文主图）
#   DATASETS=leg_counting,zebra_puzzles,color_cube,humaneval bash scripts/run_maintable_pipeline.sh llama32_1b
#
#   # 全数据集 + 跳 GPU（已有 raw）
#   SKIP_GPU=1 bash scripts/run_maintable_pipeline.sh llama32_1b
#
#   # 四模型
#   for m in qwen25_3b llama32_1b llama31_8b qwen3_8b; do
#     bash scripts/run_maintable_pipeline.sh "$m"
#   done
set -euo pipefail
source "$(dirname "$0")/env.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_TAG="${1:?model_tag}"
SKIP_GPU="${SKIP_GPU:-0}"
PREPARE_DATA="${PREPARE_DATA:-1}"

echo "=== Maintable pipeline: $MODEL_TAG ==="

if [[ "$PREPARE_DATA" == "1" ]]; then
  echo "[1/4] Check datasets..."
  bash "$ROOT/scripts/download_tokur_datasets.sh" || true
  bash "$ROOT/scripts/download_logic_code_datasets.sh" || true
  echo "  Generate API variants if missing (requires API key):"
  echo "    bash scripts/generate_api_variants.sh \${DATASETS}"
fi

if [[ "$SKIP_GPU" != "1" ]]; then
  echo "[2/4] GPU experiments (3 seeds)..."
  bash "$ROOT/scripts/run_maintable_3seed.sh" "$MODEL_TAG"
else
  echo "[2/4] Skip GPU (SKIP_GPU=1)"
fi

echo "[3/4] Recompute + aggregate tables..."
bash "$ROOT/scripts/aggregate_maintable.sh" "$MODEL_TAG"

echo "[4/4] Done → paper/maintable/${MODEL_TAG}/"
