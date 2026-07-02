#!/usr/bin/env bash
# 下载 Meta-Llama-3.1-8B-Instruct 并转换为 TFB（PANDA 实验所需）
# 用法: bash scripts/download_llama31_8b.sh
set -euo pipefail
source "$(dirname "$0")/activate_env.sh"

BASE_MODEL="${PANDA_MODELS}/Meta-Llama-3.1-8B-Instruct"
TFB_MODEL="${PANDA_MODELS}/TFB-Llama-3.1-8B-Instruct"
MS_ID="LLM-Research/Meta-Llama-3.1-8B-Instruct"

mkdir -p "${PANDA_MODELS}"

if [[ ! -f "${BASE_MODEL}/config.json" ]]; then
  echo ">>> 下载基础模型 ${MS_ID} -> ${BASE_MODEL}"
  pip install -q modelscope
  # 仅下载 HuggingFace 格式权重，跳过 15GB original/*.pth
  modelscope download --model "${MS_ID}" --local_dir "${BASE_MODEL}" \
    --exclude "original/*"
else
  echo ">>> 基础模型已存在: ${BASE_MODEL}"
fi

if [[ ! -f "${TFB_MODEL}/config.json" ]]; then
  echo ">>> 转换为 TFB 模型 -> ${TFB_MODEL}"
  python "${TOKUR_ROOT}/convert_to_tfb.py" \
    --model-path "${BASE_MODEL}" \
    --output-path "${TFB_MODEL}" \
    --architecture llama \
    --rank 8
else
  echo ">>> TFB 模型已存在: ${TFB_MODEL}"
fi

echo ">>> 完成。运行 PANDA pipeline 示例:"
echo "  PANDA_8B_SEQUENTIAL=1 PANDA_FAST=1 bash scripts/run_panda_model_pipeline.sh llama31_8b ${TFB_MODEL}"
