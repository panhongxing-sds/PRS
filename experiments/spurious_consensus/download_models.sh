#!/usr/bin/env bash
# 下载非 Qwen 的较新长上下文 Instruct 模型（ModelScope）
set -euo pipefail
PRS_MODELS="${PRS_MODELS:-/root/autodl-tmp/prs-models}"
mkdir -p "$PRS_MODELS"

download_ms() {
  local ms_id="$1" local_dir="$2"
  if [[ -f "${local_dir}/config.json" ]] && ls "${local_dir}"/*.safetensors &>/dev/null 2>&1; then
    echo "[skip] 已存在 ${local_dir}"
    return 0
  fi
  echo "[download] ${ms_id} -> ${local_dir}"
  modelscope download --model "${ms_id}" --local_dir "${local_dir}" --exclude "original/*"
}

# Microsoft Phi-4-mini (128k, 2025-02) + Google Gemma-3-4B (128k, 2025-03)
download_ms "LLM-Research/Phi-4-mini-instruct" "${PRS_MODELS}/Phi-4-mini-instruct"
download_ms "google/gemma-3-4b-it" "${PRS_MODELS}/gemma-3-4b-it"

echo ""
echo "=== 实验用模型 ==="
for d in \
  TFB-Qwen2.5-3B-Instruct \
  TFB-Llama-3.2-1B-Instruct \
  Phi-4-mini-instruct \
  gemma-3-4b-it; do
  if [[ -f "${PRS_MODELS}/${d}/config.json" ]]; then
    du -sh "${PRS_MODELS}/${d}"
  else
    echo "[missing] ${d}"
  fi
done
