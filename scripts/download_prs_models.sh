#!/usr/bin/env bash
# 下载并转换 PRS/TokUR 四模型 TFB 权重
# 用法:
#   bash scripts/download_prs_models.sh all
#   bash scripts/download_prs_models.sh qwen3b llama1b qwen8b
set -euo pipefail
source "$(dirname "$0")/activate_env.sh"

pip install -q modelscope

download_and_convert() {
  local tag="$1" ms_id="$2" base_name="$3" tfb_name="$4" arch="$5"
  local base_dir="${PRS_MODELS}/${base_name}"
  local tfb_dir="${PRS_MODELS}/${tfb_name}"

  if [[ -f "${base_dir}/config.json" ]]; then
    echo ">>> [${tag}] 基础模型已存在: ${base_dir}"
  else
    echo ">>> [${tag}] 下载 ${ms_id} -> ${base_dir}"
    mkdir -p "${base_dir}"
    modelscope download --model "${ms_id}" --local_dir "${base_dir}" \
      --exclude "original/*"
  fi

  if [[ -f "${tfb_dir}/config.json" ]]; then
    echo ">>> [${tag}] TFB 已存在: ${tfb_dir}"
  else
    echo ">>> [${tag}] 转换 TFB -> ${tfb_dir}"
    python "${TOKUR_ROOT}/convert_to_tfb.py" \
      --model-path "${base_dir}" \
      --output-path "${tfb_dir}" \
      --architecture "${arch}" \
      --rank 8
  fi
  echo ">>> [${tag}] 完成"
}

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 || "${TARGETS[0]}" == "all" ]]; then
  TARGETS=(qwen3b llama1b llama8b qwen8b)
fi

mkdir -p "${PRS_MODELS}"

for tag in "${TARGETS[@]}"; do
  case "${tag}" in
    qwen3b)
      download_and_convert qwen3b \
        "Qwen/Qwen2.5-3B-Instruct" \
        "Qwen2.5-3B-Instruct" \
        "TFB-Qwen2.5-3B-Instruct" \
        qwen2 ;;
    llama1b)
      download_and_convert llama1b \
        "LLM-Research/Llama-3.2-1B-Instruct" \
        "Llama-3.2-1B-Instruct" \
        "TFB-Llama-3.2-1B-Instruct" \
        llama ;;
    llama8b)
      download_and_convert llama8b \
        "LLM-Research/Meta-Llama-3.1-8B-Instruct" \
        "Meta-Llama-3.1-8B-Instruct" \
        "TFB-Llama-3.1-8B-Instruct" \
        llama ;;
    qwen8b)
      download_and_convert qwen8b \
        "Qwen/Qwen3-8B" \
        "Qwen3-8B" \
        "TFB-Qwen3-8B" \
        qwen2 ;;
    *)
      echo "未知模型标签: ${tag}（可选: qwen3b llama1b llama8b qwen8b all）" >&2
      exit 1 ;;
  esac
done

echo ""
echo "=== 模型目录 ==="
ls -lh "${PRS_MODELS}"/TFB-* 2>/dev/null | head -20 || ls -la "${PRS_MODELS}"
