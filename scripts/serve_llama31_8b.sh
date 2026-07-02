#!/usr/bin/env bash
# 用 vLLM 启动 Meta-Llama-3.1-8B-Instruct（OpenAI 兼容 API）
# 用法: bash scripts/serve_llama31_8b.sh [--port 8002] [--max-model-len 8192]
set -euo pipefail
source "$(dirname "$0")/activate_env.sh"

PORT="8002"
MAX_MODEL_LEN="8192"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    -h|--help)
      echo "用法: $0 [--port PORT] [--max-model-len LEN]"
      exit 0
      ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

MODEL_PATH="${PANDA_MODELS}/Meta-Llama-3.1-8B-Instruct"
[[ -f "${MODEL_PATH}/config.json" ]] || {
  echo "模型未下载: ${MODEL_PATH}" >&2
  echo "请先运行: bash scripts/download_llama31_8b.sh" >&2
  exit 1
}

LOG_DIR="${PANDA_ROOT}/../logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/vllm-llama31-8b-${PORT}.log"

echo "启动 Llama-3.1-8B @ 0.0.0.0:${PORT}"
echo "模型: ${MODEL_PATH}"
echo "日志: ${LOG_FILE}"

# 若 GPU 已被其它 vLLM 占用，请先停止旧服务或换端口/GPU
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  2>&1 | tee -a "${LOG_FILE}"
