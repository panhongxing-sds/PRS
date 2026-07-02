#!/usr/bin/env bash
# 4090 机器一键部署：依赖、迁移 Qwen 样本、冒烟测试
set -euo pipefail
cd "$(dirname "$0")"

export PANDA_ROOT="${PANDA_ROOT:-/root/PANDA}"
export PANDA_MODELS="${PANDA_MODELS:-/root/autodl-tmp/panda-models}"

echo "=== [1/6] Python 依赖 ==="
pip install -q -U pip
pip install -q -r requirements.txt

echo "=== [2/6] vLLM（可选，默认跳过）==="
if [[ "${INSTALL_VLLM:-0}" == "1" ]]; then
  if ! python -c "import vllm" 2>/dev/null; then
    pip install -q "vllm>=0.8.0" || echo "[warn] vLLM 安装失败，将使用 HF sample.py"
  fi
else
  echo "[skip] 未装 vLLM；采样用 HF: ./run_sampling.sh 或 ./run_phase1.sh"
fi

echo "=== [3/6] 模型检查 ==="
for d in \
  TFB-Qwen2.5-3B-Instruct \
  TFB-Llama-3.2-1B-Instruct \
  Phi-4-mini-instruct \
  gemma-3-4b-it; do
  if [[ -f "$PANDA_MODELS/$d/config.json" ]]; then
    du -sh "$PANDA_MODELS/$d"
  else
    echo "[missing] $PANDA_MODELS/$d — 先运行 ./download_models.sh"
    exit 1
  fi
done

echo "=== [4/6] 迁入 Qwen 样本 ==="
if [[ -d /root/aul-study/data ]]; then
  python migrate_qwen_samples.py
  python check_samples.py --tag qwen25_3b || true
else
  echo "[skip] 无 /root/aul-study/data"
fi

chmod +x run_sampling.sh run_sampling_vllm.sh run_phase1.sh download_models.sh
mkdir -p data/samples logs/phase1 figures results

echo "=== [5/6] GPU 检查 ==="
if ! nvidia-smi &>/dev/null; then
  echo "[warn] 未检测到 GPU；跳过冒烟测试。上机后运行:"
  echo "  CUDA_VISIBLE_DEVICES=0 ./setup_4090.sh --smoke"
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "${1:-}" == "--smoke" ]] || [[ "${SMOKE:-}" == "1" ]]; then
  echo "=== [6/6] 冒烟测试 aime_2024 (k=4) ==="
  CFG=/tmp/sc_smoke_$$.yaml
  python - <<PY
import yaml, os
cfg = yaml.safe_load(open("config.yaml"))
cfg["model"]["tag"] = "llama32_1b"
cfg["model"]["path"] = os.environ["PANDA_MODELS"] + "/TFB-Llama-3.2-1B-Instruct"
cfg["sampling"]["k"] = 4
yaml.dump(cfg, open("$CFG", "w"))
PY
  CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config "$CFG" --benchmark aime_2024 --k 4 --max-tokens 512
  rm -f "$CFG"
  echo "✓ 冒烟通过"
else
  echo "=== [6/6] 跳过冒烟（加 --smoke 或 SMOKE=1 启用）==="
  echo ""
  echo "部署完成。开始全量采样:"
  echo "  GPU=0 ./run_phase1.sh"
  echo "或单模型:"
  echo "  MODEL_TAG=phi4_mini MODEL_PATH=\$PANDA_MODELS/Phi-4-mini-instruct ./run_sampling.sh"
fi
