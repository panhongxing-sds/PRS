#!/usr/bin/env bash
# Convert local / shared base models to TFB format (copy weights + SVD basis_vectors).
set -euo pipefail

ROOT="/home/phx/TokUR"
OUT_BASE="/HDDDATA/phx/model"
LOG_DIR="${OUT_BASE}/_logs/tfb_convert"
mkdir -p "$LOG_DIR"

convert_one() {
  local name="$1"
  local src="$2"
  local dst="$3"
  local arch="$4"
  local log="${LOG_DIR}/${name}.log"

  if [[ -f "${dst}/config.json" ]] && ls "${dst}"/*.safetensors &>/dev/null; then
    echo "[skip] ${name} already exists at ${dst}"
    return 0
  fi

  echo "[start] ${name}: ${src} -> ${dst} (${arch})"
  python3 "${ROOT}/convert_to_tfb.py" \
    --model-path "$src" \
    --output-path "$dst" \
    --architecture "$arch" \
    --rank 8 \
    --bayes-noise right \
    2>&1 | tee "$log"
  echo "[done] ${name}"
}

# Smallest first for quick validation
convert_one "llama-3.2-1b" \
  "/HDDDATA/XieeeHuiii/model/Llama-3.2-1B-Instruct" \
  "${OUT_BASE}/TFB-Llama-3.2-1B-Instruct" \
  "llama"

convert_one "qwen3-8b" \
  "/HDDDATA/phx/model/Qwen3-8B" \
  "${OUT_BASE}/TFB-Qwen3-8B" \
  "qwen2"

convert_one "mistral-7b" \
  "/HDDDATA/phx/model/Mistral-7B-Instruct" \
  "${OUT_BASE}/TFB-Mistral-7B-Instruct" \
  "llama"

convert_one "llama-3.1-8b" \
  "/HDDDATA/XieeeHuiii/model/Llama-3.1-8B-Instruct" \
  "${OUT_BASE}/TFB-Llama-3.1-8B-Instruct" \
  "llama"

convert_one "phi35-mini" \
  "/HDDDATA/phx/model/Phi-3.5-mini-instruct" \
  "${OUT_BASE}/TFB-Phi-3.5-mini-instruct" \
  "phi3"

convert_one "opt-6.7b" \
  "/HDDDATA/phx/model/OPT-6.7B" \
  "${OUT_BASE}/TFB-OPT-6.7B" \
  "opt"

echo "All conversions finished. Outputs under ${OUT_BASE}/TFB-*"
