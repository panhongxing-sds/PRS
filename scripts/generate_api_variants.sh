#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Generate API rephrases: 100 questions × 8 variants per dataset.
#
# Requires OPENAI_API_KEY or DEEPSEEK_API_KEY (OpenAI-compatible endpoint).
# Optional: source configs/qaac_shubiaobiao.env
#
# Usage:
#   bash scripts/generate_api_variants.sh deepscaler,aime24
#   bash scripts/generate_api_variants.sh gsm8k --max-samples 100 --resume
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f configs/qaac_shubiaobiao.env ]]; then
  # shellcheck disable=SC1091
  source configs/qaac_shubiaobiao.env
fi

if [[ -z "${OPENAI_API_KEY:-}" && -z "${DEEPSEEK_API_KEY:-}" && -f /home/phx/DuoRoute/configs/embedding_api.local.yaml ]]; then
  export OPENAI_API_KEY="$(
    python3 - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("/home/phx/DuoRoute/configs/embedding_api.local.yaml").read_text())
print(cfg["embedding"]["api_key"])
PY
  )"
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY or DEEPSEEK_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${DEEPSEEK_BASE_URL:-https://api.shubiaobiao.cn/v1}}"

DATASETS="${1:-minerva,math500,gsm8k,deepscaler}"
shift || true

BENCH=$PRS_OUTPUTS/qaac_api_bench
LOG_DIR="${BENCH}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/generate_$(echo "$DATASETS" | tr ',' '_')_$(date +%Y%m%d_%H%M%S).log"

echo "API variants → datasets=$DATASETS log=$LOG"
nohup python3 -m prs.token_qaac.generate_variants \
  --datasets "$DATASETS" \
  --out-dir "$BENCH" \
  --max-samples 200 \
  --n-rephrases 8 \
  --workers 4 \
  --resume \
  "$@" \
  > "$LOG" 2>&1 &

echo "PID=$!  tail -f $LOG"
