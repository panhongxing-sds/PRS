#!/usr/bin/env bash
# End-to-end TokUR smoke: 1 minerva sample via official vLLM path.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export VLLM_USE_V1=0
export NUM_GPUS=1
export GPU_IDS=(0)
export PARALLEL_SHARDS=0
export BATCH_SIZE=1
export TOKUR_GPU_MEM_UTIL=0.85
LOG="${PANDA_OUTPUTS}/../logs/smoke_tokur_vllm.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

echo "=== smoke_tokur_vllm START $(date) ==="
"${TOKUR_PY:-$TOKUR_VENV/bin/python}" -c "import vllm, bayesian_transformer, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"

SMOKE_OUT="${PANDA_OUTPUTS}/smoke_tokur_llama1b"
SRC="${PANDA_OUTPUTS}/maintable_llama32_1b/seed41/minerva/raw_runs"
rm -rf "$SMOKE_OUT"
mkdir -p "$SMOKE_OUT/seed41/minerva/raw_runs"
python3 - <<PY
import shutil
from pathlib import Path
src, dst = Path("$SRC"), Path("$SMOKE_OUT/seed41/minerva/raw_runs")
files = sorted(p for p in src.glob("*.json") if ".partial" not in p.name and ".error" not in p.name)
if not files:
    raise SystemExit(f"no raw_runs under {src}; sync panda-outputs first")
shutil.copy2(files[0], dst / files[0].name)
print("copied", files[0].name)
PY

OUT_DIR="$SMOKE_OUT/seed41" PANDA_MODEL_TAG=llama32_1b DATASET=minerva TOKUR_SEED=41 \
  bash "$PANDA_ROOT/scripts/run_tokur_official_maintable.sh"

test -s "$SMOKE_OUT/seed41/minerva/tokur_baseline.jsonl"
python3 - <<PY
import json
r = json.loads(open("$SMOKE_OUT/seed41/minerva/tokur_baseline.jsonl").readline())
assert r.get("scoring_mode", "").startswith("official_vllm"), r
assert r.get("tokur_eu_sum") is not None
print("SMOKE OK", r["id"], "eu=", r["tokur_eu_sum"])
PY
touch "$(dirname "$LOG")/TOKUR_SMOKE_OK" 2>/dev/null || true
echo "=== smoke_tokur_vllm DONE $(date) ==="
