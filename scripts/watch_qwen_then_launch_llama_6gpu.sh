#!/usr/bin/env bash
# Wait for Qwen random300 Phase B (300/300), then launch Llama 6-GPU all + metrics watcher.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"

QWEN_OUT="/root/autodl-tmp/panda-outputs/maintable_qwen25_3b_deepscaler_random300/seed41"
LLAMA_OUT_BASE="/root/autodl-tmp/panda-outputs/maintable_llama32_1b_deepscaler_random300"
LOG="$LLAMA_OUT_BASE/logs"
META="$ROOT/paper/analysis/deepscaler_random300_meta.json"
VLLM_PY="${VLLM_PYTHON:-/root/miniconda3/bin/python}"
mkdir -p "$LOG"


count_finals() {
  "$VLLM_PY" - "$QWEN_OUT" "$META" << 'PY'
import json, sys
from pathlib import Path
out, meta = Path(sys.argv[1]), Path(sys.argv[2])
ids = json.loads(meta.read_text())["ids"]
n = 0
raw = out / "deepscaler" / "raw_runs"
for i in ids:
    p = raw / f"deepscaler_{i}.json"
    if not p.is_file():
        continue
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if d.get("summary_metrics") is not None or d.get("weight_perturb_runs"):
        n += 1
print(n)
PY
}

echo "=== watch_qwen_then_launch_llama $(date -Iseconds) ===" | tee -a "$LOG/qwen_wait.log"
while true; do
  n=$(count_finals)
  running=$(pgrep -af 'run_panda_experiment.*maintable_qwen25_3b_deepscaler_random300' 2>/dev/null | grep -vc -- '--mode metrics' || true)
  echo "$(date -Iseconds) qwen finals=$n/300 ase_procs=$running" | tee -a "$LOG/qwen_wait.log"
  if [[ "$n" -ge 300 ]]; then
    echo "Qwen 300/300 — launching Llama 6-GPU" | tee -a "$LOG/qwen_wait.log"
    break
  fi
  if [[ "${running:-0}" -eq 0 ]] && [[ "$n" -ge 295 ]]; then
    echo "Qwen ASE idle with n=$n — launching Llama" | tee -a "$LOG/qwen_wait.log"
    break
  fi
  sleep 45
done

# Brief GPU settle after Qwen
sleep 15
for _ in $(seq 1 30); do
  if pgrep -af 'run_panda_experiment.*maintable_qwen25_3b_deepscaler_random300' 2>/dev/null | grep -q -- '--mode metrics'; then
    break
  fi
  if pgrep -af 'run_panda_experiment.*maintable_qwen25_3b_deepscaler_random300' 2>/dev/null | grep -v -- '--mode metrics' | grep -q .; then
    sleep 10
  else
    break
  fi
done

nohup bash "$ROOT/scripts/run_random300_llama_6gpu.sh" all \
  >> "$LOG/llama_random300_6gpu_nohup.log" 2>&1 &
echo "Llama orchestrator PID=$!" | tee -a "$LOG/qwen_wait.log"

nohup env OUT_BASE="$LLAMA_OUT_BASE" bash "$ROOT/scripts/watch_random300_finish.sh" \
  >> "$LOG/llama_watch_finish_nohup.log" 2>&1 &
echo "Llama metrics watcher PID=$!" | tee -a "$LOG/qwen_wait.log"
