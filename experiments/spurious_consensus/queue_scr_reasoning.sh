#!/usr/bin/env bash
# 等 1.5B/supervisor 结束后，三卡保存 88 道 SCR 题的 full reasoning + token 级信息。
set -uo pipefail
cd "$(dirname "$0")"
LOG=logs/scr_reasoning
mkdir -p "$LOG"
SUP_LOG="$LOG/queue.log"

log(){ echo "[scr-reason $(date -Iseconds)] $*" | tee -a "$SUP_LOG"; }

log "等待 supervisor (pid 2892) 及 1.5B 采样进程结束..."
while kill -0 2892 2>/dev/null; do sleep 30; done
while pgrep -f "sample_vllm.*qwen25_15b" >/dev/null 2>&1; do sleep 30; done
sleep 15
TAU="${TAU:-1.0}"
log "GPU 空闲，启动 SCR reasoning+token 保存 (tau>=$TAU, K=64, 三卡)..."

pkill -f resample_scr_reasoning 2>/dev/null || true

pids=()
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s nohup python3 save_scr_reasoning.py \
    --tau "$TAU" --shard-id "$s" --num-shards 3 --k 64 --resume \
    > "$LOG/shard${s}.log" 2>&1 &
  pids+=($!)
  log "shard $s -> GPU$s pid $!"
done

for p in "${pids[@]}"; do wait "$p" || true; done

log "合并 manifest..."
python3 - <<'PY'
import json
from pathlib import Path
TAU=0.95
root = Path(f"data/scr_reasoning/qwen25_7b/t{TAU:.2f}".replace(".", ""))
mans = sorted(root.glob("manifest.shard*.json"))
all_q = []
for m in mans:
    d = json.load(open(m))
    all_q.extend(d["questions"])
out = {
    "model": "qwen25_7b",
    "n_questions": len(all_q),
    "k": 64,
    "tau": TAU,
    "description": f"清洗后 SCR@{TAU}；每题含 text、token ids、逐 token logprob",
    "questions": sorted(all_q, key=lambda x: x["id"]),
}
json.dump(out, open(root / "manifest.json", "w"), ensure_ascii=False, indent=2)
print(f"manifest.json: {len(all_q)} questions")
PY

log "=== SCR reasoning 保存完成 ==="
