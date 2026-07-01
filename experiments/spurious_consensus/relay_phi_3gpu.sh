#!/usr/bin/env bash
# 自动接力：等 llama(GPU0) 结束后，把 phi 三卡分片续跑 deepscaler。
# 不打断当前正在跑的进程；llama 跑完才动手。
set -uo pipefail
cd "$(dirname "$0")"
export PRS_MODELS="${PRS_MODELS:-/root/autodl-tmp/prs-models}"
export PRS_ROOT="${PRS_ROOT:-/root/PRS}"
LOG_DIR=logs/phase1
PHI_PATH="$PRS_MODELS/Phi-4-mini-instruct"
LLAMA_PID="$(cat $LOG_DIR/llama32_1b.pid 2>/dev/null || echo '')"

echo "[relay $(date -Iseconds)] 等待 llama(pid=$LLAMA_PID) 结束..."
while [[ -n "$LLAMA_PID" ]] && kill -0 "$LLAMA_PID" 2>/dev/null; do
  sleep 60
done
echo "[relay $(date -Iseconds)] llama 已结束，GPU0 释放。准备 phi 三卡续跑。"
sleep 10

# 停掉现有 phi 两卡（让位给统一的三卡重分片），保留已产出的样本
pkill -9 -f "sc_phi4_mini" 2>/dev/null || true
pkill -9 -f "run_shard.sh" 2>/dev/null || true
sleep 8

# 三卡各跑一个 shard（NUM_SHARDS=3）。已完成的 id 由各 shard 文件 --resume 跳过；
# 旧 shard0/1 的样本先并入 shard 文件命名空间，避免重复。
python3 - <<'PY'
import json
from pathlib import Path
d = Path("data/samples")
qs = [json.loads(l) for l in Path("data/questions/deepscaler.jsonl").read_text().splitlines() if l.strip()]
id2idx = {q["id"]: i for i, q in enumerate(qs)}
# 汇总旧 2-shard 已完成行
done = {}
for s in (0, 1):
    f = d / f"samples_phi4_mini_seed41_deepscaler.shard{s}.jsonl"
    if f.exists():
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done[r["id"]] = r
# 按新的 3-shard 归属重新分配已完成行，写入新 shard 文件作为 resume 起点
buckets = {0: [], 1: [], 2: []}
for rid, r in done.items():
    idx = id2idx.get(rid)
    if idx is None:
        continue
    buckets[idx % 3].append(r)
for s in (0, 1, 2):
    out = d / f"samples_phi4_mini_seed41_deepscaler.s3_{s}.jsonl"
    with out.open("w") as f:
        for r in buckets[s]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"s3_{s}: 预置 {len(buckets[s])} 已完成行")
PY

# 三卡各一 shard（NUM_SHARDS=3），输出后缀 .s3_{id} 避免与旧 2-shard 撞名。
for s in 0 1 2; do
  gpu=$s
  nohup env GPU=$gpu MODEL_TAG=phi4_mini MODEL_PATH="$PHI_PATH" \
    SHARD_ID=$s NUM_SHARDS=3 K_CHUNK=64 PROMPT_BATCH=8 OUT_SUFFIX=".s3_${s}" \
    PRS_MODELS="$PRS_MODELS" PRS_ROOT="$PRS_ROOT" \
    bash run_shard.sh > "$LOG_DIR/phi4_mini.s3_${s}.log" 2>&1 &
  echo $! > "$LOG_DIR/phi4_mini.s3_${s}.pid"
  echo "[relay] phi shard $s/3 -> GPU$gpu pid $!"
  sleep 3
done
echo "[relay $(date -Iseconds)] phi 三卡分片已启动。"
