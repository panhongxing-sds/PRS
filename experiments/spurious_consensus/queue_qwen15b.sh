#!/usr/bin/env bash
# 链式排队：等 0.5B 三分片启动并结束后，自动用三卡分片跑 Qwen2.5-1.5B。
set -uo pipefail
cd "$(dirname "$0")"
LOG_DIR=logs/qwen15b
mkdir -p "$LOG_DIR"
PREV_DIR=logs/qwen05b
MODEL_PATH=/root/autodl-tmp/prs-models/Qwen2.5-1.5B-Instruct

echo "[queue $(date -Iseconds)] 等待 0.5B 启动 (shard pid 文件出现)..." | tee -a "$LOG_DIR/queue.log"
# 等 0.5B 的 shard pid 文件出现（即 7B 已完成、0.5B 已启动）
until [ -f "$PREV_DIR/shard0.pid" ] && [ -f "$PREV_DIR/shard1.pid" ] && [ -f "$PREV_DIR/shard2.pid" ]; do
  sleep 30
done
PREV_PIDS="$(cat "$PREV_DIR"/shard0.pid "$PREV_DIR"/shard1.pid "$PREV_DIR"/shard2.pid)"
echo "[queue $(date -Iseconds)] 0.5B 运行中 pid=$PREV_PIDS，等待其结束..." | tee -a "$LOG_DIR/queue.log"
for p in $PREV_PIDS; do
  while kill -0 "$p" 2>/dev/null; do sleep 30; done
done
sleep 10
echo "[queue $(date -Iseconds)] 0.5B 完成，启动 1.5B 三卡分片" | tee -a "$LOG_DIR/queue.log"

for s in 0 1 2; do
  nohup env GPU=$s MODEL_TAG=qwen25_15b MODEL_PATH="$MODEL_PATH" \
    SHARD_ID=$s NUM_SHARDS=3 K_CHUNK=64 PROMPT_BATCH=16 \
    PRS_ROOT=/root/PRS \
    bash run_shard.sh > "$LOG_DIR/shard${s}.log" 2>&1 &
  echo "[queue] 1.5B shard $s -> GPU$s pid $!" | tee -a "$LOG_DIR/queue.log"
  echo $! > "$LOG_DIR/shard${s}.pid"
done
echo "[queue $(date -Iseconds)] 1.5B 已全部启动" | tee -a "$LOG_DIR/queue.log"
