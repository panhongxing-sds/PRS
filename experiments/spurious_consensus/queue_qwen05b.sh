#!/usr/bin/env bash
# 等 7B 三分片跑完，自动用三卡分片跑 Qwen2.5-0.5B。
set -uo pipefail
cd "$(dirname "$0")"
LOG_DIR=logs/qwen05b
mkdir -p "$LOG_DIR"
WAIT_PIDS="${WAIT_PIDS:?需传入 7B 三个 shard 的 pid，空格分隔}"
MODEL_PATH=/root/autodl-tmp/panda-models/Qwen2.5-0.5B-Instruct

echo "[queue $(date -Iseconds)] 等待 7B 分片结束: $WAIT_PIDS" | tee -a "$LOG_DIR/queue.log"
for p in $WAIT_PIDS; do
  while kill -0 "$p" 2>/dev/null; do sleep 30; done
done
sleep 10
echo "[queue $(date -Iseconds)] 7B 完成，启动 0.5B 三卡分片" | tee -a "$LOG_DIR/queue.log"

for s in 0 1 2; do
  nohup env GPU=$s MODEL_TAG=qwen25_05b MODEL_PATH="$MODEL_PATH" \
    SHARD_ID=$s NUM_SHARDS=3 K_CHUNK=64 PROMPT_BATCH=16 \
    PANDA_ROOT=/root/PANDA \
    bash run_shard.sh > "$LOG_DIR/shard${s}.log" 2>&1 &
  echo "[queue] 0.5B shard $s -> GPU$s pid $!" | tee -a "$LOG_DIR/queue.log"
  echo $! > "$LOG_DIR/shard${s}.pid"
done
echo "[queue $(date -Iseconds)] 0.5B 已全部启动" | tee -a "$LOG_DIR/queue.log"
