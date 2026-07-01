#!/usr/bin/env bash
# 等指定 PID 结束后，用更新后的 benchmark 列表续跑。
set -euo pipefail
wait_pid="${1:?pid}"
shift
label="${1:?label}"
shift
LOG_DIR="${LOG_DIR:-logs/phase1}"
mkdir -p "$LOG_DIR"
log="$LOG_DIR/${label}.relaunch.log"

echo "[relaunch $(date -Iseconds)] 等待 pid=$wait_pid ($label)..." | tee -a "$log"
while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
sleep 5
echo "[relaunch $(date -Iseconds)] 启动 $label: $*" | tee -a "$log"
cd "$(dirname "$0")"
exec "$@" >> "$log" 2>&1
