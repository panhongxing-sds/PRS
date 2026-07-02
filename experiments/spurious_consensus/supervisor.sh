#!/usr/bin/env bash
# 统一调度：顺序跑 7B -> 0.5B -> 1.5B，三卡分片。
# 每个模型跑完后校验完整性(按每分片应有题数)，缺则自动 --resume 重跑，
# 直到真正完整才进入下一个模型。无人值守、可断点续跑。
set -uo pipefail
cd "$(dirname "$0")"
export PANDA_ROOT=/root/PANDA
LOG=logs/supervisor
mkdir -p "$LOG"
SUP_LOG="$LOG/supervisor.log"

log(){ echo "[sup $(date -Iseconds)] $*" | tee -a "$SUP_LOG"; }

# 校验某 tag 是否三分片 × 三 benchmark 全部完整。返回 0=完整。
complete(){
  python3 - "$1" <<'PY'
import sys, json
from pathlib import Path
tag = sys.argv[1]
QB = {"deepscaler":2000, "gpqa_diamond":198, "aime_2024":30}
qdir = Path("data/questions")
for b in QB:
    rows = [json.loads(l) for l in (qdir/f"{b}.jsonl").read_text().splitlines() if l.strip()]
    for s in range(3):
        exp = len(rows[s::3])
        f = Path(f"data/samples/samples_{tag}_seed41_{b}.shard{s}.jsonl")
        ids = set()
        if f.exists():
            for l in f.read_text().splitlines():
                if l.strip():
                    ids.add(json.loads(l)["id"])
        if len(ids) < exp:
            print(f"INCOMPLETE {tag} {b} shard{s}: {len(ids)}/{exp}")
            sys.exit(1)
sys.exit(0)
PY
}

# 跑一个模型直到完整（最多 30 轮 resume）。
count_samples(){ cat data/samples/samples_"$1"_seed41_*.jsonl 2>/dev/null | wc -l; }

run_model(){
  local tag="$1" path="$2" batch="${3:-16}"
  local stall=0 prev=-1
  for attempt in $(seq 1 15); do
    if complete "$tag" >/dev/null 2>&1; then
      log "$tag 已完整 ✓"; return 0
    fi
    local before; before=$(count_samples "$tag")
    if [ "$before" -le "$prev" ]; then
      stall=$((stall+1))
    else
      stall=0
    fi
    prev=$before
    if [ "$stall" -ge 2 ]; then
      log "$tag 连续 2 轮无新增样本(疑似卡死)，跳过该模型继续后续"; return 1
    fi
    log "$tag 第 $attempt 轮：三卡分片 resume (batch=$batch, 当前 $before 行)"
    local pids=()
    for s in 0 1 2; do
      env GPU=$s MODEL_TAG="$tag" MODEL_PATH="$path" \
        SHARD_ID=$s NUM_SHARDS=3 K_CHUNK=64 PROMPT_BATCH=$batch \
        PANDA_ROOT=/root/PANDA \
        bash run_shard.sh > "$LOG/${tag}.shard${s}.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    sleep 8
  done
  log "$tag 达到最大重试仍未完整，继续后续模型"; return 1
}

PANDA_MODELS=/root/autodl-tmp/panda-models

log "开始校验+补跑链路 (max_model_len 已修正为 5120)"

# 阶段1：7B 补全（deepscaler 已完整，补 gpqa/aime shard1 缺口）
run_model qwen25_7b  "$PANDA_MODELS/Qwen2.5-7B-Instruct"  8   || true

# 阶段2：0.5B
run_model qwen25_05b "$PANDA_MODELS/Qwen2.5-0.5B-Instruct" 16 || true

# 阶段3：1.5B
run_model qwen25_15b "$PANDA_MODELS/Qwen2.5-1.5B-Instruct" 16 || true

log "=== 全部完成 ==="
for tag in qwen25_7b qwen25_05b qwen25_15b; do
  if complete "$tag" >/dev/null 2>&1; then log "$tag: 完整 ✓"; else log "$tag: 仍不完整 ✗"; complete "$tag" | tee -a "$SUP_LOG"; fi
done

# 阶段4：保存 SCR 题 full reasoning + token（默认 tau=0.95，67题）
TAU=1.0
log "开始保存 SCR reasoning+tokens (p_top=1.0 共42题, K=64)..."
SCR_PIDS=()
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s python3 save_scr_reasoning.py \
    --tau "$TAU" --shard-id "$s" --num-shards 3 --k 64 --resume \
    > "$LOG/scr_reasoning.shard${s}.log" 2>&1 &
  SCR_PIDS+=($!)
  log "scr_reasoning shard $s -> GPU$s pid $!"
done
for p in "${SCR_PIDS[@]}"; do wait "$p" || true; done
python3 - <<PY
import json
from pathlib import Path
TAU=$TAU
root = Path(f"data/scr_reasoning/qwen25_7b/t{TAU:.2f}".replace(".", ""))
mans = sorted(root.glob("manifest.shard*.json"))
all_q = []
for m in mans:
    all_q.extend(json.load(open(m))["questions"])
json.dump({
    "model": "qwen25_7b",
    "tau": TAU,
    "n_questions": len(all_q),
    "k": 64,
    "description": f"清洗后 SCR@{TAU}；每题含 text、prompt/completion token ids、逐 token logprob",
    "questions": sorted(all_q, key=lambda x: x["id"]),
}, open(root / "manifest.json", "w"), ensure_ascii=False, indent=2)
print(f"manifest: {len(all_q)} questions @ tau={TAU}")
PY
log "SCR reasoning+tokens 保存完成 → data/scr_reasoning/qwen25_7b/t095/"
