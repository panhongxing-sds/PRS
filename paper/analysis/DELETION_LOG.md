# 删除执行日志

- **执行时间**: 2026-07-02
- **授权**: 用户确认删除包 **A + B**（见 `DELETION_CANDIDATES.md`）
- **执行者**: 当前 agent（会话内）；大项 scr500 maintable 在本次执行前已不存在（疑为 prior agent `25eb2ca5` 或更早手动清理）

## 磁盘 before / after

| 时刻 | `df -h /root/autodl-tmp` | 已用字节 (`df -B1`) |
|------|---------------------------|---------------------|
| **删除前** | 200G 总 / **182G 已用** / **19G 可用** / 91% | 195,182,374,912 |
| **删除后** | 200G 总 / **182G 已用** / **19G 可用** / 91% | 195,226,624,000 |

> 删除后「已用」字节略增（约 +44 MB），因 **Llama random300 baseline 仍在写入**（未杀进程）。本次 `rm` 释放的空间与并发写入相抵，`df` 净变化不能代表删除量。

## `du` 摘要（删除后）

| 路径 | 大小 |
|------|------|
| `/root/autodl-tmp/PANDA` | 7.9G |
| `/root/autodl-tmp/panda-outputs` | 93G |
| `maintable_qwen25_3b_deepscaler_random300` | 2.5G（保留） |
| `maintable_llama32_1b_deepscaler_random300` | 2.4G（保留，运行中） |

## 本次会话实际删除（`du -sb` 合计）

**5,098,338 字节 ≈ 4.86 MiB ≈ 0.005 GB**

### 按路径（字节 → 路径）

| 字节 | 路径 |
|------|------|
| 4,397,639 | `/tmp/vllm_smoke_scr500` |
| 248,385 | `PANDA/experiments/spurious_consensus/data/questions/deepscaler_scr500.jsonl` |
| 160,928 | `PANDA/third_party/TokUR/datasets/deepscaler_scr500_only_backup.jsonl` |
| 160,928 | `PANDA/third_party/TokUR/datasets/deepscaler_scr500.jsonl` |
| 83,781 | `panda-outputs/logs/sc_k9_deepscaler_qwen25_3b.log` |
| 13,967 | `panda-outputs/logs/sc_k9_deepscaler_wrapper.log` |
| 8,076 | `PANDA/experiments/.../samples_qwen25_3b_seed41_deepscaler_k9.jsonl` |
| 7,542 | `PANDA/paper/analysis/deepscaler_scr500_meta.json` |
| 7,099 | `panda-outputs/maintable_qwen25_3b_deepscaler_random300_strong` |
| 2,918 | `panda-outputs/logs/sc_k9_watchdog_parent.log` |
| 2,108 | `PANDA/paper/analysis/random300_strong_perturb_setup.json` |
| 1,395 | `PANDA/experiments/.../samples_qwen25_3b_seed41_deepscaler_k9.shard0.jsonl` |
| 1,140 | `PANDA/paper/analysis/scr500_panda_setup.json` |
| 731 | `PANDA/paper/analysis/scr500_panda_run_status.json` |
| 684 | `/tmp/scr500_config.yaml` |
| 572 | `PANDA/paper/analysis/scr500_k9_stats.json` |
| 364 | `PANDA/paper/analysis/panda_scr500_status.txt` |
| 81 | `panda-outputs/logs/sc_k9_deepscaler_qwen25_3b.meta` |
| 0 | `panda-outputs/logs/kill_vllm_while_ase.log` |
| 0 | `panda-outputs/gpu0_sc_only.lock` |

## 删除前已缺失（计入包 A/B 清单，本次未 `rm`）

以下在扫描时已 **MISSING**，按 `DELETION_CANDIDATES.md` 估算约 **~975 MB**（主要为 scr500 maintable）：

- `/root/autodl-tmp/panda-outputs/maintable_qwen25_3b_deepscaler_scr500/`（清单 **974M**）
- `panda-outputs/qaac_api_bench/deepscaler_scr500/`
- `panda-outputs/bench/deepscaler_scr500.jsonl`
- scr500 K=9 样本与 `.bak64`、`deepscaler_scr500_k9.jsonl` 空题集
- `/tmp/sc_k9_scr500.yaml`、`/tmp/sc_k9_deepscaler.yaml`
- `panda-outputs/sc_k9_scr500.pid`
- 若干 scr500 专用 `panda-outputs/logs/*scr500*` 文件

## 明确保留（未删）

- random300 maintable / 分析 / K=9 正式样本 / `deepscaler_random300_meta.json` 等
- K=64 与主表 N=8121 相关 deepscaler 样本
- Llama random300 运行进程与输出目录
- `PANDA/scripts/*.CANCELLED_BY_USER`（3 个，按用户偏好保留）

## 删除后校验

- `find /root/autodl-tmp -iname '*scr500*'`：**无匹配**
- Llama ASE 进程：**仍在运行**（未 `kill`）

## 汇总

| 指标 | 值 |
|------|-----|
| **本会话释放** | **~0.005 GB**（5,098,338 B） |
| **包 A+B 清单中此前已清** | **~0.96 GB**（估，主项 maintable scr500） |
| **包 A+B 现状态** | 清单路径已全部不存在或已删；scr500 名残留已清零 |
## 2026-07-02 — LARGE_FILES_AUDIT priority-1 items 1–6

**Operator:** automated cleanup (user confirmed)  
**Llama baseline ASE:** PID 91602 (`run_panda_experiment --mode metrics`, `maintable_llama32_1b_deepscaler_random300`) — **not killed**; verified running after deletes.

### Disk (`df -h /root/autodl-tmp`)
| When | Size | Used | Avail | Use% |
|------|------|------|-------|------|
| Before | 200G | 182G | 19G | 91% |
| After | 200G | 135G | 66G | 68% |

**Approx. freed (df): ~47G**

### Per-path `du -sb` (before → after)
| # | Path | Before (bytes) | After |
|---|------|----------------|-------|
| 1 | `/root/autodl-tmp/panda-vllm-venv` | 12430320336 | deleted |
| 2 | `/root/autodl-tmp/archive/panda_full_20260615.tar.zst` | 12197000663 | deleted |
| 3 | `/root/autodl-tmp/PANDA/.tokur_venv` | 7833987194 | deleted |
| 4 | `/root/autodl-tmp/pip_cache` | 6936461216 | deleted |
| 5 | `/root/autodl-tmp/panda-models/.vllm_ready` | 6171966293 | deleted |
| 6 | `/root/autodl-tmp/tmp` (contents) | 5472032420 | 10 (empty dir) |

**Sum of before `du -sb`: 51041828022 bytes (~47.53 GiB)**

### Actions
- `rm -rf` items 1, 3, 4, 5; `rm -f` item 2; `find tmp -mindepth 1 -delete` equivalent via `rm -rf` children (no open files under `tmp` at delete time).
- **Not deleted:** priority-1 item 7+ (pip-cache duplicate, cursor-server, raw_runs, etc.).

### `.vllm_ready` script updates (before rm item 5)
- `PANDA/scripts/run_random300_6gpu.sh`: default `MODEL` → `/root/autodl-tmp/panda-models/TFB-Qwen2.5-3B-Instruct`
- `PANDA/paper/analysis/random300_panda_setup.json`: `model_path_vllm` → same main Qwen path
- `PANDA/scripts/restart_llama_random300_after_gpu_reset.sh`: already `MODEL_PATH=/root/autodl-tmp/panda-models/TFB-Llama-3.2-1B-Instruct` (unchanged)

