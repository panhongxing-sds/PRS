# 删除候选清单（仅盘点，未执行任何删除）

- **扫描范围**: `/root/autodl-tmp/PANDA`、`/root/autodl-tmp/panda-outputs`（另含与废弃 scr500/SC@9 相关的 `/tmp` 临时项）
- **扫描时间**: 2026-07-02
- **状态**: **未删除任何文件**。请逐项确认后再执行删除。

---

## 明确保留（勿删）

| 路径 / 模式 | 说明 |
|-------------|------|
| `panda-outputs/maintable_qwen25_3b_deepscaler_random300/seed41` | Qwen random300 baseline 已完成 (300/300) |
| `panda-outputs/maintable_llama32_1b_deepscaler_random300/seed41` | Llama random300 baseline 进行中或已完成 |
| `experiments/spurious_consensus/data/samples/samples_*deepscaler_random300_k9.jsonl` | SC@9 random300 正式样本（Qwen 300 行、Llama 300 行） |
| `paper/analysis/deepscaler_random300_meta.json` | random300 元数据 |
| `paper/analysis/random300_qwen_results_analysis.md` | 论文/分析 |
| `paper/analysis/random300_confident_wrong.json` | 论文/分析 |
| `paper/analysis/figures/` | 论文图 |
| `paper/analysis/random300_sc9_panda_join.jsonl` | join jsonl |
| `paper/analysis/random300_fair_baselines.json` | fair baselines |
| `experiments/spurious_consensus/data/samples/samples_*_deepscaler.jsonl`（非 scr500/random300_k9 子集） | 主表 N=8121 相关 deepscaler K=64 样本等 |
| `scripts/run_random300_6gpu.sh`、`scripts/watch_random300_finish.sh` | 当前活跃脚本 |

---

## 类别 1：scr500 主实验输出（已改做 random300）

| 路径 | 大小 | 删除理由 | 删除风险 |
|------|------|----------|----------|
| `/root/autodl-tmp/panda-outputs/maintable_qwen25_3b_deepscaler_scr500/` | **974M** | 用户已放弃 scr500；`seed41/deepscaler/raw_runs/` 约 **510** 个 `*.partial.json`（未完成 HF/vLLM 续跑），无完整 metrics | 无法再恢复 scr500 主表 baseline；random300 为 canonical |
| `/root/autodl-tmp/panda-outputs/qaac_api_bench/deepscaler_scr500/` | **896K** | 仅 scr500 变体 bench（`variants.jsonl`） | 若论文不再引用 scr500 QAAC，无影响 |
| `/root/autodl-tmp/panda-outputs/bench/deepscaler_scr500.jsonl` | **160K** | scr500 专用 bench 题集副本 | random300 使用 `bench/deepscaler_random300.jsonl` |

**小计约 975M**

---

## 类别 2：已取消的 strong perturb 输出

| 路径 | 大小 | 删除理由 | 删除风险 |
|------|------|----------|----------|
| `/root/autodl-tmp/panda-outputs/maintable_qwen25_3b_deepscaler_random300_strong/` | **8.0K** | 用户已取消 strong；仅 `logs/strong_orchestrator.log`（~7KB）与空 lock，无 seed 数据 | 丢失 strong 编排日志；实验已取消 |

---

## 类别 3：scr500 / 失败 SC@9 GPU 部分样本

| 路径 | 行数 / 大小 | 删除理由 | 删除风险 |
|------|-------------|----------|----------|
| `PANDA/experiments/spurious_consensus/data/samples/samples_qwen25_3b_seed41_deepscaler_scr500_k9.jsonl` | 500 行 / **188K** | scr500 路线废弃；非 random300 canonical | scr500 K=9 分析需重跑 |
| `.../samples_qwen25_3b_seed41_deepscaler_scr500_k9_fresh.jsonl` | 243 行 / **72K** | 中途 GPU 重试部分输出 | 同上 |
| `.../samples_qwen25_3b_seed41_deepscaler_scr500_k9.jsonl.bak64` | 0 字节 | 空备份 | 无 |
| `.../samples_qwen25_3b_seed41_deepscaler_k9.jsonl` | 30 行 / **8.0K** | 早期 deepscaler K=9 失败/中断；已被 `deepscaler_random300_k9`（300 行）取代 | 丢失失败尝试记录 |
| `.../samples_qwen25_3b_seed41_deepscaler_k9.shard0.jsonl` | 5 行 / **4.0K** | 分片 partial | 同上 |

**小计约 272K**

---

## 类别 4：空备份 / 0 字节占位（可选）

| 路径 | 大小 | 删除理由 | 删除风险 |
|------|------|----------|----------|
| `samples_qwen25_3b_seed41_deepscaler_scr500_k9.jsonl.bak64` | 0 | 已在类别 3 列出 | 无 |
| `samples_gemma3_4b_seed41_aime_2024.jsonl` | 0 | 空 SC 占位 | 若脚本期望文件存在可能需 touch 重建 |
| `samples_phi4_mini_seed41_aime_2024.jsonl` | 0 | 空 SC 占位 | 同上 |

**建议**: 仅删 scr500 相关 0 字节；两个 aime 占位默认 **不删**，除非确认无 pipeline 依赖。

---

## 类别 5：scr500 bench / 题集副本（random300 为 canonical）

| 路径 | 大小 | 删除理由 | 删除风险 |
|------|------|----------|----------|
| `PANDA/experiments/spurious_consensus/data/questions/deepscaler_scr500.jsonl` | **244K** (500 题) | 与 random300 题集不同；scr500 已弃 | 复现 scr500 需重新导出 |
| `PANDA/experiments/spurious_consensus/data/questions/deepscaler_scr500_k9.jsonl` | **0** | 空文件 | 无 |
| `PANDA/third_party/TokUR/datasets/deepscaler_scr500.jsonl` | **160K** | scr500 数据集副本 | TokUR 脚本若硬编码路径会报错 |
| `PANDA/third_party/TokUR/datasets/deepscaler_scr500_only_backup.jsonl` | **160K** | 重复备份 | 同上 |

**小计约 564K**（与类别 1 bench 文件内容重叠但路径不同）

---

## 类别 6：陈旧日志（scr500 / 失败 SC@9）

| 路径 | 大小 | 删除理由 | 删除风险 |
|------|------|----------|----------|
| `panda-outputs/maintable_qwen25_3b_deepscaler_scr500/logs/hf_deepscaler_scr500*.log`（含 resume2–4） | **~340K**（整目录 logs） | scr500 续跑已停（pid 内容为 `STOPPED`） | 丢失排障日志 |
| `.../logs/vllm_deepscaler_scr500.log` | （含上） | vLLM scr500  smoke/采样 | 同上 |
| `.../logs/sample_vllm_guard.log` | 0 | 空日志 | 无 |
| `panda-outputs/logs/sc_k9_deepscaler_scr500_qwen25_3b.log` | **140K** |  abandoned scr500 SC@9 | 无 |
| `panda-outputs/logs/sc_k9_deepscaler_scr500_qwen25_3b.meta` | 4K | 元数据 | 无 |
| `panda-outputs/logs/sc_k9_deepscaler_scr500_watchdog.log` | 4K | watchdog | 无 |
| `panda-outputs/logs/panda_scr500_status.txt` | 4K | scr500 panda 状态 | 无 |
| `panda-outputs/logs/sc_k9_deepscaler_qwen25_3b.log` | **84K** | 早期 vLLM 初始化失败日志（非 random300 成功路径） | 丢失失败记录 |
| `panda-outputs/logs/sc_k9_deepscaler_qwen25_3b.meta` | 4K | 同上 | 低 |
| `panda-outputs/logs/sc_k9_deepscaler_wrapper.log` | 16K | SC@9 包装脚本日志（无 random300 关键字） | 低 |
| `panda-outputs/logs/sc_k9_watchdog_parent.log` | 4K | watchdog 父进程 | 低 |
| `panda-outputs/logs/kill_vllm_while_ase.log` | 0 | 空 | 无 |
| `PANDA/paper/analysis/panda_scr500_status.txt` | 4K | 与 outputs 重复态 scr500 状态 | 低 |

**小计约 460K**（maintable scr500 logs 与类别 1 目录重叠，磁盘只计一次）

---

## 类别 7：`/tmp` 与已取消脚本标记

| 路径 | 大小 | 删除理由 | 删除风险 |
|------|------|----------|----------|
| `/tmp/vllm_smoke_scr500/` | **4.2M** | scr500 vLLM smoke 残留 | 无 |
| `/tmp/scr500_config.yaml` | 4K | scr500 临时配置 | 无 |
| `/tmp/sc_k9_scr500.yaml` | 4K |  abandoned scr500 SC@9 | 无 |
| `/tmp/sc_k9_deepscaler.yaml` | 4K | 临时 SC@9 配置（当前 random300 已跑完） | 重跑 SC@9 需重建 yaml |
| `PANDA/scripts/run_random300_qwen_strong_6gpu.sh.CANCELLED_BY_USER` | 4K | 用户取消 strong | 丢失「已取消」标记（可保留作审计） |
| `PANDA/scripts/run_random300_llama_strong_6gpu.sh.CANCELLED_BY_USER` | 4K | 同上 | 同上 |
| `PANDA/scripts/watch_llama_baseline_then_strong_6gpu.sh.CANCELLED_BY_USER` | 4K | 同上 | 同上 |

**建议**: CANCELLED 后缀脚本 **默认保留** 作历史标记；若删仅省 **12K**。

**小计（/tmp 约 4.2M + 可选脚本 12K）**

---

## 类别 8：scr500 论文/分析元数据（小文件）

| 路径 | 大小 | 删除理由 | 删除风险 |
|------|------|----------|----------|
| `PANDA/paper/analysis/deepscaler_scr500_meta.json` | 8K | scr500 实验元数据 | 丢失 scr500 实验记录 |
| `PANDA/paper/analysis/scr500_k9_stats.json` | 4K | scr500 K=9 统计 | 同上 |
| `PANDA/paper/analysis/scr500_panda_setup.json` | 4K |  setup | 同上 |
| `PANDA/paper/analysis/scr500_panda_run_status.json` | 4K |  run status | 同上 |
| `PANDA/paper/analysis/random300_strong_perturb_setup.json` | 4K | 已取消 strong 的 setup | 丢失 strong 设计记录 |

**小计约 24K**（可选保留作文档）

---

## 类别 9：孤儿 pid / lock

| 路径 | 内容 | 删除理由 | 删除风险 |
|------|------|----------|----------|
| `panda-outputs/sc_k9_scr500.pid` | PID 75273 | 进程 **已不存在** | 无 |
| `panda-outputs/maintable_qwen25_3b_deepscaler_scr500/logs/hf_deepscaler_scr500.pid` | 文本 `STOPPED` | 任务已停 | 无 |
| `panda-outputs/gpu0_sc_only.lock` | 空文件 |  stale lock | 若有新 job 会重建 lock |

**另**: `maintable_qwen25_3b_deepscaler_scr500/logs/launch_phase_b_resume3.sh`（1.1K）为 outputs 内临时启动脚本，随类别 1 目录删除即可。

---

## 未列入默认删除（需单独决定）

| 路径 | 大小 | 说明 |
|------|------|------|
| `panda-outputs/_oom_probe_llama31_8b/` | **18M** | 2025-06-13 Llama OOM 探测，与 scr500/random300 无直接关联 |
| `panda-outputs/*.pkl`（根目录 9 个 cache） | **~9.1M** | 2025-06-15 信号挖掘缓存；可能仍被分析脚本使用 |
| `panda-outputs/gpu_plan_logs/` | **128K** | 2026-07-02 GPU 计划日志 |
| `PANDA/paper/analysis/logs/` | 小 | CPU/GPU 分析运行日志 |

---

## 汇总（磁盘去重后）

| 档位 | 包含内容 | 约计 |
|------|----------|------|
| **推荐删除包 A** | 类别 1 + 2 + 3 + 5 + `/tmp` scr500 + 类别 9 | **~980 MB（≈0.96 GB）** |
| **+ 日志与元数据包 B** | 类别 6 + 8 + 类别 7 的 CANCELLED 脚本 | **+ ~0.5 MB** |
| **+ 可选包 C** | `_oom_probe_llama31_8b` | **+ 18 MB → 合计 ~1.0 GB** |

- **候选路径/组数（推荐包 A）**: **22** 组（含 1 个大目录树、若干单文件）
- **若含包 B**: **~37** 组
- **最大单项**: `maintable_qwen25_3b_deepscaler_scr500/`（974M，510 partial raw runs）

---

## 确认方式（给用户）

请回复例如：

- `确认删除包 A` / `确认删除 A+B` / `仅删除 scr500 主目录`
- 或指定 **保留** CANCELLED 脚本、scr500 元数据 json 等例外

**Agent 不得在收到明确确认前执行 `rm`。**
