# PANDA random300 — 后续实验与工程计划

> **范围说明：** 本文仅列 **实验、分析、产物、Git 同步** 等待办；**不包含** 论文正文撰写、Related Work、排版润色等写作任务。

**最后更新：** 2026-07-02

---

## 磁盘与机器

| 项 | 说明 |
|----|------|
| 可用空间 | `/root/autodl-tmp` 约 **54–55G** 可用（74% 已用）；7B 权重 + raw_runs 会占用显著空间，跑前确认 `panda-models/` 与 `panda-outputs/` 体量 |
| GPU | **6× RTX 5090**，6-way shard（Phase A vLLM + Phase B HF ASE） |
| 公平预算 | `SE_SAMPLES=0`，`N_REPHRASES=4`，`WEIGHT_SEEDS=42,43,44,45` → **9 decodes/题**；cohort **300 ids（seed=42）** |

---

## 关键路径

| 用途 | 路径 |
|------|------|
| 环境变量 | `source PANDA/scripts/env.sh` → `PANDA_ROOT`、`PANDA_OUTPUTS`、`PANDA_MODELS` |
| 输出根目录 | `/root/autodl-tmp/panda-outputs` |
| 模型目录 | `/root/autodl-tmp/panda-models`（如 `TFB-Qwen2.5-7B-Instruct`） |
| random300 cohort | `PANDA/paper/analysis/deepscaler_random300_meta.json` |
| variants | `panda-outputs/qaac_api_bench/deepscaler_random300/variants.jsonl` |
| Qwen 3B 跑数 | `panda-outputs/maintable_qwen25_3b_deepscaler_random300/seed41` |
| Qwen 7B 跑数 | `panda-outputs/maintable_qwen25_7b_deepscaler_random300/seed41` |
| Llama 1B 跑数 | `panda-outputs/maintable_llama32_1b_deepscaler_random300/seed41` |
| 7B 下载日志 | `panda-outputs/download_qwen25_7b_tfb.log` |
| 主 launch 脚本 | `PANDA/scripts/run_random300_qwen25_7b_6gpu.sh` |
| 3B 参考脚本 | `PANDA/scripts/run_random300_6gpu.sh` |
| Plan B 指标 | `PANDA/paper/analysis/compute_random300_planb_metrics.py` |
| 已有指标 JSON | `PANDA/paper/analysis/random300_planb_metrics.json` |
| join / 图 | `random300_sc9_panda_join.jsonl`，`paper/analysis/figures/random300_planb_comparison.png` 等 |

---

## 已完成（摘要）

- [x] 工程重命名：PRS→PANDA、`ase`→`panda.core`、`PANDA_*` 环境变量
- [x] random300 cohort（300 ids，seed=42）与 variants 固定
- [x] **Qwen2.5-3B random300：** SC@9 + PANDA **300/300**；metrics；Plan B 分析（AUROC + fair vote + confwrong）
- [x] **Llama-3.2-1B random300：** SC@9 + PANDA **300/300**（见下方「可忽略」）
- [x] 磁盘清理；`.vllm_ready` 相关流程已移除/不再依赖
- [x] 分析产物：`random300_planb_metrics.json`、figures、`random300_sc9_panda_join.jsonl`（及 Llama join）

---

## 进行中

- [ ] **Qwen2.5-7B：** 基座权重约 **15G** 已落盘（`prs-models/Qwen2.5-7B-Instruct`），但 **2026-07-02 13:18** ModelScope 报错退出（缺 `vocab.json`、`tokenizer.json`）；**TFB 未转换**；**random300 实验未启动（0/300）**，无 7B vLLM/HF 进程
- [ ] **Watcher（PID 143767，存活）：** 每 30s 等待 `prs-models/TFB-Qwen2.5-7B-Instruct/config.json`，就绪后执行 `run_random300_qwen25_7b_6gpu.sh`（日志：`panda-outputs/download_qwen25_7b_tfb.log`）
- [x] **7B random300 SC@9 配对文件：** `samples_qwen25_7b_seed41_deepscaler_random300_k9.jsonl` **300 行**（无 random300 专用 K=64 分片文件；全量 deepscaler 三分片仍在）
- [ ] **GitHub：** push 仍失败（见下文 P1）

---

## 待办（按优先级）

### P0 — 阻塞 cross-scale 主结论

- [ ] **Qwen2.5-7B random300：** 完整 **Phase A（vLLM）+ Phase B（HF ASE）** → 300/300 finals + `summary.jsonl`
- [ ] **7B SC@9 配对：** 确认/补齐 K=64 分片合并（若尚未有完整 `samples_qwen25_7b_*_random300_k9.jsonl`）；CPU subsample 取前 9（与 3B 同 protocol）
- [ ] **重跑 Plan B 汇总：** 7B 完成后执行 `compute_random300_planb_metrics.py`，更新 **3B vs 7B** cross-scale 表（`random300_planb_metrics.json` + summary md）
- [ ] **动机图：** 若 7B 数字显著改变叙事，按 `random300_motivation_figure_spec.md` 更新 figure（否则可跳过）

**7B 粗算 ETA（6×5090，参考 3B 已跑通时长 × 1.5–2.5）：**

| 阶段 | 粗估 wall time |
|------|----------------|
| 下载 + TFB | 视网速，权重 ~15G 量级（进行中） |
| Phase A vLLM | 约 **45–90 min**（7B 较 3B 慢，6  shard 并行） |
| Phase B HF ASE | 约 **2–3.5 h**（3B Phase B ~75–90 min；7B 约 1.5–2.5×） |
| metrics + watcher 配对 | **+5–15 min** |
| **合计（TFB 就绪后）** | 约 **3–5 h** 端到端 |

> 若单 shard 卡住或磁盘不足，ETA 顺延；跑前 `nvidia-smi` + `df -h /root/autodl-tmp`。

### P1 — 仓库与可选清理

- [ ] **GitHub push：** 推送 **代码 + `paper/analysis/`**（及必要 experiments 小文件）；**排除** `models/`、`outputs/`、大规模 `raw_runs`、本地权重
  - **当前状态（2026-07-02 13:23）：** 分支 `main` **ahead 1**（`8e239af`）；直连 `github.com` **443 超时**；`gitclone.com` / `mirror.ghproxy.com` / `ghproxy.com` 镜像 push **均未成功**（镜像超时或 `could not read Username`）；本机 **无 GitHub SSH 公钥**。需在本机配置 **PAT + HTTPS** 或 **SSH key** 后再 push；勿提交 `models/`、`outputs/`、大 raw_runs
- [ ] **（可选）** JSON schema 字段重命名：`T_ASE` → 与 PANDA 命名一致（仅 metadata，不影响已跑 raw_runs）

### P2 — 已取消 / 不必做

- ~~strong perturb `weight_sigma=0.06`~~（用户取消，脚本已 `.CANCELLED_BY_USER`）
- ~~scr500 random300 队列~~
- ~~Qwen3-8B random300 Plan B~~（已 pivot 到 Qwen2.5-7B）

---

## 总 checklist（复制跟踪）

```
进行中
[ ] 7B 补齐 tokenizer + TFB 完成（下载曾失败）
[ ] Watcher 待 TFB 后启动 run_random300_qwen25_7b_6gpu.sh
[x] 7B random300 SC@9 k9.jsonl（300 行）

P0
[ ] 7B Phase A 300/300
[ ] 7B Phase B 300/300 + summary.jsonl
[ ] 7B SC@9 join（K=64 合并如需要）
[ ] compute_random300_planb_metrics（含 3B vs 7B）
[ ] 动机 figure（按需）

P1
[ ] git push（无 models/raw_runs）
[ ] （可选）T_ASE schema rename
```

---

## 写作时可忽略（实验边界）

| 内容 | 建议 |
|------|------|
| **Llama-3.2-1B random300** | 已跑满 300/300，可作 **小模型边界 / 附录**；**主叙事 cross-scale 以 Qwen 3B vs 7B 为准**，不必在正文展开 Llama 细节 |
| TokUR / 主表 36 jobs | 与本 random300 Plan B 线独立；见 `GPU_PLAN.md` 若仍需主表 EU 列 |
| Qwen3-8B、scr500、σ=0.06 | 已取消，勿再启动相关 watcher |

---

## 7B 就绪后推荐命令（备忘）

```bash
cd /root/autodl-tmp/PANDA
source scripts/env.sh
export PANDA_OUTPUTS=/root/autodl-tmp/panda-outputs
export PANDA_MODELS=/root/autodl-tmp/panda-models

# 若 watcher 未自动拉起：
bash scripts/run_random300_qwen25_7b_6gpu.sh all

# 7B 300/300 且 join 就绪后：
python paper/analysis/compute_random300_planb_metrics.py
```

---

## 参考文档（同目录）

- `random300_planb_results_summary.md` — 当前 3B（+ Llama）Plan B 结果摘要  
- `random300_qwen_scale_plan.md` — 7B 选型与 ETA 依据  
- `random300_status.txt` — 历史进度与 incident 记录  
- `GPU_PLAN.md` / `FAST_PATH.md` — 主表 TokUR 与 CPU 快路径（与 random300 并行时可读）
