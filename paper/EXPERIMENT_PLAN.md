# PANDA 实验方案与表格预案

> 本文档预先列出 PANDA 论文需要完成的全部实验、对应数据来源、运行命令与**待填空表格**。表内数字格式统一为 `AUROC`（或百分比 `71.77 ± 0.12`，3 seeds mean±std）。所有"对错"标签使用 **clean label**（`label_wrong_clean`）。

---
## 0. 方法-代码口径备注（写表前必读）

> **定稿 PANDA 见 `METHOD.md`**。主表 = **LODO OOF LR(\(D_\text{base}, T_{\mathrm{ent\_prox\_lin}}\))**；\(F_\text{resp}\) 仅诊断。

| 符号 | 定义 | 代码字段 |
|------|------|----------|
| \(D_\text{base}\) | 8 票等权否决率（strict `math_equal`） | `base_disagree` / bd |
| \(T_{\mathrm{ent\_prox\_lin}}\) | 8 扰动 run **近答案加权** process-token entropy（linear \(w_t=(t+1)/n\)） | `T_ent_prox_lin` |
| **PANDA** | LODO OOF LR(\(D_\text{base}, T_{\mathrm{ent\_prox\_lin}}\)) → \(\hat p\) | 脚本计算 |
| \(F_\text{resp}\)（诊断） | 8 扰动 ASE，与 bd 冗余 | `F_resp` |

| 扰动集成 \(K\) | **\(K=R+W=8\)**（4 text + 4 weight） | text_runs + weight_runs |

> **公平采样预算**：见下文 SE/TokUR 说明（与旧版相同）。Baselines（DeepConf 等）仍从 `summary.jsonl` 读取；**仅 PANDA 行及 F/bd/T 分量行重算**。
>
> **PANDA 聚合**：`python PANDA/scripts/aggregate_panda_v2.py` → `paper/maintable/panda_v2_results.json`
>
> **评估协议**：留一 **数据集** OOF logistic regression（train 另外 2 个数学集 → test 当前集；\(\mu,\sigma\) 仅在 train 估计）。报 AUROC / AUPRC / ACC*（ACC* = 分数最低 50% 样本的正确率）。

---

## 0b. PANDA 定稿公式（摘要）

\[
\text{PANDA}(x)=\sigma\big(w_1 D_\text{base}+w_2 T_{\mathrm{ent\_prox\_lin}}+b\big)
\]

- **\(D_\text{base}\)** = bd = \((1/8)\sum_i \mathbf{1}[\neg\text{math\_equal}(a_i,a_0)]\)
- **\(T_{\mathrm{ent\_prox\_lin}}\)** = 8 扰动 run 上 pre-answer reasoning 的 linear 近答案加权 mean entropy（越高 → 越靠近答案仍不确定 → 越不可靠）

---

## 0c. 定稿验证摘要（自动更新）

> N=8121（4 模型 × 3 seeds × 3 数学集）。复现：`python PANDA/scripts/aggregate_panda_v2.py`

| 配置 | macro LODO AUROC | Δ vs bd-only |
|------|:--:|:--:|
| bd-only | 0.897 | — |
| **PANDA = bd + T_ent_prox_lin** | **0.904** | **+0.007** |

---
## 1. 实验总览

| 实验 | 目的 | 表 | 计算 | GPU? |
|------|------|----|------|------|
| E1 主表（数学） | PANDA vs baseline + TokUR EU；**4 模型** | 表 1a–1d | 3-seed mean±std | Qwen2.5-3B seed1 已有，仅补 seed2/3 + 其余模型 |
| E2 主表（逻辑） | 跨域泛化；**仅主模型 Qwen2.5-3B** | 表 2 | 同上 | 需生成 |
| E3 跨模型 | 4 模型稳健性（数学集）；逻辑仅主模型 | 表 3 | 同上 | 需生成 |
| E4 组件消融 | PANDA 去一个组件 | 表 4 | LR 重拟合，CPU | 否（有 raw） |
| E5 扰动预算 | R / W 数量影响 | 表 5 | 子集重算，CPU | 否 |
| E6 λ 敏感性 | $\lambda_a,\lambda_t$ 稳健性 | 表 6 | 后处理，CPU | 否 |
| E7 扰动强度 | $\sigma$ / rank | 表 7 | 需重跑 | 是 |

**模型**（`configs/panda_models.yaml`）：Qwen2.5-3B(主) / Llama-3.2-1B / Llama-3.1-8B / Qwen3-8B
**数学数据集**：minerva(272 全量), math500(**300**), gsm8k(**300**) ——**定稿档 MAX_SAMPLES=300**
**逻辑数据集**：leg_counting, zebra_puzzles, color_cube（各 **300**）
**Seeds**：41, 42, 43（`DEFAULT_EXPERIMENT_SEEDS`）
**GPU 预算**：**¥3,000**（5090-32GB ¥2.88/h）；引擎 **vLLM 混合**

### 1.1 执行优先级（先 TokUR + PANDA，再扩模型）

> 主表核心对比是 **PANDA vs TokUR EU**；其余 10 个 baseline 随 PANDA 管线 Phase B 自动产出，不单独排队。

| 优先级 | 内容 | 目的 | 估 5090 卡时 |
|--------|------|------|------------:|
| **P0** | Qwen2.5-3B × 数学 3 集 × 3 seeds → **PANDA 全管线 + TokUR EU** | 主表 1a 可填 | ~280 h |
| **P1** | Qwen2.5-3B × 逻辑 3 集 × 3 seeds → PANDA + TokUR | 主表 2a 可填 | ~130 h |
| **P2** | 其余 3 模型 × 数学 3 集（Llama-3.2-1B / Llama-3.1-8B / Qwen3-8B） | 跨模型表 3 | ~350 h |
| **P3** | 聚合 3-seed 主表 + CPU 消融（E4/E5/E6） | 论文表格 | CPU |

**当前进度（2026-06-15）**：
- P0 ✅ / P1 ✅（Qwen2.5-3B 数学+逻辑）
- P2：**Llama-3.2-1B ✅**（2616/2616）；**Llama-3.1-8B ✅**（2616/2616，表 1c 已填）；**Qwen3-8B 进行中**（GSM8K 缺 seed43；其余基本完成）
- 后台：`run_p2_continue.sh` + `watchdog_phases.sh` + `monitor_progress.sh`（日志 `logs/p2_continue.log` / `logs/watchdog.log` / `logs/monitor.log`）
- 5090-32GB 上 8B **双流 OOM 已确认**，固定单流 SPG=1、**SE=0**
| **延后** | E7 扰动强度、失败分析 | 附录/可选 | 另计 |

**P0 最小闭环**（预算紧时只跑这个也能写主结论）：
1. **Phase A（vLLM）**：clean + R + SE → `*.partial.json`
2. **Phase B（HF）**：weight 分支 → **PANDA 两分量 + 11 baselines** → 完整 `raw_runs/*.json`
3. **Phase C（官方 TokUR venv）**：`run_tokur_official_maintable.sh` → export jsonl → `greedy_unc_single_batch_refine.py` → `tokur_baseline.jsonl`（`scoring_mode=official_vllm`）。**环境搭建与 5090 故障排查见 §14**。

> **禁止**主表使用 `score_tokur_baseline.py`（post-hoc HF 近似，与官方 TFB 扰动/生成路径不一致）。

已有 **Qwen2.5-3B seed1**（500 档旧 raw）：CPU 子采样对齐 300 题 + GPU 仅补 SE backfill。

---
## 2. 主表（E1：数学三数据集）

每个数据集三列 **AUROC | AUPRC | ACC***。Baselines 不变；**PANDA 及两分量（bd / T_ent_prox_lin）为 LODO OOF 结果**。

> **自动更新**：`update_experiment_plan_panda_v2.py`（数据源 `panda_v2_results.json`）

### 表 1a：Qwen2.5-3B

> **数据状态**：PANDA = LODO OOF LR(`D_base`/bd, `T_ent_prox_lin`)。n=2207。 复现：`python PANDA/scripts/aggregate_panda_v2.py`

#### 3-seed 平均 (strict label, LODO OOF)

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 60.10 ± 3.32 | 41.40 ± 2.54 | 78.27 ± 1.86 | 46.14 ± 1.33 | 8.94 ± 1.74 | 91.29 ± 1.12 | 61.70 ± 1.89 | 71.57 ± 1.72 | 46.25 ± 0.66 |
| PE | 59.24 ± 3.04 | 40.15 ± 1.73 | 76.31 ± 2.83 | 47.72 ± 1.31 | 11.30 ± 0.13 | 92.18 ± 0.86 | 66.29 ± 1.57 | 74.80 ± 1.86 | 48.63 ± 1.60 |
| LL | 58.68 ± 2.96 | 38.86 ± 1.67 | 76.87 ± 2.47 | 47.31 ± 1.47 | 6.97 ± 0.19 | 92.41 ± 1.16 | 63.77 ± 1.65 | 71.31 ± 1.95 | 47.95 ± 1.93 |
| Self-Certainty | 59.21 ± 2.99 | 40.13 ± 1.71 | 76.87 ± 2.47 | 47.71 ± 1.32 | 10.54 ± 1.18 | 92.18 ± 0.86 | 66.23 ± 1.56 | 74.82 ± 1.88 | 48.63 ± 1.60 |
| DeepConf | 65.11 ± 3.84 | 45.43 ± 3.44 | 81.05 ± 2.16 | 57.42 ± 3.67 | 10.04 ± 0.99 | 93.30 ± 0.96 | 69.38 ± 0.96 | 75.71 ± 0.72 | 52.03 ± 1.79 |
| **PANDA (Ours)** | 93.40 ± 0.52 | 80.70 ± 3.28 | 97.49 ± 0.01 | 78.77 ± 1.86 | 62.85 ± 3.30 | 96.65 ± 0.01 | 86.50 ± 1.30 | 91.53 ± 0.95 | 67.33 ± 2.40 |
| F_resp | 89.31 ± 0.63 | 70.29 ± 1.18 | 97.49 ± 0.01 | 79.35 ± 1.46 | 41.72 ± 3.21 | 97.77 ± 0.31 | 79.74 ± 0.79 | 84.51 ± 0.32 | 60.18 ± 2.51 |
| bd | 92.73 ± 0.35 | 79.40 ± 2.96 | 97.77 ± 0.39 | 80.07 ± 1.54 | 50.64 ± 4.36 | 97.77 ± 0.31 | 84.63 ± 1.61 | 88.36 ± 1.64 | 66.30 ± 3.12 |
| T_ent_prox_lin | 73.05 ± 0.90 | 44.49 ± 0.53 | 87.46 ± 0.73 | 68.33 ± 0.99 | 29.58 ± 1.67 | 95.76 ± 0.32 | 74.18 ± 0.17 | 81.23 ± 0.55 | 55.77 ± 1.64 |

#### seed41

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 55.95 | 38.28 | 76.47 | 46.33 | 11.19 | 91.28 | 60.76 | 70.29 | 46.46 |
| PE | 55.27 | 37.92 | 73.11 | 47.67 | 11.29 | 91.95 | 65.04 | 73.18 | 49.49 |
| LL | 54.84 | 36.95 | 73.95 | 47.29 | 7.23 | 91.95 | 63.38 | 70.83 | 48.48 |
| Self-Certainty | 55.31 | 37.92 | 73.95 | 47.64 | 11.29 | 91.95 | 65.02 | 73.16 | 49.49 |
| DeepConf | 59.69 | 40.57 | 78.15 | 58.08 | 9.77 | 93.96 | 70.23 | 75.45 | 54.55 |
| **PANDA (Ours)** | 93.04 | 76.32 | 97.48 | 79.05 | 64.32 | 96.64 | 88.26 | 92.73 | 70.71 |
| F_resp | 88.47 | 68.64 | 97.48 | 79.65 | 42.02 | 97.99 | 80.85 | 84.92 | 63.64 |
| bd | 92.30 | 75.52 | 98.32 | 80.90 | 55.20 | 97.99 | 86.78 | 90.29 | 70.71 |
| T_ent_prox_lin | 71.97 | 44.26 | 86.55 | 68.48 | 27.62 | 95.97 | 74.18 | 80.51 | 57.58 |

#### seed42

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 60.27 | 41.42 | 77.50 | 47.67 | 8.68 | 92.67 | 60.00 | 70.42 | 45.36 |
| PE | 59.79 | 40.41 | 75.83 | 49.35 | 11.46 | 93.33 | 65.33 | 73.82 | 46.39 |
| LL | 59.17 | 38.61 | 76.67 | 49.11 | 6.81 | 94.00 | 61.97 | 69.20 | 45.36 |
| Self-Certainty | 59.73 | 40.39 | 76.67 | 49.35 | 11.46 | 93.33 | 65.25 | 73.85 | 46.39 |
| DeepConf | 68.09 | 48.12 | 81.67 | 61.55 | 11.37 | 94.00 | 68.04 | 74.99 | 50.52 |
| **PANDA (Ours)** | 94.13 | 84.22 | 97.50 | 76.36 | 58.28 | 96.67 | 86.08 | 91.47 | 65.98 |
| F_resp | 89.51 | 70.89 | 97.50 | 77.43 | 37.64 | 97.33 | 79.34 | 84.14 | 57.73 |
| bd | 93.15 | 82.70 | 97.50 | 77.91 | 44.76 | 97.33 | 84.22 | 88.50 | 63.92 |
| T_ent_prox_lin | 73.00 | 45.23 | 87.50 | 67.05 | 29.42 | 96.00 | 73.96 | 81.35 | 53.61 |

#### seed43

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 64.08 | 44.50 | 80.83 | 44.42 | 6.96 | 89.93 | 64.34 | 74.01 | 46.94 |
| PE | 62.66 | 42.14 | 80.00 | 46.14 | 11.14 | 91.28 | 68.50 | 77.40 | 50.00 |
| LL | 62.04 | 41.01 | 80.00 | 45.52 | 6.87 | 91.28 | 65.96 | 73.90 | 50.00 |
| Self-Certainty | 62.58 | 42.09 | 80.00 | 46.13 | 8.87 | 91.28 | 68.43 | 77.45 | 50.00 |
| DeepConf | 67.54 | 47.59 | 83.33 | 52.64 | 8.98 | 91.95 | 69.88 | 76.69 | 51.02 |
| **PANDA (Ours)** | 93.02 | 81.54 | 97.50 | 80.88 | 65.95 | 96.64 | 85.16 | 90.39 | 65.31 |
| F_resp | 89.96 | 71.33 | 97.50 | 80.97 | 45.49 | 97.99 | 79.04 | 84.45 | 59.18 |
| bd | 92.75 | 79.97 | 97.50 | 81.39 | 51.94 | 97.99 | 82.89 | 86.28 | 64.29 |
| T_ent_prox_lin | 74.16 | 43.98 | 88.33 | 69.46 | 31.69 | 95.30 | 74.38 | 81.84 | 56.12 |

### 表 1b：Llama-3.2-1B

> **数据状态**：PANDA = LODO OOF LR(`D_base`/bd, `T_ent_prox_lin`)。n=2054。 复现：`python PANDA/scripts/aggregate_panda_v2.py`

#### 3-seed 平均 (strict label, LODO OOF)

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 54.94 ± 1.20 | 71.00 ± 1.16 | 33.55 ± 1.35 | 55.07 ± 0.41 | 49.46 ± 1.32 | 59.73 ± 1.10 | 63.39 ± 2.00 | 92.54 ± 0.65 | 15.30 ± 0.49 |
| PE | 54.23 ± 1.07 | 70.48 ± 1.06 | 33.23 ± 1.32 | 54.89 ± 0.41 | 48.89 ± 1.10 | 59.51 ± 0.84 | 63.74 ± 2.22 | 92.67 ± 0.61 | 16.04 ± 1.89 |
| LL | 53.38 ± 1.17 | 69.91 ± 1.24 | 32.59 ± 1.34 | 54.60 ± 0.41 | 48.38 ± 1.13 | 59.06 ± 1.64 | 62.96 ± 2.49 | 92.53 ± 0.77 | 15.30 ± 1.39 |
| Self-Certainty | 53.85 ± 1.15 | 69.96 ± 1.24 | 33.55 ± 1.35 | 54.61 ± 0.41 | 48.74 ± 1.12 | 59.28 ± 0.84 | 63.07 ± 2.20 | 92.56 ± 0.55 | 15.67 ± 1.84 |
| DeepConf | 61.76 ± 0.87 | 74.36 ± 0.77 | 43.23 ± 0.65 | 59.84 ± 0.54 | 55.30 ± 1.16 | 62.86 ± 1.92 | 72.12 ± 1.80 | 94.82 ± 0.14 | 16.78 ± 2.36 |
| **PANDA (Ours)** | 84.36 ± 1.19 | 89.98 ± 1.16 | 57.10 ± 0.83 | 89.60 ± 0.85 | 86.00 ± 1.89 | 89.49 ± 0.32 | 75.75 ± 3.99 | 95.40 ± 1.31 | 17.91 ± 0.92 |
| F_resp | 82.02 ± 1.16 | 86.17 ± 1.27 | 56.77 ± 0.40 | 87.49 ± 0.86 | 80.03 ± 2.06 | 85.68 ± 1.38 | 75.12 ± 2.23 | 94.93 ± 0.60 | 17.90 ± 1.74 |
| bd | 85.04 ± 1.17 | 88.90 ± 1.06 | 59.35 ± 0.95 | 88.83 ± 0.82 | 82.82 ± 1.76 | 86.58 ± 1.45 | 75.51 ± 4.72 | 95.03 ± 1.66 | 17.91 ± 0.92 |
| T_ent_prox_lin | 58.04 ± 0.64 | 67.82 ± 0.88 | 42.26 ± 0.40 | 72.06 ± 0.80 | 65.81 ± 1.72 | 72.93 ± 0.63 | 69.64 ± 0.94 | 94.04 ± 0.78 | 17.16 ± 1.02 |

#### seed41

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 53.50 | 69.54 | 33.98 | 54.54 | 49.92 | 58.39 | 66.21 | 92.57 | 15.73 |
| PE | 53.32 | 69.14 | 33.01 | 54.34 | 49.12 | 58.39 | 66.85 | 92.84 | 17.98 |
| LL | 52.17 | 68.24 | 33.01 | 54.07 | 48.68 | 57.05 | 66.49 | 93.08 | 16.85 |
| Self-Certainty | 52.66 | 68.30 | 33.98 | 54.08 | 48.94 | 58.39 | 66.15 | 92.68 | 17.98 |
| DeepConf | 62.31 | 73.44 | 43.69 | 60.50 | 56.68 | 63.09 | 73.84 | 94.88 | 17.98 |
| **PANDA (Ours)** | 84.59 | 89.63 | 58.25 | 89.90 | 86.94 | 89.26 | 77.19 | 95.70 | 19.10 |
| F_resp | 83.67 | 87.96 | 57.28 | 87.35 | 80.41 | 83.89 | 72.54 | 94.19 | 17.98 |
| bd | 85.20 | 88.86 | 58.25 | 89.20 | 83.77 | 84.56 | 77.55 | 95.74 | 19.10 |
| T_ent_prox_lin | 57.14 | 66.60 | 41.75 | 71.49 | 66.73 | 72.48 | 68.35 | 92.94 | 17.98 |

#### seed42

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 56.45 | 72.37 | 34.95 | 55.15 | 50.79 | 59.73 | 62.11 | 93.31 | 14.61 |
| PE | 55.73 | 71.74 | 34.95 | 55.01 | 50.10 | 59.73 | 61.78 | 93.31 | 13.48 |
| LL | 54.97 | 71.20 | 33.98 | 54.66 | 49.58 | 59.06 | 61.09 | 93.07 | 13.48 |
| Self-Certainty | 55.40 | 71.28 | 34.95 | 54.66 | 49.99 | 59.06 | 61.16 | 93.17 | 13.48 |
| DeepConf | 62.44 | 75.32 | 43.69 | 59.17 | 55.37 | 60.40 | 69.64 | 94.96 | 13.48 |
| **PANDA (Ours)** | 82.80 | 88.77 | 56.31 | 90.46 | 87.70 | 89.26 | 79.76 | 96.83 | 16.85 |
| F_resp | 81.10 | 85.14 | 56.31 | 88.61 | 82.34 | 85.91 | 74.84 | 95.65 | 15.73 |
| bd | 83.53 | 87.63 | 59.22 | 89.59 | 84.33 | 87.92 | 80.00 | 96.61 | 16.85 |
| T_ent_prox_lin | 58.41 | 68.62 | 42.72 | 73.19 | 67.30 | 72.48 | 70.00 | 94.44 | 15.73 |

#### seed43

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 54.88 | 71.09 | 31.73 | 55.53 | 47.65 | 61.07 | 61.85 | 91.73 | 15.56 |
| PE | 53.65 | 70.55 | 31.73 | 55.32 | 47.43 | 60.40 | 62.60 | 91.85 | 16.67 |
| LL | 53.01 | 70.29 | 30.77 | 55.07 | 46.86 | 61.07 | 61.31 | 91.43 | 15.56 |
| Self-Certainty | 53.47 | 70.31 | 31.73 | 55.09 | 47.28 | 60.40 | 61.88 | 91.83 | 15.56 |
| DeepConf | 60.54 | 74.32 | 42.31 | 59.87 | 53.85 | 65.10 | 72.87 | 94.62 | 18.89 |
| **PANDA (Ours)** | 85.69 | 91.55 | 56.73 | 88.43 | 83.35 | 89.93 | 70.31 | 93.67 | 17.78 |
| F_resp | 81.31 | 85.39 | 56.73 | 86.51 | 77.33 | 87.25 | 77.98 | 94.94 | 20.00 |
| bd | 86.38 | 90.22 | 60.58 | 87.69 | 80.35 | 87.25 | 68.99 | 92.73 | 17.78 |
| T_ent_prox_lin | 58.56 | 68.25 | 42.31 | 71.50 | 63.39 | 73.83 | 70.57 | 94.73 | 17.78 |

### 表 1c：Llama-3.1-8B

> **数据状态**：PANDA = LODO OOF LR(`D_base`/bd, `T_ent_prox_lin`)。n=2131。 复现：`python PANDA/scripts/aggregate_panda_v2.py`

#### 3-seed 平均 (strict label, LODO OOF)

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 57.45 ± 0.75 | 54.33 ± 0.67 | 64.18 ± 0.30 | 57.41 ± 1.48 | 18.63 ± 1.82 | 93.29 ± 0.00 | 64.60 ± 1.67 | 78.66 ± 1.25 | 40.22 ± 1.38 |
| PE | 58.21 ± 0.78 | 55.39 ± 0.80 | 63.89 ± 1.33 | 56.90 ± 1.57 | 18.62 ± 3.09 | 93.29 ± 0.00 | 63.54 ± 1.47 | 78.02 ± 1.02 | 39.50 ± 1.75 |
| LL | 56.57 ± 0.88 | 52.46 ± 1.66 | 62.09 ± 0.74 | 56.82 ± 1.70 | 17.54 ± 3.51 | 93.51 ± 0.32 | 62.96 ± 1.16 | 77.26 ± 0.53 | 39.50 ± 1.75 |
| Self-Certainty | 57.30 ± 0.89 | 54.17 ± 0.93 | 63.59 ± 0.91 | 56.89 ± 1.58 | 18.59 ± 3.10 | 93.29 ± 0.00 | 63.37 ± 1.52 | 77.90 ± 1.03 | 39.50 ± 1.75 |
| DeepConf | 69.42 ± 0.65 | 66.15 ± 2.89 | 71.36 ± 2.39 | 75.97 ± 1.51 | 25.14 ± 2.85 | 98.21 ± 0.63 | 68.62 ± 1.12 | 82.96 ± 0.61 | 40.93 ± 2.17 |
| **PANDA (Ours)** | 89.89 ± 0.24 | 85.96 ± 1.27 | 88.96 ± 1.46 | 87.32 ± 0.45 | 66.65 ± 1.81 | 98.66 ± 0.00 | 76.66 ± 1.39 | 85.21 ± 2.16 | 49.11 ± 0.25 |
| F_resp | 84.26 ± 0.56 | 73.01 ± 2.23 | 85.39 ± 1.72 | 88.87 ± 0.74 | 51.10 ± 5.96 | 98.43 ± 0.63 | 75.84 ± 2.17 | 84.86 ± 2.50 | 48.76 ± 2.06 |
| bd | 90.11 ± 0.30 | 84.88 ± 1.31 | 90.46 ± 1.77 | 89.73 ± 0.34 | 61.52 ± 2.13 | 98.43 ± 0.63 | 77.50 ± 1.06 | 85.80 ± 0.97 | 49.47 ± 0.75 |
| T_ent_prox_lin | 57.39 ± 1.12 | 48.16 ± 0.90 | 63.90 ± 2.74 | 60.50 ± 0.32 | 21.86 ± 0.35 | 94.18 ± 0.32 | 56.65 ± 2.43 | 73.43 ± 0.88 | 36.30 ± 3.18 |

#### seed41

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 56.85 | 54.74 | 63.96 | 56.91 | 16.25 | 93.29 | 62.39 | 76.92 | 40.86 |
| PE | 57.69 | 55.86 | 63.06 | 56.51 | 16.32 | 93.29 | 61.68 | 76.61 | 39.78 |
| LL | 56.34 | 52.99 | 61.26 | 56.14 | 14.61 | 93.96 | 61.35 | 77.00 | 39.78 |
| Self-Certainty | 57.04 | 54.43 | 63.06 | 56.51 | 16.32 | 93.29 | 61.40 | 76.48 | 39.78 |
| DeepConf | 68.67 | 63.55 | 72.07 | 74.78 | 24.96 | 98.66 | 68.52 | 82.82 | 40.86 |
| **PANDA (Ours)** | 90.22 | 87.49 | 88.29 | 86.69 | 64.09 | 98.66 | 76.50 | 86.96 | 49.46 |
| F_resp | 84.07 | 72.47 | 85.59 | 88.44 | 49.19 | 97.99 | 77.38 | 85.66 | 51.61 |
| bd | 90.33 | 86.00 | 90.09 | 89.26 | 58.52 | 97.99 | 77.24 | 85.79 | 50.54 |
| T_ent_prox_lin | 58.16 | 48.59 | 66.67 | 60.65 | 21.40 | 94.63 | 55.79 | 73.21 | 37.63 |

#### seed42

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 56.98 | 53.38 | 63.96 | 55.90 | 19.00 | 93.29 | 66.43 | 79.76 | 41.49 |
| PE | 57.63 | 54.27 | 65.77 | 55.20 | 16.55 | 93.29 | 65.26 | 79.00 | 41.49 |
| LL | 55.62 | 50.21 | 63.06 | 55.17 | 15.53 | 93.29 | 64.02 | 76.79 | 41.49 |
| Self-Certainty | 56.36 | 52.93 | 64.86 | 55.17 | 16.48 | 93.29 | 65.10 | 78.88 | 41.49 |
| DeepConf | 69.34 | 64.74 | 73.87 | 78.10 | 28.71 | 98.66 | 70.04 | 83.77 | 43.62 |
| **PANDA (Ours)** | 89.78 | 84.39 | 90.99 | 87.71 | 67.89 | 98.66 | 75.05 | 82.17 | 48.94 |
| F_resp | 83.69 | 70.58 | 87.39 | 89.91 | 59.16 | 97.99 | 72.77 | 81.48 | 46.81 |
| bd | 89.69 | 83.04 | 92.79 | 90.05 | 62.75 | 97.99 | 76.35 | 84.63 | 48.94 |
| T_ent_prox_lin | 58.20 | 46.90 | 64.86 | 60.05 | 21.95 | 93.96 | 59.96 | 74.60 | 39.36 |

#### seed43

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 58.51 | 54.86 | 64.60 | 59.41 | 20.65 | 93.29 | 64.97 | 79.30 | 38.30 |
| PE | 59.31 | 56.05 | 62.83 | 58.99 | 22.98 | 93.29 | 63.68 | 78.44 | 37.23 |
| LL | 57.74 | 54.18 | 61.95 | 59.15 | 22.47 | 93.29 | 63.50 | 78.00 | 37.23 |
| Self-Certainty | 58.49 | 55.16 | 62.83 | 58.99 | 22.98 | 93.29 | 63.61 | 78.34 | 37.23 |
| DeepConf | 70.25 | 70.17 | 68.14 | 75.02 | 21.75 | 97.32 | 67.29 | 82.29 | 38.30 |
| **PANDA (Ours)** | 89.68 | 86.00 | 87.61 | 87.56 | 67.96 | 98.66 | 78.44 | 86.51 | 48.94 |
| F_resp | 85.03 | 75.97 | 83.19 | 88.27 | 44.94 | 99.33 | 77.37 | 87.45 | 47.87 |
| bd | 90.31 | 85.60 | 88.50 | 89.87 | 63.29 | 99.33 | 78.90 | 87.00 | 48.94 |
| T_ent_prox_lin | 55.81 | 48.98 | 60.18 | 60.80 | 22.23 | 93.96 | 54.20 | 72.47 | 31.91 |

### 表 1d：Qwen3-8B

> **数据状态**：PANDA = LODO OOF LR(`D_base`/bd, `T_ent_prox_lin`)。n=1729。 复现：`python PANDA/scripts/aggregate_panda_v2.py`

#### 3-seed 平均 (strict label, LODO OOF)

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 84.31 ± 10.11 | 55.65 ± 15.88 | 92.60 ± 5.28 | 76.64 ± 4.50 | 48.11 ± 19.99 | 91.43 ± 4.66 | 62.63 ± 3.45 | 81.81 ± 3.27 | 38.23 ± 4.49 |
| PE | 85.04 ± 9.58 | 56.47 ± 16.40 | 92.59 ± 5.24 | 77.62 ± 4.60 | 51.40 ± 17.41 | 91.68 ± 4.62 | 66.70 ± 4.11 | 83.28 ± 3.85 | 40.53 ± 4.01 |
| LL | 85.08 ± 9.54 | 55.06 ± 15.42 | 92.59 ± 5.24 | 75.48 ± 6.62 | 47.67 ± 20.48 | 90.88 ± 3.62 | 66.19 ± 4.45 | 83.09 ± 3.93 | 40.12 ± 4.00 |
| Self-Certainty | 85.05 ± 9.57 | 56.49 ± 16.41 | 92.59 ± 5.24 | 77.62 ± 4.60 | 51.37 ± 17.38 | 91.68 ± 4.62 | 66.70 ± 4.10 | 83.26 ± 3.85 | 40.53 ± 4.01 |
| DeepConf | 84.29 ± 11.37 | 78.08 ± 15.54 | 90.91 ± 6.43 | 80.05 ± 4.87 | 62.75 ± 8.48 | 92.69 ± 3.49 | 66.74 ± 2.78 | 84.21 ± 1.98 | 43.64 ± 2.52 |
| **PANDA (Ours)** | 97.75 ± 0.87 | 70.54 ± 35.74 | 100.00 ± 0.00 | 93.06 ± 5.06 | 75.19 ± 27.01 | 98.60 ± 0.15 | 92.67 ± 2.15 | 94.57 ± 2.96 | 55.68 ± 1.52 |
| F_resp | 91.12 ± 5.05 | 62.95 ± 26.94 | 94.61 ± 3.92 | 79.68 ± 2.29 | 52.61 ± 18.43 | 90.32 ± 6.08 | 85.64 ± 0.57 | 92.85 ± 0.28 | 54.17 ± 3.72 |
| bd | 97.48 ± 1.30 | 67.81 ± 37.85 | 100.00 ± 0.00 | 92.14 ± 5.80 | 72.85 ± 27.15 | 98.60 ± 0.15 | 92.90 ± 1.76 | 95.05 ± 2.02 | 56.05 ± 1.32 |
| T_ent_prox_lin | 72.81 ± 5.12 | 38.56 ± 24.33 | 84.50 ± 11.00 | 75.14 ± 0.76 | 44.54 ± 18.67 | 88.98 ± 7.98 | 53.41 ± 1.75 | 74.34 ± 2.78 | 30.77 ± 1.28 |

#### seed41

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 74.61 | 64.69 | 88.00 | 77.30 | 59.30 | 86.36 | 66.10 | 83.92 | 44.58 |
| PE | 75.87 | 66.75 | 89.00 | 78.31 | 60.44 | 86.36 | 71.31 | 86.67 | 45.78 |
| LL | 76.00 | 64.35 | 89.00 | 77.62 | 57.90 | 86.36 | 71.11 | 86.41 | 44.58 |
| Self-Certainty | 75.90 | 66.80 | 89.00 | 78.33 | 60.43 | 86.36 | 71.30 | 86.65 | 45.78 |
| DeepConf | 73.47 | 65.85 | 86.00 | 81.81 | 66.27 | 90.15 | 70.47 | 86.18 | 46.99 |
| **PANDA (Ours)** | 98.40 | 96.20 | 100.00 | 96.35 | 93.53 | 98.48 | 89.96 | 90.83 | 57.83 |
| F_resp | 89.89 | 84.82 | 93.00 | 76.63 | 63.37 | 84.85 | 86.22 | 92.52 | 59.04 |
| bd | 98.20 | 93.81 | 100.00 | 95.82 | 90.54 | 98.48 | 90.77 | 92.61 | 57.83 |
| T_ent_prox_lin | 68.52 | 55.21 | 78.00 | 76.14 | 57.85 | 85.61 | 51.31 | 71.11 | 32.53 |

#### seed42

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 80.07 | 68.92 | 89.80 | 81.79 | 64.99 | 90.30 | 63.86 | 84.33 | 35.23 |
| PE | 80.99 | 69.34 | 88.78 | 82.88 | 66.72 | 91.04 | 67.44 | 85.27 | 39.77 |
| LL | 80.99 | 67.50 | 88.78 | 82.30 | 66.03 | 91.04 | 67.12 | 85.30 | 40.91 |
| Self-Certainty | 81.00 | 69.34 | 88.78 | 82.87 | 66.63 | 91.04 | 67.46 | 85.26 | 39.77 |
| DeepConf | 79.41 | 68.38 | 86.73 | 84.93 | 70.92 | 90.30 | 65.96 | 84.96 | 40.91 |
| **PANDA (Ours)** | 98.32 | 95.42 | 100.00 | 96.91 | 95.03 | 98.51 | 95.23 | 98.07 | 54.55 |
| F_resp | 85.65 | 79.03 | 90.82 | 80.25 | 67.79 | 87.31 | 84.86 | 92.82 | 50.00 |
| bd | 98.58 | 95.33 | 100.00 | 96.65 | 93.51 | 98.51 | 95.07 | 97.55 | 55.68 |
| T_ent_prox_lin | 69.91 | 56.32 | 75.51 | 74.98 | 57.64 | 81.34 | 55.60 | 77.89 | 29.55 |

#### seed43

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 98.26 | 33.33 | 100.00 | 70.83 | 20.03 | 97.62 | 57.92 | 77.19 | 34.88 |
| PE | 98.26 | 33.33 | 100.00 | 71.67 | 27.05 | 97.62 | 61.33 | 77.89 | 36.05 |
| LL | 98.26 | 33.33 | 100.00 | 66.53 | 19.10 | 95.24 | 60.34 | 77.56 | 34.88 |
| Self-Certainty | 98.26 | 33.33 | 100.00 | 71.67 | 27.05 | 97.62 | 61.35 | 77.87 | 36.05 |
| DeepConf | 100.00 | 100.00 | 100.00 | 73.40 | 51.07 | 97.62 | 63.80 | 81.50 | 43.02 |
| **PANDA (Ours)** | 96.52 | 20.00 | 100.00 | 85.90 | 37.01 | 98.81 | 92.80 | 94.83 | 54.65 |
| F_resp | 97.83 | 25.00 | 100.00 | 82.15 | 26.67 | 98.81 | 85.83 | 93.20 | 53.49 |
| bd | 95.65 | 14.29 | 100.00 | 83.96 | 34.48 | 98.81 | 92.87 | 94.98 | 54.65 |
| T_ent_prox_lin | 80.00 | 4.17 | 100.00 | 74.31 | 18.14 | 100.00 | 53.33 | 74.02 | 30.23 |

## 3. 主表（E2：逻辑）

参考图布局：逻辑题缺隐藏状态/采样类 baseline 时留 `—`。

### 表 2a：逻辑数据集（主模型 Qwen2.5-3B；仅此一模型证跨域泛化）

> **数据集范围调整**：原计划逻辑三集（zebra_puzzles / leg_counting / color_cube），现 **剔除 zebra_puzzles**。原因：zebra 的 gold 是结构化网格（`{"header":[...],"rows":[...]}`），而模型输出为自由文本，常给出多种可能排列或漏命名；自由文本→网格的可靠判分需强制固定输出格式并重新生成（耗 GPU）或研究级解析器，启发式判分不可信，故暂从主表剔除，待后续以"强制网格输出格式 + 精确比对"方案补齐。保留 **leg_counting + color_cube** 两集。

3-seed (41/42/43) mean ± std，strict label 子集（`label_drop` 排除；逻辑题 drop=0）。复现：`python scripts/aggregate_math_strict.py --datasets color_cube,leg_counting`。

| Method | Leg Counting AUROC | AUPRC | ACC* | Color Cube AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|
| CoT (Lower-Bound) | — | — | — | — | — | — |
| PE | 61.86 ± 0.84 | 77.90 ± 1.26 | 33.33 ± 2.49 | 61.82 ± 0.34 | 76.30 ± 1.22 | 37.49 ± 0.74 |
| LL | 58.16 ± 1.10 | 76.03 ± 1.55 | 34.00 ± 1.63 | 61.40 ± 1.02 | 77.10 ± 1.80 | 37.99 ± 0.93 |
| Self-Certainty | 61.83 ± 0.84 | 77.87 ± 1.24 | 33.33 ± 2.49 | 61.69 ± 0.44 | 76.17 ± 1.22 | 37.49 ± 0.74 |
| DeepConf | 67.06 ± 1.56 | 81.39 ± 1.69 | 38.67 ± 0.94 | 63.67 ± 0.49 | 78.61 ± 0.88 | 39.83 ± 0.85 |
| TokUR EU | — | — | — | — | — | — |
| **PANDA (Ours)** | **76.51 ± 8.15** | **88.53 ± 4.45** | **42.25 ± 5.78** | —¹ | —¹ | —¹ |

样本量/准确率：leg_counting n=100/seed，acc≈28–30%；color_cube n=300/seed（seed42=234），acc≈29–31%。两集 strict drop≈0。

¹ **color_cube 的 PANDA 暂缺**：PANDA 在 color_cube 上 **98% 样本为 `nan`**（颜色词答案使 `formula_skeleton`/数值轨迹等数学专属特征为 nan，导致最终 logistic 组合 nan），仅余 ~2% 可算、不可信，故留空。需为非数字答案分支重拟合/降维 PANDA 后再填。leg_counting PANDA 用 **5-fold OOF LR**（单数据集无法 LODO）；**76.51 ± 8.15** AUROC，仍领先 DeepConf 67.06。TokUR EU 不在 summary 字段中，留空待官方 TokUR 评分接入。

> **判分修复①（string 数据集）**：修复严重 bug —— 旧 raw 判分对所有 string 题统一用 `math_equal`，而它对非数字串会误返回 `True`，导致 **color_cube 等被全部判对**（旧 acc≈100% 为假）。现改为 **mode-aware 严格判分**（生成 `record.py` + 离线 `scripts/fix_logic_extraction.py`），并从 `full_response` 重抽取答案。
>
> **判分修复②（MANUAL_RELABEL 跨模型误用）**：人工核验白名单 `MANUAL_RELABEL_IDS`（minerva 14 题）此前对所有模型按 id 强制判对，但它仅在 Qwen2.5-3B 上核验过。已给 `grade_answer` 加 `model` 参数，**白名单仅对 Qwen2.5-3B 生效**；其余模型（Llama/Qwen3-8B）这些题回归 strict 判定（多为 drop），避免分数虚高。

---

## 4. 跨模型汇总（E3）

仅报 PANDA 主指标 AUROC（mean±std），验证跨模型稳健。数学三集跑全 4 模型；逻辑三集仅主模型（其余模型留 `—`）。

### 表 3：PANDA AUROC × 模型 × 数据集

| Model | MATH-500 | GSM8K | Minerva |
|-------|--:|--:|--:|
| **Qwen2.5-3B** | 93.40 ± 0.52 | 78.77 ± 1.86 | 86.50 ± 1.30 |
| Llama-3.2-1B | 84.36 ± 1.19 | 89.60 ± 0.85 | 75.75 ± 3.99 |
| Llama-3.1-8B | 89.89 ± 0.24 | 87.32 ± 0.45 | 76.66 ± 1.39 |
| Qwen3-8B | 97.75 ± 0.87 | 93.06 ± 5.06 | 92.67 ± 2.15 |

> 3-seed mean±std，strict label，PANDA = LODO(bd, T_ent_prox_lin)。

---

## 5. 组件消融（E4）

**消融 = 完整 PANDA 去掉恰好一个分量**，对剩余分量 **重新 LODO OOF LR**。报 AUROC + Δ（相对 full PANDA）。

### 表 4：组件消融（Qwen2.5-3B，LODO OOF）

| Variant | MATH-500 AUROC | Δ | GSM8K AUROC | Δ | Minerva AUROC | Δ |
|---------|--:|--:|--:|--:|--:|--:|
| **PANDA (full)** | 93.40 ± 0.52 | 0 | 78.77 ± 1.86 | 0 | 86.50 ± 1.30 | 0 |
| − D_base → T | 73.05 | -20.35 | 68.33 | -10.43 | 74.18 | -12.32 |
| − T → D_base | 92.73 | -0.67 | 80.07 | +1.30 | 84.63 | -1.86 |
| + F legacy (附录) | 93.42 | +0.02 | 78.72 | -0.05 | 86.49 | -0.00 |

### 表 4b：逐步消融（Qwen2.5-3B，LODO OOF AUROC）

| 配置 | MATH-500 | GSM8K | Minerva |
|------|--:|--:|--:|
| D_base（raw AUROC） | 92.73 ± 0.35 | 80.07 ± 1.54 | 84.63 ± 1.61 |
| T_ent_prox_lin（raw） | 73.05 ± 0.90 | 68.33 ± 0.99 | 74.18 ± 0.17 |
| **PANDA = LODO(D_base, T_ent_prox_lin)** | 93.40 ± 0.52 | 78.77 ± 1.86 | 86.50 ± 1.30 |
> **Drop-one 消融**：相对 full，Δ<0 表示去掉该分量后下降。

---

## 6. 扰动预算消融（E5）

固定一支为默认满额，扫另一支数量，报 PANDA 与 $F_\text{resp}$ 的 AUROC（CPU 子集重算）。

### 表 5a：Text rephrase 数 R（W=8，Qwen2.5-3B）

| R | MATH-500 | GSM8K | Minerva |
|---|--:|--:|--:|
| 0 |  |  |  |
| 2 |  |  |  |
| 4 |  |  |  |
| 8 |  |  |  |

### 表 5b：Weight perturb 数 W（R=8）

| W | MATH-500 | GSM8K | Minerva |
|---|--:|--:|--:|
| 0 |  |  |  |
| 2 |  |  |  |
| 4 |  |  |  |
| 8 |  |  |  |

---

## 7. λ 敏感性（E6）

CPU 后处理：`compute_panda_from_row(row, lambda_a=, lambda_r=)`。代表数据集 MATH-500。

### 表 6a：$\lambda_a$ 扫描（$\lambda_t{=}0.03$）

| $\lambda_a$ | 0.01 | 0.03 | **0.05** | 0.10 | 0.20 |
|-------------|--:|--:|--:|--:|--:|
| AUROC |  |  |  |  |  |

### 表 6b：$\lambda_t$ 扫描（$\lambda_a{=}0.05$）

| $\lambda_t$ | 0.01 | **0.03** | 0.05 | 0.10 |
|-------------|--:|--:|--:|--:|
| AUROC |  |  |  |  |

---

## 8. 扰动强度（E7，需 GPU 重跑）

### 表 7：Weight $\sigma$ / rank（R=8, W=8，MATH-500）

| $\sigma$ / rank | AUROC | AUPRC | ACC* |
|-----------------|--:|--:|--:|
| 0.01 / r=4 |  |  |  |
| **0.03 / r=4**（默认） |  |  |  |
| 0.05 / r=4 |  |  |  |
| 0.03 / r=8 |  |  |  |

---

## 9. 运行命令速查（定稿档 MAX_SAMPLES=300）

```bash
cd /mnt/afs/L202500372/PANDA
source scripts/env.sh
export MAX_SAMPLES=300   # 默认已写入 configs/panda_models.yaml 与编排脚本

# ========== P0：主模型 PANDA + 官方 TokUR（数学，最优先）==========
# Phase A(vLLM) + B(HF weight+metrics) + C(official TokUR) 一体
DATASETS=minerva,math500,gsm8k bash scripts/run_maintable_vllm.sh qwen25_3b

# 分批/续跑（重复执行自动跳过已完成）
SEEDS=41 DATASETS=math500 bash scripts/run_maintable_vllm.sh qwen25_3b
SKIP_VLLM=1 bash scripts/run_maintable_vllm.sh qwen25_3b   # 只补 HF：weight + metrics
SKIP_HF=1   bash scripts/run_maintable_vllm.sh qwen25_3b   # 只跑 vLLM：clean+R+SE
PANDA_SKIP_TOKUR=1 bash scripts/run_maintable_vllm.sh qwen25_3b  # 跳过 Phase C

# 单独补官方 TokUR（需 Phase B 已完成 raw_runs）
OUT_DIR=$PANDA_OUTPUTS/maintable_qwen25_3b/seed41 PANDA_MODEL_TAG=qwen25_3b \
  DATASET=math500 TOKUR_SEED=41 bash scripts/run_tokur_official_maintable.sh

# seed1 已有 500 档 raw：子采样对齐 300 + 补 SE
SEEDS=41 MAX_SAMPLES=300 SKIP_VLLM=1 bash scripts/run_maintable_vllm.sh qwen25_3b

# ========== P1：主模型逻辑（PANDA + TokUR）==========
bash scripts/download_logic_code_datasets.sh
bash scripts/generate_api_variants.sh leg_counting,zebra_puzzles,color_cube
DATASETS=leg_counting,zebra_puzzles,color_cube \
  bash scripts/run_maintable_vllm.sh qwen25_3b

# ========== P2：其余 3 模型数学 ==========
for m in llama32_1b llama31_8b qwen3_8b; do
  DATASETS=minerva,math500,gsm8k bash scripts/run_maintable_vllm.sh "$m"
done

# ========== P3：聚合主表（CPU）==========
for m in qwen25_3b llama32_1b llama31_8b qwen3_8b; do
  bash scripts/aggregate_maintable.sh "$m"
done

# --- E4 组件消融（CPU，有 raw 后跑）---
bash scripts/run_ablation_cpu.sh

# --- E5 扰动预算（CPU）---
python -m panda.core.ablation_recompute --out-dir $PANDA_OUTPUTS/panda_full \
  --datasets math500,gsm8k,minerva --budget-sweep \
  --output $PANDA_OUTPUTS/ABLATION_budget_sweep.json

# --- E7 扰动强度（GPU，预算充裕再跑）---
python -m panda.core.run_panda_experiment --dataset math500 --mode all \
  --max-samples 300 --weight-sigma 0.05 --weight-rank 4 \
  --out-dir $PANDA_OUTPUTS/ablation_sigma05
```

输出位置：
- 主表：`paper/maintable/{model}/maintable.{md,tex}`
- 消融：`$PANDA_OUTPUTS/ABLATION_component.md`、`ABLATION_budget_sweep.json`

---

## 10. 统计与报告规范

- **指标**：AUROC（主）、AUPRC（不平衡集补充）、ACC*（Top-50% accuracy，TokUR 协议）
- **标签**：clean label；附录可附 raw label 与 `relabeled` 计数（Minerva 重点）
- **种子**：3 seeds(41/42/43) 报 mean±std；可选 `bootstrap_auroc` 给 95% CI
- **置信度类方法**（SC/DeepConf/P(True)）AUROC 计算时取负（`BASELINE_INVERT_KEYS` 已自动处理）

---



> 无编造结果：本计划所有表格均为空待填（`TBD`），符合 no-fabrication 要求。
> 失败分析（stable-wrong 子集）本轮按用户要求暂缓。

---

## 12. 数据集题量与算力估算（A100 / H100 / RTX 5090）

### 12.1 每数据集题量（定稿档 `MAX_SAMPLES=300`）

| 数据集 | 域 | 每 seed 题量 | 来源 |
|--------|----|----:|------|
| Minerva | math | 272 | 全集（`MAX_SAMPLES_MINERVA=272`，不受 cap 影响） |
| MATH-500 | math | **300** | `max_samples=300`（截断自 500） |
| GSM8K | math | **300** | `max_samples=300`（截断自 1319） |
| Zebra Puzzles | logic | **300** | `max_samples=300` |
| Color Cube | logic | **300** | 同上 |
| Leg Counting | logic | **300** | 同上 |
| **数学合计/模型/seed** | | **872** | 272+300+300，4 模型都跑 |
| **逻辑合计/seed（仅主模型）** | | **900** | 300×3，仅 Qwen2.5-3B 跑 |

**定稿方案口径**：
- 数学 3 集（**872**/seed）× **4 模型** × 3 seeds；**优先 P0 跑 Qwen2.5-3B PANDA + TokUR**。
- **Qwen2.5-3B seed1 已有**（旧 500 档 raw）：CPU **子采样前 300** 对齐新口径；R=W=8→4 子集重算免 GPU；缺 SE 需 GPU backfill（约 872 题×8 次，≈ +38 A100-h）。
- 逻辑 3 集（900/seed）**仅主模型 Qwen2.5-3B** × 3 seeds（P1）。
- 每题 **17 次解码**（`1+R(4)+W(4)+SE(8)`），统一采样预算 K=8。
- **执行顺序**：P0 主模型数学 PANDA+TokUR → P1 逻辑 → P2 其余模型 → P3 CPU 消融。

### 12.2 成本模型与假设

| 参数 | 取值 | 说明 |
|------|------|------|
| 每题生成次数 | **17**（`1 + R(4) + W(4) + SE(8)`） | 公平采样预算 K=8；SE 同等 8 样本 |
| 平均输出 token | 800 | 数学推理均值；Qwen3-8B reasoning 可能 ×1.5–2 |
| 每题 token 量 | $17\times800=13{,}600$ | |

**单流解码吞吐假设**（bf16，HF generate + `topk-save 10` 开销，保守，token/s）：

| 模型 | A100-80G | H100-80G | RTX 5090-32G |
|------|---------:|---------:|-------------:|
| Llama-3.2-1B | 55 | 88 | 50 |
| Qwen2.5-3B | 40 | 64 | 36 |
| Llama-3.1-8B / Qwen3-8B | 28 | 45 | 25 |

> 带宽比例：H100≈1.6×A100，5090≈0.9×A100。`PANDA_FAST=1`（数据集并行+分片）只缩短**墙钟**，卡时总量不变。

**各模型生成题量**（300 档）：

| 模型 | 题量 | token 量 |
|------|----:|----:|
| Llama-3.2-1B（数学 3 seed） | 872×3 = 2616 | 35.6M |
| Qwen2.5-3B（数学补 2 seed + 逻辑 3 seed） | 872×2 + 900×3 = 4444 | 60.4M |
| Llama-3.1-8B（数学 3 seed） | 2616 | 35.6M |
| Qwen3-8B（数学 3 seed） | 2616 | 35.6M |

### 12.3 主实验卡时（300 档定稿口径）

| 模型 | A100 (h) | H100 (h) | 5090 (h) | 优先级 |
|------|---------:|---------:|---------:|--------|
| **Qwen2.5-3B（含逻辑，PANDA+TokUR）** | **≈400** | **≈250** | **≈445** | **P0+P1** |
| Llama-3.2-1B | 180 | 112 | 198 | P2 |
| Llama-3.1-8B | 353 | 220 | 397 | P2 |
| Qwen3-8B | 353 | 220 | 397 | P2 |
| **小计** | **≈1286** | **≈802** | **≈1437** |

加 TokUR EU + INSIDE/P(True) 前向 ≈ **+15%**（TokUR 在 P0 即跑，已含）。

| 全计划合计（300 档，3 seeds，vLLM 前 HF 基准） | A100 | H100 | 5090 |
|------|---------:|---------:|---------:|
| **GPU-hours** | **≈1480** | **≈920** | **≈1650** |

> 相对 500 推荐档（2150 A100-h）约 **−31%**。CPU 消融（E4/E5/E6）不占 GPU。误差带 ±40%。

### 12.4 换算参考
- 8×A100：≈ **11 节点·天**；8×H100：≈ **7 节点·天**；8×5090：≈ **12.6 节点·天**（单卡 ≈100 天，须多卡）。

### 12.5 进一步降本（在定稿基础上）

| # | 方案 | A100 节省 | 后合计 |
|---|------|----------:|-------:|
| 1 | **8B 模型仅 1 seed**（1B/3B 保 3 seed） | −约 790 h | ≈**1380** |
| 2 | **Qwen3-8B 设 `--max-new-tokens 1024`** | 防 reasoning 长尾，避免 +约 500 h 风险 | — |
| 3 | 逻辑改用 Llama-3.2-1B 主模型 | 微降（约 −90 h） | — |
| — | 复用 Qwen2.5-3B seed1 | （已计入定稿口径） | — |

> **已采纳**：降 R/W 到 4（K=8）；**MAX_SAMPLES=300**；**vLLM 混合管线（§12.6）**；**先 P0 PANDA+TokUR 再扩模型**。
> 300 档下 4 模型 3 seed 全跑可纳入 ¥3000 预算，无需再砍 8B seed。

### 12.6 三阶段管线（PANDA vLLM 混合 + 官方 TokUR）

**PANDA 路径**：纯解码三路（clean + R + SE，13/17 次）走 vLLM；权重扰动（W=4）+ INSIDE 走 HF。

**TokUR 路径（官方，独立）**：`third_party/TokUR/run/greedy_unc_single_batch_refine.py`（vLLM + TFB bayesian_transformer），**不用** PANDA 的 `score_tokur_baseline` 近似。

**三阶段架构（均可续跑、可分批）**：
- **Phase A（vLLM，PANDA）**：`panda.core.run_vllm_phase` → `raw_runs/{id}.partial.json`
- **Phase B（HF，PANDA）**：`run_panda_experiment --resume` → 补 weight 分支 + metrics → `raw_runs/{id}.json`
- **Phase C（官方 TokUR venv）**：`run_tokur_official_maintable.sh` → export → greedy_unc → `tokur_baseline.jsonl`
- 编排：`scripts/run_maintable_vllm.sh`（`SKIP_VLLM`/`SKIP_HF`/`PANDA_SKIP_TOKUR` 控制各阶段）

**保真度说明（写进 limitation/附录）**：vLLM 仅返回 top-k logprobs，非全词表分布。因此：
- **精确**：选中 token logprob、top-k pairs、top-2 margin、rank → **LL / DeepConf / Self-Certainty 与 HF 一致**；答案类指标（F_resp / U_Ecc / U_Deg / SE）完全不受影响。
- **近似**：全词表预测熵用 top-k 重归一化熵代替 → **PE / SAR / T-branch ATU 为近似**。
- **不受影响**：weight 分支走 HF，故 **S_ans / S_tr（PANDA 核心）完全精确**。

**预期加速**：13/17≈76% 解码可批量化，全管线含 HF 前向约 **1.7–2.6×**（视 batch 与 vLLM 吞吐）。

| 方案（300 档） | 数学 A100-h | 逻辑 A100-h | 合计 |
|------|----:|----:|----:|
| 纯 HF | ≈1260 | ≈235 | ≈1495 |
| vLLM 混合（保守 1.7×） | ≈740 | ≈138 | **≈880** |
| vLLM 混合（中等 2.2×） | ≈575 | ≈107 | **≈680** |
| **P0 only**（Qwen 数学 PANDA+TokUR） | ≈245 | — | **≈245** |

> 安装：`pip install vllm`（按本机 CUDA/torch 版本匹配）。未安装时 HF 管线照常工作。

### 12.7 预算报价（报销用，封顶 ¥3,000）

卡时单价：**RTX 5090-32GB ¥2.88/h**、**A100-40GB ¥3.28/h**。
老师批复预算：**¥3,000**（按 5090 计 ≈ **1040 卡时**可用）。

**保守口径**（往高报，可辩护）：
- vLLM 取**最保守 1.7× 加速**；
- 失败重跑 / SE backfill / reasoning 长尾余量 ×1.2。

| 方案（300 档） | 基准 5090-h | 含余量 5090-h | **5090 费用** | 是否在 ¥3000 内 |
|------|----:|----:|----:|:---:|
| **P0 only**（Qwen 数学 PANDA+TokUR，3 seed） | ≈280 | ≈336 | **≈¥970** | ✅ |
| P0+P1（+逻辑 PANDA+TokUR） | ≈410 | ≈492 | **≈¥1,420** | ✅ |
| **全计划**（4 模型 3 seed + 逻辑，vLLM） | ≈880 | ≈1056 | **≈¥2,940** | ✅ 刚好 |
| 纯 HF 全计划（无 vLLM） | ≈1650 | ≈1980 | ≈¥5,700 | ❌ |

**建议报销额**：按老师封顶 **报 ¥3,000**（说明：300 题/集 × vLLM 混合 × 优先 PANDA+TokUR，含 20% 重跑余量）。

**分阶段花钱**（预算紧时）：
1. 先花 **~¥1,000** 跑 P0 → 主表 PANDA vs TokUR 可填
2. 再花 **~¥400** 跑 P1 → 逻辑跨域可填
3. 余 **~¥1,600** 跑 P2 三模型 + 重跑缓冲

> CPU 消融（E4/E5/E6）不占 GPU、不计费。E7 扰动强度从余量出，不够则延后。

---

## 13. 实际执行环境与加速方案（4× RTX 5090，AutoDL）

> 本节记录**本轮实际跑实验的机器配置、加速做法与续跑命令**，下次开机直接照搬即可。

### 13.1 机器与引擎约束

| 项 | 取值 |
|----|------|
| GPU | **4× RTX 5090-32GB**（Blackwell，**sm_120**） |
| 引擎 | **纯 HF 分片管线**（`run_panda_experiment --mode generate`，自动算 PANDA + 11 baselines） |
| 为何不用 vLLM（PANDA Phase A） | PANDA 主表走 **纯 HF**；5090（sm_120）上 PANDA 不需 vLLM |
| TokUR Phase C | 需 **fork vLLM 0.7.3** 独立 venv；5090 可源码编译 vLLM，但 **transformers 版本** 仍阻塞 smoke（见 **§14**） |
| Python | `/root/miniconda3/bin/python`，torch 2.8.0+cu128，attn=sdpa |
| 路径 | `PANDA_ROOT=/root/autodl-tmp/PANDA`、`PANDA_MODELS=/root/autodl-tmp/panda-models`、`PANDA_OUTPUTS=/root/autodl-tmp/panda-outputs` |

### 13.2 核心加速：每卡打包多解码流（multishard）

**问题**：每题 17 次解码中，9 次（1 clean + 4 text + 4 weight）是 **batch=1 顺序贪心解码**（SE 的 8 样本已用 `num_return_sequences` 批处理）。单流 batch=1 自回归解码是**延迟瓶颈**，单卡 GPU 利用率仅 **~54%**，算力一半闲置。

**做法**：**每张物理卡并行跑多个独立分片进程**（同一模型，不同 `--shard-id`），用第二条流填满空闲算力。分片按 `shard % NGPU` 轮转绑定到 `CUDA_VISIBLE_DEVICES`，各进程 `--device cuda:0` + `--resume`，**数值完全一致、无精度影响**。

| 模型规模 | 每卡流数 `SHARDS_PER_GPU` | 总分片(4卡) | 单进程显存 | 4卡显存占用 | 实测利用率 | 加速 |
|----------|:--:|:--:|:--:|:--:|:--:|:--:|
| **≤3B**（Qwen2.5-3B / Llama-3.2-1B） | **2** | 8 | ~7.5–8.5 GB | ~15–17 GB / 32 GB | 54% → **90%** | **≈1.6–1.75×** |
| **8B**（Llama-3.1-8B / Qwen3-8B） | **1** | 4 | **~16.5 GB**（5090 实测） | ~16.5 GB / 32 GB | ~99% | 5090 **双流 OOM**；A100 40G 可试双流 |

> 3 流/卡收益递减（利用率已 90% 近饱和）且长解码并发易 OOM，**2 流/卡是甜点**。8B 模型默认 1 流/卡，显存确认有余量后可试 `SHARDS_PER_GPU=2`。

### 13.3 执行脚本（已写入 `/root/autodl-tmp/`）

| 脚本 | 作用 |
|------|------|
| `run_model_multishard.sh` | **单模型** multishard 跑数（按 `MODEL_TAG` 自动选模型路径与每卡流数；env：`DATASETS` / `SEEDS` / `SHARDS_PER_GPU` / `NGPU`） |
| `run_all_phases.sh` | **主编排**：P0 → P1 → P2 → P3（全程 `--resume`） |
| `run_p2_continue.sh` | **P2 续跑**（1B 已完成时从 llama31_8b 起） |
| `watchdog_phases.sh` | 监控 master 进程，挂掉自动重启（默认 `MASTER_SCRIPT=run_p2_continue.sh`） |
| `monitor_progress.sh` | 每 `INTERVAL`(默认600s) 打印进度快照 |

### 13.4 续跑命令（断点恢复，重复执行自动跳过已完成）

```bash
cd /root/autodl-tmp

# P2 续跑（当前推荐：1B 已完成）
nohup bash run_p2_continue.sh > logs/p2_continue.log 2>&1 &
nohup bash watchdog_phases.sh >> logs/watchdog.log 2>&1 &
INTERVAL=600 nohup bash monitor_progress.sh >> logs/monitor.log 2>&1 &

# 进度速查（任意时刻）
for s in $(seq 0 7); do tail -1 $PANDA_OUTPUTS/maintable_qwen25_3b/seed41/logs/minerva_shard$s.log; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 安全停止（再次启动会 --resume 续上）
# ⚠️ 两个坑：
#  (1) 必须先杀中间层 run_model_multishard.sh，否则它孤儿化后继续循环、每轮 wait 后
#      重启分片，导致多套分片叠加抢 GPU。
#  (2) pkill -f 'run_model_multishard.sh' 会匹配并杀掉执行 pkill 的 shell 自己（命令行
#      含同串），命令半路自杀 → 编排器反而没死。用 [r] 括号技巧防自匹配，或按 PID 杀。
# 顺序：先编排器（确认死透）后 python。
pkill -9 -f '[r]un_all_phases.sh'; pkill -9 -f '[r]un_model_multishard.sh'; sleep 3
ps -eo pid,cmd | grep -E '[r]un_model_multishard|[r]un_all_phases'   # 必须为空
for p in $(pgrep -f '[p]rs.ase.run_panda_experiment'); do kill -9 $p; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 确认 4 卡归零
```

### 13.5b 两项末尾空转优化（已落地）

| 优化 | 问题 | 做法 | 效果 |
|------|------|------|------|
| **动态取题队列** (`--dynamic-claim`) | 固定分片 `records[shard::N]` 下题目难度不均，快分片做完空等最慢分片 | 所有分片扫全表，用 `os.open(O_CREAT\|O_EXCL)` 原子认领下一道未完成题（claims/{id}.claim，stale 1800s 可回收防崩溃残留） | 末尾空转从"整个分片余量"降到"≤8 道在飞题" |
| **后台指标计算** | `recompute_metrics` 单线程 CPU ~13 min，期间 GPU 全闲 | 每数据集 metrics 改为 `( ... ) &` 后台跑，GPU 立刻进入下一数据集；脚本末尾 `wait` 保证 P3 前算完 | 数据集间切换 GPU 空闲从 ~13 min 降到 ~20 s |

### 13.5 实测吞吐与时间重估（2025-06-12，4×5090，2流/卡）

**实测稳态**（Qwen2.5-3B，8 分片，189s 窗口去除 resume 干扰）：

| 数据集档 | max_new_tokens | 聚合吞吐(8分片) | 单分片/题 |
|----------|:--:|:--:|:--:|
| minerva（实测） | 2048 | **5.08 题/分** | ~94 s |
| math500（外推，同 2048） | 2048 | ~5.0 题/分 | ~96 s |
| gsm8k（外推，1024 且短答案） | 1024 | ~8 题/分 | ~60 s |

> 比 §12 保守假设（~1.3 题/分）**快约 4×**：8 个 SE 样本是**批处理**（非 17 次纯顺序，等效约 10 个解码 pass）；5090 实测 token/s 高于假设；多数题提前 EOS。

**各阶段墙钟重估**（4 卡并行；3B/1B 用 2流/卡=8分片，8B 用 1流/卡=4分片）：

| 阶段 | 内容 | 题量 | 估吞吐 | **墙钟** |
|------|------|----:|:--:|:--:|
| **P0** | Qwen2.5-3B 数学×3seed | 2616 | 5–8 题/分 | **≈7–8 h** |
| **P1** | Qwen2.5-3B 逻辑×3seed | 2100 | ~5 题/分 | **≈7 h** |
| **P2a** | Llama-3.2-1B 数学×3seed | 2616 | ~7–8 题/分 | **≈5–6 h** |
| **P2b** | Llama-3.1-8B 数学×3seed（1流/卡） | 2616 | ✅ 已完成（2026-06-13） | **≈11 h 墙钟** |
| **P2c** | Qwen3-8B 数学×3seed（1流/卡，可能 reasoning 长） | 2616 | **~1–1.5 题/分*** | **≈30–45 h*** |
| P3 | CPU 聚合 4 模型 | — | — | 分钟级 |
| | | | **合计** | **≈ 80–105 h（约 3.5–4.5 天）** |

> *8B 实测：Llama-3.1-8B P2b 约 **11 h** 墙钟（4 卡 1 流/卡，SE=0）；Qwen3-8B 进行中。2流/卡在 5090 32GB 上 OOM。
>
> **GPU-h ≈ 360–420**（4卡全程），5090 @ ¥2.88/h ≈ **¥1,050–1,200**，远在 ¥3000 预算内（原 §12.7 全计划估 ≈¥2,940，因 3B 实测提速而下修）。


### 13.6 已知数据缺口 / 待办

- **`leg_counting` 全集即 100 题**（数据集本身大小，非缺口）；`zebra_puzzles` / `color_cube` 各 300。P1 逻辑该集产出 100 题为正常。
- **TokUR EU 主表列**：5090 上 vLLM fork **已编译成功**，但 smoke 因 `transformers` 与 `bayesian_transformer` 不兼容未过；详见 **§14**。推荐在 **4090/A100** 新环境用预编译 wheel + `transformers==4.53.3` 补齐；PANDA `raw_runs` 已齐的模型可离线补跑。
- 续跑墙钟估计：P0(Qwen数学) ≈ 40 h（2流/卡加速后），全四模型按上表外推。

---

## 14. TokUR 官方环境：故障原因、新机器部署与 5090 补救

> **Phase C 专用**：TokUR 与 PANDA 主表 **必须分 venv**；读已有 `raw_runs`，写 `tokur_baseline.jsonl`（`scoring_mode=official_vllm`）。  
> **禁止**主表用 `score_tokur_baseline.py`（post-hoc HF 近似）。

### 14.1 为什么本机 TokUR 跑不了（分层）

| 层次 | 问题 | 本机状态 |
|------|------|----------|
| **硬件** | RTX **5090 = sm_120**；TokUR 依赖的 fork **vLLM 0.7.3 无 sm_120 预编译 wheel**，只能源码编译（cutlass / flash-attn / flashmla 等） | **已解决**：`build_tokur_vllm_5090.sh` 编译成功，`Successfully built vllm` |
| **依赖** | `bayesian_transformer/vllm_models/tfb_llama.py` 需 `from transformers.utils import LossKwargs`；**transformers≥4.54 已删除 `LossKwargs`**（4.46–4.53 存在） | **未解决（当前 blocker）**：venv 里为 **4.57.6** → smoke 报 `ImportError: LossKwargs` |
| **架构** | PANDA（HF）与 TokUR（vLLM）版本栈相反：PANDA 用 torch **2.8** + transformers **4.57**；TokUR 官方 era 为 torch **~2.4** + vLLM **0.7.3** + transformers **≤4.53** | 两套环境 **不可共用** conda / 同一 venv |

**smoke 失败日志特征**（`/root/autodl-tmp/logs/build_tokur_vllm_5090.log`）：

```text
ImportError: cannot import name 'LossKwargs' from 'transformers.utils'
```

**标记文件**（均未生成即 TokUR 队列会一直等）：

| 文件 | 含义 |
|------|------|
| `logs/TOKUR_VLLM_READY` | vLLM + bayesian_transformer import + CUDA tensor OK |
| `logs/TOKUR_SMOKE_OK` | 1 题 minerva 端到端官方 TokUR 通过 |
| `logs/P2_TOKUR_DONE` | P2 两 8B 模型 TokUR 队列跑完 |

相关脚本：`/root/autodl-tmp/build_tokur_vllm_5090.sh`、`smoke_tokur_vllm.sh`、`run_p2_tokur_queue.sh`。

### 14.2 PANDA 与 TokUR 环境对照

| | PANDA 主表（Phase B，HF） | TokUR（Phase C，官方 vLLM） |
|--|------------------------|----------------------------|
| 入口 | `panda.core.run_panda_experiment` | `run_tokur_official_maintable.sh` → `greedy_unc_single_batch_refine.py` |
| Python | `/root/miniconda3`（system） | 独立 venv：`.tokur_venv` 或 `.tokur_venv_5090` |
| torch | **2.8.0+cu128**（5090 友好） | **~2.4.0+cu12x**（与 vLLM 0.7.3 wheel 匹配） |
| transformers | **4.57.x**（PANDA 用） | **4.53.3**（保留 `LossKwargs`） |
| vLLM | 不需要 | **haizhou-shi/vllm** @ `61c6a5a79664882a8ab1c9af3ff78677911516dc` |
| GPU 用法 | 4 卡 HF 分片 | vLLM greedy；`enforce_eager=True`（禁用 CUDAGraph，否则 Bayesian 扰动失效） |

### 14.3 推荐硬件（新申请卡）

| 优先级 | GPU | 说明 |
|:--:|-----|------|
| **最推荐** | **RTX 4090 24GB** / **A100 40GB** / **H800** | 可用官方 **预编译 vLLM wheel**，无需源码编译，约 30 min 搭好 |
| 可用但麻烦 | **RTX 5090**（sm_120） | 必须跑 `build_tokur_vllm_5090.sh` 源码编译 + pin transformers |
| 不推荐 | 过老架构（sm_70 等） | vLLM 0.7.3 / flash-attn 支持有限 |

**显存粗估**（单模型 greedy + bayesian 扰动）：1B ~6–8 GB；3B ~12–16 GB；8B ~24–32 GB。

**申请模板（可直接粘贴）**：

```text
用途：TokUR 官方 vLLM 推理（ICLR 2026 baseline，Phase C）
GPU：1–4× RTX 4090 24GB 或 1× A100 40GB（避免 5090 sm_120 源码编译）
系统：Ubuntu 22.04，CUDA 12.4，Python 3.11
软件：独立 venv — vLLM 0.7.3 fork + transformers==4.53.3 + torch 2.4
数据：~50GB（TFB 权重 panda-models/ + 已完成 raw_runs panda-outputs/）
说明：不与 PANDA HF 生成共用环境；只读 raw_runs 写 tokur_baseline.jsonl
```

### 14.4 新环境部署（4090 / A100，约 30 分钟）

**Step 0 — 同步数据**（TokUR 只需 Phase B 产物，不必同步 PANDA conda / 5090 编译产物）：

```bash
# 必需
PANDA/                          # 代码 + third_party/TokUR
panda-models/TFB-*              # TFB 权重
panda-outputs/maintable_*/      # 含 */*/raw_runs/

# 不必同步
# miniconda PANDA 环境、.tokur_venv_5090、vllm-tokur/build/
```

**Step 1 — 独立 venv**（或一键：`bash scripts/setup_tokur_vllm.sh`）

```bash
cd /root/autodl-tmp/PANDA
python3.11 -m venv .tokur_venv
source .tokur_venv/bin/activate
pip install -U pip setuptools wheel
```

**Step 2 — fork vLLM（官方预编译 wheel 路径）**

```bash
git clone https://github.com/haizhou-shi/vllm.git /tmp/vllm-tokur
cd /tmp/vllm-tokur
export VLLM_COMMIT=61c6a5a79664882a8ab1c9af3ff78677911516dc
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install -e .
```

**Step 3 — pin transformers + bayesian_transformer**

```bash
pip install "transformers==4.53.3" "numpy>=1.19.2,<2"
pip install -e /root/autodl-tmp/PANDA/third_party/TokUR/bayesian_transformer
# requirements.txt 中 vllm==0.7.3 与 fork 冲突时以 fork 为准
pip install -r /root/autodl-tmp/PANDA/third_party/TokUR/requirements.txt 2>/dev/null || true
```

**Step 4 — 验证**

```bash
export TOKUR_VENV=/root/autodl-tmp/PANDA/.tokur_venv
export VLLM_USE_V1=0
python -c "import vllm, bayesian_transformer, torch; print('OK', vllm.__version__, torch.__version__)"
bash scripts/smoke_tokur_vllm.sh   # 成功 → logs/TOKUR_SMOKE_OK
```

**Step 5 — 跑主表 TokUR**（PANDA `raw_runs` 已存在即可）

```bash
# 单任务
OUT_DIR=/root/autodl-tmp/panda-outputs/maintable_qwen25_3b/seed41 \
PANDA_MODEL_TAG=qwen25_3b DATASET=math500 TOKUR_SEED=41 \
TOKUR_VENV=/root/autodl-tmp/PANDA/.tokur_venv \
bash /root/autodl-tmp/PANDA/scripts/run_tokur_official_maintable.sh

# 批量（8B 完成后）
TOKUR_VENV=/root/autodl-tmp/PANDA/.tokur_venv \
bash /root/autodl-tmp/run_p2_tokur_queue.sh
```

**可优先补 TokUR 的数据**（PANDA Phase B 已齐）：Qwen2.5-3B（2616/2616）、Llama-3.2-1B（2616/2616）；Llama-3.1-8B / Qwen3-8B 随 P2 推进逐步可补。

### 14.5 若新机器仍是 5090

1. 使用已有脚本：`bash /root/autodl-tmp/build_tokur_vllm_5090.sh`（含 cutlass / flashmla / flash-attn 本地镜像、`flock` 防并行编译）。
2. **必须** pin transformers：

```bash
/root/autodl-tmp/PANDA/.tokur_venv_5090/bin/pip install "transformers==4.53.3"
bash /root/autodl-tmp/smoke_tokur_vllm.sh
```

3. 若 runtime 仍报错，考虑 venv 内降级 torch 至 **2.4.x** 并重编 vLLM（torch 2.8 + vLLM 0.7.3 fork 非官方组合）。

5090 编译依赖环境变量（脚本已内置）：

```bash
VLLM_CUTLASS_SRC_DIR=PANDA/third_party/cutlass-v3.8.0
FLASH_MLA_SRC_DIR=PANDA/third_party/flashmla-575f772
VLLM_FLASH_ATTN_SRC_DIR=PANDA/third_party/vllm-flash-attn-9bfa986
TORCH_CUDA_ARCH_LIST=12.0
```

### 14.6 本机最小补救（不换卡，可选）

vLLM 已编好，仅差 transformers：

```bash
/root/autodl-tmp/PANDA/.tokur_venv_5090/bin/pip install "transformers==4.53.3"
/root/autodl-tmp/PANDA/.tokur_venv_5090/bin/python -c "import vllm, bayesian_transformer; print('OK')"
bash /root/autodl-tmp/smoke_tokur_vllm.sh
```

通过后 `run_p2_tokur_queue.sh` 会自动推进；否则可 **`pkill -f run_p2_tokur_scheduler`** 避免空等 vLLM。

### 14.7 与 PANDA 实验分工（推荐）

| 机器 | 任务 |
|------|------|
| **现有 4×5090** | 继续 P2 PANDA 8B 主表（Llama-3.1-8B → Qwen3-8B） |
| **新 4090/A100** | 仅 TokUR Phase C；`rsync panda-outputs/` + `panda-models/TFB-*` 即可 |
| **数据衔接** | TokUR 读 `maintable_*/seed*/{dataset}/raw_runs/`，写同目录 `tokur_baseline.jsonl` |

### 14.8 5090 源码编译曾遇到的问题（备忘）

| 现象 | 处理 |
|------|------|
| GitHub clone cutlass / flashmla 超时 | 预克隆至 `third_party/cutlass-v3.8.0` 等，设 `VLLM_CUTLASS_SRC_DIR` / `FLASH_MLA_SRC_DIR` |
| flash-attn 缺 `cutlass/numeric_types.h` | `ln -sfn cutlass-v3.8.0/include vllm-flash-attn-9bfa986/csrc/cutlass/include` |
| `Error: not a CMake build directory` | 清理 `vllm-tokur/` 下 in-source 残留：`CMakeCache.txt`、`CMakeFiles/` |
| 多 watchdog 并行编译冲突 | `build_tokur_vllm_5090.sh` 用 `flock`；编译期间勿并行启动多个 build |
