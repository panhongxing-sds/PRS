# PRS 实验方案与表格预案

> 本文档预先列出 PRS 论文需要完成的全部实验、对应数据来源、运行命令与**待填空表格**。表内数字格式统一为 `AUROC`（或百分比 `71.77 ± 0.12`，3 seeds mean±std）。所有"对错"标签使用 **clean label**（`label_wrong_clean`）。

---

## 0. 方法-代码口径备注（写表前必读）

当前主表/消融按 **代码实际实现** 出数，与 `\section{Method}` 存在待对齐处：

| 组件 | 口径（已定） | 代码字段 |
|------|----------|----------|
| 扰动集成 $K$ | **$K=R+W=8$**（R=4 text + W=4 weight；两轴独立，非笛卡尔积） | text_runs(R) + weight_runs(W) |

> **公平采样预算（关键）**：所有采样/扰动类方法统一 **K=8** 个样本，确保对比公平：
> - PRS / U_Ecc / U_Deg：共享同一组 R+W=8 扰动样本（4 text 改写 + 4 weight 扰动，均为完整解码）；
> - SE：8 个独立高温样本（`--se-samples 8`，温度 **0.3** / top-p 0.95，完整解码；`SE_TEMP=0.3` 或 `--se-temperature 0.3`）。**生成阶段**仅写 `high_temp_sample_runs`，`baseline_SE_*` 用 `math_equal` 占位（不占 GPU/CPU）；**主表官方 SE** 在 dataset 结束后 `recompute_metrics --from-cache --recompute-se` 用 **NLI**（`microsoft/deberta-v2-xlarge-mnli` 双向蕴含，对 `full_response` 聚类）。首次需下载 NLI 模型：`HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/deberta-v2-xlarge-mnli', local_dir='/root/autodl-tmp/prs-models/deberta-v2-xlarge-mnli')"`（另需 `pip install sentencepiece`）。NLI 缺模型时回退 `math_equal_nli_fallback`。
> - **TokUR EU：官方实现**（`third_party/TokUR` + `run_tokur_official_maintable.sh`）——vLLM greedy 解码时由 TFB `bayesian_transformer` 原生累计 EU；`num_samples` / `bayes_sigma` / `basis_idx` **取自 TFB checkpoint**（如 Qwen2.5-3B 默认 num_samples=5），**不与 PRS 的 K=8 强行对齐**；
> - 单次方法（PE/LL/Self-Certainty/DeepConf/INSIDE/P(True)）：仅 1 次 greedy/前向，本就不采样，不参与采样数对齐。
>
> 每题 **PRS 管线解码** = 1(clean) + 4(R) + 4(W) + 8(SE) = **17 次**。TokUR 为**独立官方管线**（greedy 生成 + 生成时 EU），不走 PRS 的 post-hoc `score_tokur_baseline` 近似路径。

| $S_\text{out}$ | **$U=1-\max_c\hat p_\text{cls}(c)$**（按代码，非归一化熵） | `F_resp`=`TW_ASE` |
| $S_\text{ans}$ | weight 分支平均 | `AltMass_final` |
| $S_\text{tr}$ | **B：domain-agnostic top-10% 局部 spread**（跨数学/逻辑统一） | `AltMass_local_spread_topk` |

> $S_\text{tr}$ 定义已统一为 **B（top-k% 局部 spread，无需 token 分类器）**，避免 filler 词稀释，且跨域通用。
> A（按域内容 token，`AltMass_local_spread_content`）与 legacy 数学版（`AltMass_local_spread_reason`）作为消融对比保留（见表 4b）。
> 待 $U$ vs $H_\text{norm}$ 实证消融：`compute_ase` 已同时产出二者，数据挂载后跑一次确认。

---

## 1. 实验总览

| 实验 | 目的 | 表 | 计算 | GPU? |
|------|------|----|------|------|
| E1 主表（数学） | PRS vs baseline + TokUR EU；**4 模型** | 表 1a–1d | 3-seed mean±std | Qwen2.5-3B seed1 已有，仅补 seed2/3 + 其余模型 |
| E2 主表（逻辑） | 跨域泛化；**仅主模型 Qwen2.5-3B** | 表 2 | 同上 | 需生成 |
| E3 跨模型 | 4 模型稳健性（数学集）；逻辑仅主模型 | 表 3 | 同上 | 需生成 |
| E4 组件消融 | PRS 去一个组件 | 表 4 | LR 重拟合，CPU | 否（有 raw） |
| E5 扰动预算 | R / W 数量影响 | 表 5 | 子集重算，CPU | 否 |
| E6 λ 敏感性 | $\lambda_a,\lambda_t$ 稳健性 | 表 6 | 后处理，CPU | 否 |
| E7 扰动强度 | $\sigma$ / rank | 表 7 | 需重跑 | 是 |

**模型**（`configs/ase_models.yaml`）：Qwen2.5-3B(主) / Llama-3.2-1B / Llama-3.1-8B / Qwen3-8B
**数学数据集**：minerva(272 全量), math500(**300**), gsm8k(**300**) ——**定稿档 MAX_SAMPLES=300**
**逻辑数据集**：leg_counting, zebra_puzzles, color_cube（各 **300**）
**Seeds**：41, 42, 43（`DEFAULT_EXPERIMENT_SEEDS`）
**GPU 预算**：**¥3,000**（5090-32GB ¥2.88/h）；引擎 **vLLM 混合**

### 1.1 执行优先级（先 TokUR + PRS，再扩模型）

> 主表核心对比是 **PRS vs TokUR EU**；其余 10 个 baseline 随 PRS 管线 Phase B 自动产出，不单独排队。

| 优先级 | 内容 | 目的 | 估 5090 卡时 |
|--------|------|------|------------:|
| **P0** | Qwen2.5-3B × 数学 3 集 × 3 seeds → **PRS 全管线 + TokUR EU** | 主表 1a 可填 | ~280 h |
| **P1** | Qwen2.5-3B × 逻辑 3 集 × 3 seeds → PRS + TokUR | 主表 2a 可填 | ~130 h |
| **P2** | 其余 3 模型 × 数学 3 集（Llama-3.2-1B / Llama-3.1-8B / Qwen3-8B） | 跨模型表 3 | ~350 h |
| **P3** | 聚合 3-seed 主表 + CPU 消融（E4/E5/E6） | 论文表格 | CPU |

**当前进度（2026-06-13 22:00）**：
- P0 ✅ / P1 ✅（Qwen2.5-3B 数学+逻辑）
- P2：**Llama-3.2-1B ✅**（2616/2616）；**Llama-3.1-8B ✅**（2616/2616，表 1c 已填）；**Qwen3-8B 进行中**（seed41 minerva）
- 后台：`run_p2_continue.sh` + `watchdog_phases.sh` + `monitor_progress.sh`（日志 `logs/p2_continue.log` / `logs/watchdog.log` / `logs/monitor.log`）
- 5090-32GB 上 8B **双流 OOM 已确认**，固定单流 SPG=1、**SE=0**
| **延后** | E7 扰动强度、失败分析 | 附录/可选 | 另计 |

**P0 最小闭环**（预算紧时只跑这个也能写主结论）：
1. **Phase A（vLLM）**：clean + R + SE → `*.partial.json`
2. **Phase B（HF）**：weight 分支 → **PRS 三组分 + 11 baselines** → 完整 `raw_runs/*.json`
3. **Phase C（官方 TokUR venv）**：`run_tokur_official_maintable.sh` → export jsonl → `greedy_unc_single_batch_refine.py` → `tokur_baseline.jsonl`（`scoring_mode=official_vllm`）。**环境搭建与 5090 故障排查见 §14**。

> **禁止**主表使用 `score_tokur_baseline.py`（post-hoc HF 近似，与官方 TFB 扰动/生成路径不一致）。

已有 **Qwen2.5-3B seed1**（500 档旧 raw）：CPU 子采样对齐 300 题 + GPU 仅补 SE backfill。

---

## 2. 主表（E1：数学四数据集）

每个数据集三列 **AUROC | AUPRC | ACC***，行含 TokUR EU + 全部 baseline + PRS 及其三组分。acc是前50%的正确率

### 表 1a：Qwen2.5-3B

> **数据状态（2026-06-12）**：P0 三数据集 × 3 seeds 已生成完毕（minerva 272/seed，math500/gsm8k 各 300/seed）。
> **判分方式（strict label, 子集）**：弃用旧的科学计数法判分（有 bug：`1.01×10⁶` 与 `2.19e6` 因指数相同被误判为正确）。改用 `prs.grading.strict_grader.strict_grade` 三态判分 `correct/wrong/drop`，**已接入管线核心 `grade_answer`**（数学数据集生效；逻辑/代码题保持原 string/code 判分，永不 drop）。能可靠判定的计入，无法可靠判定的（残缺/带单位/含自由变量的符号表达式等）**直接剔除（`label_drop=1`）**，指标只在干净子集上计算。科学计数法完整解析尾数+指数后数值比较（相对容差 1e-3）。另叠加人工核verified 的 `MANUAL_RELABEL_IDS` 白名单（格式等价的少数 Minerva 题强制判对、不剔除）。
> **存储**：`summary.jsonl`/`features.jsonl` 已写入 strict 的 `label_wrong_clean` + `label_drop`；P3 聚合器 `analyze_maintable` 与 `aggregate_math_strict.py` 均排除 `label_drop=1`。
> **未跑（留 `—`）**：SE、$U_{Ecc}$、$U_{Deg}$（当前非标准实现，标准版需用 NLI 对完整生成过程做语义聚类，待重算）；CoT（不计入）；P(True) / INSIDE（GPU 基线，需 `score_gpu_baselines.py` 额外前向/隐藏态，未跑）；**TokUR EU**（Phase C 环境未就绪，见 **§14**）。
> **复现**：`python PRS/scripts/aggregate_math_strict.py`

#### 样本剔除统计（label_drop=1 的题目被排除）

| seed | MATH-500 保留/总数 (剔除%) | GSM8K 保留/总数 (剔除%) | Minerva 保留/总数 (剔除%) |
|--|--:|--:|--:|
| seed41 | 239/300 (20.3%) | 299/300 (0.3%) | 198/272 (27.2%) |
| seed42 | 241/300 (19.7%) | 300/300 (0.0%) | 195/272 (28.3%) |
| seed43 | 240/300 (20.0%) | 299/300 (0.3%) | 196/272 (27.9%) |

#### 3-seed 平均 (41/42/43, mean ± std, strict label, 子集)

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SE | — | — | — | — | — | — | — | — | — |
| SAR | 60.10 ± 3.32 | 41.40 ± 2.54 | 78.27 ± 1.86 | 46.14 ± 1.33 | 8.94 ± 1.74 | 91.29 ± 1.12 | 61.70 ± 1.89 | 71.57 ± 1.72 | 46.25 ± 0.66 |
| $U_{Ecc}$ | — | — | — | — | — | — | — | — | — |
| $U_{Deg}$ | — | — | — | — | — | — | — | — | — |
| P(True) | — | — | — | — | — | — | — | — | — |
| INSIDE | — | — | — | — | — | — | — | — | — |
| PE | 59.24 ± 3.04 | 40.15 ± 1.73 | 76.31 ± 2.83 | 47.72 ± 1.31 | 11.30 ± 0.13 | 92.18 ± 0.86 | 66.29 ± 1.57 | 74.80 ± 1.86 | 48.63 ± 1.60 |
| LL | 58.68 ± 2.96 | 38.86 ± 1.67 | 76.87 ± 2.47 | 47.31 ± 1.47 | 6.97 ± 0.19 | 92.41 ± 1.16 | 63.77 ± 1.65 | 71.31 ± 1.95 | 47.95 ± 1.93 |
| Self-Certainty | 59.21 ± 2.99 | 40.13 ± 1.71 | 76.87 ± 2.47 | 47.71 ± 1.32 | 10.54 ± 1.18 | 92.18 ± 0.86 | 66.23 ± 1.56 | 74.82 ± 1.88 | 48.63 ± 1.60 |
| DeepConf | 65.11 ± 3.84 | 45.43 ± 3.44 | 81.05 ± 2.16 | 57.42 ± 3.67 | 10.04 ± 0.99 | 93.30 ± 0.96 | 69.38 ± 0.96 | 75.71 ± 0.72 | 52.03 ± 1.79 |
| TokUR EU | — | — | — | — | — | — | — | — | — |
| **PRS (Ours)** | **88.07 ± 0.81** | **59.95 ± 1.33** | **98.05 ± 0.39** | 77.77 ± 1.38 | **51.40 ± 3.37** | 96.65 ± 0.01 | **83.58 ± 1.41** | **90.51 ± 0.72** | **63.59 ± 1.57** |
| F_resp | 87.32 ± 0.31 | 60.50 ± 1.52 | 98.05 ± 0.39 | 79.28 ± 1.47 | 41.69 ± 3.21 | 96.20 ± 0.33 | 82.26 ± 1.23 | 87.64 ± 0.57 | 63.59 ± 1.57 |
| D_ans | 77.20 ± 4.28 | 48.67 ± 5.41 | 90.24 ± 2.63 | 66.08 ± 2.31 | 26.30 ± 2.42 | 97.54 ± 0.32 | 76.92 ± 2.12 | 82.24 ± 0.99 | 61.54 ± 2.70 |
| D_reason | 70.85 ± 0.46 | 44.96 ± 0.92 | 86.07 ± 0.42 | 66.30 ± 0.38 | 24.54 ± 0.39 | 95.76 ± 0.32 | 70.90 ± 0.57 | 79.79 ± 0.84 | 51.01 ± 1.09 |

#### seed41（strict label, 子集）

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 55.95 | 38.28 | 76.47 | 46.33 | 11.19 | 91.28 | 60.76 | 70.29 | 46.46 |
| PE | 55.27 | 37.92 | 73.11 | 47.67 | 11.29 | 91.95 | 65.04 | 73.18 | 49.49 |
| LL | 54.84 | 36.95 | 73.95 | 47.29 | 7.23 | 91.95 | 63.38 | 70.83 | 48.48 |
| Self-Certainty | 55.31 | 37.92 | 73.95 | 47.64 | 11.29 | 91.95 | 65.02 | 73.16 | 49.49 |
| DeepConf | 59.69 | 40.57 | 78.15 | 58.08 | 9.77 | 93.96 | 70.23 | 75.45 | 54.55 |
| **PRS (Ours)** | 87.36 | 58.40 | 98.32 | 78.19 | 50.32 | 96.64 | 85.57 | 91.32 | 65.66 |
| F_resp | 87.31 | 58.56 | 98.32 | 79.57 | 41.98 | 95.97 | 83.98 | 87.99 | 65.66 |
| D_ans | 71.72 | 42.53 | 86.55 | 68.26 | 27.64 | 97.32 | 79.06 | 83.64 | 63.64 |
| D_reason | 70.30 | 44.76 | 85.71 | 65.79 | 24.28 | 95.30 | 70.25 | 78.80 | 52.53 |

#### seed42（strict label, 子集）

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 60.27 | 41.42 | 77.50 | 47.67 | 8.68 | 92.67 | 60.00 | 70.42 | 45.36 |
| PE | 59.79 | 40.41 | 75.83 | 49.35 | 11.46 | 93.33 | 65.33 | 73.82 | 46.39 |
| LL | 59.17 | 38.61 | 76.67 | 49.11 | 6.81 | 94.00 | 61.97 | 69.20 | 45.36 |
| Self-Certainty | 59.73 | 40.39 | 76.67 | 49.35 | 11.46 | 93.33 | 65.25 | 73.85 | 46.39 |
| DeepConf | 68.09 | 48.12 | 81.67 | 61.55 | 11.37 | 94.00 | 68.04 | 74.99 | 50.52 |
| **PRS (Ours)** | 89.20 | 61.64 | 98.33 | 75.91 | 47.93 | 96.67 | 82.72 | 90.65 | 61.86 |
| F_resp | 87.71 | 60.67 | 98.33 | 77.36 | 37.62 | 96.67 | 81.66 | 88.10 | 61.86 |
| D_ans | 82.17 | 55.70 | 92.50 | 67.11 | 28.36 | 98.00 | 74.03 | 81.41 | 57.73 |
| D_reason | 70.82 | 46.19 | 86.67 | 66.72 | 25.10 | 96.00 | 70.83 | 79.72 | 50.52 |

#### seed43（strict label, 子集）

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 64.08 | 44.50 | 80.83 | 44.42 | 6.96 | 89.93 | 64.34 | 74.01 | 46.94 |
| PE | 62.66 | 42.14 | 80.00 | 46.14 | 11.14 | 91.28 | 68.50 | 77.40 | 50.00 |
| LL | 62.04 | 41.01 | 80.00 | 45.52 | 6.87 | 91.28 | 65.96 | 73.90 | 50.00 |
| Self-Certainty | 62.58 | 42.09 | 80.00 | 46.13 | 8.87 | 91.28 | 68.43 | 77.45 | 50.00 |
| DeepConf | 67.54 | 47.59 | 83.33 | 52.64 | 8.98 | 91.95 | 69.88 | 76.69 | 51.02 |
| **PRS (Ours)** | 87.64 | 59.82 | 97.50 | 79.21 | 55.96 | 96.64 | 82.45 | 89.56 | 63.27 |
| F_resp | 86.95 | 62.28 | 97.50 | 80.91 | 45.47 | 95.97 | 81.14 | 86.83 | 63.27 |
| D_ans | 77.71 | 47.77 | 91.67 | 62.88 | 22.91 | 97.32 | 77.69 | 81.68 | 63.27 |
| D_reason | 71.43 | 43.95 | 85.83 | 66.38 | 24.25 | 95.97 | 71.64 | 80.85 | 50.00 |


### 表 1b：Llama-3.2-1B

> **数据状态（2026-06-13）**：P2 数学三数据集 × 3 seeds **已全部生成完毕**（minerva 272/seed，math500/gsm8k 各 300/seed）。判分/聚合口径同表 1a（strict label 子集；SE/U_Ecc/U_Deg/CoT/P(True)/INSIDE/TokUR 留空）。
> **复现**：`python PRS/scripts/aggregate_math_strict.py --base prs-outputs/maintable_llama32_1b`

#### 样本剔除统计（label_drop=1 的题目被排除）

| seed | MATH-500 保留/总数 (剔除%) | GSM8K 保留/总数 (剔除%) | Minerva 保留/总数 (剔除%) |
|--|--:|--:|--:|
| seed41 | 206/300 (31.3%) | 299/300 (0.3%) | 179/272 (34.2%) |
| seed42 | 207/300 (31.0%) | 298/300 (0.7%) | 178/272 (34.6%) |
| seed43 | 209/300 (30.3%) | 298/300 (0.7%) | 180/272 (33.8%) |

#### 3-seed 平均 (41/42/43, mean ± std, strict label, 子集)

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SE | — | — | — | — | — | — | — | — | — |
| SAR | 54.94 ± 1.20 | 71.00 ± 1.16 | 33.55 ± 1.35 | 55.07 ± 0.41 | 49.46 ± 1.32 | 59.73 ± 1.10 | 63.39 ± 2.00 | 92.54 ± 0.65 | 15.30 ± 0.49 |
| $U_{Ecc}$ | — | — | — | — | — | — | — | — | — |
| $U_{Deg}$ | — | — | — | — | — | — | — | — | — |
| P(True) | — | — | — | — | — | — | — | — | — |
| INSIDE | — | — | — | — | — | — | — | — | — |
| PE | 54.23 ± 1.07 | 70.48 ± 1.06 | 33.23 ± 1.32 | 54.89 ± 0.41 | 48.89 ± 1.10 | 59.51 ± 0.84 | 63.74 ± 2.22 | 92.67 ± 0.61 | 16.04 ± 1.89 |
| LL | 53.38 ± 1.17 | 69.91 ± 1.24 | 32.59 ± 1.34 | 54.60 ± 0.41 | 48.38 ± 1.13 | 59.06 ± 1.64 | 62.96 ± 2.49 | 92.53 ± 0.77 | 15.30 ± 1.39 |
| Self-Certainty | 53.85 ± 1.15 | 69.96 ± 1.24 | 33.55 ± 1.35 | 54.61 ± 0.41 | 48.74 ± 1.12 | 59.28 ± 0.84 | 63.07 ± 2.20 | 92.56 ± 0.55 | 15.67 ± 1.84 |
| DeepConf | 61.76 ± 0.87 | 74.36 ± 0.77 | 43.23 ± 0.65 | 59.84 ± 0.54 | 55.30 ± 1.16 | 62.86 ± 1.92 | 72.12 ± 1.80 | 94.82 ± 0.14 | 16.78 ± 2.36 |
| TokUR EU | — | — | — | — | — | — | — | — | — |
| **PRS (Ours)** | **80.22 ± 0.72** | **82.53 ± 0.97** | **56.78 ± 1.07** | **88.72 ± 0.83** | **84.57 ± 2.33** | **88.14 ± 0.32** | **76.98 ± 2.59** | **95.93 ± 0.96** | **17.90 ± 1.74** |
| F_resp | 81.19 ± 0.61 | 84.66 ± 0.72 | 57.42 ± 1.29 | 87.49 ± 0.86 | 80.04 ± 2.06 | 86.13 ± 0.32 | 78.37 ± 1.27 | 95.44 ± 0.72 | 17.90 ± 1.74 |
| D_ans | 60.81 ± 0.65 | 66.84 ± 0.98 | 48.39 ± 0.82 | 72.05 ± 0.51 | 65.28 ± 1.81 | 73.15 ± 1.98 | 57.83 ± 7.65 | 89.34 ± 3.60 | 13.06 ± 0.50 |
| D_reason | 54.68 ± 0.49 | 69.06 ± 0.99 | 39.03 ± 0.28 | 69.63 ± 1.01 | 63.67 ± 0.95 | 72.93 ± 1.14 | 62.15 ± 1.27 | 91.67 ± 1.02 | 13.80 ± 1.85 |

#### seed41（strict label, 子集）

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 53.50 | 69.54 | 33.98 | 54.54 | 49.92 | 58.39 | 66.21 | 92.57 | 15.73 |
| PE | 53.32 | 69.14 | 33.01 | 54.34 | 49.12 | 58.39 | 66.85 | 92.84 | 17.98 |
| LL | 52.17 | 68.24 | 33.01 | 54.07 | 48.68 | 57.05 | 66.49 | 93.08 | 16.85 |
| Self-Certainty | 52.66 | 68.30 | 33.98 | 54.08 | 48.94 | 58.39 | 66.15 | 92.68 | 17.98 |
| DeepConf | 62.31 | 73.44 | 43.69 | 60.50 | 56.68 | 63.09 | 73.84 | 94.88 | 17.98 |
| **PRS (Ours)** | 80.86 | 82.81 | 58.25 | 88.35 | 84.25 | 87.92 | 74.41 | 95.19 | 17.98 |
| F_resp | 81.86 | 84.90 | 59.22 | 87.35 | 80.46 | 85.91 | 77.05 | 94.91 | 17.98 |
| D_ans | 60.34 | 65.71 | 49.51 | 72.58 | 66.48 | 71.14 | 52.68 | 87.06 | 12.36 |
| D_reason | 54.04 | 67.99 | 38.83 | 68.27 | 62.33 | 71.81 | 60.37 | 90.23 | 14.61 |

#### seed42（strict label, 子集）

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 56.45 | 72.37 | 34.95 | 55.15 | 50.79 | 59.73 | 62.11 | 93.31 | 14.61 |
| PE | 55.73 | 71.74 | 34.95 | 55.01 | 50.10 | 59.73 | 61.78 | 93.31 | 13.48 |
| LL | 54.97 | 71.20 | 33.98 | 54.66 | 49.58 | 59.06 | 61.09 | 93.07 | 13.48 |
| Self-Certainty | 55.40 | 71.28 | 34.95 | 54.66 | 49.99 | 59.06 | 61.16 | 93.17 | 13.48 |
| DeepConf | 62.44 | 75.32 | 43.69 | 59.17 | 55.37 | 60.40 | 69.64 | 94.96 | 13.48 |
| **PRS (Ours)** | 79.21 | 81.23 | 56.31 | 89.87 | 87.57 | 88.59 | 80.53 | 97.29 | 15.73 |
| F_resp | 80.39 | 83.69 | 56.31 | 88.61 | 82.34 | 86.58 | 80.09 | 96.45 | 15.73 |
| D_ans | 60.36 | 66.73 | 47.57 | 72.21 | 66.63 | 72.48 | 68.65 | 94.42 | 13.48 |
| D_reason | 54.79 | 68.82 | 38.83 | 70.68 | 64.46 | 72.48 | 62.81 | 92.50 | 11.24 |

#### seed43（strict label, 子集）

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 54.88 | 71.09 | 31.73 | 55.53 | 47.65 | 61.07 | 61.85 | 91.73 | 15.56 |
| PE | 53.65 | 70.55 | 31.73 | 55.32 | 47.43 | 60.40 | 62.60 | 91.85 | 16.67 |
| LL | 53.01 | 70.29 | 30.77 | 55.07 | 46.86 | 61.07 | 61.31 | 91.43 | 15.56 |
| Self-Certainty | 53.47 | 70.31 | 31.73 | 55.09 | 47.28 | 60.40 | 61.88 | 91.83 | 15.56 |
| DeepConf | 60.54 | 74.32 | 42.31 | 59.87 | 53.85 | 65.10 | 72.87 | 94.62 | 18.89 |
| **PRS (Ours)** | 80.59 | 83.55 | 55.77 | 87.93 | 81.90 | 87.92 | 76.01 | 95.32 | 20.00 |
| F_resp | 81.31 | 85.39 | 56.73 | 86.51 | 77.33 | 85.91 | 77.98 | 94.94 | 20.00 |
| D_ans | 61.74 | 68.09 | 48.08 | 71.36 | 62.72 | 75.84 | 52.16 | 86.53 | 13.33 |
| D_reason | 55.22 | 70.38 | 39.42 | 69.94 | 64.23 | 74.50 | 63.26 | 92.29 | 15.56 |

### 表 1c：Llama-3.1-8B

> **数据状态（2026-06-13）**：P2 数学三数据集 × 3 seeds **已全部生成完毕**（minerva 272/seed，math500/gsm8k 各 300/seed）。判分/聚合口径同表 1a（strict label 子集；本 run **SE=0** 未生成 SE 列；U_Ecc/U_Deg/CoT/P(True)/INSIDE/TokUR 留空）。
> **复现**：`python PRS/scripts/aggregate_math_strict.py --base prs-outputs/maintable_llama31_8b`

#### 样本剔除统计（label_drop=1 的题目被排除）

| seed | MATH-500 保留/总数 (剔除%) | GSM8K 保留/总数 (剔除%) | Minerva 保留/总数 (剔除%) |
|--|--:|--:|--:|
| seed41 | 223/300 (25.7%) | 299/300 (0.3%) | 187/272 (31.2%) |
| seed42 | 222/300 (26.0%) | 298/300 (0.7%) | 188/272 (30.9%) |
| seed43 | 227/300 (24.3%) | 298/300 (0.7%) | 189/272 (30.5%) |

#### 3-seed 平均 (41/42/43, mean ± std, strict label, 子集)

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SE | — | — | — | — | — | — | — | — | — |
| SAR | 57.45 ± 0.75 | 54.33 ± 0.67 | 64.18 ± 0.30 | 57.41 ± 1.48 | 18.63 ± 1.82 | 93.29 ± 0.00 | 64.60 ± 1.67 | 78.66 ± 1.25 | 40.22 ± 1.38 |
| $U_{Ecc}$ | — | — | — | — | — | — | — | — | — |
| $U_{Deg}$ | — | — | — | — | — | — | — | — | — |
| P(True) | — | — | — | — | — | — | — | — | — |
| INSIDE | — | — | — | — | — | — | — | — | — |
| PE | 58.21 ± 0.78 | 55.39 ± 0.80 | 63.89 ± 1.33 | 56.90 ± 1.57 | 18.62 ± 3.09 | 93.29 ± 0.00 | 63.54 ± 1.47 | 78.02 ± 1.02 | 39.50 ± 1.75 |
| LL | 56.57 ± 0.88 | 52.46 ± 1.66 | 62.09 ± 0.74 | 56.82 ± 1.70 | 17.54 ± 3.51 | 93.51 ± 0.32 | 62.96 ± 1.16 | 77.26 ± 0.53 | 39.50 ± 1.75 |
| Self-Certainty | 57.30 ± 0.89 | 54.17 ± 0.93 | 63.59 ± 0.91 | 56.89 ± 1.58 | 18.59 ± 3.10 | 93.29 ± 0.00 | 63.37 ± 1.52 | 77.90 ± 1.03 | 39.50 ± 1.75 |
| DeepConf | 69.42 ± 0.65 | 66.15 ± 2.89 | 71.36 ± 2.39 | 75.97 ± 1.51 | 25.14 ± 2.85 | 98.21 ± 0.63 | 68.62 ± 1.12 | 82.96 ± 0.61 | 40.93 ± 2.17 |
| TokUR EU | — | — | — | — | — | — | — | — | — |
| **PRS (Ours)** | **84.58 ± 0.56** | **72.08 ± 2.11** | **85.68 ± 1.36** | **86.38 ± 0.72** | **57.73 ± 7.35** | **97.99 ± 0.00** | **75.95 ± 1.77** | **86.17 ± 2.89** | **48.76 ± 1.53** |
| F_resp | 84.26 ± 0.56 | 73.01 ± 2.23 | 86.58 ± 1.36 | 88.87 ± 0.74 | 51.10 ± 5.96 | 98.66 ± 0.00 | 75.84 ± 2.17 | 84.86 ± 2.50 | 47.70 ± 2.06 |
| D_ans | 68.45 ± 0.60 | 52.04 ± 1.41 | 77.04 ± 2.56 | 67.59 ± 2.26 | 29.65 ± 1.23 | 96.20 ± 0.63 | 68.59 ± 1.04 | 78.18 ± 2.59 | 46.98 ± 1.09 |
| D_reason | 54.92 ± 0.82 | 49.10 ± 1.63 | 59.12 ± 2.41 | 62.04 ± 0.63 | 19.29 ± 0.90 | 94.63 ± 0.55 | 52.73 ± 1.43 | 70.78 ± 1.18 | 33.09 ± 1.43 |

#### seed41（strict label, 子集）

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 56.85 | 54.74 | 63.96 | 56.91 | 16.25 | 93.29 | 62.39 | 76.92 | 40.86 |
| PE | 57.69 | 55.86 | 63.06 | 56.51 | 16.32 | 93.29 | 61.68 | 76.61 | 39.78 |
| LL | 56.34 | 52.99 | 61.26 | 56.14 | 14.61 | 93.96 | 61.35 | 77.00 | 39.78 |
| Self-Certainty | 57.04 | 54.43 | 63.06 | 56.51 | 16.32 | 93.29 | 61.40 | 76.48 | 39.78 |
| DeepConf | 68.67 | 63.55 | 72.07 | 74.78 | 24.96 | 98.66 | 68.52 | 82.82 | 40.86 |
| **PRS (Ours)** | 84.16 | 71.85 | 85.59 | 85.76 | 57.09 | 97.99 | 76.61 | 86.10 | 50.54 |
| F_resp | 84.07 | 72.47 | 86.49 | 88.44 | 49.19 | 98.66 | 77.38 | 85.66 | 50.54 |
| D_ans | 67.62 | 50.68 | 78.38 | 64.40 | 28.51 | 96.64 | 67.18 | 74.53 | 48.39 |
| D_reason | 55.87 | 48.79 | 61.26 | 62.93 | 19.14 | 95.30 | 50.84 | 69.16 | 32.26 |

#### seed42（strict label, 子集）

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 56.98 | 53.38 | 63.96 | 55.90 | 19.00 | 93.29 | 66.43 | 79.76 | 41.49 |
| PE | 57.63 | 54.27 | 65.77 | 55.20 | 16.55 | 93.29 | 65.26 | 79.00 | 41.49 |
| LL | 55.62 | 50.21 | 63.06 | 55.17 | 15.53 | 93.29 | 64.02 | 76.79 | 41.49 |
| Self-Certainty | 56.36 | 52.93 | 64.86 | 55.17 | 16.48 | 93.29 | 65.10 | 78.88 | 41.49 |
| DeepConf | 69.34 | 64.74 | 73.87 | 78.10 | 28.71 | 98.66 | 70.04 | 83.77 | 43.62 |
| **PRS (Ours)** | 84.20 | 69.61 | 87.39 | 87.39 | 67.03 | 97.99 | 73.53 | 82.67 | 48.94 |
| F_resp | 83.69 | 70.58 | 88.29 | 89.91 | 59.16 | 98.66 | 72.77 | 81.48 | 46.81 |
| D_ans | 68.98 | 51.44 | 79.28 | 69.19 | 29.08 | 95.30 | 68.94 | 80.30 | 46.81 |
| D_reason | 53.87 | 47.27 | 60.36 | 61.56 | 20.45 | 93.96 | 54.31 | 71.22 | 35.11 |

#### seed43（strict label, 子集）

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SAR | 58.51 | 54.86 | 64.60 | 59.41 | 20.65 | 93.29 | 64.97 | 79.30 | 38.30 |
| PE | 59.31 | 56.05 | 62.83 | 58.99 | 22.98 | 93.29 | 63.68 | 78.44 | 37.23 |
| LL | 57.74 | 54.18 | 61.95 | 59.15 | 22.47 | 93.29 | 63.50 | 78.00 | 37.23 |
| Self-Certainty | 58.49 | 55.16 | 62.83 | 58.99 | 22.98 | 93.29 | 63.61 | 78.34 | 37.23 |
| DeepConf | 70.25 | 70.17 | 68.14 | 75.02 | 21.75 | 97.32 | 67.29 | 82.29 | 38.30 |
| **PRS (Ours)** | 85.37 | 74.77 | 84.07 | 85.98 | 49.06 | 97.99 | 77.71 | 89.74 | 46.81 |
| F_resp | 85.03 | 75.97 | 84.96 | 88.27 | 44.94 | 98.66 | 77.37 | 87.45 | 45.74 |
| D_ans | 68.75 | 53.99 | 73.45 | 69.19 | 31.35 | 96.64 | 69.66 | 79.71 | 45.74 |
| D_reason | 55.01 | 51.23 | 55.75 | 61.63 | 18.27 | 94.63 | 53.04 | 71.95 | 31.91 |

### 表 1d：Qwen3-8B

> **数据状态**：未开始（待 Llama-3.1-8B 完成后自动启动）。

结构同表 1a（待 P2 完成后填入）。

---

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
| **PRS (Ours)** | **80.50 ± 1.06** | **90.77 ± 1.43** | **50.00 ± 1.63** | —¹ | —¹ | —¹ |

样本量/准确率：leg_counting n=100/seed，acc≈28–30%；color_cube n=300/seed（seed42=234），acc≈29–31%。两集 strict drop≈0。

¹ **color_cube 的 PRS 暂缺**：PRS 在 color_cube 上 **98% 样本为 `nan`**（颜色词答案使 `formula_skeleton`/数值轨迹等数学专属特征为 nan，导致最终 logistic 组合 nan），仅余 ~2% 可算、不可信，故留空。需为非数字答案分支重拟合/降维 PRS 后再填。leg_counting（数字答案）PRS 正常且**显著领先**所有 baseline（80.50 vs DeepConf 67.06）。TokUR EU 不在 summary 字段中，留空待官方 TokUR 评分接入。

> **判分修复①（string 数据集）**：修复严重 bug —— 旧 raw 判分对所有 string 题统一用 `math_equal`，而它对非数字串会误返回 `True`，导致 **color_cube 等被全部判对**（旧 acc≈100% 为假）。现改为 **mode-aware 严格判分**（生成 `record.py` + 离线 `scripts/fix_logic_extraction.py`），并从 `full_response` 重抽取答案。
>
> **判分修复②（MANUAL_RELABEL 跨模型误用）**：人工核验白名单 `MANUAL_RELABEL_IDS`（minerva 14 题）此前对所有模型按 id 强制判对，但它仅在 Qwen2.5-3B 上核验过。已给 `grade_answer` 加 `model` 参数，**白名单仅对 Qwen2.5-3B 生效**；其余模型（Llama/Qwen3-8B）这些题回归 strict 判定（多为 drop），避免分数虚高。

---

## 4. 跨模型汇总（E3）

仅报 PRS 主指标 AUROC（mean±std），验证跨模型稳健。数学三集跑全 4 模型；逻辑三集仅主模型（其余模型留 `—`）。

### 表 3：PRS AUROC × 模型 × 数据集

| Model | MATH-500 | GSM8K | Minerva | Zebra | Leg | Color |
|-------|--:|--:|--:|--:|--:|--:|
| **Qwen2.5-3B**（主） | **88.07 ± 0.81** | **77.77 ± 1.38** | **83.58 ± 1.41** | — | **80.50 ± 1.06** | —¹ |
| Llama-3.2-1B | **80.22 ± 0.72** | **88.72 ± 0.83** | **76.98 ± 2.59** | — | — | — |
| Llama-3.1-8B | **84.58 ± 0.56** | **86.38 ± 0.72** | **75.95 ± 1.77** | — | — | — |
| Qwen3-8B | — | — | — | — | — | — |

> 数学三集：3-seed mean±std，strict label 子集，指标 = PRS AUROC。逻辑仅主模型 Qwen2.5-3B（Zebra 已剔除；Color Cube PRS 暂缺¹ 同表 2a）。Llama-3.1-8B 在 MATH-500/GSM8K 上 PRS 与 Qwen2.5-3B 同量级（84.58 vs 88.07；86.38 vs 77.77），Minerva 略低（75.95 vs 83.58）。Llama-3.2-1B GSM8K PRS（88.72）仍最高。

---

## 5. 组件消融（E4）

**消融 = 完整 PRS 去掉恰好一个组件**，对剩余组件**重新拟合 logistic 权重**（因默认 $\lambda$ 极小，朴素置零无信息）。报 AUROC + ΔAUROC（相对 full PRS）。

### 表 4：组件消融（Qwen2.5-3B，3 seeds mean±std）

| Variant | MATH-500 AUROC | Δ | GSM8K AUROC | Δ | Minerva AUROC | Δ |
|---------|--:|--:|--:|--:|--:|--:|
| **PRS (full)** |  | 0 |  | 0 |  | 0 |
| − $F_\text{resp}$ |  |  |  |  |  |  |
| − $S_\text{ans}$ |  |  |  |  |  |  |
| − $S_\text{tr}$ |  |  |  |  |  |  |
| $F_\text{resp}{=}$T-ASE（去 weight 分支） |  |  |  |  |  |  |
| $F_\text{resp}{=}$W-ASE（去 text 分支） |  |  |  |  |  |  |

**逐行假设**：
- − $F_\text{resp}$：删除答案级碎裂主项后大幅下降，证明其为主干信号。
- − $S_\text{ans}$：高难集（Minerva）下降更多，承诺点竞争对难题更关键。
- − $S_\text{tr}$：整体小幅下降，验证推理内竞争为补充信号。
- T-ASE：限制为 text 分支后下降，证明 weight 扰动有独立贡献。
- W-ASE：限制为 weight 分支后下降，证明 text 改写不可省，TW 联合最优。

### 表 4b：$S_\text{tr}$ 定义消融（B vs A vs legacy）

验证 trace-drift 的 token 选择策略。主表用 B；本表说明跨域统一定义的合理性。

| $S_\text{tr}$ 定义 | MATH-500 | GSM8K | Zebra | Color Cube |
|--------------------|--:|--:|--:|--:|
| **B：top-10% 局部 spread（默认）** |  |  |  |  |
| A：按域内容 token |  |  |  |  |
| legacy：数学 token only |  |  |  |  |

- **B**（`AltMass_local_spread_topk`）：跨域统一、无需分类器、抗 filler 稀释。
- **A**（`AltMass_local_spread_content`）：数学→数字/算符；逻辑→实词（去停用词）。
- **legacy**（`AltMass_local_spread_reason`）：仅数学 token 均值；逻辑上信号偏弱。
- 假设：B 在数学/逻辑任务上整体最稳；legacy 在非数学任务退化。

---

## 6. 扰动预算消融（E5）

固定一支为默认满额，扫另一支数量，报 PRS 与 $F_\text{resp}$ 的 AUROC（CPU 子集重算）。

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

CPU 后处理：`compute_prs_from_row(row, lambda_a=, lambda_r=)`。代表数据集 MATH-500。

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
cd /mnt/afs/L202500372/PRS
source scripts/env.sh
export MAX_SAMPLES=300   # 默认已写入 configs/ase_models.yaml 与编排脚本

# ========== P0：主模型 PRS + 官方 TokUR（数学，最优先）==========
# Phase A(vLLM) + B(HF weight+metrics) + C(official TokUR) 一体
DATASETS=minerva,math500,gsm8k bash scripts/run_maintable_vllm.sh qwen25_3b

# 分批/续跑（重复执行自动跳过已完成）
SEEDS=41 DATASETS=math500 bash scripts/run_maintable_vllm.sh qwen25_3b
SKIP_VLLM=1 bash scripts/run_maintable_vllm.sh qwen25_3b   # 只补 HF：weight + metrics
SKIP_HF=1   bash scripts/run_maintable_vllm.sh qwen25_3b   # 只跑 vLLM：clean+R+SE
ASE_SKIP_TOKUR=1 bash scripts/run_maintable_vllm.sh qwen25_3b  # 跳过 Phase C

# 单独补官方 TokUR（需 Phase B 已完成 raw_runs）
OUT_DIR=$PRS_OUTPUTS/maintable_qwen25_3b/seed41 PRS_MODEL_TAG=qwen25_3b \
  DATASET=math500 TOKUR_SEED=41 bash scripts/run_tokur_official_maintable.sh

# seed1 已有 500 档 raw：子采样对齐 300 + 补 SE
SEEDS=41 MAX_SAMPLES=300 SKIP_VLLM=1 bash scripts/run_maintable_vllm.sh qwen25_3b

# ========== P1：主模型逻辑（PRS + TokUR）==========
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
python -m prs.ase.ablation_recompute --out-dir $PRS_OUTPUTS/ase_full \
  --datasets math500,gsm8k,minerva --budget-sweep \
  --output $PRS_OUTPUTS/ABLATION_budget_sweep.json

# --- E7 扰动强度（GPU，预算充裕再跑）---
python -m prs.ase.run_ase_experiment --dataset math500 --mode all \
  --max-samples 300 --weight-sigma 0.05 --weight-rank 4 \
  --out-dir $PRS_OUTPUTS/ablation_sigma05
```

输出位置：
- 主表：`paper/maintable/{model}/maintable.{md,tex}`
- 消融：`$PRS_OUTPUTS/ABLATION_component.md`、`ABLATION_budget_sweep.json`

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
- 数学 3 集（**872**/seed）× **4 模型** × 3 seeds；**优先 P0 跑 Qwen2.5-3B PRS + TokUR**。
- **Qwen2.5-3B seed1 已有**（旧 500 档 raw）：CPU **子采样前 300** 对齐新口径；R=W=8→4 子集重算免 GPU；缺 SE 需 GPU backfill（约 872 题×8 次，≈ +38 A100-h）。
- 逻辑 3 集（900/seed）**仅主模型 Qwen2.5-3B** × 3 seeds（P1）。
- 每题 **17 次解码**（`1+R(4)+W(4)+SE(8)`），统一采样预算 K=8。
- **执行顺序**：P0 主模型数学 PRS+TokUR → P1 逻辑 → P2 其余模型 → P3 CPU 消融。

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

> 带宽比例：H100≈1.6×A100，5090≈0.9×A100。`ASE_FAST=1`（数据集并行+分片）只缩短**墙钟**，卡时总量不变。

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
| **Qwen2.5-3B（含逻辑，PRS+TokUR）** | **≈400** | **≈250** | **≈445** | **P0+P1** |
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

> **已采纳**：降 R/W 到 4（K=8）；**MAX_SAMPLES=300**；**vLLM 混合管线（§12.6）**；**先 P0 PRS+TokUR 再扩模型**。
> 300 档下 4 模型 3 seed 全跑可纳入 ¥3000 预算，无需再砍 8B seed。

### 12.6 三阶段管线（PRS vLLM 混合 + 官方 TokUR）

**PRS 路径**：纯解码三路（clean + R + SE，13/17 次）走 vLLM；权重扰动（W=4）+ INSIDE 走 HF。

**TokUR 路径（官方，独立）**：`third_party/TokUR/run/greedy_unc_single_batch_refine.py`（vLLM + TFB bayesian_transformer），**不用** PRS 的 `score_tokur_baseline` 近似。

**三阶段架构（均可续跑、可分批）**：
- **Phase A（vLLM，PRS）**：`prs.ase.run_vllm_phase` → `raw_runs/{id}.partial.json`
- **Phase B（HF，PRS）**：`run_ase_experiment --resume` → 补 weight 分支 + metrics → `raw_runs/{id}.json`
- **Phase C（官方 TokUR venv）**：`run_tokur_official_maintable.sh` → export → greedy_unc → `tokur_baseline.jsonl`
- 编排：`scripts/run_maintable_vllm.sh`（`SKIP_VLLM`/`SKIP_HF`/`ASE_SKIP_TOKUR` 控制各阶段）

**保真度说明（写进 limitation/附录）**：vLLM 仅返回 top-k logprobs，非全词表分布。因此：
- **精确**：选中 token logprob、top-k pairs、top-2 margin、rank → **LL / DeepConf / Self-Certainty 与 HF 一致**；答案类指标（F_resp / U_Ecc / U_Deg / SE）完全不受影响。
- **近似**：全词表预测熵用 top-k 重归一化熵代替 → **PE / SAR / T-branch ATU 为近似**。
- **不受影响**：weight 分支走 HF，故 **S_ans / S_tr（PRS 核心）完全精确**。

**预期加速**：13/17≈76% 解码可批量化，全管线含 HF 前向约 **1.7–2.6×**（视 batch 与 vLLM 吞吐）。

| 方案（300 档） | 数学 A100-h | 逻辑 A100-h | 合计 |
|------|----:|----:|----:|
| 纯 HF | ≈1260 | ≈235 | ≈1495 |
| vLLM 混合（保守 1.7×） | ≈740 | ≈138 | **≈880** |
| vLLM 混合（中等 2.2×） | ≈575 | ≈107 | **≈680** |
| **P0 only**（Qwen 数学 PRS+TokUR） | ≈245 | — | **≈245** |

> 安装：`pip install vllm`（按本机 CUDA/torch 版本匹配）。未安装时 HF 管线照常工作。

### 12.7 预算报价（报销用，封顶 ¥3,000）

卡时单价：**RTX 5090-32GB ¥2.88/h**、**A100-40GB ¥3.28/h**。
老师批复预算：**¥3,000**（按 5090 计 ≈ **1040 卡时**可用）。

**保守口径**（往高报，可辩护）：
- vLLM 取**最保守 1.7× 加速**；
- 失败重跑 / SE backfill / reasoning 长尾余量 ×1.2。

| 方案（300 档） | 基准 5090-h | 含余量 5090-h | **5090 费用** | 是否在 ¥3000 内 |
|------|----:|----:|----:|:---:|
| **P0 only**（Qwen 数学 PRS+TokUR，3 seed） | ≈280 | ≈336 | **≈¥970** | ✅ |
| P0+P1（+逻辑 PRS+TokUR） | ≈410 | ≈492 | **≈¥1,420** | ✅ |
| **全计划**（4 模型 3 seed + 逻辑，vLLM） | ≈880 | ≈1056 | **≈¥2,940** | ✅ 刚好 |
| 纯 HF 全计划（无 vLLM） | ≈1650 | ≈1980 | ≈¥5,700 | ❌ |

**建议报销额**：按老师封顶 **报 ¥3,000**（说明：300 题/集 × vLLM 混合 × 优先 PRS+TokUR，含 20% 重跑余量）。

**分阶段花钱**（预算紧时）：
1. 先花 **~¥1,000** 跑 P0 → 主表 PRS vs TokUR 可填
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
| 引擎 | **纯 HF 分片管线**（`run_ase_experiment --mode generate`，自动算 PRS + 11 baselines） |
| 为何不用 vLLM（PRS Phase A） | PRS 主表走 **纯 HF**；5090（sm_120）上 PRS 不需 vLLM |
| TokUR Phase C | 需 **fork vLLM 0.7.3** 独立 venv；5090 可源码编译 vLLM，但 **transformers 版本** 仍阻塞 smoke（见 **§14**） |
| Python | `/root/miniconda3/bin/python`，torch 2.8.0+cu128，attn=sdpa |
| 路径 | `PRS_ROOT=/root/autodl-tmp/PRS`、`PRS_MODELS=/root/autodl-tmp/prs-models`、`PRS_OUTPUTS=/root/autodl-tmp/prs-outputs` |

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
for s in $(seq 0 7); do tail -1 $PRS_OUTPUTS/maintable_qwen25_3b/seed41/logs/minerva_shard$s.log; done
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
for p in $(pgrep -f '[p]rs.ase.run_ase_experiment'); do kill -9 $p; done
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
- **TokUR EU 主表列**：5090 上 vLLM fork **已编译成功**，但 smoke 因 `transformers` 与 `bayesian_transformer` 不兼容未过；详见 **§14**。推荐在 **4090/A100** 新环境用预编译 wheel + `transformers==4.53.3` 补齐；PRS `raw_runs` 已齐的模型可离线补跑。
- 续跑墙钟估计：P0(Qwen数学) ≈ 40 h（2流/卡加速后），全四模型按上表外推。

---

## 14. TokUR 官方环境：故障原因、新机器部署与 5090 补救

> **Phase C 专用**：TokUR 与 PRS 主表 **必须分 venv**；读已有 `raw_runs`，写 `tokur_baseline.jsonl`（`scoring_mode=official_vllm`）。  
> **禁止**主表用 `score_tokur_baseline.py`（post-hoc HF 近似）。

### 14.1 为什么本机 TokUR 跑不了（分层）

| 层次 | 问题 | 本机状态 |
|------|------|----------|
| **硬件** | RTX **5090 = sm_120**；TokUR 依赖的 fork **vLLM 0.7.3 无 sm_120 预编译 wheel**，只能源码编译（cutlass / flash-attn / flashmla 等） | **已解决**：`build_tokur_vllm_5090.sh` 编译成功，`Successfully built vllm` |
| **依赖** | `bayesian_transformer/vllm_models/tfb_llama.py` 需 `from transformers.utils import LossKwargs`；**transformers≥4.54 已删除 `LossKwargs`**（4.46–4.53 存在） | **未解决（当前 blocker）**：venv 里为 **4.57.6** → smoke 报 `ImportError: LossKwargs` |
| **架构** | PRS（HF）与 TokUR（vLLM）版本栈相反：PRS 用 torch **2.8** + transformers **4.57**；TokUR 官方 era 为 torch **~2.4** + vLLM **0.7.3** + transformers **≤4.53** | 两套环境 **不可共用** conda / 同一 venv |

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

### 14.2 PRS 与 TokUR 环境对照

| | PRS 主表（Phase B，HF） | TokUR（Phase C，官方 vLLM） |
|--|------------------------|----------------------------|
| 入口 | `prs.ase.run_ase_experiment` | `run_tokur_official_maintable.sh` → `greedy_unc_single_batch_refine.py` |
| Python | `/root/miniconda3`（system） | 独立 venv：`.tokur_venv` 或 `.tokur_venv_5090` |
| torch | **2.8.0+cu128**（5090 友好） | **~2.4.0+cu12x**（与 vLLM 0.7.3 wheel 匹配） |
| transformers | **4.57.x**（PRS 用） | **4.53.3**（保留 `LossKwargs`） |
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
数据：~50GB（TFB 权重 prs-models/ + 已完成 raw_runs prs-outputs/）
说明：不与 PRS HF 生成共用环境；只读 raw_runs 写 tokur_baseline.jsonl
```

### 14.4 新环境部署（4090 / A100，约 30 分钟）

**Step 0 — 同步数据**（TokUR 只需 Phase B 产物，不必同步 PRS conda / 5090 编译产物）：

```bash
# 必需
PRS/                          # 代码 + third_party/TokUR
prs-models/TFB-*              # TFB 权重
prs-outputs/maintable_*/      # 含 */*/raw_runs/

# 不必同步
# miniconda PRS 环境、.tokur_venv_5090、vllm-tokur/build/
```

**Step 1 — 独立 venv**（或一键：`bash scripts/setup_tokur_vllm.sh`）

```bash
cd /root/autodl-tmp/PRS
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
pip install -e /root/autodl-tmp/PRS/third_party/TokUR/bayesian_transformer
# requirements.txt 中 vllm==0.7.3 与 fork 冲突时以 fork 为准
pip install -r /root/autodl-tmp/PRS/third_party/TokUR/requirements.txt 2>/dev/null || true
```

**Step 4 — 验证**

```bash
export TOKUR_VENV=/root/autodl-tmp/PRS/.tokur_venv
export VLLM_USE_V1=0
python -c "import vllm, bayesian_transformer, torch; print('OK', vllm.__version__, torch.__version__)"
bash scripts/smoke_tokur_vllm.sh   # 成功 → logs/TOKUR_SMOKE_OK
```

**Step 5 — 跑主表 TokUR**（PRS `raw_runs` 已存在即可）

```bash
# 单任务
OUT_DIR=/root/autodl-tmp/prs-outputs/maintable_qwen25_3b/seed41 \
PRS_MODEL_TAG=qwen25_3b DATASET=math500 TOKUR_SEED=41 \
TOKUR_VENV=/root/autodl-tmp/PRS/.tokur_venv \
bash /root/autodl-tmp/PRS/scripts/run_tokur_official_maintable.sh

# 批量（8B 完成后）
TOKUR_VENV=/root/autodl-tmp/PRS/.tokur_venv \
bash /root/autodl-tmp/run_p2_tokur_queue.sh
```

**可优先补 TokUR 的数据**（PRS Phase B 已齐）：Qwen2.5-3B（2616/2616）、Llama-3.2-1B（2616/2616）；Llama-3.1-8B / Qwen3-8B 随 P2 推进逐步可补。

### 14.5 若新机器仍是 5090

1. 使用已有脚本：`bash /root/autodl-tmp/build_tokur_vllm_5090.sh`（含 cutlass / flashmla / flash-attn 本地镜像、`flock` 防并行编译）。
2. **必须** pin transformers：

```bash
/root/autodl-tmp/PRS/.tokur_venv_5090/bin/pip install "transformers==4.53.3"
bash /root/autodl-tmp/smoke_tokur_vllm.sh
```

3. 若 runtime 仍报错，考虑 venv 内降级 torch 至 **2.4.x** 并重编 vLLM（torch 2.8 + vLLM 0.7.3 fork 非官方组合）。

5090 编译依赖环境变量（脚本已内置）：

```bash
VLLM_CUTLASS_SRC_DIR=PRS/third_party/cutlass-v3.8.0
FLASH_MLA_SRC_DIR=PRS/third_party/flashmla-575f772
VLLM_FLASH_ATTN_SRC_DIR=PRS/third_party/vllm-flash-attn-9bfa986
TORCH_CUDA_ARCH_LIST=12.0
```

### 14.6 本机最小补救（不换卡，可选）

vLLM 已编好，仅差 transformers：

```bash
/root/autodl-tmp/PRS/.tokur_venv_5090/bin/pip install "transformers==4.53.3"
/root/autodl-tmp/PRS/.tokur_venv_5090/bin/python -c "import vllm, bayesian_transformer; print('OK')"
bash /root/autodl-tmp/smoke_tokur_vllm.sh
```

通过后 `run_p2_tokur_queue.sh` 会自动推进；否则可 **`pkill -f run_p2_tokur_scheduler`** 避免空等 vLLM。

### 14.7 与 PRS 实验分工（推荐）

| 机器 | 任务 |
|------|------|
| **现有 4×5090** | 继续 P2 PRS 8B 主表（Llama-3.1-8B → Qwen3-8B） |
| **新 4090/A100** | 仅 TokUR Phase C；`rsync prs-outputs/` + `prs-models/TFB-*` 即可 |
| **数据衔接** | TokUR 读 `maintable_*/seed*/{dataset}/raw_runs/`，写同目录 `tokur_baseline.jsonl` |

### 14.8 5090 源码编译曾遇到的问题（备忘）

| 现象 | 处理 |
|------|------|
| GitHub clone cutlass / flashmla 超时 | 预克隆至 `third_party/cutlass-v3.8.0` 等，设 `VLLM_CUTLASS_SRC_DIR` / `FLASH_MLA_SRC_DIR` |
| flash-attn 缺 `cutlass/numeric_types.h` | `ln -sfn cutlass-v3.8.0/include vllm-flash-attn-9bfa986/csrc/cutlass/include` |
| `Error: not a CMake build directory` | 清理 `vllm-tokur/` 下 in-source 残留：`CMakeCache.txt`、`CMakeFiles/` |
| 多 watchdog 并行编译冲突 | `build_tokur_vllm_5090.sh` 用 `flock`；编译期间勿并行启动多个 build |
