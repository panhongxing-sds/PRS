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
> - SE：8 个独立高温样本（`--se-samples 8`，温度 0.7 / top-p 0.95，完整解码）；
> - **TokUR EU：8 次权重扰动 teacher-forcing 前向**（`num_samples=8`，对固定回复做前向估 EU，**非解码**，故卡时远低于上面三类；用其论文自带的 rank8/σ0.1/q,k,v,o 扰动配置，仅对齐样本数）；
> - 单次方法（PE/LL/Self-Certainty/DeepConf/INSIDE/P(True)）：仅 1 次 greedy/前向，本就不采样，不参与采样数对齐。
>
> 每题**解码**生成 = 1(clean) + 4(R) + 4(W) + 8(SE) = **17 次**；TokUR 另加 8 次前向（teacher-forcing，便宜）。降预算（R=W=8→4）经验上几乎不掉点，且让 SE/TokUR 都在 K=8 下公平。
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
**数学数据集**：minerva(272 全量), math500(500), gsm8k(500) ——推荐档
**逻辑数据集**：leg_counting, zebra_puzzles, color_cube（各取全量，上限 500）
**Seeds**：41, 42, 43（`DEFAULT_EXPERIMENT_SEEDS`）

---

## 2. 主表（E1：数学四数据集）

每个数据集三列 **AUROC | AUPRC | ACC***，行含 TokUR EU + 全部 baseline + PRS 及其三组分。CoT 仅 ACC*。

### 表 1a：Qwen2.5-3B

| Method | MATH-500 AUROC | AUPRC | ACC* | GSM8K AUROC | AUPRC | ACC* | Minerva AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CoT (Lower-Bound) | — | — |  | — | — |  | — | — |  |
| SE |  |  |  |  |  |  |  |  |  |
| SAR |  |  |  |  |  |  |  |  |  |
| $U_{Ecc}$ |  |  |  |  |  |  |  |  |  |
| $U_{Deg}$ |  |  |  |  |  |  |  |  |  |
| P(True) |  |  |  |  |  |  |  |  |  |
| INSIDE |  |  |  |  |  |  |  |  |  |
| PE |  |  |  |  |  |  |  |  |  |
| LL |  |  |  |  |  |  |  |  |  |
| Self-Certainty |  |  |  |  |  |  |  |  |  |
| DeepConf |  |  |  |  |  |  |  |  |  |
| TokUR EU |  |  |  |  |  |  |  |  |  |
| **PRS (Ours)** |  |  |  |  |  |  |  |  |  |


### 表 1b–1d：Llama-3.2-1B / Llama-3.1-8B / Qwen3-8B
结构同表 1a（复制即可）。

---

## 3. 主表（E2：逻辑）

参考图布局：逻辑题缺隐藏状态/采样类 baseline 时留 `—`。

### 表 2a：逻辑三数据集（主模型 Qwen2.5-3B；仅此一模型证跨域泛化）

| Method | Zebra Puzzles AUROC | AUPRC | ACC* | Leg Counting AUROC | AUPRC | ACC* | Color Cube AUROC | AUPRC | ACC* |
|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CoT (Lower-Bound) | — | — |  | — | — |  | — | — |  |
| PE |  |  |  |  |  |  |  |  |  |
| LL |  |  |  |  |  |  |  |  |  |
| Self-Certainty |  |  |  |  |  |  |  |  |  |
| DeepConf |  |  |  |  |  |  |  |  |  |
| TokUR EU |  |  |  |  |  |  |  |  |  |
| **PRS (Ours)** |  |  |  |  |  |  |  |  |  |

---

## 4. 跨模型汇总（E3）

仅报 PRS 主指标 AUROC（mean±std），验证跨模型稳健。数学三集跑全 4 模型；逻辑三集仅主模型（其余模型留 `—`）。

### 表 3：PRS AUROC × 模型 × 数据集

| Model | MATH-500 | GSM8K | Minerva | Zebra | Leg | Color |
|-------|--:|--:|--:|--:|--:|--:|
| **Qwen2.5-3B**（主） |  |  |  |  |  |  |
| Llama-3.2-1B |  |  |  | — | — | — |
| Llama-3.1-8B |  |  |  | — | — | — |
| Qwen3-8B |  |  |  | — | — | — |

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

## 9. 运行命令速查

```bash
cd /mnt/afs/L202500372/PRS
export PYTHONPATH=src

# --- E1 主表（数学，3 seeds 聚合）---
# (A) 纯 HF 管线
for m in qwen25_3b llama32_1b llama31_8b qwen3_8b; do
  bash scripts/run_maintable_pipeline.sh "$m"            # GPU 生成 + 聚合
done
SKIP_GPU=1 bash scripts/run_maintable_pipeline.sh qwen25_3b   # 仅聚合已有 raw

# (B) vLLM 混合管线（推荐，clean+R+SE 走 vLLM；weight/TokUR 走 HF）
for m in qwen25_3b llama32_1b llama31_8b qwen3_8b; do
  DATASETS=minerva,math500,gsm8k bash scripts/run_maintable_vllm.sh "$m"
done
# 分批/续跑：可按 seed、按数据集切分，重复执行自动跳过已完成
SEEDS=41 DATASETS=math500 bash scripts/run_maintable_vllm.sh qwen25_3b
SKIP_VLLM=1 bash scripts/run_maintable_vllm.sh qwen25_3b   # 只补 weight(HF)+metrics
SKIP_HF=1   bash scripts/run_maintable_vllm.sh qwen25_3b   # 只跑 vLLM 生成阶段

# --- E2 逻辑 ---
bash scripts/download_logic_code_datasets.sh
bash scripts/generate_api_variants.sh leg_counting,zebra_puzzles,color_cube
DATASETS=leg_counting,zebra_puzzles,color_cube \
  bash scripts/run_maintable_vllm.sh qwen25_3b

# --- E4 组件消融（CPU）---
bash scripts/run_ablation_cpu.sh         # → ABLATION_component.md
# 3-seed：对 seed41/42/43 各自 out-dir 分别跑后再 mean±std

# --- E5 扰动预算（CPU）---
python -m prs.ase.ablation_recompute --out-dir $PRS_OUTPUTS/ase_full \
  --datasets math500,gsm8k,minerva --budget-sweep \
  --output $PRS_OUTPUTS/ABLATION_budget_sweep.json

# --- E6 λ 敏感性（CPU）---
# 对 enriched features.jsonl 逐 (λ_a, λ_r) 调 compute_prs_from_row 再算 AUROC

# --- E7 扰动强度（GPU）---
python -m prs.ase.run_ase_experiment --dataset math500 --mode all \
  --weight-sigma 0.05 --weight-rank 4 --out-dir $PRS_OUTPUTS/ablation_sigma05
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
- **CoT**：仅 ACC*（greedy 准确率下界），AUROC/AUPRC 置 `—`

---



> 无编造结果：本计划所有表格均为空待填（`TBD`），符合 no-fabrication 要求。
> 失败分析（stable-wrong 子集）本轮按用户要求暂缓。

---

## 12. 数据集题量与算力估算（A100 / H100 / RTX 5090）

### 12.1 每数据集题量（`run_maintable_3seed.sh` 实际口径）

| 数据集 | 域 | 每 seed 题量 | 来源 |
|--------|----|----:|------|
| Minerva | math | 272 | 全集（`MAX_SAMPLES_MINERVA=272`） |
| MATH-500 | math | 500 | 全集 |
| GSM8K | math | 500 | `max_samples=500`（截断自 1319） |
| Zebra Puzzles | logic | ≤500 | 取仓库全量(上限 500)，需下载确认 |
| Color Cube | logic | ≤500 | 同上 |
| Leg Counting | logic | ≤500 | 同上（原仓库可能 <500） |
| **数学合计/模型/seed** | | **1272** | 272+500+500，4 模型都跑 |
| **逻辑合计/seed（仅主模型）** | | **≤1500** | 仅 Qwen2.5-3B 跑 |

**定稿方案口径**：
- 数学 3 集（1272/seed）× **4 模型** × 3 seeds；其中 **Qwen2.5-3B 的 seed1 已有**（500/272 全量）。
  - seed1 旧 raw 是 R=W=8（16 扰动样本），新口径 K=8 只需在 CPU 上**取前 4+4 子集重算**（免 GPU）；
  - 但 seed1 缺 SE，需 GPU **backfill SE=8**（`--se-samples 8 --resume` 自动补，约 1272 题×8 次，≈ +55 A100-h）。
- 逻辑 3 集（≤1500/seed）**仅主模型 Qwen2.5-3B** × 3 seeds。
- 每题 **17 次解码**（`1+R(4)+W(4)+SE(8)`），统一采样预算 K=8。

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

**各模型生成题量**（逻辑按代表值 zebra500+color500+leg200≈1200/seed 估）：

| 模型 | 题量 | token 量 |
|------|----:|----:|
| Llama-3.2-1B（数学 3 seed） | 1272×3 = 3816 | 51.9M |
| Qwen2.5-3B（数学补 2 seed + 逻辑 3 seed） | 1272×2 + 1200×3 = 6144 | 83.6M |
| Llama-3.1-8B（数学 3 seed） | 3816 | 51.9M |
| Qwen3-8B（数学 3 seed） | 3816 | 51.9M |

### 12.3 主实验卡时（推荐档定稿口径）

| 模型 | A100 (h) | H100 (h) | 5090 (h) |
|------|---------:|---------:|---------:|
| Llama-3.2-1B | 262 | 164 | 288 |
| Qwen2.5-3B（含逻辑） | 581 | 363 | 645 |
| Llama-3.1-8B | 515 | 320 | 577 |
| Qwen3-8B | 515 | 320 | 577 |
| **小计** | **≈1873** | **≈1167** | **≈2087** |

加 TokUR EU + INSIDE/P(True) 前向 ≈ **+15%**；加 E7 扰动强度 ≈ +20/13/22 h。

| 全计划合计（推荐档，3 seeds，无 SE） | A100 | H100 | 5090 |
|------|---------:|---------:|---------:|
| **GPU-hours** | **≈2150** | **≈1360** | **≈2420** |

> 逻辑若取满 500×3 则 A100 升至 ≈2240。CPU 消融（E4/E5/E6）不占 GPU。误差带 ±40%。

### 12.4 换算参考
- 8×A100：≈ **11 节点·天**；8×H100：≈ **7 节点·天**；8×5090：≈ **12.6 节点·天**（单卡 ≈100 天，须多卡）。

### 12.5 进一步降本（在定稿基础上）

| # | 方案 | A100 节省 | 后合计 |
|---|------|----------:|-------:|
| 1 | **8B 模型仅 1 seed**（1B/3B 保 3 seed） | −约 790 h | ≈**1380** |
| 2 | **Qwen3-8B 设 `--max-new-tokens 1024`** | 防 reasoning 长尾，避免 +约 500 h 风险 | — |
| 3 | 逻辑改用 Llama-3.2-1B 主模型 | 微降（约 −90 h） | — |
| — | 复用 Qwen2.5-3B seed1 | （已计入定稿口径） | — |

> **已采纳**：降 R/W 到 4（K=8），换来 SE 公平接入；总生成仍 17 次/题（4+4+8+1）。经验上 R=W=8→4 几乎不掉点。
> 推荐执行：定稿口径 + 方案 1（8B 单 seed）+ **vLLM 混合管线（§12.6）**。

### 12.6 vLLM 混合管线（已实现）

为压缩卡时，**纯解码三路（clean + R 文本改写 + SE 高温采样，共 13/17 次）改用 vLLM 批量生成**；权重扰动分支（W=4）、TokUR EU、INSIDE 仍走 HF（vLLM 无法逐样本注入权重噪声 / 暴露隐藏态）。

**两阶段架构（均可续跑、可分批）**：
- **Phase A（vLLM）**：`prs.ase.run_vllm_phase` 批量生成 clean+R+SE，按 chunk 写 `raw_runs/{id}.partial.json`。
- **Phase B（HF）**：现有 `prs.ase.run_ase_experiment --mode all --resume` 读取 partial，**只补 weight 分支**，再算 metrics → 完整 `raw_runs/{id}.json`。
- 编排脚本：`scripts/run_maintable_vllm.sh`（`SKIP_VLLM=1` 只跑 B，`SKIP_HF=1` 只跑 A）。

**保真度说明（写进 limitation/附录）**：vLLM 仅返回 top-k logprobs，非全词表分布。因此：
- **精确**：选中 token logprob、top-k pairs、top-2 margin、rank → **LL / DeepConf / Self-Certainty 与 HF 一致**；答案类指标（F_resp / U_Ecc / U_Deg / SE）完全不受影响。
- **近似**：全词表预测熵用 top-k 重归一化熵代替 → **PE / SAR / T-branch ATU 为近似**。
- **不受影响**：weight 分支走 HF，故 **S_ans / S_tr（PRS 核心）完全精确**。

**预期加速**：13/17≈76% 解码可批量化，全管线含 HF 前向约 **1.7–2.6×**（视 batch 与 vLLM 吞吐）。

| 方案 | 数学 A100-h | 逻辑 A100-h | 合计 |
|------|----:|----:|----:|
| 纯 HF（定稿） | ≈1826 | ≈391 | ≈2217 |
| vLLM 混合（保守 1.7×） | ≈1070 | ≈230 | ≈1300 |
| vLLM 混合（中等 2.2×） | ≈830 | ≈178 | ≈1010 |
| vLLM 混合 + 8B 单 seed | ≈550–700 | ≈178 | **≈750–900** |

> 安装：`pip install vllm`（按本机 CUDA/torch 版本匹配）。未安装时 HF 管线照常工作。

### 12.7 预算报价（报销用）

卡时单价：**RTX 5090-32GB ¥2.88/h**、**A100-40GB ¥3.28/h**。

**保守口径**（往高报，可辩护）：
- 不计 vLLM 不确定加速时按纯 HF；计 vLLM 时取**最保守 1.7× 加速**；
- A100-40GB 带宽比 80GB 低约 20% → 卡时 ×1.2；
- 失败重跑 / 调参 / reasoning 长尾余量 ×1.3。

| 方案 | 基准卡时(A100/5090) | A100-40GB 预算 | 5090-32GB 预算 |
|------|------|---:|---:|
| 纯 HF（全部跑完） | 2150 / 2420 | 3354 h → **¥11,000** | 3146 h → **¥9,100** |
| **vLLM 混合（保守 1.7×）** | 1300 / 1460 | 2028 h → **¥6,700** | 1900 h → **¥5,500** |
| vLLM 混合 + 8B 单 seed | ≈850 / 960 | ≈1326 h → **¥4,350** | ≈1250 h → **¥3,600** |

**建议报销额**：
- 走 vLLM 混合（推荐）：**A100 报 ¥7,000 / 5090 报 ¥5,500**；想要不超封顶 → 统一 **¥7,500**。
- 若不确定能否用 vLLM、要最稳：按纯 HF 报 **¥12,000** 封顶。

> 实际执行用 vLLM 混合，省下即结余。CPU 消融（E4/E5/E6）不占 GPU、不计费。
