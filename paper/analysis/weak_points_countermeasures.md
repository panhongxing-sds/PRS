# PANDA/PANDA 论文薄弱项深度分析与对策

> **生成日期**：2026-07-02  
> **数据基础**：`cpu_results.{json,md}`、`mechanism_results.json`、`ablation_macro.json`、`panda_v2_results.json`、`process_signal_mining.{json,md}`、`experiments/spurious_consensus/CPU_ANALYSIS_RESULTS.md`  
> **补充分析**：本报告撰写时运行的 Python 复分析（bd0_wrong 分解、SC-proxy vs bd、per-model weight acc、bd=0 子集 SE vs PANDA）

---

## 执行摘要

| 薄弱项 | 严重度 | 推荐路径 | 预估成本 |
|--------|--------|----------|----------|
| 1. Hesitation 偏弱 | 中 | A+B（写作降级 + bd=0 子集诚实报告） | CPU，0 GPU |
| 2. Collapse 子集增益未闭环 | 高 | A+B（动机用 SC 实验、主结果诚实 scope） | 0 GPU |
| 3. 机制信号（T0）偏弱 | 中 | A+B（改叙事为 bd 正交补集） | 0 GPU |
| 4. Qwen3-8B weight 扰动 | 低-中 | A+B（分模型/分数据集披露） | 0 GPU |
| 5. Macro SE 很强 | 中 | A+B（子集分解 + 公平预算叙事） | 0 GPU |
| 6. 动机-方法逻辑缺口 (P0) | **最高** | A+B（SC-proxy 对比图 + 改写 Intro） | CPU 1h |

**跨项协同**：将 hesitation 从「核心贡献」降为「bd 补集上的轻量过程正则」，同时用 SC failure mode（Item 2+6）作动机、用 perturbation **类型**（text vs weight vs soft bd）差异作方法卖点，可一次性缓解 Item 1/2/3/5/6。

---

## 补充分析结果（本次运行）

### bd0_wrong 分解（n=96，cache-exact）

| 维度 | 分布 |
|------|------|
| **定义** | bd=0 且 label_wrong（所有扰动 run 给出同一错误答案） |
| **按模型** | qwen25_3b: 38, llama32_1b: 28, llama31_8b: 24, qwen3_8b: 6 |
| **按数据集** | gsm8k: 39, math500: 34, minerva: 23 |
| **hesitation** | wrong mean=0.193 vs correct mean=0.152（Δ=+0.041）；hes-only AUROC=**0.626** |
| **AUROC NaN 原因** | 子集内标签全为 wrong=1，无法算 ranking metric；应改为 bd0 内 wrong vs correct 对比 |

### bd=0 子集 LODO AUROC（N=2960，错率 3.2%）

| 方法 | AUROC |
|------|------:|
| bd / PANDA（等价，bd=0 时 bd 无方差） | 0.623 |
| hesitation (T_ent_prox_lin) | 0.623 |
| **SE (baseline_SE_H)** | **0.708** |
| gsm8k bd=0 上 hes | 0.505（近随机） |
| math500 bd=0 上 hes | 0.720 |

### SC-proxy vs PANDA dissent（N=8121，CPU）

| 信号 | AUROC | 与 bd_full 相关 |
|------|------:|----------------:|
| SC-proxy（text_answers 的 1−p_top） | 0.842 | ρ=0.809 |
| bd_text（仅文本轴） | 0.902 | — |
| bd_weight（仅权重轴） | 0.836 | — |
| bd_full（R+W 合并） | 0.922 | — |

**关键**：bd=0 时 SC-proxy 恒为 0（100%）；与 bd=0 定义一致——二者在「硬答案一致」子集上同时失明。

### Per-model weight 准确率

| 模型 | base | weight | mean bd_weight | mean bd_text |
|------|-----:|-------:|---------------:|-------------:|
| qwen25_3b | 72.5% | 83.9% | 0.181 | 0.354 |
| llama32_1b | 38.2% | 65.5% | 0.360 | 0.683 |
| llama31_8b | 65.1% | 82.9% | 0.236 | 0.436 |
| **qwen3_8b** | **60.6%** | **62.9%** | **0.425** | **0.518** |

qwen3_8b 分数据集：minerva base=30.2%/wt=34.8%；math500 74.4%/71.1%；gsm8k 72.8%/74.4%。

### se_consistent_wrong 状态

当前 summary 扫描：**n=0**（`TW_ASE_H_norm==0` 的错题不存在）。此前 TODO 中 n≈96 来自 **bd0_wrong** 的混淆。论文中应统一术语，不再声称「SE 一致但 perturbation 无 dissent」子集。

---

### Item 1: Hesitation 在标题/方法中偏强，但消融贡献弱

**Why it's weak (data evidence with numbers)**

- 全量 macro LODO：w/o Hesitation AUROC **0.897** vs full **0.904**（Δ=**−0.007**）；w/o Dissent **0.704**（Δ=**−0.200**）。Hesitation 边际贡献约为 Dissent 的 **3.5%**。
- w/o Prox（仅去 hesitation 通道）0.896（Δ=−0.009），与 w/o Hesitation 几乎等价。
- bd 与 hesitation Spearman **ρ=0.429**——中等相关，非独立机制。
- **bd=0 子集**（36% 样本）：SE **0.708** > PANDA/hes **0.623**；hes 在 gsm8k bd=0 上仅 **0.505**。
- bd0_wrong（n=96）：hes 可在 bd0 内区分 wrong/correct，AUROC=**0.626**，但绝对增益有限。
- `process_signal_mining`：当前 `T_ent_prox_lin` macro LODO=**0.704**，远低于 bd（0.897）；互补候选 `W_n_tokens_avg`+bd 可达 **0.907**（+0.003 vs PANDA），但需改方法。

**Reviewer attack scenario**

> 「摘要把 *hesitation* 与 *dissent* 并列，但消融显示去掉 hesitation 几乎不影响（−0.7pt），去掉 dissent 崩溃（−20pt）。你们是否用 process 叙事包装了一个 weak feature？bd=0 上 SE 还比你们好。」

**Countermeasure options (ranked)**

- **A: Writing/reframing only**
  - 摘要/标题：改为强调 **perturbation-induced dissent (bd)** + **dual-axis ensemble**；hesitation 降为「trace-level complement when answer-level signals tie」。
  - 建议句：*"Answer drift (bd) carries most of the gain (−20.0 AUROC when removed); trace hesitation adds a modest +0.7 AUROC, mainly on the bd=0 tie-set where SE remains competitive (0.708 vs 0.623)."*
  - 消融表脚注：标注 bd=0 占 36.4%，解释 bd 常数导致 PANDA≈hes。
- **B: Re-analysis of existing data (CPU)**
  - 报告 bd0 内 hes vs SE 的 **paired comparison**（已有：hes 0.626, SE 0.708 on bd=0 with SE join n=1452）。
  - 从 `process_signal_mining.json` 引用：`T_formula_skeleton_entropy` bd=0 AUROC=**0.814**（ρ_bd=0.286）作为「可替代 hesitation 的候选」放在 Discussion/Limitation，**不改主方法**。
  - 输出一张 **subset ablation 表**：all / bd>0 / bd=0 三行，展示 dissent vs hesitation 贡献不对称。
- **C: New GPU experiments**
  - 仅当审稿人要求替换 hesitation：**不建议** submission 前改公式；`W_n_tokens_avg`+bd 的 +0.003 不值得重跑主表。

**Recommended path**

**A + B**：写作降级 hesitation + 子集诚实表；不改主方法。引用 process_signal_mining 作为 future work。

---

### Item 2: PANDA 在 collapse 子集上的增益未闭环

**Why it's weak (data evidence with numbers)**

- SC 实验（K=64）：deepscaler/gpqa/aime，6 模型，N=2200；Qwen-7B SCR@0.9 占错题 **10.8%**，n≥8 后 SCR **平台**（11.5% @ n=64）。
- PANDA 主实验：minerva/math500/gsm8k，4 模型，N=8121——**数据集与模型均不对齐**。
- `se_consistent_wrong` 在 PANDA 数据中 **n=0**（非 bd0_wrong 的 96）；无法直接定义「SC collapse ∩ PANDA 可检测」子集。
- bd0_wrong n=96（2.5% of all wrong）：PANDA 在 bd=0 上 AUROC=0.623，**不优于 SE 0.708**。
- SC 侧：p_top≥0.9 子集可靠率 Qwen-7B 仅 **82.5%**；PANDA 侧：无等价 p_top 字段。

**Reviewer attack scenario**

> 「你们用 SC 实验论证 collapse，但从未在同一批题上跑 PANDA。96 题 bd0_wrong 上 AUROC 无法计算且 bd=0 子集 SE 更强——凭什么说 PANDA 解决了 self-consistency collapse？」

**Countermeasure options (ranked)**

- **A: Writing/reframing only**
  - **拆成两个 claim**：（i）SC 实验 **诊断** collapse 存在（failure mode，§4.2）；（ii）PANDA **缓解** answer-level UQ 但对 bd=0 tie-set 有限（limitation）。
  - 建议段：*"We do not claim PANDA eliminates spurious consensus; we show SC hits an irreducible SCR plateau (§4.2) while PANDA improves single-pass UQ via cross-axis perturbation. Direct evaluation on SCR@τ questions requires matched sampling and is left to future work."*
  - 复用 `experiments/spurious_consensus/figures/fig_main.png`, `nsweep_irreducible.png`（GitHub 已有，0 GPU）。
- **B: Re-analysis of existing data (CPU)**
  - 定义 **operational analog**：bd0_wrong（n=96）≈「perturbation-consistent wrong」；报告 hes AUROC=0.626、与 SC-proxy 在 bd=0 上同样为 0 的 **structural analogy**。
  - Matched-budget 叙事：SC@8 ≈ 8 decodes vs PANDA ~9（1+8 perturb），引用 `CPU_ANALYSIS_RESULTS.md` §4。
  - 在 bd0_wrong 上报告 **non-AUROC metrics**：mean hes、按模型/数据集分解（见补充分析表）。
- **C: New GPU experiments**
  - **最小 scope（仅当被拒后）**：在 SC 共同题 N=2200 上，对 **1 个模型**（Qwen-7B 或最接近的 qwen3_8b）跑 PANDA pipeline，报告 SCR@0.9 子集 AUROC。
  - **Submission 前不建议**：成本高、模型不完全匹配；用 A+B 足够防御。

**Recommended path**

**A + B**：SC 作 motivation/failure mode；PANDA 主表保持 math 三数据集；Limitations 明确 bd0_wrong 与 SCR 的操作性对应及 n=96 样本限制。

---

### Item 3: 机制信号（temporal / T0）偏弱

**Why it's weak (data evidence with numbers)**

- `mechanism_results.json`：`auroc_T0_to_split` = **0.501**（≈随机）；`auroc_T_full` = 0.539。
- 条件子集：`auroc_T0` all=0.517, bd_eq_0=0.571, bd_gt_0=0.538——均接近随机。
- Spearman：`T0~bd` = **0.012**, `T0~y` = 0.029——T0 与 dissent/标签几乎无关。
- **对照**：`auroc_F_to_split` = **0.962**；quadrant 中 bd 维度强：`T_low_bd_high` wrong_rate=**0.753** vs `T_low_bd_low` **0.096**。
- Ladder：`L2_dc_T0_bd` = 0.903 vs `L0_dc_min` = 0.676；`residual_bd_given_T0` = **0.917**。
- `process_signal_mining`：temporal 类信号 macro LODO 最高约 0.77（`T_math_token_flip_mean`），低于 bd 0.897。

**Reviewer attack scenario**

> 「Prefix re-query / temporal hesitation 故事不成立——T0 预测 split AUROC 0.50。机制分析是否 post-hoc？」

**Countermeasure options (ranked)**

- **A: Writing/reframing only**
  - 将机制 claim 从「T0 预测未来失败」改为「**bd 与 local token competition 正交**；F（fragmentation）可预测 split，T0 单独不行」。
  - 建议句：*"Temporal proximity to the answer span (T0) alone does not predict failure (AUROC≈0.50), but answer drift bd remains predictive even conditional on T0 (residual AUROC 0.917). Quadrant analysis shows wrong answers concentrate in high-bd regions regardless of T0 level."*
  - 用 quadrant 图（2×2 T0×bd）替代 temporal causality 图作为主机制图。
- **B: Re-analysis of existing data (CPU)**
  - 已有 `mechanism_results.json` 的 ladder + residual + quadrants——整理成一张 **mechanism summary table** 放入 appendix。
  - 报告 heterogeneity：`mean_bd_text`=0.440 vs `mean_bd_weight`=0.260；text 轴贡献更大，与 ablation w/o Text (−0.042) > w/o Weight (−0.034) 一致。
- **C: New GPU experiments**
  - **不需要**，除非审稿人要求 interventional prefix re-query；现有 predict_split 已是 observational 上限。

**Recommended path**

**A + B**：弱化 T0 causal narrative，强化 bd⊥T0 + F 预测 split + residual analysis。Intro 中「provisional answer shifts then fragmentation」改为 **hypothesis/exploratory** 语气。

---

### Item 4: Qwen3-8B weight 扰动准确率低，「扰动不破坏生成」叙事弱

**Why it's weak (data evidence with numbers)**

- 全局：qwen3_8b weight acc **62.9%** vs base **60.6%**（Δ=+2.3pt）；对比 qwen25_3b **83.9%** vs 72.5%。
- qwen3 minerva：**30.2%→34.8%**（最难子集，weight 扰动几乎不保持答案）。
- mean bd_weight=**0.425** vs qwen25 **0.181**——weight 轴 dissent 高但可能含 **noise 而非 signal**。
- 消融仍显示 w/o Weight −**0.034** macro（四模型混合）；qwen3 单模型 ab_full−ab_no_T 增益仍主要来自 bd。
- base wrong 时 weight acc 未显著高于 base ok（需按模型条件报告）。

**Reviewer attack scenario**

> 「Weight perturbation 声称探测 epistemic instability，但 Qwen3-8B 上 63% 答案保持率——扰动主要在破坏生成，而非探测 reliability。Low-rank noise 是否 meaningful？」

**Countermeasure options (ranked)**

- **A: Writing/reframing only**
  - 改为 **ensemble diversity** 叙事：weight 轴提供与 text 轴 **低相关** 的扰动（ρ_bd_text vs bd_weight 分离报告）；不要求「保持原答案」。
  - 建议句：*"Weight perturbations need not preserve the base answer; they expose an independent instability axis (mean bd_weight=0.43 for Qwen3-8B vs 0.18 for Qwen2.5-3B). Gains on Qwen3-8B are driven primarily by text rephrasing and answer-drift scoring."*
  - Impl details：披露 σ、r'、扰动层；说明 Qwen3 对 weight noise 更敏感。
- **B: Re-analysis of existing data (CPU)**
  - 主文 **per-model 小表**：base / weight acc / mean bd_text / mean bd_weight（已有）。
  - 分数据集 breakdown（minerva 弱、gsm8k 强）解释 heterogeneity。
  - 报告 w/o Text vs w/o Weight 消融 + qwen3 上 bd≈PANDA（hes 贡献小）。
- **C: New GPU experiments**
  - 仅 qwen3：**σ 敏感性**（0.01/0.03/0.05）或减少 W——ROI 低，放 rebuttal。
  - **不建议** submission 前做 Q/K vs V/O 对比（GPU_PLAN 已剔除）。

**Recommended path**

**A + B**：分模型披露 + 改「preserve generation」为「independent axis」；主结论不过度依赖 weight 轴。

---

### Item 5: Macro SE 很强，整体增益看似边际

**Why it's weak (data evidence with numbers)**

- Collapse LODO all：PANDA **0.904** vs SE **0.890**（Δ=**+0.014**）；bd alone **0.897**。
- prs_v2 macro mean：Self-Certainty **0.626**, DeepConf 0.692, PANDA **0.867**, F_resp 0.830。
- **子集分化**：bd>0 上 bd AUROC **0.859**；bd=0 上 SE **0.708** > PANDA **0.623**。
- w/o Fusion **0.877** < SE 0.890——说明 **bd+hes 线性融合** 必要，但 fusion 本身不是 learned（z-score sum）。
- 36 cell mean vs macro LODO 可能有差异——以 LODO 为准更 conservative。

**Reviewer attack scenario**

> 「+1.4pt over semantic entropy isn't exciting. bd≈PANDA（0.867 vs 0.867 cell mean）。你们是否只是 rebrand SE + extra decode？」

**Countermeasure options (ranked)**

- **A: Writing/reframing only**
  - 强调 **(i) 公平预算**（~9 decodes vs SC@8）；（ii）**子集互补**——PANDA 在 bd>0（64% wrong-rate 63%）主导，SE 在 bd=0 tie-set 更好；（iii）**vs TokUR** 仍是大增益（tex 中 +17.5pt）。
  - 建议句：*"PANDA does not uniformly dominate SE: on perturbation-consistent errors (bd=0, 36% of items), SE remains stronger (0.708 vs 0.623). Overall gains (+1.4pt LODO) come from detecting soft answer drift (bd>0) that entropy over final answers misses."*
  - 突出 w/o Dissent 0.704 vs w/o Hesitation 0.897——**bd/soft dissent 是 SE 没有的**。
- **B: Re-analysis of existing data (CPU)**
  - 产出 **2×2 表**：{all, bd=0, bd>0} × {SE, bd, PANDA, SC-proxy}。
  - SC-proxy AUROC 0.842 < bd_full 0.922——说明 **soft bd** 优于 hard SC。
  - 引用 w/o Fusion 0.877 说明 multi-signal 必要。
- **C: New GPU experiments**
  - 不需要；可选 bootstrap CI on Δ=0.014（CPU）。

**Recommended path**

**A + B**：Honest subset table + soft-vs-hard dissent + TokUR 主对比；不把 +1.4pt 作 headline，作「complementary to SE」。

---

### Item 6: 动机-方法逻辑缺口 (P0) — SC dissent vs perturbation dissent

**Why it's weak (data evidence with numbers)**

- SC（answer-level 1−p_top）与 PANDA bd（soft answer drift）在概念上 **重叠**；CPU 显示 ρ(sc_text, bd_text)=**0.846**，AUROC sc_proxy **0.842** vs bd_text **0.902**。
- bd=0 ⟺ sc_text=0（100%）——**同一 collapse 结构**。
- 关键图 **SC vs perturbation dissent** 未做（`CPU_ANALYSIS_RESULTS.md` §11 标记 ❌）。
- Intro 批评 SC 但 PANDA 的 F_resp/s_mode 也是 answer-level fragmentation——需区分 **hard vs soft** 与 **cross-axis**。

**Reviewer attack scenario**

> 「你们反对 answer-level SC，但 bd 本质上也是 ensemble disagreement。为何不直接多采样？SC@8 已达 AUROC 0.77 且与 PANDA 预算相当。」

**Countermeasure options (ranked)**

- **A: Writing/reframing only**
  - **三层区分**：（1）SC = same-model temperature，same prompt family；（2）text perturb = semantic rephrase；（3）bd = **token-level mass on competing answers**, not just vote count.
  - 建议 Intro 改写：*"Self-consistency measures hard vote fragmentation (1−p_top); PANDA adds (i) cross-axis perturbations, (ii) soft probability drift bd even when votes agree, and (iii) weight-space probes unreachable by sampling alone."*
  - Fairness 段：PANDA ~9 decodes vs SC@8；SC 不能探测 weight epistemics 或 rephrase robustness。
- **B: Re-analysis of existing data (CPU)** — **P0 必做**
  - **Figure: SC-proxy vs bd scatter / AUROC bars**（本次已算：sc 0.842, bd_text 0.902, bd_full 0.922）。
  - **2×2 panel**：（a）SC high & bd low（collapse）；（b）SC low & bd high（soft drift，PANDA 独有）。
  - 用现有 summary.jsonl 的 text_answers 作 SC-proxy；**无需 SC K=64 raw**。
  - 可选：bd_text − sc_text 残差 AUROC（CPU，~30s）。
- **C: New GPU experiments**
  - 理想图：SC K=64 与 PANDA 同一题 overlay——需新采样或 matched 题集。
  - **Submission 最小**：B 的 proxy 图 + SC failure mode 引用；C 仅 rebuttal。

**Recommended path**

**A + B（必须）**：Intro/§4.2 逻辑改写 + CPU 生成 SC-proxy vs bd 对比图。这是 **blocking writing task**，0 GPU。

---

## 跨项协同策略

| 协同 | 涉及 Item | 做法 |
|------|-----------|------|
| **Demote hesitation, elevate perturbation taxonomy** | 1, 3, 6 | 方法叙事中心改为 dual-axis + soft bd；hes 降为 tie-set complement |
| **SC failure mode → honest scope** | 2, 5, 6 | SC 实验只证 collapse 存在；PANDA 证 soft drift + 不完全解决 bd=0 |
| **Subset-conditioned claims** | 1, 2, 5 | 统一 bd=0 / bd>0 分解，避免 macro 平均掩盖 |
| **Mechanism: orthogonal signals** | 3, 6 | T0 弱 + bd 强 + F 预测 split → 「multi-facet instability」 |
| **Per-model heterogeneity** | 4, 5 | qwen3 weight 弱不影响 macro story；text+bd 仍 0.947 |

---

## Submission 前修复优先级

| 优先级 | 任务 | 类型 | 预估 |
|--------|------|------|------|
| **P0** | Item 6：SC-proxy vs bd 图 + Intro/§4.2 动机改写 | A+B CPU | 2–4 h 写作 |
| **P1** | Item 2：SC failure mode 段落 + 图（GitHub 已有） | A | 1–2 h |
| **P1** | Item 5：subset 表（all/bd=0/bd>0 × methods） | B CPU | 30 min |
| **P2** | Item 1：hesitation 降级 + 消融脚注 | A | 1 h |
| **P2** | Item 3：mechanism 改 quadrant/residual 叙事 | A+B | 1 h |
| **P3** | Item 4：qwen3 per-model 披露 | A+B | 30 min |
| **GPU** | TokUR EU 主表（非本报告范围，但是 blocking） | GPU | 见 GPU_PLAN |

**不建议 submission 前做**：DeepScaleR 上跑 PANDA、替换 hesitation 公式、σ 敏感性、SC vs PANDA 同一 raw 对比（除非 rebuttal）。

---

## 可引用的关键数字速查

```
Macro LODO:  PANDA=0.904  bd=0.897  hes=0.704  SE=0.890
Ablation:    w/o Dissent=0.704  w/o Hesitation=0.897  w/o Text=0.862  w/o Weight=0.871
bd=0 LODO:   PANDA=0.623  SE=0.708  hes=0.623  (n=2960)
bd0_wrong:   n=96  hes AUC(vs bd0 correct)=0.626
SC-proxy:    AUROC=0.842  bd_text=0.902  bd_full=0.922  ρ(sc,bd_text)=0.846
Mechanism:   T0→split=0.501  F→split=0.962  residual_bd|T0=0.917
Qwen3-8B:    wt_acc=62.9%  base=60.6%  bd_weight=0.425
SC (7B):     SCR@0.9=10.8%  AUROC@64=0.772  plateau n≥8
```

---

## 附录：建议新增 CPU 脚本（可选）

```bash
# 生成 subset 表 + SC-proxy 图数据
python scripts/run_cpu_paper_analyses.py  # 已有
# 建议扩展：--export-subset-table --sc-proxy-fig
```

扩展 `run_cpu_paper_analyses.py` 输出：
- `paper/analysis/subset_comparison.json`
- `paper/figures/sc_proxy_vs_bd.json`

---

*报告版本：v1.0 | 分析脚本运行于 2026-07-02*
