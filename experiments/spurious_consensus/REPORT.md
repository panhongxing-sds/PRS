# 虚假共识（Spurious Consensus）实验报告

**主题**：多采样（self-consistency）能改善 answer-level 不确定性量化（UQ），但**无法消除高置信度的稳定错误**——一类"自信地错"的系统性盲区。

**日期**：2026-06-26  ·  **数据**：6 个开源模型 × 3 个推理基准 × 每题 K=64 采样（共同题 N=2228）

---

## 0. 一句话结论（TL;DR）

> 在数学/科学推理任务上，对答案做多采样投票（self-consistency）确实让"答案对不对"的不确定性信号（`1 − p_top`）更可判别（AUROC 从 ~0.66 升到 ~0.78），但有一类题——模型**反复给出同一个错误答案、且高度一致**——这类"虚假共识"**不会随采样数增加而消失**，构成一个**不可约的下界**。更反直觉的是：**模型越强，这类自信错误占比越高**（0.5B 0.8% → 7B 10.5%）；这些错误**跨模型共享**；且对 7B SCR@1.0 子集，**64 条 reasoning/token 路径可完全不同却仍同错**——虚假共识是答案级现象，不是复制粘贴。

---

## 1. 研究问题与命题

**核心命题**：self-consistency 提升聚合层面的 UQ 指标，却留下一类无法被采样消除的"高置信度稳定错误"（spurious consensus）。这类错误是 selective prediction（选择性回答 / 拒答）的根本障碍。

具体研究问题：

- **RQ1**：随采样数 `n` 增加，answer-level UQ 的判别力（AUROC of `1 − p_top`）如何变化？
- **RQ2**：高置信度错误率 `SCR@τ`（majority 判错且 `p_top ≥ τ`）随 `n` 如何变化？是否存在不可约下界？
- **RQ3**：哪些题最容易产生 spurious consensus？是否与"题目难度 × 模型能力"相关？
- **RQ4**：spurious consensus 是单模型特有的偏差，还是题目内禀的、可跨模型迁移的现象？

---

## 2. 实验设置

### 2.1 模型矩阵（构成一条能力梯度）

| Tag | 模型 | 参数 | 后端 | 采样完成 | majority@64（共同题） |
|-----|------|------|------|:--------:|----------------------:|
| `qwen25_05b` | TFB-Qwen2.5-0.5B-Instruct | 0.5B | vLLM | ✅ 2228/2228 | **25.6%** |
| `llama32_1b` | TFB-Llama-3.2-1B-Instruct | 1B | vLLM | ✅ 2228/2228 | **29.6%** |
| `qwen25_15b` | TFB-Qwen2.5-1.5B-Instruct | 1.5B | vLLM | ✅ 2228/2228 | **37.8%** |
| `phi4_mini` | Phi-4-mini-instruct | 3.8B | vLLM | ✅ 2228/2228 | **39.4%** |
| `qwen25_3b` | TFB-Qwen2.5-3B-Instruct | 3B | vLLM | ✅ 2228/2228 | **45.3%** |
| `qwen25_7b` | TFB-Qwen2.5-7B-Instruct | 7B | vLLM | ✅ 2228/2228 | **52.1%** |

> 注：gemma-3-4b 因吞吐过低被移除。"TFB-"前缀模型在采样前由 `sampling_utils.py` 自动剥离非标准权重（`basis_vectors` 等）并改写 config，以标准架构在 vLLM 上加载。§4.1–4.2 的 n-sweep 仍基于早期 **3 模型清洗后**子集（Llama / Phi / Qwen-3B）；§4.7 起扩展为 **6 模型 stored correct** 全量对比。

### 2.2 数据集（每模型 ~2228 题）

| Benchmark | 题数 | 判分 | 说明 |
|-----------|-----:|------|------|
| deepscaler | 2000 | math | DeepScaleR 子集（由 3000 裁剪） |
| gpqa_diamond | 198 | mcq | GPQA Diamond（研究生级科学选择题） |
| aime_2024 | 30 | math | AIME 2024（竞赛数学） |

> 三个数据集是三模型都跑齐、可跨模型比较的"主实验集"。competition_math / math_level4plus / minerva 因模型覆盖不全（仅部分模型采样），不纳入主实验。

### 2.3 采样协议

- 每题 **K=64** 个独立样本，`temperature=0.5`，`top_p=0.95`，`seed=41`（单 seed）。
- 采样后用 PANDA 的 `extract_math_answer` / boxed / MCQ 规则抽取答案，`math_equal` 判分。

### 2.4 数据清洗（`clean_samples.py`）

小模型输出不稳定，抽取阶段会混入垃圾（整段推理被误当答案、截断的 LaTeX、非法选项等）。清洗流程：

1. **垃圾识别**：长度 > 180、反斜杠 > 20、命中 `## Step` / 截断 `\frac{`/`\boxed{` 等模式 → 判为垃圾，置空。
2. **规范化**：`canonicalize_answer` 合并格式等价答案（如 `1/2` 与 `\frac12`），使共识统计不被表面格式割裂。
3. **重判分**：用 `math_equal_clean` 重新判定 correct。
4. **质量门控**：有效答案 < 32 或垃圾占比 > 35% 的题打 `label_drop=1`，分析时剔除。

原始数据备份为 `*.jsonl.bak`，清洗结果写回原路径。

---

## 3. 指标定义（方法学）

对每题的 n 个样本：

- **`p_top`**：出现频率最高的答案（majority）所占比例。`p_top ∈ [1/n, 1]`，越高表示越"自信/一致"。
- **`u_ans = 1 − p_top`**：answer-level 不确定性。作为"答案可能错"的预警分数。
- **majority 判错（wrong）**：majority 答案与标准答案不等。
- **`SCR@τ`（Spurious Consensus Rate）**：在判错的题中，`p_top ≥ τ` 的比例（即"自信地错"占错题的比例）。主用 τ=0.9。
- **AUROC of `u_ans`**：以"majority 是否判错"为标签、`u_ans` 为分数的 ROC-AUC，衡量 UQ 信号能否区分对/错题。
- **外生难度（exogenous difficulty）**：用**其他模型**在该题的平均逐样本正确率作为难度锚（leave-one-out），独立于被评估模型自身，避免循环定义。值越高 = 越容易。
- **n-sweep**：对每题从 K=64 个样本中**有放回**抽 `n ∈ {2,4,8,16,32,64}` 个，bootstrap 重复多次取均值，模拟"采样预算为 n"时的指标——免重复采样，自带误差。
- **budget-stable-wrong**：majority 在**所有** n 预算下都判错的题，占 wrong@64 的比例（单 seed 下对"稳定错误"的近似度量）。

---

## 4. 主要结果

### 4.1 发现 F1 — 多采样改善 UQ，但很快饱和、甚至略降

`AUROC(1 − p_top)` 随 n 的变化（见 `figures/nsweep_irreducible.png` 左图）：

| n | Llama-1B | Phi-3.8B | Qwen-3B |
|---|---------|---------|---------|
| 2 | 0.656 | 0.668 | 0.718 |
| 4 | 0.756 | 0.730 | 0.771 |
| 8 | 0.778 | 0.737 | 0.772 |
| 16 | 0.781 | 0.734 | 0.771 |
| 32 | 0.774 | 0.727 | 0.765 |
| 64 | 0.765 | 0.723 | 0.761 |

→ 从 n=2 到 n=8 提升明显，**之后饱和，n>16 后甚至轻微下降**。说明"多采样改善 UQ"是有上限的，单纯堆采样预算的边际收益迅速归零。

### 4.2 发现 F2（核心）— 虚假共识存在不可约下界，多采样无法消除

`SCR@0.9`（占当下 n 的错题比例）随 n 的变化（见 `figures/nsweep_irreducible.png` 右图）：

| n | Llama-1B | Phi-3.8B | Qwen-3B |
|---|---------|---------|---------|
| 2 | 9.0% | 18.8% | 25.1% |
| 4 | 1.6% | 6.5% | 10.8% |
| 8 | 0.3% | 2.8% | 6.1% |
| 16 | 0.2% | 2.9% | 6.4% |
| 32 | 0.2% | 2.9% | 6.5% |
| 64 | 0.1% | **2.6%** | **6.1%** |

两个关键现象：

1. **强模型（Phi/Qwen）的 SCR 在 n≥8 后停止下降，进入平台**（Phi ≈ 2.6–2.9%，Qwen ≈ 6.0–6.5%）。继续加采样到 64 也压不下去——这就是**不可约下界**。
2. **弱模型（Llama）的 SCR 持续衰减趋近 0**（0.1%）。

> **解读**：弱模型的错误是"噪声型"——分散、不稳定，多采样投票能洗掉；强模型的错误是"偏差型"——稳定、一致，多采样只会让错误共识更稳固。**self-consistency 能治噪声，治不了偏差。**

补充证据：**budget-stable-wrong** 占 wrong@64 的比例 = Llama 99.3% / Phi 98.9% / Qwen 98.3%。即几乎所有最终错误，在任何采样预算下都是错的——错误几乎不随预算波动。

### 4.3 发现 F3 — 模型越强，虚假共识占比越高

在六模型共同题（N=2228，stored correct，剔除 `label_drop`）上，SCR 沿能力轴**严格单调上升**：

| 模型 | majority@64 | AUROC | SCR@0.9 | SCR@0.95 | SCR@1.0 | p≥0.9 可靠率 |
|------|------------:|------:|--------:|---------:|--------:|-------------:|
| Qwen-0.5B | 25.6% | 0.774 | 11 (**0.8%**错) | 8 (0.6%) | 2 (0.2%) | 86.2% |
| Llama-1B | 29.6% | 0.739 | 1 (**0.1%**错) | 0 | 0 | 97.6% |
| Qwen-1.5B | 37.8% | 0.773 | 31 (**2.6%**错) | 19 (1.6%) | 5 (0.4%) | 85.7% |
| Phi-3.8B | 39.4% | 0.713 | 35 (**2.6%**错) | 21 (1.6%) | 4 (0.3%) | 79.3% |
| Qwen-3B | 45.3% | 0.758 | 71 (**5.8%**错) | 51 (4.2%) | 22 (1.8%) | 82.4% |
| **Qwen-7B** | **52.1%** | **0.776** | **88 (10.5%错)** | **67 (8.0%)** | **42 (5.0%)** | **82.6%** |

括号内为 SCR 占该模型**错题**比例。早期 3 模型清洗后子集（见 `figures/capability_vs_scr.png`）趋势一致：Llama 0.1% → Phi 2.6% → Qwen-3B 5.8%。

→ 能力提升不是把错误消除，而是把"可检测的分散错误"转化成"不可检测的自信错误"。**7B 在 p≥0.9 时覆盖近 29% 的题，但可靠率仅 ~83%**——高置信筛选仍不可靠。

### 4.4 发现 F4 — 虚假共识是跨模型共享的"题目陷阱"

在三模型共同题（N=2218）上：

- **Phi ∩ Qwen** 同时 SCR 的有 **9 题，其中 8 题错答逐字相同**。
- Qwen 的 71 道 SCR 题中：**Phi 有 62 道也判错、39 道给出与 Qwen 相同的错答**；Llama 有 56 道也错、25 道同一错答。

→ 不同架构、不同厂商的模型在同一批题上被吸引到**同一个错误答案**，远超随机一致。说明这类错误不主要在模型权重里，而在**题目结构 ×（人类先验式）推理捷径**里——存在"模型无关的陷阱题"。

### 4.5 发现 F5 — 置信度在能力边界处系统性失效（校准崩塌）

同样是 `p_top ≥ 0.9` 的"高度自信"，按外生难度分层后的**实际正确率**（见 `figures/confidence_collapse.png`）：

| 模型 | 极难(<10%) | 难(10–25%) | 中(25–45%) | 易(>45%) |
|------|-----------|------------|------------|----------|
| Llama-1B | 0% (n=1) | — | 100% (n=1) | 100% (n=39) |
| Phi-3.8B | **0%** (n=16) | 27% (n=15) | 60% (n=10) | 97% (n=128) |
| Qwen-3B | **25%** (n=51) | 36% (n=22) | 83% (n=54) | 96% (n=276) |

→ 在容易题上"自信 = 可靠"（96–97%），但在边缘/难题带，**同样的自信只有 0–36% 可靠**，置信度与正确率脱钩甚至反向。

> **方法学启示**：一个漂亮的全局 AUROC 会**掩盖**这种局部失效。用单一阈值做拒答的范式，在能力边界带会恰好放行最该拒答的题。**UQ 应分难度分层报告。**

### 4.6 发现 F6 — 存在"危险难度窗口"：自信错误集中在略高于自身能力的题

SCR 错题 vs 普通分散错题（`p_top<0.5`）的外生难度对比（值越高 = 越容易）：

| 模型 | SCR 错题外生难度 | 普通分散错题外生难度 |
|------|-----------------:|---------------------:|
| Phi-3.8B | **0.159** (n=35) | 0.127 (n=1034) |
| Qwen-3B | **0.163** (n=71) | 0.084 (n=807) |

→ 高共识错误反而落在**相对更容易**的题上（别的模型做得更对），而非超纲的最难题。即虚假共识发生在"**够得着、但会栽**"的边缘窗口；模型对超纲题反而错得分散（心虚）。

### 4.7 六模型 deepscaler 子集（N=2000）

| 模型 | majority@64 | SCR@0.9 占错 | SCR@1.0 占错 |
|------|------------:|-------------:|-------------:|
| Qwen-0.5B | 25.6% | 0.4% | 0.0% |
| Llama-1B | 29.8% | 0.1% | 0.0% |
| Qwen-1.5B | 39.4% | 2.2% | 0.5% |
| Phi-3.8B | 40.4% | 2.4% | 0.3% |
| Qwen-3B | 46.9% | 6.2% | 2.1% |
| Qwen-7B | 54.6% | 11.4% | 5.7% |

### 4.8 发现 F7（新增）— SCR 是答案级共识，不是 reasoning/token 级复制粘贴

对 Qwen-7B **SCR@1.0 最严子集**（42 题，64/64 全给同一错答）重采样完整 reasoning + token logprob（`data/scr_reasoning/qwen25_7b/t100/`，每题 K=64）：

**Reasoning 一致性**

| 指标 | 结果 |
|------|------|
| 64/64 reasoning 文本完全相同 | **0/42** |
| 64/64 token id 序列完全相同 | **0/42** |
| 前 200 字符 reasoning 全部相同 | **0/42** |
| 与首条 reasoning 相似度 sim→1st（均值 / 中位） | **0.393 / 0.375** |
| sim < 0.5（路径高度分散） | **67%** 题 |

**Token-level 统计**

| 指标 | 结果 | 含义 |
|------|------|------|
| 共同前缀 token 长度（均值 / 中位） | **6.8 / 1** | 第 1–7 个 token 后即分叉 |
| 前 50 token 位置共识度 | **0.608** | 开头部分共享，很快发散 |
| pairwise 首个分歧位置 | **26.5** token | 分歧极早 |
| 跨 trace logprob 标准差 | **0.167** | 同位置有一定波动 |
| 平均 token logprob | **−0.066** | 整体高置信生成 |
| logprob < −1.0 占比 | **1.5%** | 几乎无「犹豫 token」 |

**机制分型**（42 题）

| 类型 | 题数 | 特征 |
|------|-----:|------|
| A 多路径汇入 | 12 | sim<0.3，token 前缀≤5，典型虚假共识 |
| B 中等同构 | 19 | 0.3≤sim<0.6，部分步骤共享 |
| C 高同构错答 | 8 | sim≥0.6，骨架相似但 unique reasoning 仍 >1 |

**跨模型共享（7B SCR@1.0 的 42 题）**

| 对比模型 | 与 7B 同错答 | 该模型也在 SCR@0.9 |
|----------|------------:|-------------------:|
| Qwen-3B | 38/42 | 17/42 |
| Qwen-1.5B | 27/42 | 13/42 |
| Qwen-0.5B | 14/42 | 4/42 |
| Phi-3.8B | 19/42 | 5/42 |
| Llama-1B | 16/42 | 0/42 |

→ **核心机制结论**：模型可走 64 条不同推理路径、以高置信 token 生成，却收敛到同一错误答案——**过程多样 + 输出一致 + 全程自信**。这解释了 answer-level UQ（`1−p_top`）对 SCR 的系统性失明；token-level 信号（logprob 抖动、路径分歧）**理论上可见**，但当前 trace 整体仍高置信，单独用 logprob 阈值未必能筛净。

> 完整逐题表见 `figures/scr7b_t100_reasoning_analysis.md`；原始 JSON 见 `results/scr7b_t100_reasoning_analysis.json`。

**Qwen-7B SCR 题数按阈值**

| 阈值 | 题数 | 占错题比 | 清单 |
|------|-----:|---------:|------|
| SCR@0.9 | 88 | 10.5% | `figures/qwen25_7b_scr_questions_clean.md` |
| SCR@0.95 | 67 | 8.0% | `figures/qwen25_7b_scr_questions_clean_t095.md` |
| SCR@1.0 | **42** | **5.0%** | `figures/qwen25_7b_scr_questions_clean_t100.md` |

---

## 5. 按 benchmark 分解

| 模型 | benchmark | 题数 | majority 准确率 | SCR@0.9（占错题） | SCR@1.0（占错题） |
|------|-----------|-----:|----------------:|------------------:|------------------:|
| Qwen-0.5B | deepscaler | 2000 | 25.6% | 0.4% | 0.0% |
| | gpqa_diamond | 198 | 29.3% | 5.0% | 1.4% |
| | aime_2024 | 30 | 0.0% | 0.0% | 0.0% |
| Llama-1B | deepscaler | 2000 | 29.8% | 0.1% | 0.0% |
| | gpqa_diamond | 198 | 31.8% | 0.0% | 0.0% |
| | aime_2024 | 30 | 3.3% | 0.0% | 0.0% |
| Qwen-1.5B | deepscaler | 2000 | 39.4% | 2.2% | 0.5% |
| | gpqa_diamond | 198 | 29.3% | 6.4% | 0.0% |
| | aime_2024 | 30 | 6.7% | 0.0% | 0.0% |
| Phi-3.8B | deepscaler | 2000 | 40.4% | 2.4% | 0.3% |
| | gpqa_diamond | 198 | 33.8% | 4.6% | 0.0% |
| | aime_2024 | 30 | 6.7% | 0.0% | 0.0% |
| Qwen-3B | deepscaler | 2000 | 46.9% | 6.2% | 2.1% |
| | gpqa_diamond | 198 | 33.8% | 3.8% | 0.0% |
| | aime_2024 | 30 | 16.7% | 0.0% | 0.0% |
| Qwen-7B | deepscaler | 2000 | 54.6% | 11.4% | 5.7% |
| | gpqa_diamond | 198 | 38.4% | 7.4% | 2.5% |
| | aime_2024 | 30 | 20.0% | 0.0% | 0.0% |

AIME 上 SCR 普遍为 0，与题量小、且模型在竞赛题上错得分散有关（属超纲区，对应 F6）。GPQA（4 选项 MCQ）天然更易出现高 `p_top`，其 SCR 与数学题不完全可比。

---

## 6. 结论与可写入论文的 Claim

1. **C1（核心）**：spurious consensus 对 test-time scaling **免疫**——self-consistency 能压低分散的噪声型错误，但对稳定的偏差型错误存在不可约下界（强模型 SCR 在 n≥8 后平台化）。可靠性提升必须来自**异质性**（跨范式/跨模型分歧、外部符号验证、检索），而非加采样。
2. **C2**：spurious consensus 随模型能力**单调上升**——是能力 scaling 的负向涌现副产品；"更强"在此维度上意味着"更自信地错"。
3. **C3**：这类错误是**题目内禀、跨模型共享的陷阱**（8/9 同时 SCR 的题错答相同），主流推理 benchmark 中混有一批系统性误导题。
4. **C4**：置信度（`p_top`）在能力边界处**系统性失效**；单一全局 AUROC 掩盖这一局部崩塌，UQ 须分难度分层评估。
5. **C5**：spurious consensus 集中在"略高于自身能力"的**危险难度窗口**，这使其可被定向探测（针对该窗口做红队，比全量评测更高效）。
6. **C6（新增）**：SCR 是**答案级**虚假共识，不是 reasoning/token 级复制粘贴——64 条 trace 可高度多样却仍同错；answer-level UQ 失明有明确的生成机制支撑。

---

## 7. 局限性（如实陈述）

- **单 seed**：仅 seed=41，无法计算跨 seed 的 stable-wrong；用 budget-stable-wrong 近似。补多 seed 可使"稳定错误"的证据更硬。
- **难度锚较粗**：外生难度由其他模型 leave-one-out 平均得到，分辨率有限。
- **判分口径混用**：§4.1–4.2 n-sweep 基于 3 模型**清洗后**数据；§4.3/4.7/4.8 六模型表基于 **stored correct**（仅剔除 `label_drop`），跨节数值不可直接横比。
- **reasoning 重采样协议差异**：t100 用固定 seed=41 一次 n=64；stored 用 `seed*100003+batch*10`，故 52% 题 majority 错答与 stored 一致，但不影响 reasoning/token 多样性结论。
- **GPQA 为 4 选项 MCQ**：答案空间小会抬高 `p_top`，其 SCR 与数学题不完全可比。
- **TFB 模型**：Llama/Qwen 为剥离改造后的权重，行为与官方权重可能有细微差异。

---

## 8. 后续工作

1. ~~**加 Qwen2.5-7B**~~ ✅ 已完成；可继续加 14B+ 验证 SCR 是否继续上升。
2. **多 seed** 计算真正的 stable-wrong（跨 seed 同一错答），把不可约性证据从 budget-stable 升级为 seed-stable。
3. **陷阱题案例研究**：人工检视 42 题 SCR@1.0 共享吸引子，归纳错误模式（漏条件、捷径、单位/边界陷阱等）。
4. **缓解实验**：测试跨模型分歧投票 / 外部验证 / token-level UQ（EU、位置共识度）能否筛出 SCR 错题（验证 C1/C6）。
5. ~~**SCR@1.0 reasoning+token 深挖**~~ ✅ 42 题 × K=64 已保存并分析（§4.8）。

---

## 附录 A：复现实验

```bash
# 环境：3×RTX4090, CUDA12.8, torch2.8.0+cu128, vllm0.10.2, transformers4.56.2
bash setup_4090.sh
bash download_models.sh
python build_questions.py && python trim_deepscaler.py
bash run_phase1_3gpu.sh                 # 三卡并行采样
python clean_samples.py --tag <tag>     # 清洗
python analyze.py --samples "data/samples/samples_<tag>_seed41_*.jsonl" \
                  --out results/<tag>_metrics.json --fig-dir figures/<tag>
```

## 附录 B：数据与图表清单

**图表（3 模型早期分析）**

- `figures/nsweep_irreducible.png` — F1+F2：AUROC 饱和 & SCR 不可约下界（**主图**）
- `figures/capability_vs_scr.png` — F3：能力 ↑ → SCR ↑
- `figures/confidence_collapse.png` — F5：高置信度按难度的可靠性崩塌
- `figures/scr_hypothesis.png` — F3/F5/F6 综合三面板

**结构化结果 JSON**

- `results/all_models_stats_raw.json` — 六模型 pooled + 分 benchmark 指标（stored correct）
- `results/scr7b_t100_reasoning_analysis.json` — 42 题 reasoning + token 分析明细
- `results/metrics_pooled.json` — 3 模型 n-sweep pooled 指标
- `results/<tag>_metrics.json` — 各模型 n-sweep 完整指标（Llama 等）

**SCR 题单**

- `figures/qwen25_7b_scr_questions_clean.md` — SCR@0.9（88 题）
- `figures/qwen25_7b_scr_questions_clean_t095.md` — SCR@0.95（67 题）
- `figures/qwen25_7b_scr_questions_clean_t100.md` — SCR@1.0（42 题）

**Reasoning + token 原始数据**

- `data/scr_reasoning/qwen25_7b/t100/{qid}.jsonl` — 42 题 × 64 行（text + token ids + logprob）
- `data/scr_reasoning/qwen25_7b/t100/manifest.json` — 索引

**分析报告**

- `figures/scr7b_t100_reasoning_analysis.md` — §4.8 完整逐题表与样例
- `analyze_scr_reasoning_t100.py` — 复现 §4.8 统计
- `save_scr_reasoning.py` — 重采样保存 reasoning/token（`--tau 1.0`）
