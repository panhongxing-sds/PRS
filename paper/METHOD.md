# PANDA 方法说明（Method）

> **PANDA uncertainty score**：在固定采样预算下，对 base 贪心答案 \(a_0\) 给出 **不可靠度分数**（越高 → 越可能错）。  
> 本文档描述 **定稿 PANDA**：**answer-level 否决** \(D_\text{base}\) + **token-level 近答案不确定性** \(T_{\mathrm{ent\_prox}}\)，留一数据集 LR 融合。  
> \(F_\text{resp}\) 保留为 **诊断/附录消融**（与 bd 高度冗余，不进入主公式）。

---

## 1. 总览

### 1.1 输入与输出

对每道题，管线已产生：

| 符号 | 含义 | 来源 |
|------|------|------|
| \(a_0\) | base 贪心解码的最终答案 | `base_generation.answer_normalized` |
| \(\{a_1^T,\ldots,a_4^T\}\) | 4 次 **text rephrase** 扰动答案 | `text_rephrase_runs[*].answer_normalized` |
| \(\{a_1^W,\ldots,a_4^W\}\) | 4 次 **weight perturb** 扰动答案 | `weight_perturb_runs[*].answer_normalized` |
| token 轨迹 | 每条 run 的逐 token entropy / logprob | `*.token_trace`, `*.answer_span` |

记 \(K=8\) 个扰动答案：

\[
\mathcal{A}_\text{pert} = \{a_1^T,\ldots,a_4^T,\, a_1^W,\ldots,a_4^W\}
\]

**PANDA 输出**：标量不可靠度 \(S_\text{PANDA}\)，由 **两个主分量** LODO logistic 融合：

\[
\boxed{\;\text{PANDA}(x)=\sigma\!\big(w_1 D_\text{base}+w_2 T_{\mathrm{ent\_prox\_lin}}+b\big)\;}
\]

| 符号 | 字段名 | 层级 | 问的问题 |
|------|--------|------|----------|
| \(D_\text{base}\) | `base_disagree` / **bd** | answer | 8 扰动 **多大程度反对** \(a_0\)？ |
| \(T_{\mathrm{ent\_prox\_lin}}\) | `T_ent_prox_lin` | token | 8 扰动 run 在 **靠近答案** 的 reasoning 上是否仍不确定？ |

\(D_\text{base}\) 即 bd：\(D_\text{base}=\frac{1}{8}\sum_i \mathbf{1}[\neg\text{math\_equal}(a_i,a_0)]\)。

权重 \(w_1,w_2,b\) 由 **留一数据集 logistic regression** 拟合（§6）。

**诊断分量（不进主 PANDA）**：\(F_\text{resp}\)（8 扰动答案 ASE，与 bd 冗余 ρ≈0.9）；legacy \(T_\text{commit}\)（cliff，MATH-500 上失效，已弃用）。

### 1.2 采样预算

- 每题解码：**1 base + 4 text + 4 weight = 9 次**（PANDA 两分量用 8 扰动 run 的 pre-answer trace；SE 等 baseline 另计 8 次高温采样，见 `EXPERIMENT_PLAN.md`）
- 两分量 **共享同一批扰动答案与 token trace**，无额外生成成本

### 1.3 两分量为何互补

| 场景 | bd | \(T_{\mathrm{ent\_prox}}\) | 解读 |
|------|:--:|:--:|------|
| 8 扰动全反对 \(a_0\)，但尾部 entropy 低 | **高** | 低 | bd 报警；过程看似已收敛 |
| 8 扰动均支持 \(a_0\)，但近答案 token 仍高熵 | 低 | **高** | bd 认为可靠；过程仍犹豫 |
| 8 扰动一致且等于 \(a_0\)，尾部低熵 | 0 | 低 | 二者均认为可靠 |
| 答案对但 reasoning 末段反复改算 | 低 | **高** | bd 盲区；prox 报警 |

bd **显式以 \(a_0\) 为锚**；\(T_{\mathrm{ent\_prox}}\) 看 **过程 token 在收束阶段的 uncertainty**，与最终投票正交。

---

## 2. 公共依赖

### 2.1 答案规范化与判等

所有 outcome-level 分量共用：

1. **抽取**：`extract_math_answer(text)` → 规范化答案字符串（优先 `\boxed{}` 内容）
2. **判等**：`math_equal(a, b)` → bool（sympy / 数值容差，与 TokUR 口径兼容）

聚类实现：`panda.core.cluster.cluster_answers(answers)`  
- 对 \(n\) 个答案两两 `math_equal`，Union-Find 得 cluster  
- 返回 `labels`（每答案簇 id）与 `cluster_sizes`

### 2.2 Pre-answer reasoning 与 answer span

\(T_{\mathrm{ent\_prox}}\) 依赖每条 run 的 **答案起始 token** \(a\)：

- 优先：响应中 `\boxed{...}` 在 token 流中的 `start_token`（`find_answer_span`, method=`boxed`）
- 回退：最后 8 个 token（method=`tail8`）

字段：`run["answer_span"]["start_token"]`  
**Pre-answer trace**：\(\text{trace}[0:a)\)，即答案 span 之前的 reasoning tokens。  
各 token 的 entropy 来自 `run["token_trace"][t]["entropy"]`（由 logprob / top-k 导出）。

### 2.3 标签与评估

- **标签**：`label_wrong_clean`（strict grader；`label_drop=1` 的样本不进指标）
- **指标**：error detection AUROC（分数越高 → 预测「错」越好）
- **融合评估**：留一 **数据集** LR（§6），3 个数学集 minerva / math500 / gsm8k 轮换作 test

---

## 3. 分量一：\(F_\text{resp}\)（响应碎片化，诊断）

### 3.1 定义

**仅对 8 个扰动答案** \(\mathcal{A}_\text{pert}\) 聚类（**不含 \(a_0\)**）。设最大簇占比为 \(p_\max\)，

\[
F_\text{resp} = 1 - p_\max
\]

等价于 **Adversarial Semantic Entropy** 的主分数 \(U\)（`compute_ase` 返回值中的 `U`）。

### 3.2 计算步骤

```
输入: text_answers[4], weight_answers[4]
1. all_pert ← concat(text_answers, weight_answers)     # |all_pert| = 8
2. labels, sizes ← cluster_answers(all_pert)           # math_equal 聚类
3. masses ← [count_c / 8 for each cluster c]
4. p_max ← max(masses)
5. F_resp ← 1 - p_max
```

**代码路径**：

```python
tw_ase = compute_ase(text_answers + weight_answers)
F_resp = tw_ase["U"]   # 写入 summary 时字段名 F_resp 或 TW_ASE
```

实现文件：`panda/core/semantic_entropy.py`, `panda/core/metrics.py`（L88–90, L144）。

### 3.3 取值与含义

| \(F_\text{resp}\) | 含义 |
|:--:|------|
| 0 | 8 个扰动答案全在同一等价类（完全一致） |
| 0.75 | 最大簇占 1/4（如 2 vs 2 vs 2 vs 2 四簇） |
| 1 − 1/8 = 0.875 | 8 个答案两两不等（8 簇） |

**注意**：\(F_\text{resp}\) 高只说明 **扰动之间** 答案分裂，不说明 \(a_0\) 是否被反对（见 bd）。

定稿 PANDA **不使用** \(F_\text{resp}\)，仅作附录消融。

---

## 4. 分量二：bd（base_disagree，否决率）

### 4.1 定义

\[
\text{bd} = \frac{1}{K}\sum_{i=1}^{K} \mathbf{1}\big[\,\neg\,\text{math\_equal}(a_i, a_0)\,\big],
\quad K=8
\]

即：8 个扰动答案中，与 base 答案 **数学不等价** 的比例。  
范围 \([0,1]\)，只有 9 个离散档位 \(\{0, \frac18, \ldots, 1\}\)。

### 4.2 计算步骤

```
输入: a0 ← base_generation.answer_normalized
      pert[] ← 8 个扰动 answer_normalized
1. votes ← 0
2. for each a in pert:
3.     if not math_equal(a, a0):
4.         votes += 1
5. bd ← votes / 8
```

**等价实现**（若已有 `semantic_cache`）：

```python
pe = record["semantic_cache"]["pairwise_equivalent"]  # 9×9, index 0 = a0
bd = 1.0 - sum(pe[0][1:9]) / 8.0
```

### 4.3 数值示例

| 扰动答案（相对 \(a_0=72\)） | bd |
|------------------------------|:--:|
| 8 票均为 72 | 0.0 |
| 6 票 90，2 票 72 | 0.75 |
| 8 票均为 90 | 1.0 |

**与 \(F_\text{resp}\) 对照**：最后一行 \(F_\text{resp}=0\)（扰动全同），\(\text{bd}=1\)（全反对 \(a_0\)）。

---

## 5. 分量三：\(T_{\mathrm{ent\_prox\_lin}}\)（近答案 process-token 不确定性）

### 5.1 动机

bd 与 \(F_\text{resp}\) 只看 **最终答案**；两条 run 可以给出相同答案，但一条在接近答案时 process token 已收敛，另一条仍在高熵犹豫。

**核心 insight**：

\[
\boxed{
\text{推理越接近最终答案，process-token uncertainty 越应下降；}
\text{若仍高熵，则更可能不可靠。}
}
\]

与 legacy \(T_\text{commit}\)（边界 cliff）不同：cliff 在 MATH-500 上单特征 AUROC ≈ 0.5，且与 bd 信息重叠；近答案加权 entropy 在三数据集上稳定增益。

### 5.2 单 run 的近答案加权 entropy

对一条 perturb run \(j\)，设 pre-answer reasoning 共 \(n\) 个 token，第 \(t\) 个 token（\(t=0\) 最远，\(t=n-1\) 最近 answer）的相对位置与权重：

\[
r_t = \frac{t+1}{n}, \qquad w_t = r_t \quad \text{(linear)}
\]

token entropy \(H_t\) 来自 `token_trace[t].entropy`。加权平均：

\[
\overline{H}_j = \frac{\sum_{t=0}^{n-1} w_t H_t}{\sum_{t=0}^{n-1} w_t}
\]

若某 token 无 entropy 或 \(n=0\)，跳过该 token；若有效权重和为 0，该 run 记 NaN。

### 5.3 ensemble 聚合

\[
T_{\mathrm{ent\_prox\_lin}}(x) = \frac{1}{|\mathcal{R}_\text{valid}|}\sum_{j \in \mathcal{R}_\text{valid}} \overline{H}_j
\]

- \(\mathcal{R} = \{\text{text rephrase runs}\} \cup \{\text{weight perturb runs}\}\)，共 **8** 条扰动 run（**不含 base**）
- \(\mathcal{R}_\text{valid}\) = \(\overline{H}_j\) 为有限值的 run
- 若 \(|\mathcal{R}_\text{valid}|=0\)，记 NaN（融合时用训练集中位数 impute）

### 5.4 伪代码

```
输入: text_rephrase_runs[4], weight_perturb_runs[4]
runs ← text_runs + weight_runs
ents ← []

for run in runs:
    trace ← run.token_trace
    a ← run.answer_span.start_token
    pre ← trace[0:a]                    # pre-answer reasoning
    n ← len(pre)
    if n == 0: continue
    s, wsum ← 0, 0
    for t in 0..n-1:
        H ← pre[t].entropy
        if H is None: continue
        w ← (t + 1) / n
        s += w * H; wsum += w
    if wsum > 0: ents.append(s / wsum)

T_ent_prox_lin ← mean(ents) if ents else NaN
```

### 5.5 取值与含义

| \(T_{\mathrm{ent\_prox\_lin}}\) | 典型含义 |
|:--:|------|
| 低 | 靠近答案的 reasoning token 已低熵 → 收束清晰 |
| 高 | 越近答案 entropy 仍高 → 过程未收敛、犹豫 |

**为何用 linear 权重**：在 late-gap、slope、ratio、exp 权重、text/weight 拆分、BD×prox 交互等 8 组变体中，linear prox 在 macro LODO 上与最佳持平（0.904），且最简单、最好解释（见 `paper/tables/table_near_answer_final.md`）。

### 5.6 与 bd 的分工

| | bd | \(T_{\mathrm{ent\_prox\_lin}}\) |
|--|-----|-----|
| 输入 | 最终答案字符串 | pre-answer `token_trace` entropy |
| 问的问题 | 扰动 **投什么票** | **靠近答案时** process token 是否仍不确定 |
| 典型独立增益 | \(a_0\) 与扰动答案不一致 | 答案一致但末段 reasoning 高熵 |

---

## 6. 分数融合

### 6.1 特征向量

\[
\mathbf{x}=\big(D_\text{base},\,T_{\mathrm{ent\_prox\_lin}}\big)^\top
\]

标签 \(y=\texttt{label\_wrong\_clean}\)。两分量方向均为 **越高 → 越不可靠**。

### 6.2 留一数据集 LR

数学集 \(\mathcal{D} = \{\text{minerva}, \text{math500}, \text{gsm8k}\}\)。

对每个 held-out 数据集 \(D_\text{test}\)：

1. 训练集 = 其余 2 个数据集上 **全部模型、全部 seed** 的样本
2. 缺失值：训练集中位数 impute
3. 标准化：\(\tilde{\mathbf{x}} = (\mathbf{x} - \boldsymbol\mu)/\boldsymbol\sigma\)（在训练集上估计）
4. 拟合 \(\Pr(y=1\mid \mathbf{x}) = \sigma(\mathbf{w}^\top \tilde{\mathbf{x}} + b)\)
5. 在 \(D_\text{test}\) 各 seed 上算 AUROC，再对 seed 平均

报告 **3 折平均 AUROC（macro）**，及相对 **仅 \(D_\text{base}\)** 的增量。

### 6.3 推理时使用

给定新样本的 \((D_\text{base}, T_{\mathrm{ent\_prox\_lin}})\)，用 **全数据拟合** 的 LR 得到 \(\text{PANDA}(x)\)。

---

## 7. 实证结果（定稿配置）

数据：4 模型（Qwen2.5-3B, Llama-3.2-1B, Llama-3.1-8B, Qwen3-8B）× 3 seeds × 3 数学集，**N=8121** 样本（strict label，2026-06-15）。  
复现：`python PANDA/scripts/validate_near_answer_final.py --use-cache`

### 7.1 留一 LR macro AUROC（4 模型 pooled，3 数据集 LODO 均值）

| 配置 | macro AUROC | Δ vs bd-only |
|------|:--:|:--:|
| bd-only | 0.897 | — |
| bd + TW_ent_sum（legacy T） | 0.896 | −0.001 |
| **PANDA = bd + T_ent_prox_lin** | **0.904** | **+0.007** |

### 7.2 分数据集（bd → bd+prox，macro 增益）

| 数据集 | bd-only | bd + prox | Δ |
|--------|:--:|:--:|:--:|
| Minerva | 0.852 | 0.863 | +0.011 |
| MATH-500 | 0.916 | 0.921 | +0.005 |
| GSM8K | 0.926 | 0.931 | +0.005 |

三数据集 **全部提升**。

### 7.3 分模型 LODO macro（bd + prox）

| 模型 | bd-only | bd + prox | Δ |
|------|:--:|:--:|:--:|
| Qwen2.5-3B | 0.918 | 0.927 | +0.010 |
| Llama-3.2-1B | 0.883 | 0.887 | +0.004 |
| Llama-3.1-8B | 0.916 | 0.916 | ±0 |
| Qwen3-8B | 0.964 | 0.966 | +0.002 |

Llama-3.1-8B 上 prox 与 bd 持平（已知 trade-off）；其余模型均有增益。

### 7.4 单特征 pooled AUROC

| 特征 | pooled AUROC |
|------|:--:|
| bd | ≈0.85+ |
| \(T_{\mathrm{ent\_prox\_lin}}\) | 0.731 |
| legacy TW_ent_sum | ≈0.73 |

prox 作为单特征弱于 bd，但与 bd 融合有 **+0.7 pp macro** 互补增益。

---

## 8. raw 记录 → summary 字段映射

### 8.1 输入：`raw_runs/{id}.json` 关键结构

```json
{
  "id": "minerva_0",
  "base_generation": {
    "answer_normalized": "72",
    "token_trace": [{"entropy": 1.2, "logprob": -0.1, "token": "...", ...}, ...],
    "answer_span": {"start_token": 142, "end_token": 145, "method": "boxed"}
  },
  "text_rephrase_runs": [ /* 4 runs, same schema */ ],
  "weight_perturb_runs": [ /* 4 runs, same schema */ ]
}
```

### 8.2 输出：`summary.jsonl` 一行（定稿 PANDA 相关）

| 字段 | 类型 | 说明 |
|------|------|------|
| `base_disagree` | float | bd，\([0,1]\) |
| `T_ent_prox_lin` | float | 8 扰动 run 近答案加权 entropy 均值 |
| `PANDA` | float | LODO-LR(\(D_\text{base}, T_{\mathrm{ent\_prox\_lin}}\)) |
| `F_resp` | float | 诊断；\(=\texttt{TW\_ASE}\) |
| `a0` | str | base 答案 |
| `label_wrong_clean` | int | 0/1 |
| `label_drop` | int | 1 则剔除 |

### 8.3 代码落点

| 步骤 | 文件 | 函数 |
|------|------|------|
| bd | `panda/core/metrics.py` | `compute_base_disagree(a0, pert)` |
| \(T_{\mathrm{ent\_prox\_lin}}\) | `scripts/validate_near_answer_token.py`（待迁入 `panda/core/`） | `prox_weighted_entropy`, `extract_features` |
| 融合 | `scripts/aggregate_panda_v2.py` | LODO OOF LR |

---

## 9. 论文消融表建议（E4）

**Drop-one**（相对 full bd+prox，Δ<0 表示去掉该分量后下降）：

| 行 | 特征 | 说明 |
|----|------|------|
| 1 | \(D_\text{base}\)（raw） | answer 否决单特征 |
| 2 | \(T_{\mathrm{ent\_prox\_lin}}\)（raw） | 近答案 token 不确定性单特征 |
| 3 | **PANDA full** | LODO-LR(\(D_\text{base}, T_{\mathrm{ent\_prox\_lin}}\)) |
| 4 | −\(D_\text{base}\) → 仅 \(T\) | drop-one |
| 5 | −\(T\) → 仅 \(D_\text{base}\) | drop-one |

附录可选：+ \(F_\text{resp}\) 三分量 legacy；legacy \(T_\text{commit}\)（cliff）对比。

---

## 10. 公式速查

\[
\boxed{
\begin{aligned}
D_\text{base} &= \frac{1}{8}\sum_{a \in \mathcal{A}_\text{pert}} \mathbf{1}[\neg\,\text{math\_equal}(a, a_0)] \\[6pt]
w_t &= \frac{t+1}{n}, \quad t=0,\ldots,n-1 \;\text{(pre-answer reasoning)} \\[6pt]
T_{\mathrm{ent\_prox\_lin}} &= \mean_{j \in \mathcal{R}_\text{pert}} \frac{\sum_t w_t H_{j,t}}{\sum_t w_t} \\[6pt]
\text{PANDA}(x) &= \sigma(w_1 D_\text{base} + w_2 T_{\mathrm{ent\_prox\_lin}} + b)
\end{aligned}
}
\]

---

*最后更新：2026-06-15 — 定稿 PANDA = LODO-LR(\(D_\text{base}\), \(T_{\mathrm{ent\_prox\_lin}}\))；\(F_\text{resp}\) 仅诊断；legacy cliff 已弃用。*
