# Qwen-7B SCR@1.0（42 题）Reasoning + Token 分析

> 数据：`data/scr_reasoning/qwen25_7b/t100/`，每题 K=64 完整 reasoning + token logprob
> 协议：temp=0.5, top_p=0.95, seed=41（与 stored 采样 seed 不同，见下方复现说明）

## 总览

- 题数：**42**（deepscaler 39 + gpqa 3）
- 重采样 majority 与 **stored SCR 错答一致**：52%（22/42）
- 64/64 **答案完全相同**（重采样）：57%
- 64/64 **reasoning 文本完全相同**（去 boxed 后、空白归一）：0%
- 64/64 **token id 序列完全相同**：0%
- 前 200 字符 reasoning 全部相同：0%
- 与首条 reasoning 平均相似度（sim_to_first）：均值 **0.393**，中位 **0.375**
- sim≥0.9（高度同构）：0% 题；sim<0.5（路径分散）：67% 题

## Token-level 总览

- 64 条 trace **共同前缀 token 长度**：均值 **6.8**，中位 **1**
- 前 50 token **位置共识度**（该位置最多 token 占比均值）：**0.608**
- pairwise **首个分歧位置**（16 条子集）：均值 **26.5** token
- 同位置跨 trace **logprob 标准差**均值：**0.167**
- 平均 token logprob：**-0.066**（整体高置信生成）
- logprob < −1.0 的 token 占比：**1.49%**（极低，几乎无「犹豫 token」）

**核心发现**：SCR@1.0 刻画的是 **答案级虚假共识**，不是 reasoning/token 级复制粘贴。绝大多数题 64 条 completion 的 token 序列互不相同，reasoning 文本也高度多样化，却在第 1–7 个 token 后就开始分叉，却能收敛到同一错误答案——**错误共识可以在不同推理路径上达成，且生成过程整体仍保持高置信。**

> **复现说明**：stored 样本用 `seed*100003+batch*10` 逐 batch 采样；t100 重采样用固定 seed=41 一次生成 n=64，故仅 52% 题 majority 错答与 stored 完全一致。分析结论基于 t100 重采样数据本身的 reasoning/token 多样性，与 SCR 机制一致。

## 逐题明细（按 sim_to_first 升序）

| id | bench | stored错答 | resample一致 | unique reason | unique tok | 共同前缀 | sim→1st | 前50共识 | 外生难度 |
|----|-------|-----------|-------------|--------------|-----------|---------|---------|---------|---------|
| `39604` | deepscaler | `\beginpmatrix` | ✗ | 64 | 64 | 1 | 0.087 | 0.45 | 0.531 |
| `39958` | deepscaler | `sqrt2` | ✗ | 63 | 63 | 0 | 0.099 | 0.42 | 0.271 |
| `39871` | deepscaler | `\beginpmatrix` | ✗ | 64 | 64 | 1 | 0.102 | 0.48 | 0.700 |
| `39645` | deepscaler | `\beginpmatrix` | ✗ | 64 | 64 | 1 | 0.137 | 0.53 | 0.756 |
| `38203` | deepscaler | `d` | ✗ | 64 | 64 | 10 | 0.148 | 0.73 | 0.193 |
| `gpqa_86` | gpqa_diamond | `A` | ✓ | 64 | 64 | 0 | 0.158 | 0.32 | 0.084 |
| `gpqa_145` | gpqa_diamond | `A` | ✓ | 64 | 64 | 32 | 0.182 | 0.78 | 0.062 |
| `24798` | deepscaler | `50` | ✓ | 64 | 64 | 2 | 0.197 | 0.47 | 0.034 |
| `1401` | deepscaler | `25` | ✓ | 64 | 64 | 0 | 0.217 | 0.33 | 0.006 |
| `40108` | deepscaler | `\beginpmatrix` | ✗ | 64 | 64 | 1 | 0.221 | 0.65 | 0.525 |
| `937` | deepscaler | `4sqrt3` | ✗ | 64 | 64 | 1 | 0.246 | 0.50 | 0.355 |
| `23641` | deepscaler | `-2sqrt2` | ✗ | 64 | 64 | 0 | 0.253 | 0.44 | 0.479 |
| `6247` | deepscaler | `1998` | ✓ | 64 | 64 | 22 | 0.257 | 0.97 | 0.000 |
| `gpqa_122` | gpqa_diamond | `B` | ✓ | 64 | 64 | 0 | 0.284 | 0.36 | 0.113 |
| `1670` | deepscaler | `-\infty,-2]\cup[3,\infty` | ✗ | 64 | 64 | 3 | 0.292 | 0.43 | 0.003 |
| `13184` | deepscaler | `32` | ✓ | 64 | 64 | 13 | 0.301 | 0.84 | 0.054 |
| `12827` | deepscaler | `26` | ✓ | 64 | 64 | 0 | 0.316 | 0.27 | 0.010 |
| `34320` | deepscaler | `\cfrac29` | ✗ | 64 | 64 | 1 | 0.333 | 0.45 | 0.434 |
| `30267` | deepscaler | `pi` | ✗ | 64 | 64 | 3 | 0.340 | 0.49 | 0.229 |
| `17214` | deepscaler | `135^\circ` | ✓ | 64 | 64 | 1 | 0.363 | 0.70 | 0.172 |
| `21529` | deepscaler | `2^49/6` | ✗ | 64 | 64 | 35 | 0.366 | 0.92 | 0.089 |
| `39627` | deepscaler | `\beginpmatrix` | ✗ | 64 | 64 | 11 | 0.384 | 0.68 | 0.481 |
| `25540` | deepscaler | `15` | ✓ | 64 | 64 | 1 | 0.392 | 0.72 | 0.000 |
| `28830` | deepscaler | `15` | ✓ | 64 | 64 | 1 | 0.392 | 0.72 | 0.000 |
| `28082` | deepscaler | `-sqrt2` | ✗ | 63 | 63 | 19 | 0.398 | 0.73 | 0.094 |
| `28566` | deepscaler | `900` | ✓ | 63 | 63 | 1 | 0.422 | 0.77 | 0.103 |
| `24723` | deepscaler | `sqrt2` | ✗ | 64 | 64 | 1 | 0.455 | 0.49 | 0.392 |
| `12473` | deepscaler | `220` | ✓ | 64 | 64 | 15 | 0.466 | 0.55 | 0.000 |
| `11874` | deepscaler | `6924764` | ✓ | 64 | 64 | 1 | 0.502 | 0.83 | 0.031 |
| `39486` | deepscaler | `95040` | ✓ | 64 | 64 | 1 | 0.504 | 0.86 | 0.072 |
| `25178` | deepscaler | `180` | ✓ | 63 | 63 | 18 | 0.540 | 0.65 | 0.044 |
| `33958` | deepscaler | `1657` | ✓ | 64 | 64 | 1 | 0.560 | 0.40 | 0.059 |
| `27761` | deepscaler | `60` | ✓ | 64 | 64 | 19 | 0.578 | 0.67 | 0.006 |
| `40209` | deepscaler | `\beginpmatrix` | ✗ | 63 | 63 | 7 | 0.586 | 0.85 | 0.787 |
| `39973` | deepscaler | `-sqrt2` | ✗ | 64 | 64 | 3 | 0.613 | 0.40 | 0.619 |
| `17252` | deepscaler | `sqrt6+2sqrt2` | ✗ | 63 | 63 | 1 | 0.629 | 0.89 | 0.181 |
| `24959` | deepscaler | `15\textand` | ✗ | 64 | 64 | 0 | 0.645 | 0.33 | 0.288 |
| `12867` | deepscaler | `2` | ✓ | 60 | 60 | 9 | 0.650 | 0.95 | 0.263 |
| `30435` | deepscaler | `0.3` | ✓ | 46 | 64 | 11 | 0.675 | 0.88 | 0.102 |
| `14993` | deepscaler | `39` | ✓ | 64 | 64 | 6 | 0.677 | 0.46 | 0.134 |
| `22551` | deepscaler | `368\textcubicfeet` | ✗ | 63 | 63 | 1 | 0.716 | 0.39 | 0.106 |
| `26836` | deepscaler | `10040_5` | ✓ | 62 | 62 | 30 | 0.811 | 0.80 | 0.072 |

## 跨模型（5 个其他模型 stored correct）

- 与 7B 给出**相同错答**的题数：Phi 19/42，Qwen-3B 38/42
- 其他模型也在 SCR@0.9 的题数：Phi 5/42，Qwen-3B 17/42

## 机制分型（基于 reasoning + token 多样性）

| 类型 | 定义 | 题数 | 代表 id |
|------|------|-----:|---------|
| A 多路径汇入 | sim<0.3 且 token 前缀≤5，典型虚假共识 | 12 | `39604` |
| B 中等同构 | 0.3≤sim<0.6，部分步骤共享 | 19 | `13184` |
| C 高同构错答 | sim≥0.6，推理骨架相似但仍非复制 | 8 | `39973` |

## 典型样例

### 最低相似度：`39604`（sim=0.087）
- stored 错答 `\beginpmatrix` → 重采样 maj `\begin{pmatrix`（100%）
- unique reasoning=64，unique token ids=64
- 共同前缀 token=1，前50位置共识=0.45

```
To solve the given matrix multiplication problem, we need to understand the pattern and properties of the matrices involved. Each matrix in the product is of the form \(\begin{pmatrix} 1 & a_i \\ 0 & 1 \end{pmatrix}\), where \(a_i\) is an odd number starting from 1 and increasing by 2 each time (i.e., 1, 3, 5, ..., 99).

Let's denote the product of these matrices as \(P\):
\[ P = \begin{pmatrix} 1
```

### 最高相似度：`26836`（sim=0.811）
- stored 错答 `10040_5` → 重采样 maj `10040_5`（100%）
- unique reasoning=62，unique token ids=62
- 共同前缀 token=30，前50位置共识=0.80

```
To convert the decimal number \(645_{10}\) to base 5, we need to repeatedly divide the number by 5 and keep track of the remainders. Here are the steps:

1. **Divide 645 by 5:**
   \[
   645 \div 5 = 129 \quad \text{with a remainder of} \quad 0
   \]
   So, the least significant digit (rightmost) in base 5 is 0.

2. **Divide 129 by 5:**
   \[
   129 \div 5 = 25 \quad \text{with a remainder of} \qu
```
