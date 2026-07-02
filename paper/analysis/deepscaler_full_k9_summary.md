# DeepScaler 全量 2000 题 SC@9 分布（Qwen2.5-3B, seed 41）

**数据源**: `/root/autodl-tmp/PANDA/experiments/spurious_consensus/data/samples/samples_qwen25_3b_seed41_deepscaler.jsonl`  
**方法**: 每题取 K=64 的**前 9** 条答案做 majority vote；众数是否正确用全行 `answer_correct_map`（与 `experiments/spurious_consensus/analyze.py` 一致）。

## 规模

- 题目数 **2000**
- 答案长度 ≥9：**2000**（全部满足）

## SC@9 结果

| 指标 | 计数 | 占比 |
|------|------|------|
| Majority 正确 | 878 | 43.90% |
| Majority 错误 | 1122 | 56.10% |
| Blind@9（0/9 单样本正确） | 693 | 34.65% |
| 至少 1 次采样正确 | 1307 | 65.35% |
| 9/9 unanimous 正确 | 319 | 15.95% |
| 9 条完全相同且为错答 | 62 | 3.10% |

## K=9 共识度 `p_top`（众数占比 / 9）

| mean | median | min | max | p25 | p75 |
|------|--------|-----|-----|-----|-----|
| 0.5562 | 0.4444 | 0.1111 | 1.0000 | 0.3333 | 0.8889 |

**直方图（题数）**: [0,0.5)=1003, [0.5,0.7)=361, [0.7,0.9)=284, [0.9,1.0]=352

Entropy@9: mean=1.1166, median=1.2149

## 与 K=64 对比

| | Majority 正确 | Majority 错误 |
|---|--------------|---------------|
| K=64 | 938 (46.9%) | 1062 (53.1%) |
| K=9 | 878 (43.9%) | 1122 (56.1%) |

- wrong@64 → correct@9：**52**（前缀下应极少）
- correct@64 → wrong@9：**112**

### 分层

**Majority wrong@64（n=1062）**

- blind@9：686 题（**64.6%**）
- `p_top@9` 均值：**0.4337**

**Majority correct@64（n=938）**

- 在 K=9 上 majority 变错：**112** 题（**11.9%**）

## 与 scr500 子集对照（便于直觉）

scr500 是从 wrong@64 中抽的 500 题；全量 2000 上 wrong@64=1062。全量 blind@9=693（34.6%），高于 scr500 子集内的 310/500=62% 是因为 scr500 **条件于**已 majority 错@64。

## 解读

1. **SC@9 准确率** 43.9% 反映仅用前 9 次采样的投票，比 K=64（46.9%）低 3.0 个百分点。
2. **`p_top` 均值 0.556**：前 9 次答案表面共识中等；histogram 显示 17.6% 题达到 ≥0.9 众数占比，其中仍含 majority 错题（虚假共识）。
3. **wrong@64 题** 中 64.6% 在前 9 次**无任何正确单样本**；这些题对 early-stop / 盲共识风险最高。
4. **correct@64 → wrong@9** 共 112 题：早期 9 次众数与最终 64 次众数不一致（前缀效应），不是 K 增加后「变错」，而是前 9 条更分散/更偏。

数值 JSON：`/root/autodl-tmp/PANDA/paper/analysis/deepscaler_full_k9_stats.json`
