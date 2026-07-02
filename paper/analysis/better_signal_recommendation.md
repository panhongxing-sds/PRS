# 更优过程信号推荐（吸取 skeleton 教训）

> 搜索脚本：`signal_search_v2.json`；全量 N=8121，macro LODO-LR，与 `aggregate_panda_v2.py` 一致。

## 吸取的教训

| 教训 | skeleton 反例 | 好信号应满足 |
|------|---------------|--------------|
| 子集强 ≠ 全局强 | bd=0 AUROC 0.86，全量替换 −0.011 | bd>0 不能拖后腿 |
| 与 bd 同源会冲突 | 同在 text rephrase 分支 | 优先 **base greedy** 或 **weight** 轴 |
| AUPRC 比 AUROC 更敏感 | 条件替换 AUPRC −0.125 | 全量 AUPRC 不能崩 |
| 长度/DeepConf 可刷分 | W_n_tokens、DC_mean | 需可解释为过程机制 |

---

## 推荐方案（按优先级）

### 🥇 方案 A：PANDA-v3 = bd + hes + `base_ans_mar_mean`（**推荐，不改叙事**）

| 指标 | PANDA | PANDA-v3 | Δ |
|------|------:|---------:|--:|
| AUROC | 0.904 | **0.908** | **+0.004** |
| AUPRC | 0.875 | **0.883** | **+0.008** |

- **`base_ans_mar_mean`**：base greedy 轨迹上 **答案 span 内 token 的 logit margin 均值**（`token_advanced.py`）
- 与 hes 同源（base + 答案附近），但测 **margin 坍缩** 而非 entropy
- Spearman(hes, ans_margin) ≈ **0.50** → 半独立，可叠加
- bd=0 子集 AUROC **0.641**（略高于 hes 0.623）

**叙事**：Hesitation（熵）+ Answer-margin collapse（边际）双通道捕捉「出答案时的过程犹豫」。

---

### 🥈 方案 B：PANDA-v2 = bd + `base_ans_mar_mean`（**替换 hes，AUPRC 最优**）

| 指标 | PANDA | PANDA-v2 | Δ |
|------|------:|---------:|--:|
| AUROC | 0.904 | 0.906 | +0.002 |
| AUPRC | 0.875 | **0.889** | **+0.015** |

- 若担心 hes 消融太弱，可用 **margin 单特征**替代 entropy
- ρ(bd) ≈ 0.46，与 hes 类似，但 **AUPRC 增益更大**

---

### 🥉 方案 C：bd + `T_mar_top10_sum_total`（**低相关备选**）

| 指标 | PANDA | 替换后 | Δ |
|------|------:|-------:|--:|
| AUROC | 0.904 | 0.905 | +0.001 |
| AUPRC | 0.875 | 0.879 | +0.005 |
| \|ρ(bd)\| | 0.43 | **0.11** | 更正交 |

- text 分支 **top-10 token margin 总和**；与 bd 几乎独立
- 单独在 bd=0/bd>0 上弱，**必须和 bd 联合 logistic** 才有增益
- 适合写「margin 型 hesitation 变体」，不宜单独替代 hes

---

## 不推荐

| 信号 | 原因 |
|------|------|
| `T_formula_skeleton_entropy` | 全量替换 −0.011；与 bd 同源；AUPRC 崩 |
| `W_n_tokens_avg` | AUROC +0.003 但为长度代理，审稿难辩护 |
| `baseline_DC_mean` | 外部 baseline 特征，非 PANDA 原生 |
| 条件路由 bd=0→skeleton | AUPRC 0.750，误报多 |

---

## 实施建议

1. **论文不改公式**：Discussion 报告 PANDA-v3 作 ablation 一行（+0.004 AUROC）
2. **若改方法**：PANDA-v2 将 `T_ent_prox_lin` 换为 `base_ans_mar_mean`，或 v3 三特征 LODO
3. **保留 hes 标题**：v3 证明 hes 非冗余；margin 是同一「answer-span process」的互补维度

复现：
```bash
source scripts/env.sh
python scripts/mine_process_signals.py  # 或见 signal_search_v2.json
```
