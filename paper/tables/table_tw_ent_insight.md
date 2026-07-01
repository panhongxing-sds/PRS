# TW_ent 叙事变体扫描

> N=8121，4 模型 LODO。对比 bd + 各 TW 变体 vs bd + TW_ent_sum。

**bd-only macro**: 0.897  |  **bd+TW_ent_sum**: 0.896

## 排名（bd+T macro）

| 特征 | bd+T | Δ vs bd | pooled AUROC | 叙事 |
|------|------:|------:|------:|------|
| `TW_tail_ent` | 0.901 | +0.003 | 0.726 | 答案前 16 个 process token 的 entropy 和（commit 前犹豫） **←best** |
| `TW_calc_xvar_sum` | 0.900 | +0.002 | 0.815 | 计算 token 对齐位置的跨 run entropy 方差之和（计算步 epistemic 分歧） |
| `TW_reject_ent` | 0.897 | -0.000 | 0.803 | 仅「否决 base 答案」的 run 的 entropy 和（否决 run 有多飘） |
| `TW_reject_calc_ent` | 0.897 | -0.000 | 0.802 | 否决 run 上计算 token entropy 和 |
| `TW_ent_delta` | 0.897 | -0.000 | 0.760 | 相对 base，8 run 全 trace |Δentropy| 累计（扰动致不确定漂移） |
| `TW_calc_ent_delta` | 0.896 | -0.001 | 0.751 | 相对 base，计算区 |Δentropy| 累计 |
| `TW_calc_ent` | 0.896 | -0.001 | 0.794 | 仅 reasoning 内计算 token（数/符/变量）entropy 和 |
| `TW_calc_top10` | 0.896 | -0.001 | 0.763 | 计算 token 中 top10% 高 entropy 之和（最犹豫的计算步） |
| `TW_ent_sum` | 0.896 | -0.002 | 0.789 | 基线：8 run 全 trace entropy 总和 ≈ref |

## Qwen2.5-3B

| 特征 | bd+T | Δ vs bd+TW_ent |
|------|------:|------:|
| `TW_tail_ent` | 0.866 | -0.007 |
| `TW_calc_xvar_sum` | 0.877 | +0.004 |
| `TW_reject_ent` | 0.855 | -0.018 |
| `TW_reject_calc_ent` | 0.855 | -0.018 |
| `TW_ent_delta` | 0.872 | -0.001 |
| `TW_calc_ent_delta` | 0.878 | +0.005 |
| `TW_calc_ent` | 0.873 | +0.000 |
| `TW_calc_top10` | 0.875 | +0.002 |
| `TW_ent_sum` | 0.873 | +0.000 |

## 建议

- **数字最好**: `TW_tail_ent`
- **相对 TW_ent_sum**: macro 0.901 vs ref 0.896 (+0.005)
