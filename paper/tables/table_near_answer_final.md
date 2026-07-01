# 近答案 token 指标 — 最后一轮（8 配置）

> N=8121，LODO OOF LR。

**bd-only**: 0.897 | **bd+TW_ent_sum**: 0.896 | **bd+T_ent_prox_lin (ref)**: 0.904

## macro LODO

| # | 配置 | macro | Δ vs bd | Qwen2.5 | Llama3.2 | Llama3.1 | Qwen3 | pooled T |
|---|------|------:|---:|---:|---:|---:|---:|---:|
| 1 prox_lin | `T_ent_prox_lin` | 0.904 | +0.007 | 0.862 | 0.832 | 0.846 | 0.945 | 0.731 |
| 2 prox_exp_tau16 | `T_ent_prox_exp_tau16` | 0.902 | +0.004 | 0.872 | 0.827 | 0.857 | 0.944 | 0.769 |
| 3 prox_exp_tau32 | `T_ent_prox_exp_tau32` | 0.900 | +0.003 | 0.871 | 0.828 | 0.851 | 0.943 | 0.736 |
| 4 late20-early50 | `T_ent_late20_minus_early50` | 0.897 | -0.000 | 0.827 | 0.835 | 0.868 | 0.939 | 0.520 |
| 5 ent_slope | `T_ent_slope` | 0.898 | +0.001 | 0.846 | 0.835 | 0.865 | 0.940 | 0.522 |
| 6 late20_ratio | `T_ent_late20_ratio` | 0.894 | -0.004 | 0.837 | 0.831 | 0.870 | 0.939 | 0.556 |
| 7 prox_text+weight | `T_ent_prox_text, T_ent_prox_weight` | 0.903 | +0.006 | 0.864 | 0.830 | 0.847 | 0.944 | 0.712 |
| 8 prox+BD×prox | `T_ent_prox_lin, I_bd_x_prox_lin` | 0.905 | +0.007 | 0.860 | 0.833 | 0.846 | 0.943 | 0.731 |

## 12 格 drop-one（fusion vs bd-only，负=融合更差）

### qwen25_3b
| ds | bd | #1 prox | #5 slope | #7 T+W | best |
|----|---:|---:|---:|---:|---|
| minerva | 84.63 | 86.50 | 84.56 | 85.84 | 1 prox_lin |
| math500 | 92.73 | 93.40 | 93.45 | 93.41 | 5 ent_slope |
| gsm8k | 80.07 | 78.77 | 75.68 | 79.86 | 3 prox_exp_tau32 |

### llama32_1b
| ds | bd | #1 prox | #5 slope | #7 T+W | best |
|----|---:|---:|---:|---:|---|
| minerva | 75.51 | 75.75 | 76.27 | 75.82 | 8 prox+BD×prox |
| math500 | 85.04 | 84.36 | 85.23 | 84.16 | 4 late20-early50 |
| gsm8k | 88.83 | 89.60 | 88.95 | 89.07 | 1 prox_lin |

### llama31_8b
| ds | bd | #1 prox | #5 slope | #7 T+W | best |
|----|---:|---:|---:|---:|---|
| minerva | 77.50 | 76.66 | 77.78 | 77.04 | 2 prox_exp_tau16 |
| math500 | 90.11 | 89.89 | 90.11 | 89.87 | 4 late20-early50 |
| gsm8k | 89.73 | 87.32 | 91.74 | 87.26 | 6 late20_ratio |

### qwen3_8b
| ds | bd | #1 prox | #5 slope | #7 T+W | best |
|----|---:|---:|---:|---:|---|
| minerva | 92.90 | 92.67 | 93.19 | 92.59 | 5 ent_slope |
| math500 | 97.48 | 97.75 | 97.28 | 97.43 | 8 prox+BD×prox |
| gsm8k | 92.14 | 93.06 | 91.41 | 93.05 | 8 prox+BD×prox |

## 判定

- **最佳**: 8 prox+BD×prox macro=0.905 (Δ vs bd +0.007)
- vs #1 prox_lin: +0.000 macro
- 12格 fusion≥bd: 6/12
- Llama3.1: 0.846 vs bd 0.858 (-0.011)
- Qwen2.5: 0.860 vs prox_lin 0.862

**定稿建议**: `PRS = LODO(bd, T_ent_prox_lin)` — 新 variant 未超过 +0.002，保留 linear。
