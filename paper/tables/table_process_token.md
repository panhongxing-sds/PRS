# Process-token 特征诊断（calc / tail × margin / entropy）

> Qwen2.5-3B, N=2207, κ=0.1, tail L=32, K=9 runs TopMean_κ 后均值。

**bd-only LODO macro**: 0.858

## 1. 单特征 pooled AUROC（auto-invert）

| 特征 | pooled AUROC | >0.6? |
|------|------:|:---:|
| T_calc_margin | 0.715 | ✓ |
| T_calc_entropy | 0.810 | ✓ |
| T_tail_margin | 0.606 | ✓ |
| T_tail_entropy | 0.673 | ✓ |

## 2. LODO macro（tok-only / bd+tok / Δ vs bd）

| 特征 | tok-only | bd+tok | Δ vs bd | drop tok Δ |
|------|------:|------:|------:|------:|
| T_calc_margin | 0.682 | 0.858 | -0.000 | +0.000 |
| T_calc_entropy | 0.742 | 0.859 | +0.001 | -0.001 |
| T_tail_margin | 0.477 | 0.856 | -0.002 | +0.002 |
| T_tail_entropy | 0.498 | 0.861 | +0.003 | -0.003 |

## 3. 分数据集 bd+tok vs bd-only

### T_calc_margin
| ds | tok pooled | bd+tok LODO | bd-only | Δ fusion | drop tok |
|----|------:|------:|------:|------:|------:|
| minerva | 0.720 | 0.856 | 0.846 | +0.010 | -0.010 |
| math500 | 0.674 | 0.923 | 0.927 | -0.004 | +0.004 |
| gsm8k | 0.652 | 0.794 | 0.801 | -0.007 | +0.007 |

### T_calc_entropy
| ds | tok pooled | bd+tok LODO | bd-only | Δ fusion | drop tok |
|----|------:|------:|------:|------:|------:|
| minerva | 0.768 | 0.862 | 0.846 | +0.016 | -0.016 |
| math500 | 0.746 | 0.930 | 0.927 | +0.003 | -0.003 |
| gsm8k | 0.715 | 0.786 | 0.801 | -0.015 | +0.015 |

### T_tail_margin
| ds | tok pooled | bd+tok LODO | bd-only | Δ fusion | drop tok |
|----|------:|------:|------:|------:|------:|
| minerva | 0.619 | 0.841 | 0.846 | -0.005 | +0.005 |
| math500 | 0.527 | 0.909 | 0.927 | -0.018 | +0.018 |
| gsm8k | 0.576 | 0.818 | 0.801 | +0.017 | -0.017 |

### T_tail_entropy
| ds | tok pooled | bd+tok LODO | bd-only | Δ fusion | drop tok |
|----|------:|------:|------:|------:|------:|
| minerva | 0.650 | 0.851 | 0.846 | +0.004 | -0.004 |
| math500 | 0.515 | 0.914 | 0.927 | -0.014 | +0.014 |
| gsm8k | 0.627 | 0.818 | 0.801 | +0.018 | -0.018 |

## 4. 结论

- **最佳 bd+tok**：tail_ent macro=0.861 (Δ=+0.003 vs bd-only 0.858)
- cliff 已弃用；本表不含 cliff。
