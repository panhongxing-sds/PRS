# BD 拆分 / 加权 / 交互 — 最后一轮验证

> N=8121，4 模型，LODO OOF LR，后处理 only。

**D_base macro**: 0.897  |  **当前 ref D_base+TW_ent**: 0.896

## 8 组配置 macro LODO

| # | 配置 | macro | Δ vs D_base | Qwen2.5 | Llama3.2 | Llama3.1 | Qwen3 |
|---|------|------:|---:|---:|---:|---:|---:|
| 1 D_base | D_base | 0.897 | +0.000 | 0.858 | 0.831 | 0.858 | 0.942 |
| 2 D_text+D_weight | D_text, D_weight | 0.898 | +0.001 | 0.855 | 0.836 | 0.857 | 0.942 |
| 3 D_text+D_weight+D_min | D_text, D_weight, D_min | 0.898 | +0.001 | 0.857 | 0.837 | 0.857 | 0.942 |
| 4 D_base+TW_ent | D_base, TW_ent_sum | 0.896 | -0.002 | 0.873 | 0.836 | 0.854 | 0.945 |
| 5 D_base+D_conf | D_base, D_conf | 0.900 | +0.002 | 0.860 | 0.829 | 0.860 | 0.944 |
| 6 D_base+D_unc | D_base, D_unc | 0.897 | -0.000 | 0.857 | 0.828 | 0.856 | 0.941 |
| 7 D_base+TW+inter | D_base, TW_ent_sum, I_bd_ent | 0.894 | -0.003 | 0.870 | 0.834 | 0.853 | 0.943 |
| 8 D_tw+D_min+TW | D_text, D_weight, D_min, TW_ent_sum | 0.898 | +0.001 | 0.874 | 0.838 | 0.856 | 0.945 |

## 分数据集 drop-one（full vs D_base-only，负=融合更好）

### qwen25_3b
| ds | D_base | best cfg | best | Δ | #4 TW |
|----|---:|---|---:|---:|---:|
| minerva | 84.63 | 8 D_tw+D_min+TW | 86.54 | +1.91pp | 86.43 |
| math500 | 92.73 | 8 D_tw+D_min+TW | 93.22 | +0.49pp | 92.91 |
| gsm8k | 80.07 | 7 D_base+TW+inter | 82.56 | +2.50pp | 82.46 |

### llama32_1b
| ds | D_base | best cfg | best | Δ | #4 TW |
|----|---:|---|---:|---:|---:|
| minerva | 75.51 | 3 D_text+D_weight+D_min | 76.21 | +0.70pp | 75.66 |
| math500 | 85.04 | 8 D_tw+D_min+TW | 86.87 | +1.84pp | 85.86 |
| gsm8k | 88.83 | 4 D_base+TW_ent | 89.25 | +0.42pp | 89.25 |

### llama31_8b
| ds | D_base | best cfg | best | Δ | #4 TW |
|----|---:|---|---:|---:|---:|
| minerva | 77.50 | D_base | 77.50 | +0.00pp | 77.13 |
| math500 | 90.11 | 2 D_text+D_weight | 90.59 | +0.48pp | 90.18 |
| gsm8k | 89.73 | 5 D_base+D_conf | 90.08 | +0.35pp | 89.01 |

### qwen3_8b
| ds | D_base | best cfg | best | Δ | #4 TW |
|----|---:|---|---:|---:|---:|
| minerva | 92.90 | 6 D_base+D_unc | 93.26 | +0.35pp | 92.71 |
| math500 | 97.48 | 8 D_tw+D_min+TW | 97.53 | +0.06pp | 97.51 |
| gsm8k | 92.14 | 4 D_base+TW_ent | 93.24 | +1.10pp | 93.24 |

## 判定

- **最佳配置**: 5 D_base+D_conf macro=0.900 (Δ=+0.002)
- 4模型 macro 超过 D_base: True
- 4模型中 ≥D_base: 3/4
- 12格 fusion≥bd-only: 10/12
- Llama3.1-8B: 0.860 vs bd 0.858 (+0.002)
- Qwen2.5 vs ref#4: 0.860 vs 0.873

**建议定稿**: `5 D_base+D_conf` — D_base, D_conf
