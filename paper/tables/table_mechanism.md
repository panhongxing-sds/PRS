# Mechanism Analysis: Token (T₀, DeepConf) vs Answer (bd, F_resp)

> N=8121 samples, 4 models × 3 math datasets × 3 seeds.
> T₀ = cliff on **base run only**; bd = 8-vote disagree rate; split = 1[bd>0].

## I. Token → Answer split (predictive)

| Predictor | AUROC → split (bd>0) |
|-----------|---------------------:|
| T0 | 0.501 |
| F | 0.962 |
| dc_min | 0.667 |

## II. Pattern enrichment P(T₀ low ∧ bd>0)

| Dataset | P(wrong) | P(correct) | Enrichment |
|---------|----------:|-----------:|-----------:|
| Minerva | 42.1% | 41.5% | 1.014× |
| MATH-500 | 39.2% | 19.4% | 2.027× |
| GSM8K | 53.0% | 16.9% | 3.130× |

## III. Run-level: P(flip | cliff bin)

| cliff bin mid | P(flip) | n |
|--------------:|--------:|--:|
| -0.236 | 31.3% | 12994 |
| -0.019 | 30.9% | 12993 |
| -0.000 | 44.6% | 12994 |
| 0.000 | 32.8% | 12993 |
| 0.075 | 35.5% | 12994 |

## IV. Conditional AUROC (wrong detection)

| Subset | n | wrong% | T0 | bd | F | dc_min |
|--------|--:|-------:|--:|--:|--:|--:|
| all | 8121 | 40.6% | 0.517 | 0.916 | 0.885 | 0.729 |
| bd_eq_0 | 3126 | 4.9% | 0.571 | 0.500 | 0.618 | 0.711 |
| bd_gt_0 | 4995 | 63.0% | 0.538 | 0.859 | 0.795 | 0.700 |
| bd0_wrong | 154 | 100.0% | — | — | — | — |
| bd0_correct | 2972 | 0.0% | — | — | — | — |

## V. Complementarity ladder (LODO LR AUROC)

- **L0_dc_min**: 0.676
- **L1_dc_T0**: 0.674
- **L2_dc_T0_bd**: 0.903
- **L3_dc_T0_bd_F**: 0.904
- **PANDA_full**: 0.895

## VI. Residual AUROC (orthogonal component)

- residual_bd_given_T0: 0.917
- residual_T0_given_bd: 0.598

## VII. Quadrant wrong rates (T₀ × bd)

| Quadrant | n | wrong% | mean bd | mean T₀ |
|----------|--:|-------:|--------:|--------:|
| T_low_bd_low | 2282 | 9.6% | 0.052 | -0.105 |
| T_low_bd_high | 1778 | 75.3% | 0.708 | -0.107 |
| T_high_bd_low | 2152 | 9.3% | 0.052 | 0.025 |
| T_high_bd_high | 1909 | 80.8% | 0.709 | 0.034 |

## VIII. Perturbation heterogeneity

- mean bd_text=0.440  mean bd_weight=0.260
- ρ(T₀, bd_text)=0.019  ρ(T₀, bd_weight)=-0.001

## Key Spearman correlations

- T0~F: -0.001
- T0~bd: 0.012
- T0~dc_min: -0.189
- T0~y: 0.029
- bd~F: 0.882
- bd~y: 0.731
