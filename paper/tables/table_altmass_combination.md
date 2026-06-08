# AltMass Combination Ablation

In-sample logistic fusion; ΔNLL vs TW-only (lower NLL = better).

## deepscaler

| Combination | NLL | ΔNLL vs TW |
|-------------|----:|-----------:|
| TW only | 0.5296 | 0.0000 |
| TW + final | 0.5204 | -0.0092 |
| TW + best reasoning (late_reason) | 0.5142 | -0.0154 |
| TW + final + best reasoning | 0.5128 | -0.0168 |
| TW + all AltMass (7) | 0.5045 | -0.0250 |
| TW + local_spread | 0.5191 | -0.0104 |

## Cross-dataset claims

- **TW + final + best reasoning** improves over **TW + final** when reasoning_dominance > 0 (esp. MATH-500).
- **TW + local_spread** ≈ **TW only** on Minerva (ΔNLL ≈ 0); does not substitute for cluster-aligned AltMass.