# AltMass Final vs Reasoning Dominance

ΔNLL_final = NLL(TW+final)−NLL(TW); ΔNLL_reason_best = best among {late_reason, equation, reason_all, numeric, step_end}. Reasoning_dominance = ΔNLL_final − ΔNLL_reason_best (**positive ⇒ reasoning helps more**). D_ratio = |ΔNLL_reason_best| / |ΔNLL_final|.

| Dataset | ΔNLL_final | ΔNLL_reason_best | best variant | Reasoning_dom | D_ratio | Taxonomy |
|---------|----------:|-----------------:|--------------|--------------:|--------:|----------|
| deepscaler | -0.0092 | -0.0154 | late_reason | 0.0062 | 1.68 | — |

## Interpretation

- **Minerva:** commitment-boundary (`AltMass_final`) carries most matched-pair / fusion gain; reasoning adds marginal ΔNLL.
- **MATH-500:** reasoning-stage variants (esp. late_reason, numeric) often beat final in AUROC and ΔNLL.
- **GSM8K:** high standalone AUROC for local_spread but near-zero fusion ΔNLL — ranking without cluster alignment.