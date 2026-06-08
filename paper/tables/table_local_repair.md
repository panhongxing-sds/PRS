# T-Triggered Local Repair (API Pilot (Prompt 2))

**Cohort:** 25 wrong + 25 correct-but-high-T (MATH-500).
**Prompt 2:** prefix recomputation at flagged step (not full rewrite).
**full_SC:** SC@3 full recomputation from question only.

Framing: test **action value** — localization for cost-effective repair.

| Dataset | Method | Acc | ΔAcc | Corr Yield | Damage | Avg Extra Tok | Gain/1k tok | McNemar p |
|---------|--------|----:|-----:|-----------:|-------:|--------------:|------------:|----------:|
| math500 | baseline_a0 | 0.500 | +0.000 | 0.0% | 0.0% | 0 | nan | 1.000 |
| math500 | random_local | 0.460 | -0.040 | 24.0% | 32.0% | 1144 | -0.0350 | 0.791 |
| math500 | Tmax_local | 0.400 | -0.100 | 24.0% | 44.0% | 988 | -0.1012 | 0.332 |
| math500 | full_SC | 0.040 | -0.460 | 4.0% | 96.0% | 1817 | -0.2532 | 0.000 |

## Success criteria (pilot)

1. **Tmax_local > random_local** in correction yield.
2. **Damage rate** not much higher than random.
3. **Gain/token** better than full_SC (does not need to beat full_SC accuracy).

- Tmax vs random correction yield: **24.0%** vs **24.0%** (✗)
- Tmax gain/1k vs full_SC: **-0.1012** vs **-0.2532** (✓)