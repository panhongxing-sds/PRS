# Overfitting / Validation: TW + AltMass Fusion

5-fold stratified OOF logistic regression (L2). C chosen by inner 3-fold CV on each train fold. AltMass NaN → train-fold median impute. ΔNLL < 0 = improvement vs TW-only. Detection: higher score = more uncertain/wrong.

## Table 1: OOF metrics by method

| Dataset | Method | OOF NLL | ΔNLL vs TW | 95% CI (ΔNLL) | OOF AURC | ΔAURC | OOF AUROC |
|---------|--------|--------:|-----------:|:-------------:|---------:|------:|----------:|
| deepscaler | TW only | 0.5342 | 0.0000 | — | 0.390 | 0.000 | 0.783 |
| deepscaler | TW + final | 0.5282 | -0.0060 | [-0.0280, 0.0157] | 0.381 | -0.009 | 0.783 |
| deepscaler | TW + all AltMass | 0.5257 | -0.0086 | [-0.0379, 0.0224] | 0.395 | 0.005 | 0.781 |
| deepscaler | TW + local_spread | 0.5292 | -0.0050 | — | 0.396 | 0.006 | 0.790 |
| deepscaler | TW + best single AltMass | 0.5349 | 0.0007 | — | 0.398 | 0.008 | 0.769 |
| deepscaler | TW + shuffled AltMass | 0.5617 | 0.0275 | — | 0.420 | 0.030 | 0.754 |
| deepscaler | TW + within-TW-bin shuffled AltMass | 0.5510 | 0.0168 | — | 0.416 | 0.026 | 0.763 |
| deepscaler | TW + random Gaussian AltMass | 0.5306 | -0.0036 | — | 0.406 | 0.015 | 0.795 |

## Table 2: Permutation null (TW + all AltMass, 1000 seeds)

p-value = fraction of null runs with ΔNLL ≤ real (more negative = better).

| Dataset | Real ΔNLL | Shuffled mean±std | Within-bin shuffle | Random Gaussian | p (column) | p (bin) | p (gauss) |
|---------|----------:|------------------:|-------------------:|----------------:|-----------:|--------:|----------:|
| deepscaler | -0.0086 | 0.0213±0.0133 | 0.0100±0.0097 | 0.0214±0.0127 | 0.013 | 0.036 | 0.008 |

## Bootstrap 95% CI (Δ vs TW-only)

- **deepscaler** / TW + all AltMass: ΔNLL [-0.0379, 0.0224], ΔAURC [-0.0369, 0.0366], ΔAUROC [-0.0369, 0.0366]
- **deepscaler** / TW + final: ΔNLL [-0.0280, 0.0157], ΔAURC [-0.0430, 0.0129], ΔAUROC [-0.0430, 0.0129]

## Coefficient stability (TW + all AltMass, scaled features)

| Dataset | Feature | mean±std coef | sign consistency % |
|---------|---------|--------------:|-------------------:|
| deepscaler | TW_ASE | 0.6422±0.0699 | 100% |
| deepscaler | AltMass_final | -0.1055±0.1250 | 80% |
| deepscaler | AltMass_reason_all | 0.0622±0.0419 | 100% |
| deepscaler | AltMass_numeric | 0.0179±0.0444 | 60% |
| deepscaler | AltMass_equation | 0.1552±0.1260 | 80% |
| deepscaler | AltMass_step_end | 0.0335±0.0575 | 60% |
| deepscaler | AltMass_late_reason | 0.3339±0.0494 | 100% |
| deepscaler | AltMass_local_spread_reason | 0.3027±0.0586 | 100% |
