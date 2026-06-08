# Prefix-Level AltMass Trajectory (T_k vs A_k)

Per reasoning step k (newline boundaries, weight-perturb branch):
- **A_k** = W-ASE from provisional answers (last numeric in prefix k) across weight runs
- **T_k** = cross-cluster AltMass on math tokens in step k

Spike threshold = 0.3; future fragmentation = ΔA > 0.05 or cluster increase.

## Main AUROC table

| Dataset | T_k→final wrong | A_k→final wrong | T_k→future frag | T_k→cluster switch | early⅓ max T | early⅓ max A |
|---------|----------------:|----------------:|----------------:|-------------------:|-------------:|-------------:|
| deepscaler | 0.502 | 0.635 | 0.455 | 0.466 | 0.503 | 0.544 |

## AUROC by relative step bin

| Dataset | T_early→wrong | T_mid→wrong | T_late→wrong | T_early→future frag |
|---------|--------------:|------------:|-------------:|--------------------:|
| deepscaler | 0.473 | 0.519 | 0.524 | 0.408 |

## Lead time (wrong samples only)

| Dataset | mean lead steps (A−T) | frac T strictly before A | rel pos T spike | rel pos A spike | early⅓ max T | early⅓ max A |
|---------|----------------------:|-------------------------:|----------------:|----------------:|-------------:|-------------:|
| deepscaler | 3.48 | 81.3% | 0.00 | 0.21 | 0.503 | 0.544 |

## Interpretation

- **Q1** T_k → future A rise: see `T_k_to_future_frag`
- **Q2** T_k → cluster switch: see `T_k_to_cluster_switch`
- **Q3** T_k → final wrong earlier than A_k: compare `early_max_T` vs `early_max_A` and lead time
