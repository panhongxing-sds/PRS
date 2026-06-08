# AltMass local_spread Negative Control

`AltMass_local_spread_reason` = top-k mass on non-realized tokens **without** cross-cluster grouping (plain local spread). Cross-cluster variants compete among answer-cluster representatives.

| Dataset | Metric | cross-cluster (best) | local_spread | Δ (local − best) |
|---------|--------|---------------------:|-------------:|-----------------:|
| deepscaler | AUROC | 0.764 | 0.677 | -0.087 |
| deepscaler | matched gap (pp) | 9.1 | 6.1 | -3.0 |
| deepscaler | ΔNLL vs TW | — | -0.0104 | — |