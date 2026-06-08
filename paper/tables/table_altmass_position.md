# AltMass Position Decomposition (Main Table)

Clean labels (`label_wrong_clean==0` = correct). ΔNLL = in-sample logistic fusion NLL(TW+variant) − NLL(TW only); **negative = improvement**. AURC = area under risk–coverage (abstain highest score first); **lower = better**.

## AUROC

| Dataset | Metric | final | late_reason | equation | reason_all | numeric | local_spread |
|---|---|---|---|---|---|---|---|
| deepscaler | AUROC | 0.739 | 0.764 | 0.724 | 0.734 | 0.708 | 0.677 |

## matched gap (pp)

| Dataset | Metric | final | late_reason | equation | reason_all | numeric | local_spread |
|---|---|---|---|---|---|---|---|
| deepscaler | matched gap (pp) | 6.1 | 9.1 | 3.0 | 0.0 | -3.0 | 6.1 |

## ΔNLL vs TW

| Dataset | Metric | final | late_reason | equation | reason_all | numeric | local_spread |
|---|---|---|---|---|---|---|---|
| deepscaler | ΔNLL vs TW | -0.0092 | -0.0154 | -0.0129 | -0.0118 | -0.0113 | -0.0104 |

## AURC

| Dataset | Metric | final | late_reason | equation | reason_all | numeric | local_spread |
|---|---|---|---|---|---|---|---|
| deepscaler | AURC | 0.462 | 0.459 | 0.457 | 0.464 | 0.477 | 0.528 |


---

# AltMass local_spread Negative Control

`AltMass_local_spread_reason` = top-k mass on non-realized tokens **without** cross-cluster grouping (plain local spread). Cross-cluster variants compete among answer-cluster representatives.

| Dataset | Metric | cross-cluster (best) | local_spread | Δ (local − best) |
|---------|--------|---------------------:|-------------:|-----------------:|
| deepscaler | AUROC | 0.764 | 0.677 | -0.087 |
| deepscaler | matched gap (pp) | 9.1 | 6.1 | -3.0 |
| deepscaler | ΔNLL vs TW | — | -0.0104 | — |