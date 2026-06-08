# AltMass Decomposition Ablation (Qwen2.5-3B, clean labels)

| Dataset | Variant | AUROC | corr(TW) | matched gap (pp) | TW+var ΔNLL |
|---------|---------|------:|---------:|-----------------:|------------:|
| minerva | AltMass_final | 0.746 | 0.650 | **+15.6** | −0.0105 |
| minerva | AltMass_late_reason | 0.722 | 0.609 | **+15.6** | −0.0083 |
| minerva | AltMass_equation | 0.717 | 0.596 | +12.5 | −0.0107 |
| minerva | AltMass_reason_all | 0.728 | 0.610 | +9.4 | −0.0081 |
| minerva | AltMass_numeric | 0.677 | 0.604 | +6.2 | −0.0079 |
| math500 | AltMass_late_reason | **0.807** | 0.767 | +8.6 | −0.0058 |
| math500 | AltMass_final | 0.793 | 0.781 | +2.9 | −0.0023 |
| math500 | AltMass_numeric | 0.798 | 0.763 | +5.7 | −0.0050 |
| gsm8k | AltMass_local_spread | 0.805 | 0.500 | +5.0 | −0.0066 |
| gsm8k | AltMass_final | 0.717 | 0.663 | 0.0 | −0.0037 |

**Note:** `AltMass_final` ≡ `W_alternative_answer_mass_topk` (corr=1.0). Best fusion: Minerva `TW+final+numeric` (−0.0110); MATH-500 `TW+final+numeric` (−0.0071).
