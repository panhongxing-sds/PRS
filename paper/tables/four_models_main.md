# Four-model main results (AUROC / AUPRC / ACC*)

ACC* = Top-50% accuracy (TokUR protocol).
Partial datasets are reported as-is (available raw/features, no 400/272 gate).

## TokUR EU

| Model | Minerva AUROC | Minerva AUPRC | Minerva ACC* | MATH-500 AUROC | MATH-500 AUPRC | MATH-500 ACC* | GSM8K AUROC | GSM8K AUPRC | GSM8K ACC* | DeepScaler AUROC | DeepScaler AUPRC | DeepScaler ACC* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B | 0.758 | 0.875 | 0.500 | 0.801 | 0.659 | 0.850 | 0.801 | 0.478 | 0.957 | 0.667 | 0.739 | 0.500 |
| Llama-3.2-1B | 0.592 | 0.840 | 0.200 | 0.727 | 0.842 | 0.430 | 0.795 | 0.801 | 0.670 | — | — | — |
| Qwen3-8B | — | — | — | — | — | — | — | — | — | — | — | — |

## PRS

| Model | Minerva AUROC | Minerva AUPRC | Minerva ACC* | MATH-500 AUROC | MATH-500 AUPRC | MATH-500 ACC* | GSM8K AUROC | GSM8K AUPRC | GSM8K ACC* | DeepScaler AUROC | DeepScaler AUPRC | DeepScaler ACC* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B | 0.822 | 0.900 | 0.575 | 0.904 | 0.856 | 0.935 | 0.961 | 0.839 | 0.995 | 0.810 | 0.858 | 0.595 |
| Llama-3.2-1B | 0.714 | 0.930 | 0.235 | 0.791 | 0.880 | 0.500 | 0.898 | 0.883 | 0.785 | 0.754 | 0.916 | 0.285 |
| Qwen3-8B | 0.802 | 0.931 | 0.351 | 0.868 | 0.807 | 0.794 | 0.903 | 0.825 | 0.910 | 0.786 | 0.959 | 0.169 |

## TW-ASE

| Model | Minerva AUROC | Minerva AUPRC | Minerva ACC* | MATH-500 AUROC | MATH-500 AUPRC | MATH-500 ACC* | GSM8K AUROC | GSM8K AUPRC | GSM8K ACC* | DeepScaler AUROC | DeepScaler AUPRC | DeepScaler ACC* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B | 0.815 | 0.887 | 0.567 | 0.896 | 0.834 | 0.940 | 0.955 | 0.807 | 0.995 | 0.806 | 0.845 | 0.585 |
| Llama-3.2-1B | 0.710 | 0.923 | 0.243 | 0.788 | 0.873 | 0.500 | 0.899 | 0.875 | 0.785 | 0.748 | 0.909 | 0.290 |
| Qwen3-8B | 0.795 | 0.924 | 0.351 | 0.866 | 0.809 | 0.778 | 0.898 | 0.811 | 0.900 | 0.785 | 0.955 | 0.169 |

## F_resp

| Model | Minerva AUROC | Minerva AUPRC | Minerva ACC* | MATH-500 AUROC | MATH-500 AUPRC | MATH-500 ACC* | GSM8K AUROC | GSM8K AUPRC | GSM8K ACC* | DeepScaler AUROC | DeepScaler AUPRC | DeepScaler ACC* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B | 0.815 | 0.887 | 0.567 | 0.896 | 0.834 | 0.940 | 0.955 | 0.807 | 0.995 | 0.806 | 0.845 | 0.585 |
| Llama-3.2-1B | 0.710 | 0.923 | 0.243 | 0.788 | 0.873 | 0.500 | 0.899 | 0.875 | 0.785 | 0.748 | 0.909 | 0.290 |
| Qwen3-8B | 0.795 | 0.924 | 0.351 | 0.866 | 0.809 | 0.778 | 0.898 | 0.811 | 0.900 | 0.785 | 0.955 | 0.169 |
