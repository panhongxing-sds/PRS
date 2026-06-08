# 主结果表（Clean labels：AUROC / AUPRC / ACC*）

ACC* = Top-50% accuracy（TokUR 协议：按**置信度**取前 50% 最自信样本的正确率，等价于对 EU/TU 等取负后降序）。

| Method | Minerva AUROC | Minerva AUPRC | Minerva ACC* | MATH-500 AUROC | MATH-500 AUPRC | MATH-500 ACC* | GSM8K AUROC | GSM8K AUPRC | GSM8K ACC* | DeepScaler AUROC | DeepScaler AUPRC | DeepScaler ACC* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TokUR EU | 0.758 | 0.875 | 0.500 | 0.801 | 0.659 | 0.850 | 0.786 | 0.474 | 0.951 | 0.712 | 0.782 | 0.535 |
| PRS | 0.822 | 0.900 | 0.575 | 0.904 | 0.856 | 0.935 | 0.961 | 0.839 | 0.995 | 0.810 | 0.858 | 0.595 |
| F_resp | 0.815 | 0.887 | 0.567 | 0.896 | 0.834 | 0.935 | 0.955 | 0.807 | 0.995 | 0.806 | 0.845 | 0.595 |
| D_ans | 0.772 | 0.847 | 0.530 | 0.808 | 0.661 | 0.865 | 0.778 | 0.435 | 0.960 | 0.765 | 0.803 | 0.565 |
