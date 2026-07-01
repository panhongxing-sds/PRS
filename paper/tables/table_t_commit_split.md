# T_commit 诊断（cliff vs 边界前/后拆分）

> N=8121，multiprocess CPU（JSON 解析无 GPU 加速意义）。

## 1. 四模型 pooled 单特征 AUROC（auto-invert）

| 特征 | AUROC | 说明 |
|------|------:|------|
| bd | 0.925 | 8 票否决 |
| T_cliff_u | 0.539 | −(lp_before−lp_after) 当前 T_commit |
| T_pre_u | 0.500 | −lp_before 边界前 |
| T_post_u | 0.564 | −lp_after 答案 token |

## 2. LODO 融合

| 配置 | AUROC |
|------|------:|
| bd + T_cliff **(当前)** | 0.895 |
| bd + T_pre | 0.896 |
| bd + T_post | 0.900 |
| bd + T_pre + T_post **(拆开)** | 0.898 |

## 3. Qwen2.5-3B 分数据集

| ds | bd | T_cliff | T_pre | T_post |
|----|---:|--------:|------:|-------:|
| minerva | 0.846 | 0.662 | 0.680 | 0.580 |
| math500 | 0.927 | 0.515 | 0.548 | 0.561 |
| gsm8k | 0.801 | 0.720 | 0.689 | 0.606 |
