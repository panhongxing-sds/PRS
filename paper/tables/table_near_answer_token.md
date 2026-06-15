# 近答案加权 process-token 指标

> N=8121。权重 $w_t\propto$ 距 answer 越近越大（linear / quad）。

bd-only=0.897, bd+TW_ent_sum=0.896

| 特征 | bd+T | Δ | Qwen2.5 | pooled | 说明 |
|------|------:|---:|---:|---:|------|
| `T_ent_prox_lin` | 0.904 | +0.007 | 0.862 | 0.731 | 近答案加权 mean entropy（linear） |
| `T_margin_prox_lin` | 0.904 | +0.007 | 0.863 | 0.679 | 近答案加权 −margin（尾部犹豫） |
| `T_ent_prox_quad` | 0.904 | +0.006 | 0.864 | 0.738 | 近答案加权 mean entropy（quad，更强调尾部） |
| `T_support_tail_miss` | 0.900 | +0.003 | 0.868 | 0.655 | 答案是否缺失于最后25% reasoning |
| `T_calc_edit_tail_pair` | 0.898 | +0.001 | 0.867 | 0.769 | 尾部计算序列 pairwise edit |
| `T_calc_edit_tail` | 0.898 | +0.001 | 0.871 | 0.759 | 最后25% reasoning 计算序列 vs base edit |
