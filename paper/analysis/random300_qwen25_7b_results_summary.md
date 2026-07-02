# Qwen2.5-7B · DeepScaleR random300 Plan B 结果摘要

**完成时间：** 2026-07-02 15:03+08（Phase B 300/300 finals）  
**OUT：** `/root/autodl-tmp/panda-outputs/maintable_qwen25_7b_deepscaler_random300/seed41`  
**同 cohort：** `paper/analysis/deepscaler_random300_meta.json`（300 ids，与 3B Plan B 相同）

## 流水线状态

| 项 | 值 |
|----|-----|
| `raw_runs` 最终 json | **300/300**（末题 `deepscaler_9931` @ 15:03） |
| `summary.jsonl` | **300** 行 |
| `watch_random300_finish.sh` | 已触发 **metrics**（15:03 起；重算 features 较慢，summary 已齐） |
| SC@9 样本 | `experiments/spurious_consensus/data/samples/samples_qwen25_7b_seed41_deepscaler_random300_k9.jsonl` |

**SC@9 数据说明：** 300 行中有 **106** 行 `answers[]` 为空（解码未写入 k9 字段），但 `answers_orig`（K=64 前缀）可用。公平对比时 SC 取 **前 9 个非空**（先 `answers`，再 `answers_orig`），与「9 次 decode 预算」一致。详见 `random300_qwen25_7b_planb_metrics.json` 内 `sc_k9_note`。

---

## 主结果（n=300，公平 9 decode）

| 指标 | SC@9 多数票 | PANDA@9 公平多数票 | PANDA greedy a0 | Δ(PANDA@9 − SC) |
|------|-------------|-------------------|-----------------|-----------------|
| 准确率 | **50.7%** (152/300) | **58.3%** (175/300) | **55.3%** (166/300) | **+7.7 pp** |

### SC@9 分布（repair 后 n=300）

- 多数票错：148；**blind@9**（9 路全错）：**94**（31.3%）
- `p_top@9` 均值：**0.646**
- 与仅填 `answers` 可算的子集（n≈194）不同；勿用旧 `random300_qwen25_7b_k9_stats.json`（121/125 与当前 k9 文件不一致）

---

## UQ（label_wrong_clean，greedy a0 为标签）

| 集合 | n | SC 1−p_top AUROC | PANDA AUROC | Δ(PANDA−SC) |
|------|---|------------------|-------------|-------------|
| **论文 eval（无 label_drop）** | **239** | 0.789 | **0.825** | **+3.60 pp** |
| 全 300（含 drop） | 300 | 0.773 | 0.784 | +1.14 pp |

AUPRC（n=239）：SC **0.747**，PANDA **0.769**。

---

## ConfWrong@τ（n=300）

| 方法 | τ=0.9（全体率 / 给定多数票错） | τ=8/9 |
|------|-------------------------------|-------|
| SC@9 | 6.7% / 13.5% | 8.7% / 17.6% |
| PANDA@9 投票 | 4.3% / 10.4% | 6.3% / 15.2% |

（精确计数见 `random300_qwen25_7b_planb_metrics.json` → `confident_wrong_*`。）

---

## 与 Qwen2.5-3B 跨尺度（同 300 ids，`random300_planb_metrics.json`）

|  | 3B SC@9 | 3B PANDA@9 | 3B Δ vote | 7B SC@9 | 7B PANDA@9 | 7B Δ vote |
|--|---------|------------|-----------|---------|------------|-----------|
| n=300 | 44.0% | 50.7% | +6.7 pp | 50.7% | 58.3% | +7.7 pp |
| greedy a0 | 42.0% | — | −2.0 pp vs SC | 55.3% | — | +4.7 pp vs SC |

| UQ n=228/239 eval | 3B Δ(PANDA−SC) | 7B Δ(PANDA−SC) |
|-------------------|----------------|----------------|
| label_wrong_clean | **+3.95 pp** (n=228) | **+3.60 pp** (n=239) |

7B 任务准确率与 PANDA@9 均高于 3B；UQ 提升幅度与 3B 同量级（约 +3.6–4.0 pp）。

---

## 产物路径

- 指标 JSON：`paper/analysis/random300_qwen25_7b_planb_metrics.json`
- SC–PANDA join：`paper/analysis/random300_sc9_panda_join_qwen25_7b.jsonl`（300 行）
- PANDA 输出：`.../maintable_qwen25_7b_deepscaler_random300/seed41/deepscaler/{raw_runs,summary.jsonl}`
- 监控日志：`.../maintable_qwen25_7b_deepscaler_random300/logs/watch_random300_finish.log`
- 3B 对照：`paper/analysis/random300_planb_metrics.json`、`random300_planb_results_summary.md`
