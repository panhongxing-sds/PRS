# Spurious Consensus 实验计划

> 项目：`/root/spurious-consensus`  
> 最后更新：2026-06-25  
> 状态：**采样与分析已完成**（Llama-3.2-1B / Phi-4-mini-3.8B / Qwen2.5-3B，各 ~2228 题）。
>
> ⚠️ 本文件为**初始计划**，记录原始设计意图。最终的实验结果、结论与图表请以
> **[`REPORT.md`](REPORT.md)** 为准。与计划的主要差异：
> - 模型从 4 个减为 **3 个**（gemma-3-4b 因吞吐过低移除）；
> - deepscaler 由 3000 裁剪为 **2000**；competition_math / math_level4plus / minerva 因模型覆盖不全已弃用，主实验集 = deepscaler+gpqa_diamond+aime_2024；
> - 采样后端改为 vLLM，单 seed=41。

---

## 0. 一句话目标

证明 **answer-level uncertainty（\(u_{\text{ans}} = 1 - p_{\text{top}}\)）随采样数 \(n\) 改善 AUROC，但无法消除高 \(p_{\text{top}}\) 稳定错答（spurious consensus）**——该现象跨模型、跨 benchmark 普遍存在，是表示/推理层面的结构性盲区，而非单纯采样预算不足。

---

## 1. 研究问题与假设

### 1.1 核心问题

| # | 问题 | 对应实验 |
|---|------|----------|
| RQ1 | 加采样能否持续提升 answer-level UQ 的判错能力？ | n-sweep AUROC 曲线 |
| RQ2 | 残余错题是否越来越「自信」？ | residual error confidence \(\mathbb{E}[p_{\text{top}} \mid \text{wrong}]\) |
| RQ3 | 高置信错题（\(p_{\text{top}}\ge0.9\)）能否被更多采样消除？ | SCR@0.9 vs \(n\) |
| RQ4 | 上述现象是否**跨模型**一致？ | 4 模型 pooled 对比 |
| RQ5 | 难题 benchmark 上盲区形态有何不同？ | per-benchmark 分型表 |

### 1.2 假设（可证伪）

- **H1（饱和）**：AUROC 在 \(n \approx 8\) 后进入平台，\(n=64\) 不再显著上升。
- **H2（虚假共识）**：错题中 split（\(p<0.3\)）占多数，但 spurious（\(p\ge0.9\)）虽少（\(\sim\)5%）却不可被 \(n\) 消除。
- **H3（跨模型）**：弱模型（1B）SCR 更高、AUROC 更低；但 H1/H2 的**定性形状**在 4 模型上重现。
- **H4（难度解耦）**：AIME 等极难题 blind 率可接近 0（离散错为主）；Deepscaler 等中等难度集 spurious 更显著——盲区不随「题难」单调增加。

---

## 2. 实验设计总览

```
Phase 0  环境与题库          [完成]
Phase 1  跨模型 K=64 采样     [进行中：3/4 模型待跑]
Phase 2  单模型分析           [Qwen 基线可参考 aul-study]
Phase 3  跨模型 pooled 对比    [待 Phase 1]
Phase 4  论文主图 + 案例      [待 Phase 3]
```

**设计原则**

- 固定采样协议（temp=0.5, top_p=0.95, K=64, seed=41），只变模型。
- 主分析在 **pooled 4262 题**上做 n-sweep；辅分析按 benchmark 分层。
- 小模型族（1B–4B）足够支撑 motivation；不跑 8B+。

---

## 3. 数据与模型

### 3.1 题库（4262 题，`data/questions/`）

| Benchmark | 题数 | 类型 | 角色 |
|-----------|-----:|------|------|
| deepscaler | 3000 | math | 主体量；spurious 主要来源 |
| competition_math_l5_500 | 500 | math | 竞赛难度 |
| math_level4plus_300 | 262 | math | MATH level≥4 |
| minerva | 272 | math | 多源数学 |
| gpqa_diamond | 198 | MCQ | 科学推理对照 |
| aime_2024 | 30 | math | 极难上限对照 |

### 3.2 模型矩阵（4 模型 × 4 厂商）

| Tag | 模型 | 厂商 | 参数量 | 上下文 | 采样状态 |
|-----|------|------|--------|--------|----------|
| `qwen25_3b` | Qwen2.5-3B-Instruct | Alibaba | 3B | 32k | **已有**（aul-study / 需迁入或重链） |
| `llama32_1b` | Llama-3.2-1B-Instruct | Meta | 1B | 128k | 待采样 |
| `phi4_mini` | Phi-4-mini-instruct | Microsoft | 3.8B | 128k | 待采样 |
| `gemma3_4b` | gemma-3-4b-it | Google | 4B | 128k | 待采样 |

模型路径见 `models.yaml`；权重目录 `$PRS_MODELS=/root/autodl-tmp/prs-models`。

### 3.3 采样协议（全实验统一）

| 参数 | 值 |
|------|-----|
| K | 64（`num_return_sequences=64`，每题一次 forward） |
| temperature | 0.5 |
| top_p | 0.95 |
| seed | 41（per-question：`seed * 100003 + qi`） |
| max_new_tokens | math 类 2048；aime_2024 4096 |
| 判分 | PRS `math_grader` / MCQ 字母匹配 |
| UQ 分数 | \(u_{\text{ans}} = 1 - p_{\text{top}}\)，\(p_{\text{top}}\) = majority 答案频率 |

---

## 4. 分阶段执行计划

### Phase 0 — 环境（已完成）

- [x] 独立仓库与 4262 题库
- [x] `sample.py` / `analyze.py` / `plot_figure.py` 流水线
- [x] 4 模型权重就位（Phi-4 7.2G + Gemma-3 8.1G 已下载）
- [ ] 将 Qwen2.5-3B 已有 `samples_*.jsonl` 迁入 `data/samples/`（或软链），避免重跑 4262×64 次

### Phase 1 — 跨模型采样（优先级最高）

**目标**：每个模型 6 个 benchmark 全量 K=64 采样。

```bash
export PRS_ROOT=/root/PRS
export PRS_MODELS=/root/autodl-tmp/prs-models
cd /root/spurious-consensus

# 按模型依次跑（单卡 4090 24GB 可跑 ≤4B）
MODEL_TAG=llama32_1b MODEL_PATH=$PRS_MODELS/TFB-Llama-3.2-1B-Instruct GPU=0 ./run_sampling.sh
MODEL_TAG=phi4_mini   MODEL_PATH=$PRS_MODELS/Phi-4-mini-instruct      GPU=0 ./run_sampling.sh
MODEL_TAG=gemma3_4b   MODEL_PATH=$PRS_MODELS/gemma-3-4b-it          GPU=0 ./run_sampling.sh
```

**产出**：`data/samples/samples_{TAG}_seed41_{benchmark}.jsonl`（每模型 6 文件，共 24 文件；不含 qwen 则 18 新文件）

**估算**（单卡 4090，按 Qwen 3B 经验外推）：

| 模型 | 相对速度 | 粗估总时长 |
|------|----------|------------|
| Llama-3.2-1B | ~1.5× 快 | 1–2 天 |
| Phi-4-mini | ~1× | 2–3 天 |
| Gemma-3-4B | ~0.8× | 3–4 天 |

**注意**

- `deepscaler` 3000 题占 70% 时间；可先跑小 benchmark 验管道，再挂 deepscaler。
- 全程 `--resume`；中断可续跑。
- 记录每文件行数 = 题数；与 `data/questions/manifest.json` 对账。

### Phase 2 — 单模型 n-sweep 分析

**目标**：每个模型一份 `results/metrics_{TAG}.json` + 主图。

```bash
# 单模型 pooled
python analyze.py \
  --samples 'data/samples/samples_qwen25_3b_seed41_*.jsonl' \
  --out results/metrics_qwen25_3b.json

python plot_figure.py \
  --metrics results/metrics_qwen25_3b.json \
  --samples 'data/samples/samples_qwen25_3b_seed41_*.jsonl' \
  --out figures/fig2_triple_qwen25_3b.png
```

对 `llama32_1b` / `phi4_mini` / `gemma3_4b` 重复。

**per-benchmark 辅表**：

```bash
python analyze_benchmarks.py \
  --samples 'data/samples/samples_{TAG}_seed41_*.jsonl' \
  --out results/benchmark_{TAG}.json
```

### Phase 3 — 跨模型对比（论文核心）

**目标**：回答 RQ4——现象是否模型无关。

| 分析项 | 方法 | 产出 |
|--------|------|------|
| AUROC@n 曲线 | 4 模型 overlay | `figures/fig_cross_model_auroc.png` |
| SCR@0.9@n 曲线 | 4 模型 overlay | `figures/fig_cross_model_scr.png` |
| K=64 错题分型 | split / moderate / spurious 堆叠柱 | `figures/fig_wrong_spectrum_4model.png` |
| 弱 vs 强 | 1B vs 3–4B 的 SCR、AUROC 差 | 正文 Table 1 |

**汇总表草稿（K=64, pooled）**

| 模型 | AUROC | wrong% | SCR@0.9 | split% | spurious% |
|------|------:|-------:|--------:|-------:|----------:|
| Qwen2.5-3B | ~0.82 | ~49% | ~5% | ~48% | ~5% |
| Llama-3.2-1B | ? | ? | ? | ? | ? |
| Phi-4-mini | ? | ? | ? | ? | ? |
| Gemma-3-4B | ? | ? | ? | ? | ? |

> Qwen 基线数字来自 `results/metrics_pooled.json`（4148 题有效子集）；全量 4262 题待迁入样本后重算。

### Phase 4 — 案例与论文素材

**目标**：2–3 个 spurious consensus 典型案例 + 1 个 split 对照。

- 从 `analyze_benchmarks.py` 的 `blind_ids` 抽取；
- 每例展示：题目摘要、gold、majority 错答、\(p_{\text{top}}\)、64 答案分布直方图；
- 优先 deepscaler / competition_math（spurious 密度高）。

**论文主图（已定稿方向）**

三联图 `figures/fig2_triple.png`：

1. **(a)** AUROC vs \(n\)（bootstrap 均值 + 95% CI）
2. **(b)** SCR@0.9 vs \(n\)（动态分母 \(|wrong@n|\)）
3. **(c)** K=64 错题 \(p_{\text{top}}\) 直方图，三区标注 0.3 / 0.9

Caption 要点：pooled 4262 题、K=64 先验采样、n-sweep 为有放回 bootstrap 子采样。

---

## 5. 指标定义（与代码一致）

| 符号/指标 | 定义 | 代码位置 |
|-----------|------|----------|
| \(p_{\text{top}}\) | majority 答案在 \(n\) 个子样本中的频率 | `analyze.py:dist_stats` |
| AUROC | 用 \(1-p_{\text{top}}\) 排序区分 majority 错/对 | `analyze.py` per-n bootstrap |
| SCR@τ | \(\|wrong@n \land p_{\text{top}}\ge\tau\| / \|wrong@n\|\) | `spurious_consensus_rate` |
| BlindSpotRate@τ | 在 **固定** \(wrong@K_{max}\) 分母上，\(\overline{p_{\text{top}}}_n\ge\tau\) 的比例 | `blindspot_rate` |
| residual error conf | \(\mathbb{E}[p_{\text{top}} \mid majority\_wrong]\) | `residual_error_conf` |
| 错题分型 | split \(p<0.3\) / moderate \(0.3\le p<0.9\) / spurious \(p\ge0.9\) | `analyze_benchmarks.py` |

**注意**：SCR（动态分母）与 BlindSpotRate（固定分母）不可混用；论文主叙事用 **SCR@0.9**。

---

## 6. 预期结果与成功标准

### 6.1 定性（必须满足）

- [ ] 4 模型 AUROC 曲线均呈「先升后平」
- [ ] 4 模型 SCR@0.9 在 \(n\ge8\) 后均呈平台（不随 \(n\) 归零）
- [ ] K=64 错题谱均呈三区：split 为主、spurious 长尾
- [ ] AIME blind 率显著低于 Deepscaler（难度解耦）

### 6.2 定量（参考 Qwen 基线，允许 ±3pt 浮动）

| 指标 | Qwen@K=64 | 期望跨模型范围 |
|------|----------|----------------|
| AUROC | ~0.82 | 0.75–0.88 |
| SCR@0.9 | ~5% | 2%–15%（1B 偏高） |
| spurious 占错题 | ~5% | 3%–10% |
| split 占错题 | ~48% | 40%–55% |

### 6.3 失败/异常处置

| 现象 | 可能原因 | 动作 |
|------|----------|------|
| 某模型 AUROC < 0.65 | 解析失败多 / prompt 不适配 | 查空答案率；调 `build_prompt` |
| SCR@0.9 > 20% | 模型过弱或 temp 过高 | 记录为 weak-model 现象，不改协议 |
| Gemma-3 加载失败 | 多模态权重 | 已用 `Gemma3ForConditionalGeneration` 纯文本路径 |
| 样本行数 < 题数 | 中断 | `--resume` 续跑 |

---

## 7. 产出清单

| 产物 | 路径 | 阶段 |
|------|------|------|
| 原始采样 | `data/samples/samples_*_seed41_*.jsonl` | P1 |
| 单模型 metrics | `results/metrics_{TAG}.json` | P2 |
| 跨 benchmark 表 | `results/benchmark_{TAG}.json` | P2 |
| 跨模型汇总 | `results/cross_model_summary.json` | P3（待写脚本） |
| 主图（单模型） | `figures/fig2_triple_{TAG}.png` | P2 |
| 主图（跨模型） | `figures/fig2_triple_pooled_4model.png` | P3 |
| 案例集 | `results/cases_spurious.jsonl` | P4 |
| 论文段落 | `sec:answer-uq-limit` 一节 | P4 |

---

## 8. 执行顺序（推荐）

```
Week 1
  Day 1   迁入 Qwen 样本 + 验 analyze 流水线
  Day 2   启动 llama32_1b 全量采样
  Day 3–4 llama32_1b deepscaler + 启动 phi4_mini

Week 2
  Day 1–3 phi4_mini 全量
  Day 4–5 gemma3_4b 全量

Week 3
  Day 1–2 4 模型 analyze + per-benchmark 表
  Day 3   跨模型对比图
  Day 4–5 案例抽取 + 论文图定稿
```

---

## 9. 待办脚本（可选增强）

| 任务 | 说明 | 优先级 |
|------|------|--------|
| `migrate_qwen_samples.sh` | 从 aul-study 软链/复制 Qwen 样本 | P0 |
| `check_samples.py` | 对账题数、空答案率、坏槽位 | P0 |
| `plot_cross_model.py` | 4 模型 AUROC/SCR overlay | P1 |
| `extract_cases.py` | 抽 spurious 案例 | P2 |
| 多 seed（41/42/43） | 稳健性；非 motivation 必需 | P3 可选 |

---

## 10. 与论文叙事的对应

| 论文 claim | 本实验证据 |
|------------|------------|
| Answer-level UQ 有上限 | AUROC 饱和曲线 |
| 更多采样不能消除盲区 | SCR@0.9 平台 |
| 盲区 = 虚假共识，非单纯难题 | 三区谱 + AIME vs Deepscaler |
| 现象非单模型偶然 | 4 厂商小模型复现 |

---

## 附录 A：Qwen 基线参考（`results/metrics_pooled.json`）

pooled 4148 题，n-sweep bootstrap B=200：

| \(n\) | AUROC | SCR@0.9 |
|------:|------:|--------:|
| 2 | 0.76 | 22% |
| 4 | 0.80 | 12% |
| 8 | 0.82 | 5% |
| 16 | 0.82 | 5% |
| 32 | 0.82 | 5% |
| 64 | 0.82 | 5% |

K=64 错题谱（占 wrong 的比例）：split ~48%，moderate ~47%，spurious ~5%。

---

## 附录 B：快速命令索引

```bash
# 采样
MODEL_TAG=phi4_mini MODEL_PATH=$PRS_MODELS/Phi-4-mini-instruct ./run_sampling.sh

# 分析
python analyze.py --samples 'data/samples/samples_phi4_mini_seed41_*.jsonl' \
  --out results/metrics_phi4_mini.json

# 主图
python plot_figure.py --metrics results/metrics_phi4_mini.json \
  --samples 'data/samples/samples_phi4_mini_seed41_*.jsonl'

# 对账
wc -l data/samples/samples_*_seed41_*.jsonl
python -c "import json; print(json.load(open('data/questions/manifest.json')))"
```
