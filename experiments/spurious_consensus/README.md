# Spurious Consensus（Self-Consistency 高温采样实验）

跨模型验证：**多采样（self-consistency）改善 answer-level AUROC，但无法消除高置信度的稳定错误（spurious consensus / self-consistent collapse）**。

> 本目录是 PRS 仓库下的独立实验子项目；采样数据已随仓库分发，分析脚本可直接运行。

## TL;DR

- **命题**：self-consistency 提升 AUROC，却留下「自信地错」的题（majority 判错且 `p_top` 高），占比随模型能力上升。
- **指标**：`u_ans = 1 − p_top`；`SCR@τ` = majority 判错且 `p_top ≥ τ` 占错题比例。
- **协议**：K=64, temp=0.5, top_p=0.95, seed=41。
- **数据**：6 模型 × 3 benchmark × 2228 共同题（见 [`DATA.md`](DATA.md)）。

## 模型（能力梯度）

| Tag | 模型 | 参数 |
|-----|------|------|
| `qwen25_05b` | TFB-Qwen2.5-0.5B-Instruct | 0.5B |
| `llama32_1b` | TFB-Llama-3.2-1B-Instruct | 1B |
| `qwen25_15b` | TFB-Qwen2.5-1.5B-Instruct | 1.5B |
| `phi4_mini` | Phi-4-mini-instruct | 3.8B |
| `qwen25_3b` | TFB-Qwen2.5-3B-Instruct | 3B |
| `qwen25_7b` | TFB-Qwen2.5-7B-Instruct | 7B |

## 数据集

| Benchmark | 题数 | 类型 |
|-----------|-----:|------|
| deepscaler | 2000 | math |
| gpqa_diamond | 198 | MCQ |
| aime_2024 | 30 | math |

## 快速开始（复现分析，无需 GPU）

```bash
cd experiments/spurious_consensus
pip install -r requirements.txt

# 六模型汇总 + n-sweep + collapse 图
python analyze_all_models.py

# 分 benchmark 统计
python analyze_benchmarks.py --samples "data/samples/samples_*"

# SCR@1.0 reasoning 多样性分析（需 scr_reasoning 数据）
python analyze_scr_reasoning_t100.py
```

输出：`results/*.json`，`figures/*.png`

## 重新采样（需 GPU + 模型权重）

```bash
bash setup_4090.sh
export PRS_MODELS=/path/to/models
python build_questions.py
python trim_deepscaler.py
GPU=0 MODEL_TAG=qwen25_3b bash run_sampling.sh
python clean_samples.py
```

## 目录结构

```
├── config.yaml / models.yaml   # 配置
├── sample_vllm.py              # vLLM K=64 高温采样（主入口）
├── sampling_utils.py           # 模型加载、TFB 权重清洗
├── clean_samples.py            # 答案清洗与重判分
├── analyze.py                  # 单模型 n-sweep / AUROC / SCR
├── analyze_all_models.py       # 六模型汇总 + 出图
├── analyze_benchmarks.py         # 分 benchmark
├── analyze_scr_reasoning_t100.py # reasoning 路径多样性
├── save_scr_reasoning.py         # SCR 子集 reasoning 重采样
├── data/                         # 题库 + 采样 + scr_reasoning（见 DATA.md）
├── results/                      # 指标 JSON
├── figures/                      # 论文图与 case study
├── REPORT.md                     # 完整实验报告
└── EXPERIMENT_PLAN.md            # 原始计划
```

## 主要结论

详见 [`REPORT.md`](REPORT.md)。核心发现：

1. SCR 随模型能力单调上升（0.5B 0.8% → 7B 10.5% 占错题）
2. SCR 在 n≥8 后进入平台，多采样无法消除
3. 64 条 reasoning 路径可完全不同却仍给出同一错答（非复制粘贴）
4. 虚假共识跨模型共享，是题目×能力诱发的内在陷阱
