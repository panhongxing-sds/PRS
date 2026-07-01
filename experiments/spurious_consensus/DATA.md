# 数据说明（Spurious Consensus / Self-Consistency 高温采样）

本目录下的 `data/` **已纳入 git**，可直接复现分析，无需重新采样。

## 目录结构

```
data/
├── questions/          # 2228 题共同题库（deepscaler + gpqa_diamond + aime_2024）
├── benchmarks/         # 原始 benchmark 元数据
├── samples/            # K=64 高温采样结果（temp=0.5, top_p=0.95, seed=41）
└── scr_reasoning/      # Qwen-7B SCR@1.0 子集 reasoning 重采样（t=1.0, K=64）
```

## 采样协议

| 参数 | 值 |
|------|-----|
| K | 64 |
| temperature | 0.5 |
| top_p | 0.95 |
| seed | 41 |
| 后端 | vLLM |
| 判分 | PRS `math_grader` / MCQ 字母匹配 |

## 模型 × 样本文件（canonical）

**主实验集**（6 模型 × 3 benchmark，共同题 N≈2228）：

| Tag | 文件模式 | 说明 |
|-----|----------|------|
| `qwen25_05b` | `samples_qwen25_05b_seed41_{bench}.shard{0,1,2}.jsonl` | 3 分片 |
| `qwen25_15b` | 同上 | 3 分片 |
| `qwen25_7b` | 同上 | 3 分片 |
| `llama32_1b` | `samples_llama32_1b_seed41_{bench}.jsonl` | 单文件 |
| `qwen25_3b` | `samples_qwen25_3b_seed41_{bench}.jsonl` | 单文件 |
| `phi4_mini` | `samples_phi4_mini_seed41_{bench}.s3_{0,1,2}.jsonl` | 3 分片 |

`{bench}` ∈ `deepscaler`, `gpqa_diamond`, `aime_2024`

**扩展采样**（额外 benchmark / 中间分片，共 54 个 jsonl 文件全量入库）：

- `qwen25_3b`：+ `minerva`, `competition_math_l5_500`, `math_level4plus_300`
- `llama32_1b`：+ `competition_math_l5_500`
- `gemma3_4b`：部分 deepscaler / aime（未完成全量）
- `phi4_mini`：+ 部分 `shard*` / `s3_0_h*` 中间运行文件

## jsonl 字段

每行一题：

- `id`, `benchmark`, `gold`, `grading`, `seed`
- `answers`: K 个抽取后的最终答案
- `correct`: K 个 0/1 判分
- `label_drop`: 1 表示垃圾答案过多，分析时剔除

## scr_reasoning/

Qwen-7B 在 SCR@1.0 子集（42 题）上的完整 reasoning + token logprob 重采样，用于 F7「过程多样、输出一致」分析。

路径：`data/scr_reasoning/qwen25_7b/t100/{question_id}.jsonl`

## 重新采样

若需从头生成，见 `README.md` 快速开始。模型权重不在 git 中，需自行下载到 `$PRS_MODELS`。
