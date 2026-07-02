# 4090 部署速查

上机后 **一条命令**（默认 HF，不装 vLLM）：

```bash
cd /root/spurious-consensus
./setup_4090.sh --smoke    # 装依赖 + 迁 Qwen 样本 + 冒烟（HF）
GPU=0 ./run_phase1.sh      # 三模型全量采样
```

可选 vLLM：`INSTALL_VLLM=1 ./setup_4090.sh` 后再用 `./run_sampling_vllm.sh`。

## 环境变量

```bash
export PANDA_ROOT=/root/PANDA
export PANDA_MODELS=/root/autodl-tmp/panda-models
```

## 脚本一览

| 脚本 | 用途 |
|------|------|
| `setup_4090.sh` | 装依赖、迁 Qwen、检查模型 |
| `run_sampling_vllm.sh` | 单模型 6 benchmark（vLLM，Gemma 自动 HF） |
| `run_phase1.sh` | Llama-1B → Phi-4 → Gemma-3 顺序全跑 |
| `migrate_qwen_samples.py` | aul-study shard → `data/samples/` |
| `check_samples.py` | 对账缺题 |
| `sample_vllm.py` | vLLM 批采样（`--batch-size` 可调） |
| `sample.py` | HF 回退 / Gemma-3 |

## 常用命令

```bash
# 对账
python check_samples.py --tag qwen25_3b
python check_samples.py --tag phi4_mini

# 单 benchmark 续跑
MODEL_TAG=phi4_mini MODEL_PATH=$PANDA_MODELS/Phi-4-mini-instruct \
  CUDA_VISIBLE_DEVICES=0 python sample_vllm.py --benchmark deepscaler --resume

# Qwen 只补 deepscaler 缺口（约 800 题）
MODEL_TAG=qwen25_3b MODEL_PATH=$PANDA_MODELS/TFB-Qwen2.5-3B-Instruct \
  ./run_sampling_vllm.sh

# OOM 时
# 编辑 config.yaml → vllm.batch_size: 2
# 或 BACKEND=hf MODEL_TAG=... ./run_sampling_vllm.sh

# 采样完成后分析
python analyze.py --samples 'data/samples/samples_phi4_mini_seed41_*.jsonl' \
  --out results/metrics_phi4_mini.json
python plot_figure.py
```

## 模型与后端

| Tag | 路径 | 采样后端 |
|-----|------|----------|
| qwen25_3b | TFB-Qwen2.5-3B-Instruct | vLLM（`.vllm_ready` 洗 config） |
| llama32_1b | TFB-Llama-3.2-1B-Instruct | vLLM |
| phi4_mini | Phi-4-mini-instruct | vLLM |
| gemma3_4b | gemma-3-4b-it | **HF**（多模态权重，自动回退） |

## 日志

`logs/phase1/{llama32_1b,phi4_mini,gemma3_4b}.log`

## 预估耗时（4090，vLLM batch=4）

| 模型 | 约 |
|------|-----|
| Llama-3.2-1B | 1–2 天 |
| Phi-4-mini | 1.5–2.5 天 |
| Gemma-3-4B (HF) | 2–4 天 |

详见 `EXPERIMENT_PLAN.md`。
