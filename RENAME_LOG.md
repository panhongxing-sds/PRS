# PANDA 重命名日志

**日期:** 2026-07-02  
**范围:** `/root/autodl-tmp/PANDA`（原 `PRS`）及同级数据目录

---

## 1. PRS → PANDA（首轮，2026-07-02 上午）

### 摘要

| 指标 | 数值 |
|------|------|
| Git 跟踪变更文件 | ~250 |
| 文本批量替换 | 223 + 17 次补丁 |
| 路径/文件 git mv | 13 |
| 单元测试 | 9 passed (`test_panda`, `test_ablation_recompute`) |

### 关键路径变更

| 旧路径 | 新路径 |
|--------|--------|
| `/root/autodl-tmp/PRS/` | `/root/autodl-tmp/PANDA/` |
| `src/prs/` | `src/panda/` |
| `src/panda/ase/prs.py` | `src/panda/ase/panda.py` |
| `tests/test_prs.py` | `tests/test_panda.py` |
| `configs/ase_models.yaml` | `configs/panda_models.yaml` |
| `scripts/aggregate_prs_v2.py` | `scripts/aggregate_panda_v2.py` |
| `paper/maintable/prs_v2_results.json` | `paper/maintable/panda_v2_results.json` |
| `paper/iclr2026/prs_iclr2026*.tex` | `paper/iclr2026/panda_iclr2026*.tex` |
| `/root/autodl-tmp/prs-models` | `/root/autodl-tmp/panda-models` |
| `/root/autodl-tmp/prs-outputs` | `/root/autodl-tmp/panda-outputs` |

### Python 包与 API

- **包名:** `prs` → `panda`
- **核心函数:** `compute_prs` → `compute_panda`，`enrich_row_with_prs` → `enrich_row_with_panda`
- **JSON 主键:** `KEY_PRS` / `"PRS"` → `KEY_PANDA` / `"PANDA"`

### 环境变量（PRS → PANDA）

| 旧 | 新 |
|----|-----|
| `PRS_ROOT` | `PANDA_ROOT` |
| `PRS_OUTPUTS` | `PANDA_OUTPUTS` |
| `PRS_MODELS` | `PANDA_MODELS` |
| `PRS_VLLM_VENV` | `PANDA_VLLM_VENV` |
| `PRS_MODEL_TAG` | `PANDA_MODEL_TAG` |
| `PRS_SE_CLUSTER` | `PANDA_SE_CLUSTER` |
| `PRS_NLI_MODEL` | `PANDA_NLI_MODEL` |
| `PRS_NLI_DEVICE` | `PANDA_NLI_DEVICE` |

### 向后兼容符号链接（仓库外，不中断在跑任务）

```
/root/autodl-tmp/PRS         -> PANDA
/root/autodl-tmp/prs-models  -> panda-models
/root/autodl-tmp/prs-outputs -> panda-outputs
```

迁移脚本: `scripts/_rename_to_panda.py`

---

## 2. ASE → PANDA / `panda.core`（第二轮，2026-07-02 下午）

论文正式方法名为 **PANDA**（*Perturbation-based Analysis of Dissent and Hesitation*），信号为 **dissent** / **hesitation** / **bd**。本轮移除代码与脚本中的 ASE 品牌，统一为 PANDA 术语；**无** `ASE_*` / `PRS_*` 环境变量向后兼容别名（用户要求 clean break）。

### 模块与文件

| 旧 | 新 |
|----|-----|
| `src/panda/ase/` | `src/panda/core/` |
| `panda.ase.*` | `panda.core.*` |
| `run_ase_experiment.py` | `run_panda_experiment.py` |
| `run_advw_ase_experiment.py` | `run_advw_panda_experiment.py` |
| `tests/test_advw_ase.py` | `tests/test_advw_panda.py` |
| `scripts/run_ase_model_pipeline.sh` | `scripts/run_panda_model_pipeline.sh` |
| `scripts/recompute_ase_metrics.sh` | `scripts/recompute_panda_metrics.sh` |
| `scripts/launch_ase_full_8gpu.sh` | `scripts/launch_panda_full_8gpu.sh` |
| `scripts/queue_ase_extra_models.sh` | `scripts/queue_panda_extra_models.sh` |
| `scripts/ase_gpu_lock.sh` | `scripts/panda_gpu_lock.sh` |

### 环境变量（ASE → PANDA，clean break）

| 旧 | 新 |
|----|-----|
| `ASE_SKIP_TOKUR` | `PANDA_SKIP_TOKUR` |
| `ASE_DYNAMIC_CLAIM` | `PANDA_DYNAMIC_CLAIM` |
| `ASE_ATTN_IMPLEMENTATION` | `PANDA_ATTN_IMPLEMENTATION` |
| `ASE_MAX_TOKENS` | `PANDA_MAX_TOKENS` |
| `ASE_MAX_TOKENS_GSM8K` | `PANDA_MAX_TOKENS_GSM8K` |
| `ASE_FAST` | `PANDA_FAST` |
| `ASE_SKIP_GENERATE` | `PANDA_SKIP_GENERATE` |
| `ASE_PARALLEL_RECOMPUTE` | `PANDA_PARALLEL_RECOMPUTE` |
| `ASE_GPU_MIN_FREE_MIB` | `PANDA_GPU_MIN_FREE_MIB` |
| `ASE_TOKUR_STRICT` | `PANDA_TOKUR_STRICT` |
| `ASE_TOKUR_PARALLEL_SHARDS` | `PANDA_TOKUR_PARALLEL_SHARDS` |
| `ASE_8B_SEQUENTIAL` | `PANDA_8B_SEQUENTIAL` |

### 输出路径与日志

| 旧 | 新 |
|----|-----|
| `outputs/ase_full` | `outputs/panda_full` |
| TokUR slug `ase_{dataset}` | `panda_{dataset}` |
| tqdm / shell: `ASE [dataset]` | `PANDA [dataset]` |
| `prs_full` 等聚合键 | `panda_full` |
| `/tmp/prs_tmp` | `/tmp/panda_tmp` |

### 刻意保留

1. **enriched JSON 列名 `T_ASE` / `TW_ASE` / `TW_ASE_H_norm` 等** — 已有 `metrics.jsonl` 产物字段；论文分析脚本仍引用这些列。新写入主键为 `PANDA`、`F_resp`、`D_ans`、`D_reason`；`resolve_f_resp` 仍可读 `TW_ASE` 作为 legacy alias。
2. **第三方 TokUR** — `third_party/TokUR/datasets/ase_*.jsonl` 上游命名未改。
3. **`.vllm_venv/`** — 虚拟环境元数据中的旧路径字符串未改。
4. **仓库外 symlink** — `/root/autodl-tmp/PRS` → `PANDA` 仍保留，供在跑 Qwen 下载 / random300 任务使用。

### 兼容说明

- **无** `ASE_*` → `PANDA_*` 双读 shim；旧 env 需手动更新。
- **无** JSON `"PRS"` 键回退（已从 `resolve_feature` 移除）。
- 在跑任务（Qwen2.5-7B modelscope 下载、Qwen3-8B `sample_vllm.py` random300）**不** import `panda.core`，不受模块重命名影响。

### 验证

```bash
cd /root/autodl-tmp/PANDA
PYTHONPATH=src python -c "import panda; from panda.core.panda import compute_panda; print(compute_panda(0.8,0.4,0.2))"
PYTHONPATH=src python -m pytest tests/ -q --ignore=tests/test_logic_code_loaders.py
```

迁移脚本: `scripts/_rename_ase_to_core.py`
