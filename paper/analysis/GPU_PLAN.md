# PANDA 论文 — 精简执行计划

> 原则：**CPU 能做的全部 CPU 做；GPU 只跑主表绕不开的 TokUR EU。**

## 已从 TODO 中剔除（不必做）

| 原 TODO | 原因 |
|---------|------|
| DeepScaleR/GPQA/AIME 新采样 | GitHub `experiments/spurious_consensus/` 已有 K=64 + 完整 CPU 分析 |
| n∈{2..64} saturation 曲线 | 同上，`CPU_ANALYSIS_RESULTS.md` + `figures/` |
| Case study 从零写 | 已有 `case_study_12473.md` |
| σ / rank 敏感性 | 审稿 nice-to-have，非 blocking |
| Q/K vs V/O 扰动对比 | 需重跑 weight branch，ROI 低 |
| 开放域 / free-form 任务 | 超出 verifiable-answer scope |
| Related Work 批量 fake cite | 写作阶段单独做，非实验 |
| 主表 relative-gain 标注 | 排版 polish，最后做 |

## CPU 任务（<60 秒，纯 cache，不联网）

```bash
source scripts/env.sh
export PANDA_OUTPUTS=/root/autodl-tmp/panda-outputs
python scripts/run_cpu_paper_analyses.py
```

**原理：** 读 `.proc_near_answer_final_cache.pkl` + `summary.jsonl`，**不读 raw JSON、不调 HuggingFace**。

**产出：**

- `paper/analysis/cpu_results.md` — weight 准确率、collapse 子集、SE vs dissent、消融
- `paper/maintable/ablation_macro.json` — 消融表数字

**Failure mode（SC K=64）：** 已在 GitHub `experiments/spurious_consensus/`，直接复用，无需重跑。

**写作可直接引用（无需新实验）：**

- Failure mode：`experiments/spurious_consensus/CPU_ANALYSIS_RESULTS.md`
- 图：`experiments/spurious_consensus/figures/fig_main.png`, `nsweep_irreducible.png`

## GPU 任务（仅 P0：TokUR EU）

主表 TokUR 列目前缺 official pipeline 输出；这是唯一 blocking GPU 项。

```bash
export PANDA_OUTPUTS=/root/autodl-tmp/panda-outputs
export PANDA_MODELS=/root/autodl-tmp/panda-models
bash scripts/run_gpu_minimal_plan.sh          # 全自动
bash scripts/run_gpu_minimal_plan.sh --dry-run  # 只看会跑什么
```

| 项 | 值 |
|----|-----|
| 任务 | Phase C only：`run_tokur_official_maintable.sh` |
| 规模 | 4 models × 3 datasets × 3 seeds = **36 jobs** |
| 前提 | raw_runs 已存在（本地已有）；TokUR venv 需 setup |
| 机器 | RTX 5090 ×1，串行即可 |
| 预估 | ~30–60 min/job → **18–36 GPU·h**（可 resume，已有 tokur 文件会 skip） |

## 写作任务（无 GPU，与实验并行）

1. Section 4.2：贴 SC 图 + 数字（GitHub 已有）
2. 动机改写 + fairness 段落（引用 `w/o Fusion` AUROC 0.877）
3. Algorithm box + impl details（R=4, W=4, σ=0.03, r'=4, LODO LR）
4. Scope：verifiable-answer reasoning

## 执行顺序

```
Day 1  bash scripts/run_gpu_minimal_plan.sh --setup-only   # TokUR venv
       python scripts/run_cpu_paper_analyses.py            # CPU 全部分析
Day 2–3 bash scripts/run_gpu_minimal_plan.sh               # TokUR 36 jobs（可过夜）
Day 4   写作：Section 4.2 + fairness + collapse 子集段落
Day 5   Related Work + Limitations
```
