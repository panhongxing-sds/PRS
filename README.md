# PANDA

**PANDA uncertainty score** — training-free LLM uncertainty / failure
detection via multi-branch perturbation (text rephrase + low-rank weight noise),
benchmarked against 11 uncertainty baselines + TokUR EU.

PANDA scores a response by three components:

- **F_resp** (`TW_ASE`): output fragmentation across the perturbation ensemble.
- **S_ans** (`AltMass_final`): answer drift / competing-answer mass (weight branch).
- **S_tr** (`AltMass_local_spread_topk`): reasoning-trace instability (domain-agnostic top-k local spread).

The full experiment plan, result-table templates, GPU-hour estimates, and the
reimbursement budget live in **[`paper/EXPERIMENT_PLAN.md`](paper/EXPERIMENT_PLAN.md)**.

---

## What's in this repo

| Path | In git | Notes |
|------|--------|-------|
| `src/panda/` | yes | Python package (pipeline, metrics, baselines, datasets) |
| `scripts/` | yes | Shell entry points |
| `configs/` | yes | Model & dataset configs (`panda_models.yaml`) |
| `tests/` | yes | unit tests (`pytest tests/`) |
| `paper/` | yes | LaTeX, result tables, `EXPERIMENT_PLAN.md` |
| `third_party/TokUR/` | yes | official [Wang-ML-Lab/TokUR](https://github.com/Wang-ML-Lab/TokUR) (unmodified) |
| `data/examples.jsonl` | yes | smoke-test data |
| `experiments/spurious_consensus/` | yes | **Self-Consistency 高温采样实验**（K=64, 6 模型 × 2228 题 + 分析脚本 + 采样数据，见 [`experiments/spurious_consensus/README.md`](experiments/spurious_consensus/README.md)） |
| `models/` | **no** | ~39 GB TFB checkpoints — download separately |
| `outputs/` | **no** | experiment runs — regenerate on GPU |

---

## 1. Setup

```bash
git clone https://github.com/panhongxing-sds/PANDA.git
cd PANDA
bash scripts/setup_after_clone.sh
source scripts/env.sh          # sets PANDA_ROOT / PYTHONPATH / PANDA_OUTPUTS / PANDA_MODELS
pytest tests/                  # 48 tests, no GPU needed
```

### 1.1 Models (not in git)

Place TFB checkpoints under `models/` (or set `PANDA_MODELS`):

```
models/TFB-Qwen2.5-3B-Instruct/     # primary model
models/TFB-Llama-3.2-1B-Instruct/
models/TFB-Llama-3.1-8B-Instruct/
models/TFB-Qwen3-8B/
```

Helper: `bash scripts/download_panda_models.sh` (and `download_llama31_8b.sh`).

### 1.2 Datasets

```bash
bash scripts/download_tokur_datasets.sh          # math: minerva / math500 / gsm8k
bash scripts/download_logic_code_datasets.sh     # logic: leg_counting / zebra_puzzles / color_cube
# Text-rephrase variants (R perturbations) need an API key:
bash scripts/generate_api_variants.sh minerva,math500,gsm8k,leg_counting,zebra_puzzles,color_cube
```

### 1.3 vLLM + official TokUR venv (required for maintable)

```bash
pip install vllm      # PANDA Phase A (match your CUDA / torch build)
bash scripts/setup_after_clone.sh   # creates .tokur_venv for official TokUR Phase C
```

TokUR baseline uses `third_party/TokUR` native `greedy_unc_single_batch_refine.py`
(`scoring_mode=official_vllm`). Do **not** use `score_tokur_baseline.py` for paper tables.

---

## 2. The experiment

**Models** (4): Qwen2.5-3B (primary) / Llama-3.2-1B / Llama-3.1-8B / Qwen3-8B
**Datasets** (6): `minerva`(272) `math500`(300) `gsm8k`(300) | `zebra_puzzles` `color_cube` `leg_counting` (各 300)
**Seeds**: 41 / 42 / 43 → mean±std
**Budget**: ¥3,000 cap on RTX 5090 (¥2.88/h) — see `paper/EXPERIMENT_PLAN.md` §12.7
**Priority**: **PANDA + TokUR EU first** (P0 main model math), then logic, then other models
**Fair sampling budget K=8** for every sampling/perturbation method:

| Method | 8 samples come from | engine |
|--------|---------------------|--------|
| PANDA / U_Ecc / U_Deg | R=4 text rephrase + W=4 weight perturb (shared) | text→vLLM, weight→HF |
| SE | 8 high-temp samples | vLLM |
| TokUR EU | official greedy + TFB EU during decode | **TokUR venv (vLLM)** |
| PE / LL / Self-Certainty / DeepConf / INSIDE / P(True) | single greedy/forward (no sampling) | HF |

Per question = 1 clean + 4 text + 4 weight + 8 SE = **17 decodes**.

---

## 3. Hybrid vLLM pipeline (how to run)

**PANDA** uses a vLLM+HF hybrid; **TokUR EU uses the official third_party/TokUR pipeline**
(separate vLLM + TFB bayesian_transformer — not the post-hoc `score_tokur_baseline` approx).

Three phases, all **resumable**:

```
Phase A  panda.core.run_vllm_phase              vLLM: clean + R + SE  → *.partial.json
Phase B  panda.core.run_panda_experiment          HF:   weight branch + metrics → raw_runs/*.json
Phase C  run_tokur_official_maintable.sh     TokUR venv: greedy_unc → tokur_baseline.jsonl
```

### Execution order (PANDA + TokUR first)

```bash
source scripts/env.sh
export MAX_SAMPLES=300   # default in configs/panda_models.yaml

# P0 — main model math: PANDA full pipeline + TokUR EU (highest priority)
DATASETS=minerva,math500,gsm8k bash scripts/run_maintable_vllm.sh qwen25_3b

# P1 — main model logic
DATASETS=leg_counting,zebra_puzzles,color_cube bash scripts/run_maintable_vllm.sh qwen25_3b

# P2 — other 3 models (math only)
for m in llama32_1b llama31_8b qwen3_8b; do
  DATASETS=minerva,math500,gsm8k bash scripts/run_maintable_vllm.sh "$m"
done

# P3 — aggregate 3-seed mean±std tables (CPU)
for m in qwen25_3b llama32_1b llama31_8b qwen3_8b; do
  bash scripts/aggregate_maintable.sh "$m"
done
```

### Run in small batches / resume

```bash
# By seed and dataset (re-invoke anytime; finished records are skipped)
SEEDS=41 DATASETS=math500 bash scripts/run_maintable_vllm.sh qwen25_3b

# Only the vLLM stage (e.g. on a vLLM-capable node)
SKIP_HF=1   bash scripts/run_maintable_vllm.sh qwen25_3b
# Only the HF weight stage + metrics (reads existing partials)
SKIP_VLLM=1 bash scripts/run_maintable_vllm.sh qwen25_3b
```

Useful env vars: `MAX_SAMPLES` (default 300), `N_REPHRASES=4`, `WEIGHT_SEEDS=42,43,44,45`,
`SE_SAMPLES=8`, `PANDA_MAX_TOKENS=2048`, `GPU_MEM_UTIL=0.90`, `CUDA_DEVICE=cuda:0`.

### Pure-HF fallback (no vLLM)

```bash
bash scripts/run_maintable_pipeline.sh qwen25_3b          # generate + aggregate
SKIP_GPU=1 bash scripts/run_maintable_pipeline.sh qwen25_3b   # aggregate existing raw only
```

### Fidelity note

vLLM returns top-k logprobs (not full vocab). Exact: chosen logprob, top-k pairs,
top-2 margin, rank → **LL / DeepConf / Self-Certainty match HF**, and all answer-based
metrics are unaffected. Approximate (top-k renormalised entropy): **PE / SAR / T-branch
ATU**. The **weight branch runs on HF**, so **PANDA's S_ans / S_tr are exact**.

---

## 4. Ablations & analysis (CPU, no GPU)

```bash
bash scripts/run_ablation_cpu.sh                       # E4 component ablation (LR refit)
python -m panda.core.ablation_recompute --budget-sweep \  # E5 perturbation-budget sweep
  --out-dir $PANDA_OUTPUTS/panda_full --datasets math500,gsm8k,minerva \
  --output $PANDA_OUTPUTS/ABLATION_budget_sweep.json
bash scripts/run_baseline_comparison.sh                # baseline comparison table
```

Outputs: main tables → `paper/maintable/{model}/maintable.{md,tex}`;
ablations → `$PANDA_OUTPUTS/ABLATION_*.{md,json}`.

---

## 5. Layout

```
src/panda/
  ase/            # pipeline: run_panda_experiment, run_vllm_phase, vllm_backend, metrics, panda, altmass_*
  baselines/      # 11 UQ baselines (registry, token_scores, sample_scores)
  datasets/       # math/logic dataset registry + loaders
  grading/        # math / logic / code answer graders
scripts/          # shell entry points (run_maintable_vllm.sh, ...)
configs/          # panda_models.yaml (models, datasets, sampling_budget)
paper/            # EXPERIMENT_PLAN.md + result tables + LaTeX
third_party/TokUR # official TokUR baseline (tracked, unmodified)
```

Paths resolve from repo root via `scripts/env.sh` and `panda.paths`.

## License

MIT — see [LICENSE](LICENSE).
