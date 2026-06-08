# PRS

**Perturbation Reliability Score (PRS)** — LLM uncertainty / failure detection via ASE multi-branch perturbation + TokUR baselines.

## What's in this repo

| Path | In git | Size (approx.) |
|------|--------|----------------|
| `src/`, `scripts/`, `configs/`, `tests/` | yes | code |
| `paper/` | yes | tables & LaTeX |
| `third_party/TokUR/` | yes | ~150 MB (baseline code + small datasets) |
| `data/examples.jsonl` | yes | smoke-test data |
| `models/` | **no** | ~39 GB TFB checkpoints — download separately |
| `outputs/` | **no** | ~85 GB experiment runs — regenerate on GPU |

## After clone

```bash
git clone https://github.com/panhongxing-sds/PRS.git
cd PRS
bash scripts/setup_after_clone.sh
source scripts/env.sh
pytest tests/
```

Download TFB model weights (not in git) and place under `models/`, e.g.:

```
models/TFB-Qwen2.5-3B-Instruct/
models/TFB-Llama-3.2-1B-Instruct/
...
```

Optional: download full benchmark datasets if not already in `third_party/TokUR/datasets/`:

```bash
bash scripts/download_tokur_datasets.sh
```

## Quick commands

```bash
# Regenerate paper tables (CPU only, needs outputs/)
MAX_JOBS=8 SKIP_RECOMPUTE=1 bash scripts/fast_four_model_tables.sh

# ASE pipeline (needs GPU + models/)
bash scripts/run_ase_model_pipeline.sh qwen25_3b models/TFB-Qwen2.5-3B-Instruct

# Official TokUR baseline (needs vLLM + GPU)
bash scripts/launch_official_tokur_all.sh
```

## Layout

```
src/prs/          # Python package
scripts/          # Shell entry points
configs/          # Model & dataset configs
paper/            # LaTeX + result tables
third_party/TokUR # Official TokUR baseline (tracked)
data/             # Small example data
models/           # TFB checkpoints (local only)
outputs/          # ASE / TokUR experiment outputs (local only)
```

Paths resolve from repo root via `scripts/env.sh` and `prs.paths`.

## License

MIT — see [LICENSE](LICENSE).
