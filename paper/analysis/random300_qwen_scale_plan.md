# Plan B random300 — Qwen scale choice

## Decision (2026-07-02)

- **Selected model:** `TFB-Qwen2.5-7B-Instruct` (Plan B SC@9 + PANDA fair budget)
- **Not running:** `TFB-Qwen3-8B` random300 (user switched back to 7B for this cohort)

## Fixed cohort

- 300 question ids: `paper/analysis/deepscaler_random300_meta.json` (seed=42)
- Variants: `panda-outputs/qaac_api_bench/deepscaler_random300/variants.jsonl`
- SC@9 (CPU, first 9 of K=64, seed 41):  
  `experiments/spurious_consensus/data/samples/samples_qwen25_7b_seed41_deepscaler_random300_k9.jsonl`
- Stats: `paper/analysis/random300_qwen25_7b_k9_stats.json`

## PANDA run

- Script: `scripts/run_random300_qwen25_7b_6gpu.sh`
- Model: `/root/autodl-tmp/prs-models/TFB-Qwen2.5-7B-Instruct`
- Output: `panda-outputs/maintable_qwen25_7b_deepscaler_random300/seed41`
- Fair budget: `SE_SAMPLES=0`, `N_REPHRASES=4`, weight seeds 42–45 → 9 decodes/question
- GPUs: 6-way shard (Phase A vLLM, Phase B HF ASE), `GPU_MEMORY_UTIL=0.75` default for 7B on 5090

## Reference baselines (same 300 ids)

| Model | SC@9 file prefix | PANDA out dir |
|-------|------------------|---------------|
| Qwen2.5-3B | `samples_qwen25_3b_*_random300_k9` | `maintable_qwen25_3b_deepscaler_random300` |
| Qwen2.5-7B | `samples_qwen25_7b_*_random300_k9` | `maintable_qwen25_7b_deepscaler_random300` |
| Llama-3.2-1B | `samples_llama32_1b_*_random300_k9` | `maintable_llama32_1b_deepscaler_random300` |

## ETA (rough, 6×5090)

- 3B random300: ~few hours (completed reference)
- **7B:** ~1.5–2.5× 3B wall time (between 3B and 8B)
- 8B: ~2–3× 3B (not scheduled for this cohort)
