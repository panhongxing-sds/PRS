# Large Files / Directories Audit

**Scan date:** 2026-07-02 (CST)  
**Scope:** `/root/autodl-tmp`, `/root` (same filesystem as autodl-tmp data disk)  
**Methods:** `du -sh` (top-level and targeted breakdowns), `find … -size +100M`, byte-accurate `du -sb` for candidates  
**Disk:** `/dev/md0` mounted on `/root/autodl-tmp` — **200G total, ~182G used, ~19G free (91%)**  
**Policy:** Advisory only — **nothing deleted** in this scan.

---

## Executive summary

| Category | Approx. size |
|----------|--------------|
| **KEEP (essential)** | ~140G (models, maintable + random300 outputs, PANDA repo, miniconda) |
| **MAY DELETE (safe / obvious junk)** | ~48G (caches, temps, duplicate venv, archive tarball, duplicate Qwen2.5 copy) |
| **CONDITIONAL (large, needs explicit sign-off)** | ~93G (`raw_runs` under `panda-outputs`; aggregated `features.jsonl` / `summary.jsonl` already present) |

---

## All major items (sorted by size, descending)

| Size | Path | Verdict | Notes |
|------|------|---------|-------|
| 93G | `/root/autodl-tmp/panda-outputs` | **KEEP** (maintable + random300) | See breakdown; ~entire tree is `raw_runs` per-dataset JSON shards |
| 39G | `panda-outputs/maintable_qwen3_8b` | **KEEP** | Active maintable (3 seeds × math500/minerva/gsm8k) |
| 26G | `panda-outputs/maintable_qwen25_3b` | **KEEP** | Includes large `zebra_puzzles` raw runs |
| 13G | `panda-outputs/maintable_llama32_1b` | **KEEP** | Llama 3.2 1B maintable |
| 12G | `panda-outputs/maintable_llama31_8b` | **KEEP** | Llama 3.1 8B maintable / baseline in progress |
| 46G | `/root/autodl-tmp/panda-models` | **KEEP** | All TFB weights + deberta NLI |
| 5.8G | `panda-models/.vllm_ready/` | **MAY DELETE** (after symlink) | Duplicate of `TFB-Qwen2.5-3B-Instruct` (~5.75G); `run_random300_6gpu.sh` points here; configs differ slightly |
| 11.6G | `/root/autodl-tmp/panda-vllm-venv` | **MAY DELETE** | Standalone vLLM/CUDA venv; **no references** in `PANDA/scripts/*.sh`; overlaps `miniconda3` + `.tokur_venv` |
| 11.4G | `/root/autodl-tmp/archive/panda_full_20260615.tar.zst` | **MAY DELETE** | 2026-06-15 full backup (manifest: PANDA + panda-outputs + logs); safe if OSS/offsite copy verified |
| 11G | `/root/miniconda3` | **KEEP** | Primary conda env (torch, vllm) |
| 7.9G | `/root/autodl-tmp/PANDA` | **KEEP** (mostly) | Repo + paper; see `.tokur_venv` below |
| 7.3G | `PANDA/.tokur_venv` | **MAY DELETE** | TokUR/Ray/CuPy stack; only needed if running TokUR baseline scripts (`run_tokur_*.sh`, `run_gpu_minimal_plan.sh`) |
| 6.5G | `/root/autodl-tmp/pip_cache` | **MAY DELETE** | Pip HTTP cache (re-download on install) |
| 5.1G | `/root/autodl-tmp/tmp` | **MAY DELETE** | Stale `pip-unpack-*` wheels and temp dirs |
| 0.92G | `/root/autodl-tmp/pip-cache` | **MAY DELETE** | Second pip cache path (duplicate of above pattern) |
| 2.5G | `panda-outputs/maintable_qwen25_3b_deepscaler_random300` | **KEEP** | random300 deepscaler (K pipeline) |
| 2.4G | `panda-outputs/maintable_llama32_1b_deepscaler_random300` | **KEEP** | random300 deepscaler |
| 0.94G | `/root/.cursor-server` | **MAY DELETE** | Remote IDE server; re-fetches on reconnect |
| 0.42G | `/root/.cache/vllm` | **MAY DELETE** | torch_compile_cache + modelinfos |
| 0.34G | `PANDA/experiments/spurious_consensus` | **MAY DELETE** | Side experiment data (337M in `data/`); not maintable |
| 18M | `panda-outputs/_oom_probe_llama31_8b` | **MAY DELETE** | OOM probe leftovers (`minerva/raw_runs` ~18M) |
| 9.1M | `panda-outputs/qaac_api_bench` | **MAY DELETE** | API bench scratch |
| 5.3M | `/root/.cursor/projects/.../agent-transcripts` | **MAY DELETE** | Cursor agent chat logs |
| 54M | `.specstory` (autodl-tmp + /root) | **MAY DELETE** | SpecStory history |
| 26M | `PANDA/.vllm_venv` | **KEEP** | Small helper venv |
| 0 | scr500 / strong output dirs | **(already gone)** | Only `.CANCELLED_BY_USER` script stubs remain (KB-scale) |

---

## `panda-outputs` detail (KEEP vs trim)

### KEEP per project policy

- All `maintable_*` trees (seeds 41–43, datasets math500, minerva, gsm8k, zebra_puzzles, etc.)
- Both `*_deepscaler_random300` trees (~4.9G combined; logs are tiny, not multi-GB vllm logs)

### Small MAY DELETE under `panda-outputs`

| Size | Path | Reason |
|------|------|--------|
| 18M | `_oom_probe_llama31_8b` | Diagnostic probe, not paper data |
| 9.1M | `qaac_api_bench` | Benchmark scratch |
| ~1M | `gpu_plan_logs`, `bench`, `logs` | Orchestration logs |

### CONDITIONAL — largest recoverable space (not “junk”)

**~93G total `raw_runs/`** under `panda-outputs` (verified: `du -ch …/raw_runs` ≈ 93G).

Each completed dataset also has **`features.jsonl` + `summary.jsonl`** (~MB scale). Deleting `raw_runs` only after:

- Tables/paper figures finalized from aggregates
- K=64 / random300 exports verified
- No need to re-run NLI or re-parse completions

Example structure:

```
maintable_qwen25_3b/seed41/zebra_puzzles/
  features.jsonl   7.7M
  summary.jsonl    7.7M
  raw_runs/        3.4G   ← bulk
```

**Do not delete** without explicit approval — this is most of the experiment record.

---

## Duplicate / redundant model weights

| Size | Item |
|------|------|
| 5.8G | `panda-models/TFB-Qwen2.5-3B-Instruct` |
| 5.8G | `panda-models/.vllm_ready/TFB-Qwen2.5-3B-Instruct` |

Same shard filenames; `config.json` differs (vLLM-ready tweak). **Recovery:** point scripts at one tree or symlink; remove the other → **~5.75G**.

Other models (Llama 3.1 8B, Qwen3 8B, Llama 3.2 1B, deberta) — **no duplicate trees detected**.

---

## Large individual files (>100MB, top samples)

| Size | File |
|------|------|
| 11.4G | `archive/panda_full_20260615.tar.zst` |
| 4.7G | `panda-models/TFB-Llama-3.1-8B-Instruct/model-0000{1,2}-of-00005.safetensors` |
| 4.6G | `panda-models/TFB-Llama-3.1-8B-Instruct/model-00003-of-00005.safetensors` |
| 3.8G | `panda-models/TFB-Qwen3-8B/model-0000{1,2}-of-00006.safetensors` |
| 3.7G | `panda-models/TFB-Qwen2.5-3B-Instruct/model-00001-of-00003.safetensors` (+ duplicate under `.vllm_ready`) |
| 2.4G | `panda-models/TFB-Llama-3.2-1B-Instruct/model.safetensors` |
| 1.7G | `panda-models/deberta-v2-xlarge-mnli/pytorch_model.bin` |
| 0.78G | `pip_cache/.../c0a5a9....body` (cached wheel) |
| 0.63G | `tmp/pip-unpack-*/nvidia_cudnn_cu12-*.whl` (multiple copies) |

---

## `/root` outside autodl-tmp (on overlay root FS)

| Size | Path | Verdict |
|------|------|---------|
| 11G | `miniconda3` | **KEEP** |
| 0.94G | `.cursor-server` | **MAY DELETE** |
| 0.45G | `.cache` (mostly vllm) | **MAY DELETE** partial |
| 43M | `.nv` | **KEEP** |
| 24M | `.specstory` | **MAY DELETE** |

---

## Top 10 deletable candidates (by recoverable GB, advisory)

Sorted for **safe or low-risk** cleanup first; **excludes** maintable/random300/model weights unless noted.

| Rank | ~GB | Path / item | Brief reason |
|------|-----|-------------|--------------|
| 1 | **11.6** | `autodl-tmp/panda-vllm-venv` | Extra full GPU Python env; scripts use conda/TokUR venv instead |
| 2 | **11.4** | `autodl-tmp/archive/panda_full_20260615.tar.zst` | Local compressed snapshot; redundant if remote backup OK |
| 3 | **7.3** | `PANDA/.tokur_venv` | TokUR-only deps; drop if TokUR baseline not running |
| 4 | **6.5** | `autodl-tmp/pip_cache` | Pip download cache |
| 5 | **5.8** | `panda-models/.vllm_ready` | Duplicate Qwen2.5 weights (fix script path first) |
| 6 | **5.1** | `autodl-tmp/tmp` | Abandoned pip unpack / temp |
| 7 | **0.92** | `autodl-tmp/pip-cache` | Second pip cache directory |
| 8 | **0.94** | `/root/.cursor-server` | IDE remote cache; regenerates |
| 9 | **0.42** | `/root/.cache/vllm` | torch compile cache |
| 10 | **0.34** | `PANDA/experiments/spurious_consensus` | Non-maintable side experiment |

**Honorable mention (not in top 10 by policy risk):** deleting all **`raw_runs`** (~**93G**) would dwarf the above but requires explicit post-analysis approval (aggregates already on disk).

**Combined “obvious junk + duplicates” (rows 1–7, 9):** ~**48G** without touching experiment outputs.

---

## Items explicitly checked — not found / negligible

- **scr500** output directories: absent (guard scripts only)
- **strong** run output trees: absent (cancelled script stubs only)
- **deepscaler k9 GPU logs**: random300 logs are **KB–156K**, not multi-GB
- **HuggingFace hub cache** (`~/.cache/huggingface`): **~13M** only
- **TokUR code** (`third_party/TokUR`): **~1.8M** (venv is the cost)

---

## Suggested cleanup order (if user approves later)

1. `tmp/` + both pip caches (~12.5G) — lowest risk  
2. `archive/*.tar.zst` if offsite backup confirmed (~11.4G)  
3. `panda-vllm-venv` after confirming no manual activation (~11.6G)  
4. Deduplicate Qwen2.5 `.vllm_ready` vs main tree (~5.8G)  
5. `.tokur_venv` only when TokUR work is finished (~7.3G)  
6. Post-paper: selective or global `raw_runs` pruning (~93G potential)

---

*Generated by disk audit scan; no files were modified or removed.*
