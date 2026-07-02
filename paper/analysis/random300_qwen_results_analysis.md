# Qwen2.5-3B DeepScaler random300: PANDA vs SC@9 (seed 41)

**Date:** 2026-07-02  
**Cohort:** 300 questions (`deepscaler_random300_meta.json`, seed 42 id draw; SC/PANDA use numeric ids aligned 1:1).

## Data sources

| Role | Path | Rows |
|------|------|------|
| PANDA summary | `panda-outputs/maintable_qwen25_3b_deepscaler_random300/seed41/deepscaler/summary.jsonl` | 300 |
| SC@9 (prefix of K=64) | `PANDA/experiments/spurious_consensus/data/samples/samples_qwen25_3b_seed41_deepscaler_random300_k9.jsonl` | 300 |
| Join (computed) | `PANDA/paper/analysis/random300_sc9_panda_join.jsonl` | 300 |
| Aggregates (computed) | `PANDA/paper/analysis/random300_fair_baselines.json` | — |
| SC cohort stats (precomputed) | `PANDA/paper/analysis/random300_k9_stats.json` | n=300, maj@9 correct=132 |

**Note:** `random300_sc9_panda_join.jsonl` and `random300_fair_baselines.json` were not present in the repo; they were generated for this analysis by joining on `meta.ids` (PANDA ids `deepscaler_{id}` ↔ SC numeric `id`).

**Macro LODO AUROC:** Not applicable on this single-dataset slice (no multi-dataset leave-one-out fold). Report **pooled** AUROC/AUPRC on the 300- or 228-question eval set.

---

## Decoding / budget asymmetry (important)

| Method | Generation protocol |
|--------|---------------------|
| **SC@9** | Existing **K=64** pool per question: **temperature=0.5**, top_p=0.95, vLLM, seed=41; SC@9 uses the **first 9** answers only; UQ score **1 − p_top** (answer-level consensus). |
| **PANDA** | Phase A vLLM + Phase B HF ASE: **1 greedy base** + **4 text rephrases + 4 weight perturbations** (9 greedy traces total, fair-budget narrative in `random300_panda_setup.json`: `N_REPHRASES=4`, weight seeds 42–45). Final answer **a0** = greedy base; UQ = **PANDA** (fusion; **F_resp** / bd / reasoning drift components in summary). |

SC majority accuracy can exceed greedy PANDA accuracy even when PANDA’s UQ targets **greedy wrongness** (`label_wrong_clean`), not SC majority wrongness.

---

## Answer accuracy (same 300 ids)

Majority@9 correctness uses the `correct[]` flags on SC samples (not raw string match to `gold`, which mis-grades ~56 items due to formatting).

| Metric | Correct / n | Accuracy |
|--------|-------------|----------|
| **SC@9 majority** | 132 / 300 | **44.0%** |
| **PANDA greedy (`is_correct_clean`, a0)** | 120 / 300 | **40.0%** |
| **Δ (PANDA − SC)** | −12 questions | **−4.0 pp** |

**Paired 2×2 (correctness):**

| | SC wrong | SC correct |
|---|----------|------------|
| **PANDA wrong** | 132 | 48 |
| **PANDA correct** | 36 | 84 |

- PANDA-only correct (SC wrong): **36**
- SC-only correct (PANDA wrong): **48**
- Both correct: **84**

**Interpretation:** On this cohort, **SC@9 wins on task accuracy**; PANDA greedy is not an accuracy improvement over early-prefix self-consistency.

---

## Wrong-answer detection (UQ)

**Primary label (PANDA paper-aligned):** `y = label_wrong_clean` (1 = greedy final wrong).  
72 questions have `label_drop=1` (ungradable / dropped labels); main-table style eval uses **n=228** kept rows.

### Pooled AUROC / AUPRC

| Score | n=300 AUROC | n=300 AUPRC | n=228 AUROC | n=228 AUPRC |
|-------|-------------|-------------|-------------|-------------|
| **SC@9: 1 − p_top** | 0.772 | 0.801 | **0.823** | 0.847 |
| **PANDA: PANDA** | **0.803** | **0.821** | **0.862** | **0.902** |
| PANDA: F_resp | 0.805 | 0.816 | 0.862 | 0.899 |
| PANDA: bd (answer_drift) | 0.717 | 0.706 | 0.788 | — |
| PANDA: reasoning_drift | 0.631 | 0.699 | — | — |
| PANDA: baseline_SC_mean (1 greedy trace) | 0.462 | — | 0.437 | — |
| PANDA: TW9_H (9-run token window) | — | — | **0.893** | — |

**Paired lift (PANDA − SC), primary label:**

- All 300: **+3.12 pp** AUROC (0.803 vs 0.772)
- Eval 228 (no drop): **+3.95 pp** AUROC (0.862 vs 0.823)

**Alternative label:** `y = SC majority wrong` (168/300 wrong).

| Score | AUROC |
|-------|-------|
| SC 1 − p_top | **0.778** |
| PANDA | 0.745 |
| F_resp | 0.746 |

When the detection target is **SC’s own majority mistake**, consensus UQ is better calibrated than PANDA (which is trained for **greedy** failure). This is expected under decoding asymmetry.

---

## Subsets (y = `label_wrong_clean`, n=300)

| Subset | n | SC AUROC | PANDA AUROC | Δ (PANDA−SC) |
|--------|---|----------|-----------|------------|
| All | 300 | 0.772 | 0.803 | **+3.1 pp** |
| bd > 0 | 172 | 0.656 | 0.651 | −0.5 pp |
| bd = 0 | 128 | 0.697 | 0.692 | −0.4 pp |
| p_top ≥ 8/9 (high consensus) | 75 | 0.521 | 0.737 | **+21.7 pp** |
| p_top ≥ 7/9 | 88 | 0.573 | 0.719 | **+14.5 pp** |
| SC majority wrong | 168 | 0.681 | 0.738 | +5.7 pp |

**High p_top@9:** 75 questions; **16/75 (21.3%)** are majority-wrong despite strong surface consensus (spurious consensus regime). Here **1 − p_top has little room to move** (AUROC ≈ 0.52); **PANDA retains signal** (AUROC ≈ 0.74).

**bd split:** On questions with answer drift, SC and PANDA are similar; PANDA’s gain is not driven by bd>0 alone—it shows up strongly on **high-consensus** items and in the full pooled metric.

---

## SC@9 distribution (random300, matches `random300_k9_stats.json`)

- Majority correct: **132**; wrong: **168**
- Any of 9 samples correct: **191**; **blind@9** (0/9 correct): **109** (36.3%)
- Mean **p_top@9**: **0.545** (min 1/9, max 1.0)

**p_top histogram (300):**

| bin | count |
|-----|-------|
| [0, 0.5) | 158 |
| [0.5, 0.7) | 54 |
| [0.7, 8/9) | 34 |
| ≥ 8/9 | 75 |

*(Recomputed blind=109; `random300_k9_stats.json` lists `blind_at_9_all_wrong: 11`, which is inconsistent with the sample file—likely a stats-script bug; use 109 here.)*

---

## Verdict for paper narrative

| Dimension | vs SC@9 | Notes |
|-----------|---------|-------|
| **Task accuracy** | **Negative** | Greedy PANDA 40.0% vs SC@9 44.0% (−4 pp); eval-228 gap larger (−7 pp) because drops are not random w.r.t. SC. |
| **UQ (greedy wrong, pooled)** | **Positive** | PANDA AUROC **+3.1–3.9 pp** over 1−p_top on matched ids; AUPRC also higher on n=228. |
| **UQ (SC majority wrong)** | **Negative** | SC consensus score wins (~0.778 vs ~0.745). |
| **High p_top / SCR-like** | **Strong positive** | PANDA much better where SC score is saturated. |

**Overall:** **Conditionally positive for PANDA**—support the claim that **path-level / multi-perturbation scores beat answer-level consensus for detecting greedy errors**, especially under **high spurious consensus**; do **not** claim PANDA beats SC@9 on **majority-vote accuracy** on this 300-id slice without caveats.

**Caveats to state explicitly:**

1. Different decoding: SC prefix @ T=0.5 vs PANDA greedy multi-perturb ASE.  
2. UQ label is **greedy final**, not SC majority.  
3. 72/300 `label_drop`; headline UQ should use **n=228** where applicable.  
4. `baseline_SC_mean` on the single greedy trace is weak (~0.44 AUROC)—PANDA gain is not from repeating SC on one trace.  
5. TW9_H is very high on n=228 (~0.89); clarify whether it is in-scope for the main claim vs PANDA.

---

## Repro

```bash
python3 - <<'PY'
# Regenerates join + fair_baselines; see analysis session 2026-07-02.
# Core logic: meta ids, maj_correct via correct[], sklearn roc_auc on label_wrong_clean.
PY
```

Artifacts: `random300_sc9_panda_join.jsonl`, `random300_fair_baselines.json`.
