# Motivation figure spec: SC@9 vs PANDA@9 (DeepScaler random300, Qwen2.5-3B)

**Data:** `paper/analysis/random300_confident_wrong.json` (computed by `paper/analysis/compute_random300_confident_wrong.py`).

**Figure script:** `paper/analysis/figures/plot_random300_confident_wrong.py`  
**Default output:** `paper/analysis/figures/random300_confident_wrong.png`

---

## Key message (what the numbers support)

1. **Fair accuracy (9-decode majority vote):** PANDA **50.7%** vs SC **44.0%** (+6.7 pp on n=300). Use this whenever comparing task performance—not greedy PANDA `a0` alone (42.0%).
2. **Confident-wrong rate (consensus on 9 answers):** At strict thresholds, rates are **similar**; PANDA is **slightly lower** at 8/9 and 7/9, **tied** at \(p_{\top}\ge 0.9\) (both **3.67%**, 11/300). Do **not** over-claim a large gap on vote-only consensus alone.
3. **PANDA’s UQ story (main text companion):** PANDA / dissent (`bd`, `F_resp`) separates wrong from right better than SC \(1-p_{\top}\) (see `random300_fair_baselines.json`: AUROC 0.803 vs 0.772). Many SC confident-wrong items have **low `bd`** (collapse on perturbations)—PANDA is designed to **flag** these, not necessarily eliminate all high-consensus errors.

---

## Metric definitions (caption-ready)

**Shared budget:** 9 decoded answers per question.

| Method | Decodes | Final prediction for accuracy |
|--------|---------|--------------------------------|
| **SC@9** | First 9 of K=64 pool (temp 0.5, seed 41) | Plurality over extracted answers; correctness from pre-graded `correct[]` flags |
| **PANDA@9** | 1 greedy base + 4 text rephrases + 4 weight perturbations | Plurality over normalized answer strings; majority correctness via `math_equal` to reference |

**Vote share:** \(p_{\top} = \max_a \#\{i : \text{ans}_i = a\} / 9\) (SC: exact string; PANDA: normalized strings from summary).

**Confident wrong (primary):** Indicator that final majority vote is **incorrect** AND \(p_{\top} \ge \tau\). Report:

- **Rate (all):** \(\#\{\text{confident wrong}\} / N\)
- **Rate (given majority wrong):** \(\#\{\text{confident wrong}\} / \#\{\text{majority wrong}\}\)

**Thresholds:** \(\tau \in \{0.9,\, 8/9,\, 7/9\}\).

**PANDA collapse proxy (appendix / secondary):** Greedy base wrong AND **answer drift** \(bd = 0\) (all 8 perturbations match \(a_0\)). On random300: **13.3%** of all questions (40/300), **23.0%** of greedy-wrong—this is **not** a lower “confident wrong” rate; it characterizes **spurious consensus under perturbation**.

---

## Recommended layout

### Main text (single column or half-width)

**Panel A — Grouped bar chart (generated PNG)**  
- X: three consensus thresholds (\(p_{\top}\ge 0.9\), \(\ge 8/9\), \(\ge 7/9\)).  
- Y: confident-wrong rate (% of **all** 300 questions).  
- Bars: SC@9 vs PANDA@9 (majority vote).  
- Optional dashed line: PANDA \(a_0\) wrong & \(bd=0\) (13.3%) labeled “collapse subset (not comparable threshold)”.  
- Subtitle or caption line: majority-vote accuracy SC 44.0% vs PANDA 50.7%.

**Panel B (small inset or adjacent)**  
- Paired bar: **majority-vote accuracy** SC vs PANDA (fair comparison).  
- Footnote: greedy PANDA 42.0% shown in appendix only.

### Appendix

- **Panel C:** Venn or table of question IDs: SC confident-wrong vs PANDA confident-wrong at \(\tau=0.9\) (11 vs 11; overlap 5; SC-only 6; PANDA-only 6; 3 SC-CW fixed by PANDA majority).  
- **Panel D:** AUROC bars from `random300_fair_baselines.json` (SC \(1-p_{\top}\) vs PANDA).  
- **Panel E:** Stacked bar of error types among **majority-wrong** only: low-\(p_{\top}\) wrong vs high-\(p_{\top}\) wrong (SC vs PANDA).

---

## n=300 vs n=228

- **Motivation figure:** use **n=300** (full random cohort).  
- **Table / UQ alignment with main table:** repeat metrics on **n=228** (`label_drop=0`) in appendix; see `metrics.n228_eval_no_label_drop` in JSON.

---

## Regenerate

```bash
cd /root/autodl-tmp/PANDA
python3 paper/analysis/compute_random300_confident_wrong.py
python3 paper/analysis/figures/plot_random300_confident_wrong.py
```

---

## Caption draft (English)

*Confident wrong answers on DeepScaler random300 (Qwen2.5-3B, 300 questions). Self-consistency (SC@9) and PANDA use the same 9-answer budget; wrongness is by majority vote. Confident wrong: majority incorrect with vote share \(p_{\top}\) above threshold. PANDA majority accuracy exceeds SC (50.7% vs 44.0%); confident-wrong rates are comparable at \(p_{\top}\ge 0.9\) and slightly lower for PANDA at looser thresholds.*
