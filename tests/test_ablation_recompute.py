"""Tests for CPU ablation recompute."""

from __future__ import annotations

import json
from pathlib import Path

from panda.core.ablation_recompute import (
    _add_ablation_scores,
    _subset_record,
    summarize_ablation,
)


def test_subset_record():
    rec = {"text_rephrase_runs": [1, 2, 3, 4], "weight_perturb_runs": [5, 6, 7, 8]}
    sub = _subset_record(rec, n_text=2, n_weight=1)
    assert len(sub["text_rephrase_runs"]) == 2
    assert len(sub["weight_perturb_runs"]) == 1


def test_add_ablation_scores():
    row = {"F_resp": 0.8, "D_ans": 0.4, "D_reason": 0.2, "T_ASE": 0.5, "W_ASE": 0.6}
    abl = _add_ablation_scores(row, lambda_a=0.05, lambda_r=0.03)
    assert abl["abl_PANDA_full"] == 0.8 + 0.05 * 0.4 + 0.03 * 0.2
    assert abl["abl_wo_F_resp"] == 0.05 * 0.4 + 0.03 * 0.2
    # F_resp branch ablation: swap TW source to T-ASE / W-ASE, keep drift terms
    assert abl["abl_F_resp_T"] == 0.5 + 0.05 * 0.4 + 0.03 * 0.2
    assert abl["abl_F_resp_W"] == 0.6 + 0.05 * 0.4 + 0.03 * 0.2


def test_summarize_ablation():
    # Separable toy set: wrong answers carry high component values.
    rows = []
    for i in range(12):
        wrong = i % 2
        v = 0.9 if wrong else 0.1
        rows.append(
            {
                "label_wrong_clean": wrong,
                "F_resp": v,
                "D_ans": v,
                "D_reason": v,
                "T_ASE": v,
                "W_ASE": v,
                "AltMass_local_spread_topk": v,
                "AltMass_local_spread_content": v,
                "AltMass_local_spread_reason": v,
            }
        )
    s = summarize_ablation(rows)
    assert s["PANDA (full)"]["auroc"] > 0.5
    assert s["PANDA (full)"]["delta"] == 0.0
    for name in (
        "-F_resp", "-D_ans", "-D_reason", "F_resp=T-ASE", "F_resp=W-ASE",
        "S_tr=topk (B)", "S_tr=content (A)", "S_tr=math (legacy)",
    ):
        assert name in s
        assert "delta" in s[name]


def test_trace_drift_variants_present():
    from panda.core.altmass_decomposition import altmass_variants_weight_branch

    def run(ans, toks):
        return {
            "answer_normalized": ans,
            "token_trace": [
                {"token": t, "topk": [[t, -0.1], [t + "x", -2.0]]} for t in toks
            ],
            "answer_span": {"start_token": len(toks) - 1, "end_token": len(toks) - 1},
        }

    runs = [run("1", ["The", " sum", " =", "1"]), run("2", ["So", " x", " =", "2"])]
    for ds in ("math500", "humaneval", "zebra_puzzles"):
        out = altmass_variants_weight_branch(runs, ds)
        assert "AltMass_local_spread_topk" in out
        assert "AltMass_local_spread_content" in out
        assert "AltMass_local_spread_reason" in out
