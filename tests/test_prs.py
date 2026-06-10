"""PRS (Perturbation Reliability Score) unit tests."""

from __future__ import annotations

import math

from prs.ase.metrics import metrics_from_record
from prs.ase.prs import (
    LAMBDA_A,
    LAMBDA_R,
    compute_prs,
    compute_prs_from_row,
    enrich_row_with_prs,
    resolve_d_ans,
    resolve_f_resp,
)


def test_compute_prs_formula():
    f, a, r = 0.8, 0.4, 0.2
    expected = f + LAMBDA_A * a + LAMBDA_R * r
    assert compute_prs(f, a, r) == expected


def test_resolve_legacy_keys():
    row = {
        "TW_ASE": 0.75,
        "AltMass_final": 0.3,
        "AltMass_local_spread_reason": 0.1,
    }
    assert resolve_f_resp(row) == 0.75
    assert resolve_d_ans(row) == 0.3
    bundle = compute_prs_from_row(row)
    assert bundle["PRS"] == compute_prs(0.75, 0.3, 0.1)


def test_resolve_prefers_new_keys():
    row = {
        "F_resp": 0.5,
        "D_ans": 0.2,
        "D_reason": 0.1,
        "TW_ASE": 0.99,
        "AltMass_final": 0.99,
    }
    assert resolve_f_resp(row) == 0.5
    assert resolve_d_ans(row) == 0.2


def test_enrich_reads_w_alternative_when_no_altmass_final():
    row = {"TW_ASE": 1.0, "W_alternative_answer_mass_topk": 0.5, "AltMass_local_spread_reason": 0.25}
    enrich_row_with_prs(row, write_legacy=False)
    assert row["D_ans"] == 0.5
    assert row["PRS"] == 1.0 + LAMBDA_A * 0.5 + LAMBDA_R * 0.25


def test_metrics_from_record_includes_prs():
    record = {
        "id": "t0",
        "dataset": "test",
        "reference": "42",
        "is_correct": False,
        "label_wrong": 1,
        "base_generation": {
            "answer_normalized": "1",
            "token_trace": [{"token": "1", "entropy": 0.1, "margin_top2": 0.9, "logprob": -0.1, "rank": 1, "topk": []}],
            "answer_span": {"start_token": 0, "end_token": 0},
            "full_response": "1",
        },
        "text_rephrase_runs": [
            {
                "answer_normalized": "2",
                "token_trace": [{"token": "2", "entropy": 0.2, "margin_top2": 0.8, "logprob": -0.2, "rank": 1, "topk": []}],
                "answer_span": {"start_token": 0, "end_token": 0},
            }
        ],
        "weight_perturb_runs": [
            {
                "answer_normalized": "3",
                "token_trace": [
                    {
                        "token": "3",
                        "entropy": 0.3,
                        "margin_top2": 0.7,
                        "logprob": -0.3,
                        "rank": 1,
                        "topk": [("9", -1.0), ("3", -0.3)],
                    }
                ],
                "answer_span": {"start_token": 0, "end_token": 0},
            },
            {
                "answer_normalized": "4",
                "token_trace": [
                    {
                        "token": "4",
                        "entropy": 0.4,
                        "margin_top2": 0.6,
                        "logprob": -0.4,
                        "rank": 1,
                        "topk": [("8", -0.5), ("4", -0.4)],
                    }
                ],
                "answer_span": {"start_token": 0, "end_token": 0},
            },
        ],
        "semantic_cache": {},
    }
    m = metrics_from_record(record)
    assert "PRS" in m
    assert "F_resp" in m
    assert m["F_resp"] == m["TW_ASE"]
    assert math.isfinite(m["F_resp"])
    assert m["F_resp"] == m["TW_ASE"]
    # D_reason needs reasoning-prefix tokens; may be nan on degenerate traces
    if math.isfinite(m["D_reason"]):
        assert math.isfinite(m["PRS"])
        assert m["PRS"] == compute_prs(m["F_resp"], m["D_ans"], m["D_reason"])
    assert "baseline_PE_mean" in m
    assert "baseline_SE_H" in m
    assert math.isfinite(m["baseline_PE_mean"])
