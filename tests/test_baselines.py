"""Tests for ISO uncertainty baselines."""

from __future__ import annotations

import math

import numpy as np
import pytest

from prs.baselines.from_record import (
    _collect_se_sample_answers,
    _collect_tw_sample_answers,
    baselines_from_record,
    enrich_row_with_baselines,
    sample_baselines_from_record,
)
from prs.baselines.sample_scores import graph_uncertainty, semantic_entropy_h, u_deg, u_ecc
from prs.baselines.token_scores import (
    deepconf_mean,
    log_likelihood_nll,
    predictive_entropy,
    sar_token_approx,
)


def _trace(ent: float, lp: float, topk: list | None = None) -> list[dict]:
    return [
        {
            "token": "42",
            "entropy": ent,
            "logprob": lp,
            "topk": topk or [("42", lp), ("41", lp - 1.0)],
        }
    ]


def test_predictive_entropy_and_nll():
    trace = _trace(0.5, -0.2)
    pe = predictive_entropy(trace)
    ll = log_likelihood_nll(trace)
    assert pe["baseline_PE_mean"] == 0.5
    assert ll["baseline_LL_nll"] == pytest.approx(0.2)


def test_deepconf_from_topk():
    trace = _trace(0.1, -0.1, topk=[("a", -0.1), ("b", -2.0)])
    dc = deepconf_mean(trace, k=2)
    assert math.isfinite(dc["baseline_DC_mean"])
    assert dc["baseline_DC_mean"] > 0


def test_semantic_entropy_uniform_clusters():
    h = semantic_entropy_h(["1", "2", "3", "4"])
    assert h["baseline_SE_num_clusters"] == 4
    assert h["baseline_SE_H"] == pytest.approx(math.log(4), rel=1e-5)


def test_graph_uncertainty_identical_answers():
    g = graph_uncertainty(["42", "42", "42"])
    assert g["baseline_U_Deg"] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(g["baseline_U_Ecc"])


def test_u_deg_formula():
    w = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert u_deg(w) == pytest.approx(0.5)


def test_se_uses_high_temp_not_tw():
    record = {
        "high_temp_sample_runs": [{"answer_normalized": "1"}, {"answer_normalized": "2"}],
        "text_rephrase_runs": [{"answer_normalized": "9"}],
        "weight_perturb_runs": [{"answer_normalized": "8"}],
    }
    se_answers = _collect_se_sample_answers(record)
    tw_answers = _collect_tw_sample_answers(record)
    assert se_answers == ["1", "2"]
    assert tw_answers == ["9", "8"]

    b = sample_baselines_from_record(record)
    assert b["baseline_SE_status"] == "ok"
    assert b["baseline_SE_num_clusters"] == 2
    assert math.isfinite(b["baseline_U_Ecc"])


def test_se_missing_without_high_temp_samples():
    record = {
        "text_rephrase_runs": [{"answer_normalized": "2"}],
        "weight_perturb_runs": [{"answer_normalized": "3"}],
    }
    b = sample_baselines_from_record(record)
    assert b["baseline_SE_status"] == "missing_high_temp_samples"
    assert math.isnan(b["baseline_SE_H"])
    assert math.isfinite(b["baseline_U_Ecc"])


def test_baselines_from_record():
    record = {
        "id": "x",
        "is_correct": False,
        "base_generation": {
            "answer_normalized": "1",
            "token_trace": _trace(0.3, -0.3),
            "answer_span": {"start_token": 0, "end_token": 0},
        },
        "high_temp_sample_runs": [
            {"answer_normalized": "2"},
            {"answer_normalized": "3"},
        ],
        "text_rephrase_runs": [{"answer_normalized": "2"}],
        "weight_perturb_runs": [{"answer_normalized": "3"}],
    }
    b = baselines_from_record(record)
    assert "baseline_PE_mean" in b
    assert "baseline_SE_H" in b
    assert math.isfinite(b["baseline_SE_H"])
    assert "baseline_U_Ecc" in b
    assert b["cot_greedy_acc"] == 0.0


def test_enrich_row_with_baselines():
    row = {"id": "x", "label_wrong_clean": 1, "is_correct_clean": False}
    record = {
        "base_generation": {"token_trace": _trace(0.4, -0.4), "answer_span": {"start_token": 0, "end_token": 0}},
        "text_rephrase_runs": [],
        "weight_perturb_runs": [],
    }
    out = enrich_row_with_baselines(row, record)
    assert out["baseline_SAR"] == sar_token_approx(_trace(0.4, -0.4))["baseline_SAR"]
    assert out["cot_greedy_acc"] == 0.0
    assert out["baseline_SE_status"] == "missing_high_temp_samples"
