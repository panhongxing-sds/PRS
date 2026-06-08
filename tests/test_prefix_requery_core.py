"""Tests for prefix re-query drift signals."""

from prs.ase.prefix_requery_core import (
    cross_prefix_drift,
    distribution_jsd,
    prefix_drift_flip,
    prefix_drift_jsd,
    stage_signals,
)


def test_jsd_identical_is_zero():
    p = {"24": 1.0}
    assert distribution_jsd(p, p) == 0.0


def test_jsd_detects_shift():
    p = {"24": 1.0}
    q = {"36": 1.0}
    assert distribution_jsd(p, q) > 0.9


def test_prefix_drift_confident_shift():
    prev = ["24", "24", "24"]
    curr = ["36", "36", "36"]
    assert prefix_drift_jsd(prev, curr) > 0.9
    assert prefix_drift_flip(prev, curr) == 1.0


def test_prefix_entropy_zero_but_shift_high():
    requery = {
        0: {0: ["24", "24"], 1: ["36", "36"], 2: ["48", "48"]},
        1: {0: ["24", "24"], 1: ["36", "36"], 2: ["48", "48"]},
    }
    runs = [{"token_trace": [{"token": "x"}], "answer_span": {"start_token": 1}}] * 2
    sig = stage_signals(runs, requery, S=3)
    assert sig["B_entropy"][1] == 0.0
    assert sig["B_shift"][1] > 0.9
    assert sig["B_flip"][1] == 1.0


def test_cross_prefix_drift_stage_zero():
    requery = {0: {0: ["1"]}}
    j, f = cross_prefix_drift(requery, 1, 0)
    assert j == 0.0 and f == 0.0
