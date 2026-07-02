"""Tests for 3-seed main-table aggregation."""

from __future__ import annotations

import pytest

from panda.core.analyze_maintable import aggregate_seed_metrics, fmt_mean_std


def test_aggregate_seed_metrics_mean_std():
    agg = aggregate_seed_metrics([0.71, 0.72, 0.72])
    assert agg["n"] == 3
    assert agg["mean"] == pytest.approx(0.7166666667, rel=1e-4)
    assert agg["std"] == pytest.approx(0.0047140452, rel=1e-3)


def test_aggregate_seed_metrics_nan():
    agg = aggregate_seed_metrics([float("nan")])
    assert agg["n"] == 0
    assert agg["mean"] != agg["mean"]


def test_fmt_mean_std_pct():
    assert fmt_mean_std(0.7177, 0.0012, as_pct=True) == "71.77 ± 0.12"
    assert fmt_mean_std(0.80, 0.0, as_pct=True) == "80.00"
    assert fmt_mean_std(float("nan"), 0.0, as_pct=True) == "—"


def test_fmt_mean_std_unit_scale():
    assert fmt_mean_std(0.758, 0.012, as_pct=False, digits=3) == "0.758 ± 0.012"
