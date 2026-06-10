"""Unit tests for the vLLM backend trace builder (no vLLM install required)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from prs.ase.vllm_backend import _topk_renorm_entropy, _trace_from_vllm_logprobs


@dataclass
class _LP:
    logprob: float
    rank: int | None = None
    decoded_token: str = ""


def test_topk_renorm_entropy_uniform():
    assert math.isclose(_topk_renorm_entropy([0.0, 0.0, 0.0, 0.0]), math.log(4), rel_tol=1e-6)


def test_topk_renorm_entropy_degenerate():
    assert _topk_renorm_entropy([]) == 0.0
    assert abs(_topk_renorm_entropy([0.0])) < 1e-9  # single token => ~zero entropy


def test_trace_fields_match_hf_schema():
    lp0 = {5: _LP(-0.1, 1, "2"), 9: _LP(-2.0, 2, "3")}
    lp1 = {7: _LP(-0.5, 1, " is"), 1: _LP(-1.5, 2, " was")}
    trace, texts = _trace_from_vllm_logprobs([5, 7], [lp0, lp1], topk_save=10)

    assert texts == ["2", " is"]
    assert len(trace) == 2
    t0 = trace[0]
    # Exact quantities preserved.
    assert t0["token_id"] == 5
    assert math.isclose(t0["logprob"], -0.1)
    assert math.isclose(t0["margin_top2"], 1.9)
    assert t0["rank"] == 1
    assert t0["topk"] == [["2", -0.1], ["3", -2.0]]
    # Required keys present for downstream baselines.
    for key in ("pos", "token", "token_id", "logprob", "entropy", "entropy_topk", "margin_top2", "rank", "topk"):
        assert key in t0


def test_trace_handles_empty_logprob_map():
    trace, texts = _trace_from_vllm_logprobs([5], [{}], topk_save=10)
    assert len(trace) == 1
    assert trace[0]["topk"] == []
    assert trace[0]["entropy"] == 0.0
