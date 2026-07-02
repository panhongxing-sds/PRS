"""Numeric trajectory extraction and cross-run disagreement (Experiment 4)."""

from __future__ import annotations

import re

import numpy as np

from panda.grading.math_grader import math_equal

_NUM_RE = re.compile(
    r"(?<![A-Za-z])(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![A-Za-z])"
)


def extract_numbers(text: str) -> list[float]:
    if not text:
        return []
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            continue
    return out


def _edit_distance(a: list[float], b: list[float], tol: float = 1e-6) -> int:
    """Levenshtein on numeric tokens with approximate equality."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = abs(a[i - 1] - b[j - 1]) <= tol * max(1.0, abs(a[i - 1]), abs(b[j - 1]))
            cost = 0 if same else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def per_run_numeric(run: dict) -> dict:
    resp = run.get("full_response") or run.get("answer_raw") or ""
    nums = extract_numbers(resp)
    final = run.get("answer_normalized") or ""
    final_nums = extract_numbers(str(final))
    final_val = final_nums[-1] if final_nums else float("nan")
    in_reasoning = False
    if final_val == final_val and nums:
        in_reasoning = any(math_equal(str(n), str(final_val)) for n in nums[:-1] if nums)
    return {
        "num_seq": nums,
        "final_num": final_val,
        "final_in_reasoning": float(in_reasoning),
        "num_count": len(nums),
    }


def branch_numeric_disagreement(runs: list[dict], prefix: str) -> dict:
    stats = [per_run_numeric(r) for r in runs]
    nan = float("nan")
    if not stats:
        return {
            f"{prefix}_num_unique_finals": nan,
            f"{prefix}_num_seq_var_mean": nan,
            f"{prefix}_num_edit_dist_mean": nan,
            f"{prefix}_num_edit_dist_max": nan,
            f"{prefix}_final_in_reasoning_rate": nan,
            f"{prefix}_num_count_avg": nan,
        }
    finals = [s["final_num"] for s in stats if s["final_num"] == s["final_num"]]
    # unique finals up to math_equal
    unique_finals: list[float] = []
    for f in finals:
        if not any(math_equal(str(f), str(u)) for u in unique_finals):
            unique_finals.append(f)
    seqs = [s["num_seq"] for s in stats if s["num_seq"]]
    edit_dists = []
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            edit_dists.append(_edit_distance(seqs[i], seqs[j]))
    seq_lens = [len(s) for s in seqs]
    seq_var = float(np.var(seq_lens)) if len(seq_lens) >= 2 else 0.0
    return {
        f"{prefix}_num_unique_finals": float(len(unique_finals)),
        f"{prefix}_num_seq_var_mean": seq_var,
        f"{prefix}_num_edit_dist_mean": float(np.mean(edit_dists)) if edit_dists else 0.0,
        f"{prefix}_num_edit_dist_max": float(np.max(edit_dists)) if edit_dists else 0.0,
        f"{prefix}_final_in_reasoning_rate": float(np.mean([s["final_in_reasoning"] for s in stats])),
        f"{prefix}_num_count_avg": float(np.mean([s["num_count"] for s in stats])),
    }


def merge_numeric_metrics(text_runs: list[dict], weight_runs: list[dict]) -> dict:
    out = {}
    out.update(branch_numeric_disagreement(text_runs, "T"))
    out.update(branch_numeric_disagreement(weight_runs, "W"))
    all_runs = list(text_runs) + list(weight_runs)
    out.update(branch_numeric_disagreement(all_runs, "TW"))
    return out
