"""Range-only UQ features: isolate rephrase_range vs clean logprob vs answer length."""

from __future__ import annotations

import numpy as np

# Internal name -> uncertainty direction (+1: higher = more likely wrong)
RANGE_ONLY_FEATURES: dict[str, int] = {
    "clean_logprob_norm": -1,
    "a0_num_tokens": 1,
    "range_raw": 1,
    "range_norm": 1,
}

RANGE_ONLY_LR_GROUP = "RangeLength-LR"
RANGE_ONLY_LR_FEATURES = ["range_norm", "a0_num_tokens"]

# Display names for reports
RANGE_ONLY_DISPLAY: dict[str, str] = {
    "clean_logprob_norm": "-clean_logprob_norm",
    "a0_num_tokens": "a0_num_tokens",
    "range_raw": "range_raw",
    "range_norm": "range_norm",
}


def compute_range_features(row: dict) -> dict[str, float]:
    """Derive range-only features from a scored features.jsonl row."""
    clean = row.get("clean_score") or {}
    rep = row.get("a0_rephrase_scores") or []
    tlp = clean.get("token_logprobs") or []

    t = float(clean.get("num_tokens", 0) or 0)
    clean_norm = float(clean.get("len_norm_logprob", float("nan")))
    if tlp:
        clean_raw = float(np.sum(tlp))
    elif np.isfinite(clean_norm) and t > 0:
        clean_raw = clean_norm * t
    else:
        clean_raw = float("nan")

    rep_arr = np.array(rep, dtype=float)
    rep_arr = rep_arr[np.isfinite(rep_arr)]
    if len(rep_arr) == 0:
        range_norm = float("nan")
    else:
        range_norm = float(rep_arr.max() - rep_arr.min())

    range_raw = range_norm * t if np.isfinite(range_norm) and t > 0 else float("nan")

    return {
        "clean_logprob_norm": clean_norm,
        "clean_logprob_raw": clean_raw,
        "a0_num_tokens": t,
        "range_norm": range_norm,
        "range_raw": range_raw,
    }


def attach_range_features(row: dict) -> dict:
    row = dict(row)
    row["range_features"] = compute_range_features(row)
    return row
