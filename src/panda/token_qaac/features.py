"""Extract token-level UQ features from rephrase scores (Self-A0 main experiment)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_range": float("nan"),
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    mn = float(arr.min())
    mx = float(arr.max())
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": mn,
        f"{prefix}_max": mx,
        f"{prefix}_range": mx - mn,
    }


def extract_token_features_self_a0(
    *,
    clean_logprob: float,
    rephrase_logprobs: list[float],
) -> dict[str, float]:
    """Main-experiment features: clean + rephrase stability + deltas."""
    clean = float(clean_logprob) if np.isfinite(clean_logprob) else float("nan")
    rep = _stats(rephrase_logprobs, "rephrase")

    mean_r = rep["rephrase_mean"]
    min_r = rep["rephrase_min"]
    range_r = rep["rephrase_range"]

    feats: dict[str, float] = {
        "tok_a0_logprob_clean": clean,
        "rephrase_mean_logprob": mean_r,
        "rephrase_std_logprob": rep["rephrase_std"],
        "rephrase_min_logprob": min_r,
        "rephrase_max_logprob": rep["rephrase_max"],
        "rephrase_range_logprob": range_r,
    }

    if np.isfinite(clean) and np.isfinite(min_r):
        feats["clean_minus_rephrase_min"] = clean - min_r
        feats["tok_a0_drop_from_clean"] = clean - min_r
    else:
        feats["clean_minus_rephrase_min"] = float("nan")
        feats["tok_a0_drop_from_clean"] = float("nan")

    if np.isfinite(clean) and np.isfinite(mean_r):
        feats["rephrase_mean_minus_clean"] = mean_r - clean
    else:
        feats["rephrase_mean_minus_clean"] = float("nan")

    if np.isfinite(range_r) and np.isfinite(mean_r) and abs(mean_r) > 1e-9:
        feats["tok_a0_range_over_abs_mean"] = range_r / abs(mean_r)
    else:
        feats["tok_a0_range_over_abs_mean"] = float("nan")

    return feats


def extract_token_features(
    *,
    clean_score: dict[str, Any],
    a0_rephrase_scores: list[float],
    a0: str = "",
    candidates: list[str] | None = None,
    candidate_rephrase_scores: list[dict[str, float]] | None = None,
    tau: float = 1.0,
) -> dict[str, float]:
    lp = clean_score.get("len_norm_logprob", float("nan"))
    return extract_token_features_self_a0(clean_logprob=float(lp), rephrase_logprobs=a0_rephrase_scores)


# LR feature groups (main experiment)
FEATURE_GROUPS: dict[str, list[str]] = {
    "CleanToken-LR": ["tok_a0_logprob_clean"],
    "Rephrase-A0-LR": [
        "rephrase_mean_logprob",
        "rephrase_std_logprob",
        "rephrase_min_logprob",
        "rephrase_range_logprob",
        "clean_minus_rephrase_min",
        "rephrase_mean_minus_clean",
        "tok_a0_range_over_abs_mean",
    ],
}
FEATURE_GROUPS["HybridToken-LR"] = (
    FEATURE_GROUPS["CleanToken-LR"] + FEATURE_GROUPS["Rephrase-A0-LR"]
)

# Primary univariate features for reporting (name -> direction for uncertainty)
PRIMARY_UNIVARIATE: dict[str, int] = {
    "tok_a0_logprob_clean": -1,
    "rephrase_mean_logprob": -1,
    "rephrase_std_logprob": 1,
    "rephrase_min_logprob": -1,
    "rephrase_range_logprob": 1,
    "clean_minus_rephrase_min": 1,
    "rephrase_mean_minus_clean": -1,
    "tok_a0_drop_from_clean": 1,
    "tok_a0_range_over_abs_mean": 1,
}

UNCERTAINTY_DIRECTION: dict[str, int] = dict(PRIMARY_UNIVARIATE)
