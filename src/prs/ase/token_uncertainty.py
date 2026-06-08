"""Token-level ATU: TokUR-inspired aggregations (sum / mean / epistemic-style)."""

from __future__ import annotations

import math

import numpy as np


def _entropy_discrete(values: list[float], eps: float = 1e-12) -> float:
    """Entropy of a non-negative vector (normalized)."""
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if s <= eps:
        return 0.0
    p = arr / s
    p = p[p > eps]
    return float(-np.sum(p * np.log(p)))


def per_generation_stats(gen: dict, top_pct: float = 0.10) -> dict:
    """Scalar stats for one generation trajectory."""
    ent = list(gen.get("token_entropies") or [])
    mar = list(gen.get("token_margins") or [])
    n = len(ent)
    if n == 0:
        nan = float("nan")
        return {
            "n_tokens": 0,
            "ent_sum": nan,
            "ent_mean": nan,
            "ent_max": nan,
            "ent_top10_sum": nan,
            "ent_top10_mean": nan,
            "mar_sum": nan,
            "mar_mean": nan,
            "mar_max": nan,
            "mar_top10_sum": nan,
            "mar_top10_mean": nan,
        }

    ent_arr = np.asarray(ent, dtype=float)
    mar_arr = np.asarray(mar, dtype=float)
    k = max(1, int(math.ceil(n * top_pct)))
    top_ent = np.partition(ent_arr, -k)[-k:]
    top_mar = np.partition(mar_arr, -k)[-k:]

    return {
        "n_tokens": n,
        "ent_sum": float(ent_arr.sum()),
        "ent_mean": float(ent_arr.mean()),
        "ent_max": float(ent_arr.max()),
        "ent_top10_sum": float(top_ent.sum()),
        "ent_top10_mean": float(top_ent.mean()),
        "mar_sum": float(mar_arr.sum()),
        "mar_mean": float(mar_arr.mean()),
        "mar_max": float(mar_arr.max()),
        "mar_top10_sum": float(top_mar.sum()),
        "mar_top10_mean": float(top_mar.mean()),
    }


def branch_aggregate(gen_stats: list[dict], prefix: str) -> dict:
    """
    Aggregate per-generation stats across adversarial samples (rephrase or weight).

    TokUR analog (without full vocab distributions):
      - ent_epi_var: Var(mean entropy across generations) — spread of uncertainty
      - ent_epi_H: H(normalized per-gen mean entropies) — disagreement distribution
      - ent_disagreement: max(mean) - min(mean) across generations
    """
    valid = [s for s in gen_stats if s.get("n_tokens", 0) > 0]
    out: dict[str, float] = {}
    if not valid:
        for suffix in (
            "ent_sum_total", "ent_mean_avg", "ent_top10_sum_total", "ent_top10_mean_avg",
            "ent_max_max", "ent_epi_var", "ent_epi_H", "ent_disagreement",
            "mar_sum_total", "mar_top10_sum_total",
            "n_tokens_avg",
        ):
            out[f"{prefix}_{suffix}"] = float("nan")
        # legacy keys
        out[f"{prefix}_ATU_entropy_top10"] = float("nan")
        out[f"{prefix}_ATU_margin_top10"] = float("nan")
        return out

    ent_sums = [s["ent_sum"] for s in valid]
    ent_means = [s["ent_mean"] for s in valid]
    ent_top10_sums = [s["ent_top10_sum"] for s in valid]
    ent_top10_means = [s["ent_top10_mean"] for s in valid]
    mar_sums = [s["mar_sum"] for s in valid]
    mar_top10_sums = [s["mar_top10_sum"] for s in valid]

    out[f"{prefix}_ent_sum_total"] = float(np.sum(ent_sums))
    out[f"{prefix}_ent_mean_avg"] = float(np.mean(ent_means))
    out[f"{prefix}_ent_top10_sum_total"] = float(np.sum(ent_top10_sums))
    out[f"{prefix}_ent_top10_mean_avg"] = float(np.mean(ent_top10_means))
    out[f"{prefix}_ent_max_max"] = float(np.max([s["ent_max"] for s in valid]))
    out[f"{prefix}_ent_epi_var"] = float(np.var(ent_means))
    out[f"{prefix}_ent_epi_H"] = _entropy_discrete(ent_means)
    out[f"{prefix}_ent_disagreement"] = float(np.max(ent_means) - np.min(ent_means))
    out[f"{prefix}_mar_sum_total"] = float(np.sum(mar_sums))
    out[f"{prefix}_mar_top10_sum_total"] = float(np.sum(mar_top10_sums))
    out[f"{prefix}_n_tokens_avg"] = float(np.mean([s["n_tokens"] for s in valid]))

    # legacy (backward compatible)
    out[f"{prefix}_ATU_entropy_top10"] = out[f"{prefix}_ent_top10_mean_avg"]
    out[f"{prefix}_ATU_margin_top10"] = float(np.mean([s["mar_top10_mean"] for s in valid]))
    return out


def aggregate_atu(
    token_entropies: list[float],
    token_margins: list[float],
    top_pct: float = 0.10,
) -> dict[str, float]:
    """Single-trajectory aggregation (legacy API)."""
    st = per_generation_stats(
        {"token_entropies": token_entropies, "token_margins": token_margins},
        top_pct=top_pct,
    )
    return {
        "ATU_entropy_top10": st["ent_top10_mean"],
        "ATU_margin_top10": st["mar_top10_mean"],
        "ATU_entropy_sum": st["ent_sum"],
        "ATU_entropy_mean": st["ent_mean"],
        "ATU_entropy_top10_sum": st["ent_top10_sum"],
    }


def merge_branch_from_gens(gens: list[dict], prefix: str, top_pct: float = 0.10) -> dict:
    """Build branch ATU dict from list of generate_with_stats outputs."""
    gen_stats = [per_generation_stats(g, top_pct=top_pct) for g in gens]
    return branch_aggregate(gen_stats, prefix)


def merge_joint_branches(text_gens: list[dict], weight_gens: list[dict], top_pct: float = 0.10) -> dict:
    """TW branch: pool all text + weight generations."""
    all_gens = list(text_gens) + list(weight_gens)
    return merge_branch_from_gens(all_gens, "TW", top_pct=top_pct)
