"""Compute ASE/ATU summary metrics from full records or legacy cache."""

from __future__ import annotations

from prs.ase.altmass_decomposition import altmass_variants_weight_branch
from prs.ase.semantic_entropy import compute_ase
from prs.ase.token_uncertainty import merge_branch_from_gens, merge_joint_branches
from prs.ase.token_advanced import merge_advanced_metrics
from prs.ase.numeric_trajectory import merge_numeric_metrics
from prs.ase.reasoning_token_features import merge_reasoning_token_metrics
from prs.ase.cluster_token_trace import merge_cluster_token_trace
from prs.ase.prs import enrich_row_with_prs
from prs.baselines.from_record import enrich_row_with_baselines


def _gens_from_runs(runs: list[dict]) -> list[dict]:
    return [
        {
            "token_entropies": r.get("token_entropies") or [t.get("entropy", 0) for t in r.get("token_trace", [])],
            "token_margins": r.get("token_margins")
            or [-t.get("margin_top2", 0) for t in r.get("token_trace", [])],
            "n_tokens": r.get("n_tokens", 0),
        }
        for r in runs
    ]


def metrics_from_record(record: dict, *, top_pct: float = 0.10) -> dict:
    """Summary metrics from a full raw_runs JSON record."""
    text_runs = record.get("text_rephrase_runs") or []
    weight_runs = record.get("weight_perturb_runs") or []
    base = record.get("base_generation") or {}

    text_answers = [r.get("answer_normalized", "") for r in text_runs]
    weight_answers = [r.get("answer_normalized", "") for r in weight_runs]

    t_ase = compute_ase(text_answers)
    w_ase = compute_ase(weight_answers)
    tw_ase = compute_ase(text_answers + weight_answers)

    t_atu = merge_branch_from_gens(_gens_from_runs(text_runs), "T", top_pct=top_pct)
    w_atu = merge_branch_from_gens(_gens_from_runs(weight_runs), "W", top_pct=top_pct)
    tw_atu = merge_joint_branches(_gens_from_runs(text_runs), _gens_from_runs(weight_runs), top_pct=top_pct)
    adv = merge_advanced_metrics(base, text_runs, weight_runs, top_pct=top_pct)
    num = merge_numeric_metrics(text_runs, weight_runs)
    rtok = merge_reasoning_token_metrics(base, text_runs, weight_runs)
    ctr = merge_cluster_token_trace(base, text_runs, weight_runs)
    altmass = altmass_variants_weight_branch(weight_runs, record.get("dataset"), top_pct=top_pct)

    sem = record.get("semantic_cache") or {}
    n_tw = sem.get("cluster_assignments", {}).get("num_clusters")

    out = {
        "id": record["id"],
        "dataset": record.get("dataset", ""),
        "is_correct": record.get("is_correct", False),
        "label_wrong": record.get("label_wrong", 1),
        "a0": base.get("answer_normalized", ""),
        "reference": record.get("reference", ""),
        "n_rephrases": len(text_runs),
        "n_weight_perturb": len(weight_runs),
        "text_answers": text_answers,
        "weight_answers": weight_answers,
        "T_ASE": t_ase["U"],
        "T_ASE_H_norm": t_ase["H_norm"],
        "T_num_clusters": t_ase["num_clusters"],
        "W_ASE": w_ase["U"],
        "W_ASE_H_norm": w_ase["H_norm"],
        "W_num_clusters": w_ase["num_clusters"],
        "TW_ASE": tw_ase["U"],
        "TW_ASE_H_norm": tw_ase["H_norm"],
        "TW_num_clusters": n_tw if n_tw is not None else tw_ase["num_clusters"],
        **{k: v for k, v in t_atu.items() if k.startswith("T_")},
        **{k: v for k, v in w_atu.items() if k.startswith("W_")},
        **{k: v for k, v in tw_atu.items() if k.startswith("TW_")},
        **adv,
        **num,
        **rtok,
        **ctr,
        **altmass,
        "atu_top_pct": top_pct,
    }
    enrich_row_with_prs(out, write_legacy=False)
    enrich_row_with_baselines(out, record)
    return out


def metrics_from_cache(cache: dict, *, top_pct: float = 0.10) -> dict:
    """Legacy: thin cache jsonl → metrics."""
    record = {
        "id": cache["id"],
        "dataset": cache.get("dataset", ""),
        "reference": cache.get("reference", ""),
        "is_correct": False,
        "label_wrong": 1,
        "base_generation": {
            "answer_normalized": (cache.get("clean") or {}).get("final_answer", ""),
            "token_entropies": (cache.get("clean") or {}).get("token_entropies", []),
            "token_margins": (cache.get("clean") or {}).get("token_margins", []),
        },
        "text_rephrase_runs": [
            {
                "answer_normalized": (x.get("gen") or x).get("final_answer", ""),
                "token_entropies": (x.get("gen") or x).get("token_entropies", []),
                "token_margins": (x.get("gen") or x).get("token_margins", []),
            }
            for x in cache.get("text", [])
        ],
        "weight_perturb_runs": [
            {
                "answer_normalized": (x.get("gen") or x).get("final_answer", ""),
                "token_entropies": (x.get("gen") or x).get("token_entropies", []),
                "token_margins": (x.get("gen") or x).get("token_margins", []),
            }
            for x in cache.get("weight", [])
        ],
    }
    from prs.grading.math_grader import math_equal

    ref = str(record["reference"]).strip()
    a0 = record["base_generation"]["answer_normalized"]
    ok = math_equal(a0, ref) if ref and a0 else False
    record["is_correct"] = ok
    record["label_wrong"] = 0 if ok else 1
    return metrics_from_record(record, top_pct=top_pct)
