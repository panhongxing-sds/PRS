"""Compute PANDA/ATU summary metrics from full records or legacy cache."""

from __future__ import annotations

from panda.core.altmass_decomposition import altmass_variants_weight_branch, drift_form_variants
from panda.core.semantic_entropy import compute_ase
from panda.core.token_uncertainty import merge_branch_from_gens, merge_joint_branches
from panda.core.token_advanced import merge_advanced_metrics
from panda.core.numeric_trajectory import merge_numeric_metrics
from panda.core.reasoning_token_features import merge_reasoning_token_metrics
from panda.core.cluster_token_trace import merge_cluster_token_trace
from panda.core.panda import enrich_row_with_panda
from panda.baselines.from_record import enrich_row_with_baselines
from panda.grading.answer_canonicalizer import grade_answer


def enrich_row_with_clean_label(row: dict, model: str | None = None) -> dict:
    """Paper tables use label_wrong_clean (sympy-aware grading), not raw label_wrong."""
    g = grade_answer(
        row.get("a0", ""),
        row.get("reference", ""),
        record_id=row.get("id"),
        dataset=row.get("dataset"),
        model=model,
    )
    row["is_correct_clean"] = g["is_correct_clean"]
    row["label_wrong_clean"] = g["label_wrong_clean"]
    row["label_drop"] = g["label_drop"]
    row["relabeled"] = g["relabeled"]
    return row


def _mean_entropy(run: dict) -> float:
    ents = run.get("token_entropies") or []
    return float(sum(ents) / len(ents)) if ents else 0.0


def _se_style_ensemble_entropy(answers: list[str], mean_ents: list[float]) -> dict:
    """SE-style scores over the perturbation ensemble (+base): full entropy and
    confidence-weighted cluster masses, keys TW9_U / TW9_H / TW9_U_conf / TW9_H_conf."""
    import math as _math

    from panda.core.cluster import cluster_answers

    n = len(answers)
    if n == 0:
        nan = float("nan")
        return {"TW9_U": nan, "TW9_H": nan, "TW9_U_conf": nan, "TW9_H_conf": nan}
    assign, sizes = cluster_answers(answers)
    masses = sorted((c / n for c in sizes.values()), reverse=True)
    h = -sum(p * _math.log(p) for p in masses if p > 0)
    w = [_math.exp(-e) for e in mean_ents]
    tot = sum(w) or 1.0
    cm: dict[int, float] = {}
    for ci, wi in zip(assign, w):
        cm[ci] = cm.get(ci, 0.0) + wi / tot
    pw = sorted(cm.values(), reverse=True)
    hw = -sum(p * _math.log(p) for p in pw if p > 0)
    return {
        "TW9_U": 1.0 - masses[0],
        "TW9_H": h,
        "TW9_U_conf": 1.0 - pw[0],
        "TW9_H_conf": hw,
    }


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
    # Ablation: same AltMass drift decomposition over the full K=8 perturbation set
    # (4 text rephrase + 4 weight). Stored with a `_tw8` suffix; default S_ans/S_tr still
    # use the weight branch (W=4). Lets us compare 4-run vs 8-run drift offline.
    altmass_tw8 = {
        f"{k}_tw8": v
        for k, v in altmass_variants_weight_branch(
            text_runs + weight_runs, record.get("dataset"), top_pct=top_pct
        ).items()
    }

    # Alternative mathematical forms of D_ans/D_reason (W4 and TW8 run sets),
    # stored for later form-selection analysis once all data is generated.
    drift_forms = drift_form_variants(weight_runs, top_pct=top_pct)
    drift_forms_tw8 = drift_form_variants(text_runs + weight_runs, top_pct=top_pct, prefix="tw8_")

    # SE-style entropy variants over the attack ensemble *including* the base answer
    # (9 samples). Full cluster entropy (not 1-max_mass) plus confidence weighting
    # (w ∝ exp(-mean token entropy), the attack-sampling analogue of SE's likelihood
    # weights). Offline eval (seed41): TW9_H/TW9_H_conf beat F_resp on all 3 math sets.
    tw9 = _se_style_ensemble_entropy(
        text_answers + weight_answers + [base.get("answer_normalized", "")],
        [_mean_entropy(r) for r in text_runs + weight_runs] + [_mean_entropy(base)],
    )

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
        **altmass_tw8,
        **tw9,
        **drift_forms,
        **drift_forms_tw8,
        "atu_top_pct": top_pct,
    }
    enrich_row_with_panda(out, write_legacy=False)
    enrich_row_with_baselines(out, record)
    enrich_row_with_clean_label(out, model=(record.get("model_info") or {}).get("model_name"))
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
    from panda.grading.math_grader import math_equal

    ref = str(record["reference"]).strip()
    a0 = record["base_generation"]["answer_normalized"]
    ok = math_equal(a0, ref) if ref and a0 else False
    record["is_correct"] = ok
    record["label_wrong"] = 0 if ok else 1
    return metrics_from_record(record, top_pct=top_pct)
