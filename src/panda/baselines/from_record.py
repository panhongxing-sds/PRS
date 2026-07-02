"""Compute all CPU baselines from one ASE raw_runs record."""

from __future__ import annotations

from panda.baselines.sample_scores import (
    graph_uncertainty_from_sequences,
    semantic_entropy_h,
)
from panda.baselines.token_scores import token_baselines_from_generation

# Official SE (Kuhn et al.): high-temperature i.i.d. samples on the *same* prompt.
SE_SAMPLE_RUN_KEYS = ("high_temp_sample_runs", "sample_runs", "semantic_sample_runs")


def _answer_from_run(run: dict) -> str:
    return (run.get("answer_normalized") or run.get("final_answer") or "").strip()


def _sequence_from_run_for_se(run: dict) -> str:
    """Official SE clusters full generations; fall back to extracted answer."""
    return (
        run.get("full_response")
        or run.get("response_text")
        or run.get("answer_raw")
        or _answer_from_run(run)
        or ""
    ).strip()


def _collect_tw_sample_answers(record: dict) -> list[str]:
    """Text + weight perturbation answers (ASE T/W branch; not used for official U_Ecc/U_Deg)."""
    text = record.get("text_rephrase_runs") or []
    weight = record.get("weight_perturb_runs") or []
    answers: list[str] = []
    for run in list(text) + list(weight):
        ans = _answer_from_run(run)
        if ans:
            answers.append(ans)
    return answers


def _collect_se_sample_sequences(record: dict) -> list[str]:
    """
    High-temperature sampled generations for official semantic entropy (NLI clustering).

    Does **not** fall back to T/W perturbation runs — those are a different
    experimental protocol (ASE), not SE (Kuhn et al.).
    """
    for key in SE_SAMPLE_RUN_KEYS:
        runs = record.get(key) or []
        if not runs:
            continue
        seqs = [_sequence_from_run_for_se(r) for r in runs]
        seqs = [s for s in seqs if s]
        if seqs:
            return seqs
    return []


def _collect_se_sample_answers(record: dict) -> list[str]:
    """Extracted final answers from SE runs (diagnostics; SE metric uses full sequences)."""
    for key in SE_SAMPLE_RUN_KEYS:
        runs = record.get(key) or []
        if not runs:
            continue
        answers = [_answer_from_run(r) for r in runs]
        answers = [a for a in answers if a]
        if answers:
            return answers
    return []


def official_nli_sample_uq_from_record(record: dict) -> dict[str, float | str | int]:
    """Official sample-UQ: SE (Kuhn) + U_Ecc/U_Deg (Lin) on the same high-temp generations."""
    se_sequences = _collect_se_sample_sequences(record)
    if not se_sequences:
        return _missing_official_sample_uq_fields()
    out = semantic_entropy_h(se_sequences, cluster_mode="nli")
    out["baseline_SE_status"] = "ok"
    out.update(graph_uncertainty_from_sequences(se_sequences, graph_mode="nli"))
    return out


def official_nli_se_from_record(record: dict) -> dict[str, float | str | int]:
    """Alias for ``official_nli_sample_uq_from_record`` (backward compatible)."""
    return official_nli_sample_uq_from_record(record)


def _missing_official_sample_uq_fields() -> dict[str, float | str | int]:
    nan = float("nan")
    return {
        "baseline_SE_H": nan,
        "baseline_SE_H_norm": nan,
        "baseline_SE_num_clusters": 0,
        "baseline_SE_status": "missing_high_temp_samples",
        "baseline_U_Ecc": nan,
        "baseline_U_Deg": nan,
        "baseline_U_Ecc_graph_mode": "nli_entail",
    }


def _missing_se_fields() -> dict[str, float | str | int]:
    """Legacy alias."""
    return _missing_official_sample_uq_fields()


def sample_baselines_from_record(record: dict) -> dict[str, float | str | int]:
    """
    Sample-tier baselines with correct provenance:
      - SE, U_Ecc, U_Deg ← high_temp_sample_runs (same pool; math_equal placeholder at generation)
    """
    out: dict[str, float | str | int] = {}

    se_sequences = _collect_se_sample_sequences(record)
    if se_sequences:
        out.update(semantic_entropy_h(se_sequences))
        out["baseline_SE_status"] = "ok"
        out.update(graph_uncertainty_from_sequences(se_sequences, graph_mode="math_equal"))
    else:
        nan = float("nan")
        out.update(_missing_official_sample_uq_fields())
        out["baseline_U_Ecc_graph_mode"] = "math_equal"

    return out


def enrich_row_with_baselines(row: dict, record: dict | None = None) -> dict:
    """
    Add baseline score fields to a metrics/features row.

    ``record`` is the full raw_runs JSON when available; otherwise uses
    ``row`` + nested ``base_generation`` only (token baselines).
    """
    rec = record if record is not None else row
    base = rec.get("base_generation") or row.get("base_generation") or {}

    row.update(token_baselines_from_generation(base))
    row.update(sample_baselines_from_record(rec))

    # CoT lower bound: greedy accuracy (diagnostic, not uncertainty)
    if "is_correct_clean" in row:
        row["cot_greedy_acc"] = float(1.0 - int(row.get("label_wrong_clean", 1)))
    elif "label_wrong_clean" in row:
        row["cot_greedy_acc"] = float(1.0 - int(row["label_wrong_clean"]))
    elif rec.get("is_correct") is not None:
        row["cot_greedy_acc"] = float(bool(rec.get("is_correct")))

    # GPU baselines: preserve if already scored externally
    for key in ("baseline_P_True", "baseline_INSIDE"):
        if key in rec and key not in row:
            row[key] = rec[key]

    return row


def baselines_from_record(record: dict) -> dict[str, float | str | int]:
    """Return only baseline fields from a raw record."""
    base = record.get("base_generation") or {}
    out: dict[str, float | str | int] = {}
    out.update(token_baselines_from_generation(base))
    out.update(sample_baselines_from_record(record))
    if record.get("is_correct") is not None:
        out["cot_greedy_acc"] = float(bool(record["is_correct"]))
    for key in ("baseline_P_True", "baseline_INSIDE"):
        if key in record:
            out[key] = record[key]
    return out
