"""Build and persist full ASE experiment records (raw + summary)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prs.ase.semantic_cache import build_semantic_cache
from prs.grading.answer_canonicalizer import grade_answer
from prs.grading.math_grader import math_equal


def adversarial_pgd_config_dict(
    *,
    seed: int,
    rank: int,
    epsilon: float,
    pgd_steps: int,
    attack_objective: str,
    attack_loss_final: float,
    target_modules: list[str],
) -> dict[str, Any]:
    return {
        "perturb_type": "adversarial_pgd_multistart",
        "perturb_seed": seed,
        "perturb_rank": rank,
        "perturb_epsilon": epsilon,
        "pgd_steps": pgd_steps,
        "attack_objective": attack_objective,
        "attack_loss_final": attack_loss_final,
        "target_modules": target_modules,
    }


def perturb_config_dict(
    *,
    seed: int,
    sigma: float,
    rank: int,
    target_suffixes: tuple[str, ...],
    target_modules: list[str],
    noise_norm: float | None = None,
) -> dict[str, Any]:
    return {
        "perturb_seed": seed,
        "perturb_type": "low_rank_gaussian",
        "perturb_scale": sigma,
        "perturb_rank": rank,
        "target_suffixes": list(target_suffixes),
        "target_modules": target_modules,
        "noise_norm": noise_norm,
        "relative_noise_norm": None,
    }


def _run_from_gen(
    run_id: str,
    gen: dict,
    *,
    source: str,
    rephrase_text: str | None = None,
    perturb_config: dict | None = None,
    reference: str = "",
) -> dict:
    ans = gen.get("answer_normalized") or gen.get("final_answer", "")
    ref = reference.strip()
    correct = math_equal(ans, ref) if ref and ans else False
    row = {
        "run_id": run_id,
        "source": source,
        "input_prompt": gen.get("input_prompt"),
        "rephrase_text": rephrase_text,
        "perturb_config": perturb_config,
        "answer_raw": gen.get("answer_raw", ans),
        "answer_normalized": ans,
        "full_response": gen.get("response_text", ""),
        "parse_success": gen.get("parse_success", bool(ans)),
        "correctness": correct,
        "token_trace": gen.get("token_trace", []),
        "answer_span": gen.get("answer_span", {}),
        "n_tokens": gen.get("n_tokens", 0),
        "decoding": gen.get("decoding", {}),
        # legacy flat lists for fast metrics
        "token_entropies": gen.get("token_entropies", []),
        "token_margins": gen.get("token_margins", []),
    }
    return row


def runs_from_high_temp_answers(
    answers: list[str],
    *,
    temperature: float,
    top_p: float,
    reference: str = "",
) -> list[dict]:
    """Minimal run rows for official SE (answers only, no token trace)."""
    decoding = {"temperature": temperature, "top_p": top_p, "do_sample": True}
    runs: list[dict] = []
    for i, ans in enumerate(answers):
        ref = reference.strip()
        correct = math_equal(ans, ref) if ref and ans else False
        runs.append(
            {
                "run_id": f"SE_{i}",
                "source": "high_temp_sample",
                "input_prompt": None,
                "rephrase_text": None,
                "perturb_config": None,
                "answer_raw": ans,
                "answer_normalized": ans,
                "full_response": "",
                "parse_success": bool(ans),
                "correctness": correct,
                "token_trace": [],
                "answer_span": {},
                "n_tokens": 0,
                "decoding": decoding,
                "token_entropies": [],
                "token_margins": [],
            }
        )
    return runs


def build_full_record(
    rec: dict,
    *,
    clean_gen: dict,
    text_runs: list[dict],
    weight_runs: list[dict],
    high_temp_runs: list[dict] | None = None,
    model_info: dict,
    experiment_config: dict,
) -> dict:
    ref = str(rec.get("reference", "")).strip()
    a0 = clean_gen.get("answer_normalized", "")
    # Mode-aware strict grading (math/string/code) so the raw stored label is not a
    # bogus math_equal verdict for non-numeric datasets (e.g. logic color words).
    g = grade_answer(
        a0, ref, record_id=rec.get("id"), dataset=rec.get("dataset"),
        model=(model_info or {}).get("model_name"),
    )
    ok = bool(g["is_correct_clean"])

    text_answers = [r["answer_normalized"] for r in text_runs]
    weight_answers = [r["answer_normalized"] for r in weight_runs]

    semantic = build_semantic_cache(a0, text_answers, weight_answers)

    return {
        "id": rec["id"],
        "dataset": rec["dataset"],
        "question": rec.get("question", ""),
        "reference": ref,
        "reference_normalized": ref,
        "is_correct": ok,
        "label_wrong": 0 if ok else 1,
        "is_correct_clean": bool(g["is_correct_clean"]),
        "label_wrong_clean": g["label_wrong_clean"],
        "label_drop": g["label_drop"],
        "relabeled": g["relabeled"],
        "model_info": model_info,
        "experiment_config": experiment_config,
        "base_generation": _run_from_gen("base", clean_gen, source="base", reference=ref),
        "text_rephrase_runs": text_runs,
        "weight_perturb_runs": weight_runs,
        "high_temp_sample_runs": high_temp_runs or [],
        "semantic_cache": semantic,
    }


def raw_run_path(out_dir: Path, dataset: str, record_id: str) -> Path:
    return out_dir / dataset / "raw_runs" / f"{record_id}.json"


def partial_run_path(out_dir: Path, dataset: str, record_id: str) -> Path:
    return out_dir / dataset / "raw_runs" / f"{record_id}.partial.json"


def summary_path(out_dir: Path, dataset: str) -> Path:
    return out_dir / dataset / "summary.jsonl"


def _atomic_write_json(path: Path, payload: dict, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_partial_record(out_dir: Path, partial: dict) -> None:
    """Checkpoint mid-question progress (base + completed text/weight runs)."""
    ds = partial["dataset"]
    rid = partial["id"]
    path = partial_run_path(out_dir, ds, rid)
    _atomic_write_json(path, partial, indent=None)
    print(f"[checkpoint] {ds}/{rid} partial saved", flush=True)


def load_partial_record(out_dir: Path, dataset: str, record_id: str) -> dict | None:
    path = partial_run_path(out_dir, dataset, record_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_partial_record(out_dir: Path, dataset: str, record_id: str) -> None:
    path = partial_run_path(out_dir, dataset, record_id)
    if path.exists():
        path.unlink()


def append_summary_line(out_dir: Path, dataset: str, record: dict, summary_metrics: dict) -> None:
    sp = summary_path(out_dir, dataset)
    sp.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "id": record["id"],
        "dataset": dataset,
        "a0": record.get("base_generation", {}).get("answer_normalized", ""),
        "reference": record.get("reference", ""),
        "is_correct": record.get("is_correct"),
        "label_wrong": record.get("label_wrong"),
        **summary_metrics,
    }
    with sp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def save_record(out_dir: Path, record: dict, summary_metrics: dict) -> None:
    """Write one raw_runs/{id}.json (includes summary_metrics for quick scan)."""
    ds = record["dataset"]
    rid = record["id"]
    path = raw_run_path(out_dir, ds, rid)
    record_out = dict(record)
    record_out["summary_metrics"] = summary_metrics
    _atomic_write_json(path, record_out)
    delete_partial_record(out_dir, ds, rid)
    append_summary_line(out_dir, ds, record_out, summary_metrics)
    print(f"[saved] {ds}/{rid} → {path}", flush=True)


def rebuild_summary_jsonl(out_dir: Path, dataset: str) -> int:
    """Rebuild summary.jsonl from all raw_runs/*.json."""
    raw_dir = out_dir / dataset / "raw_runs"
    sp = summary_path(out_dir, dataset)
    if not raw_dir.exists():
        return 0
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        if p.name.endswith(".error.json") or p.name.endswith(".partial.json"):
            continue
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        try:
            rec = json.loads(text)
        except json.JSONDecodeError:
            continue
        sm = rec.get("summary_metrics") or {}
        rows.append(
            {
                "id": rec["id"],
                "dataset": dataset,
                "a0": rec.get("base_generation", {}).get("answer_normalized", ""),
                "reference": rec.get("reference", ""),
                "is_correct": rec.get("is_correct"),
                "label_wrong": rec.get("label_wrong"),
                **sm,
            }
        )
    sp.parent.mkdir(parents=True, exist_ok=True)
    with sp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def record_exists(out_dir: Path, dataset: str, record_id: str) -> bool:
    return raw_run_path(out_dir, dataset, record_id).exists()


def load_record(out_dir: Path, dataset: str, record_id: str) -> dict:
    return json.loads(raw_run_path(out_dir, dataset, record_id).read_text(encoding="utf-8"))
