"""Load originals, API rephrases, and candidate answers."""

from __future__ import annotations

import json
from pathlib import Path

from prs.datasets.loaders import load_dataset_records
from prs.datasets.registry import get_dataset_spec, list_dataset_ids, normalize_dataset_id
from prs.grading.math_grader import extract_math_answer, math_equal
from prs.grading.tokur_records import _extract_answer_text, _normalize_id
from prs.paths import TOKUR_ROOT


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _is_math_dataset(dataset: str) -> bool:
    try:
        return get_dataset_spec(dataset).grading == "math"
    except ValueError:
        return dataset in ("math500", "minerva", "deepscaler", "gsm8k")


def _unique_answers(answers: list[str], dataset: str) -> list[str]:
    out: list[str] = []
    seen_norm: set[str] = set()
    for ans in answers:
        a = extract_math_answer(ans) if _is_math_dataset(dataset) else ans.strip()
        if not a:
            continue
        key = a.strip().lower()
        if key in seen_norm:
            continue
        seen_norm.add(key)
        out.append(a)
    return out


def unique_answers(answers: list[str], dataset: str) -> list[str]:
    return _unique_answers(answers, dataset)


def load_variants_by_id(variants_path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for row in _load_jsonl(variants_path):
        uid = _normalize_id(row.get("unique_id", row.get("id", "")))
        variants = row.get("variants") or []
        # Drop duplicate original; keep rephrases only when first equals original
        original = (row.get("original") or variants[0] if variants else "").strip()
        rephrases = []
        for v in variants:
            v = v.strip()
            if not v:
                continue
            if v == original and rephrases:
                continue
            rephrases.append(v)
        if original and (not rephrases or rephrases[0] != original):
            rephrases = [original] + [v for v in rephrases if v != original]
        by_id[uid] = {
            "question": original,
            "rephrases": rephrases[1:] if len(rephrases) > 1 else rephrases,
            "all_variants": rephrases,
        }
    return by_id


def load_qaac_results(results_path: Path, dataset: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for row in _load_jsonl(results_path):
        uid = _normalize_id(row.get("unique_id", row.get("id", "")))
        variants = row.get("variants") or []
        extracted = row.get("extracted_answers") or []
        a0 = extracted[0] if extracted else row.get("majority_answer", "")
        rephrase_answers = extracted[1:] if len(extracted) > 1 else []
        by_id[uid] = {
            "id": uid,
            "dataset": dataset,
            "question": variants[0] if variants else "",
            "a0": a0,
            "is_correct": bool(row.get("is_correct", False)),
            "label_wrong": 0 if row.get("is_correct", False) else 1,
            "reference": row.get("gold_answer", ""),
            "rephrase_answers": rephrase_answers,
            "variants": variants,
        }
    return by_id


def load_tfttcl_originals(
    dataset: str,
    data_root: Path,
    *,
    only_ids: set[str] | None = None,
) -> dict[str, dict]:
    """Load gold + question from TF-TTCL MATH JSON (aligned with api bench idx ids)."""
    by_id: dict[str, dict] = {}
    if dataset == "minerva":
        rows = json.loads((data_root / "minerva.json").read_text(encoding="utf-8"))
        for i, r in enumerate(rows):
            uid = f"minerva_{i}"
            question = str(r.get("instruction", "")).strip()
            ref = str(r.get("output", "")).strip()
            by_id[uid] = {
                "id": uid,
                "dataset": dataset,
                "question": question,
                "reference": ref,
                "a0": "",
                "is_correct": False,
                "label_wrong": 1,
            }
        return by_id

    if dataset == "gsm8k":
        rows = json.loads((data_root / "gsm8k.json").read_text(encoding="utf-8"))
        for i, r in enumerate(rows):
            uid = f"gsm8k_{i}"
            parts = [str(r.get("instruction", "")).strip(), str(r.get("input", "")).strip()]
            question = "\n".join(p for p in parts if p)
            ref = str(r.get("output", "")).strip()
            by_id[uid] = {
                "id": uid,
                "dataset": dataset,
                "question": question,
                "reference": ref,
                "a0": "",
                "is_correct": False,
                "label_wrong": 1,
            }
        return by_id

    if dataset == "math500":
        path = data_root / "math500.json"
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        for i, r in enumerate(rows):
            uid = _normalize_id(r.get("unique_id", str(i)))
            question = str(r.get("problem") or r.get("instruction") or "").strip()
            if r.get("input"):
                question = "\n".join(p for p in [question, str(r.get("input")).strip()] if p)
            ref = str(r.get("answer") or r.get("output") or "").strip()
            by_id[uid] = {
                "id": uid,
                "dataset": dataset,
                "question": question,
                "reference": ref,
                "a0": "",
                "is_correct": False,
                "label_wrong": 1,
            }
        return by_id

    if dataset == "aime24":
        rows = json.loads((data_root / "aime24.json").read_text(encoding="utf-8"))
        for i, r in enumerate(rows):
            uid = f"aime24_{i}"
            question = str(r.get("instruction", "")).strip()
            ref = str(r.get("output", "")).strip()
            by_id[uid] = {
                "id": uid,
                "dataset": dataset,
                "question": question,
                "reference": ref,
                "a0": "",
                "is_correct": False,
                "label_wrong": 1,
            }
        return by_id

    if dataset == "deepscaler":
        ds_path = TOKUR_ROOT / "datasets" / "deepscaler.jsonl"
        for line in ds_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            uid = str(r.get("unique_id", "")).replace("/", "_").replace(".json", "")
            uid = f"deepscaler_{uid}"
            if only_ids is not None and uid not in only_ids:
                continue
            question = str(r.get("problem") or r.get("question") or "").strip()
            ref = str(r.get("answer") or r.get("solution") or "").strip()
            by_id[uid] = {
                "id": uid,
                "dataset": dataset,
                "question": question,
                "reference": ref,
                "a0": "",
                "is_correct": False,
                "label_wrong": 1,
            }
        return by_id

    try:
        spec = get_dataset_spec(dataset)
        if spec.domain in ("logic", "code"):
            return {
                r["id"]: r
                for r in load_dataset_records(spec.id, tfttcl_root=data_root, max_samples=None)
            }
    except ValueError:
        pass

    raise ValueError(f"Unsupported dataset for TF-TTCL load: {dataset}")


def load_tokur_originals(tokur_path: Path, dataset: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for row in _load_jsonl(tokur_path):
        if row.get("dataset", dataset) != dataset and dataset not in row.get("dataset", dataset):
            pass
        uid = _normalize_id(row.get("id", ""))
        response = row.get("response", "")
        a0 = _extract_answer_text(response, dataset)
        ref = row.get("reference", "")
        is_correct = bool(row.get("is_correct", math_equal(a0, ref)))
        by_id[uid] = {
            "id": uid,
            "dataset": dataset,
            "question": row.get("question", ""),
            "prompt": row.get("prompt", ""),
            "response": response,
            "a0": a0,
            "reference": ref,
            "is_correct": is_correct,
            "label_wrong": 0 if is_correct else 1,
        }
    return by_id


def build_records(
    *,
    dataset: str,
    variants_path: Path,
    results_path: Path | None = None,
    tokur_path: Path | None = None,
    tfttcl_root: Path | None = None,
    max_samples: int | None = None,
) -> list[dict]:
    variants = load_variants_by_id(variants_path)
    variant_rows = _load_jsonl(variants_path)
    records_by_id: dict[str, dict] = {}

    if results_path and results_path.exists():
        records_by_id = load_qaac_results(results_path, dataset)

    if tokur_path and tokur_path.exists():
        for uid, rec in load_tokur_originals(tokur_path, dataset).items():
            if uid not in records_by_id:
                records_by_id[uid] = rec

    ds_norm = normalize_dataset_id(dataset) if dataset in list_dataset_ids() else dataset
    only_ids = set(variants.keys()) if ds_norm == "deepscaler" else None
    if tfttcl_root and tfttcl_root.exists():
        for uid, rec in load_tfttcl_originals(ds_norm, tfttcl_root, only_ids=only_ids).items():
            if uid not in records_by_id:
                records_by_id[uid] = rec
            else:
                base = records_by_id[uid]
                for k in ("question", "reference"):
                    if not base.get(k):
                        base[k] = rec.get(k, "")

    out: list[dict] = []
    for row in variant_rows:
        uid = _normalize_id(row.get("unique_id", row.get("id", "")))
        v = variants.get(uid)
        if not v or uid not in records_by_id:
            continue
        rec = dict(records_by_id[uid])
        rec["id"] = uid
        rec["dataset"] = dataset
        rec["question"] = rec.get("question") or v["question"]
        rec["rephrases"] = v["rephrases"]
        rec["all_variants"] = v["all_variants"]

        cand_answers = [rec.get("a0", "")]
        cand_answers.extend(rec.get("rephrase_answers") or [])
        rec["candidates"] = _unique_answers([a for a in cand_answers if a], dataset)
        if rec.get("a0") and rec["a0"] not in rec["candidates"]:
            rec["candidates"] = [rec["a0"]] + rec["candidates"]
        out.append(rec)

    if max_samples is not None:
        out = out[:max_samples]
    return out