"""Load benchmark rows into unified ASE record dicts."""

from __future__ import annotations

import json
import random
from pathlib import Path

from prs.datasets.registry import DatasetSpec, get_dataset_spec
from prs.grading.tokur_records import _normalize_id
from prs.paths import TOKUR_ROOT

DEFAULT_TFTTCL = Path("/home/phx/TF-TTCL/data/MATH")
DEFAULT_TOKUR_DATA = TOKUR_ROOT / "datasets"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _compose_question(row: dict, spec: DatasetSpec) -> str:
    if spec.domain == "code":
        prompt = row.get("prompt") or row.get("problem") or row.get("question") or ""
        return str(prompt).strip()

    if spec.id == "zebra_puzzles":
        parts: list[str] = []
        for key in ("introduction", "clues", "question", "format_instructions"):
            val = row.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                parts.append("\n".join(str(x) for x in val))
            else:
                parts.append(str(val))
        if parts:
            return "\n\n".join(p.strip() for p in parts if p.strip())
        return str(row.get("problem") or row.get("question") or "").strip()

    for key in ("problem", "question", "instruction", "input"):
        if row.get(key):
            q = str(row[key]).strip()
            if key == "input" and row.get("instruction"):
                q = "\n".join(p for p in [str(row["instruction"]).strip(), q] if p)
            return q
    return ""


def _reference_answer(row: dict, spec: DatasetSpec) -> str:
    if spec.domain == "code":
        return str(row.get("canonical_solution") or row.get("answer") or row.get("solution") or "").strip()
    if spec.id == "zebra_puzzles" and row.get("solution") is not None:
        return json.dumps(row["solution"], ensure_ascii=False, sort_keys=True)
    for key in ("answer", "output", "solution", "gold", "target"):
        if row.get(key) is not None:
            return str(row[key]).strip()
    return ""


def _record_id(row: dict, spec: DatasetSpec, index: int) -> str:
    raw = row.get("unique_id") or row.get("task_id") or row.get("id")
    if raw is not None:
        uid = _normalize_id(str(raw))
        if spec.id == "deepscaler" and not uid.startswith("deepscaler_"):
            uid = f"deepscaler_{uid}"
        return uid
    if spec.id == "humaneval" and row.get("task_id"):
        return _normalize_id(str(row["task_id"]))
    return f"{spec.id}_{index}"


def load_tokur_jsonl_rows(
    path: Path,
    spec: DatasetSpec,
    *,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    rows = _load_jsonl(path)
    if not rows and path.with_suffix(".json").exists():
        text = path.with_suffix(".json").read_text(encoding="utf-8").strip()
        rows = json.loads(text) if text.startswith("[") else _load_jsonl(path.with_suffix(".json"))

    rng = random.Random(seed)
    if spec.id in ("math500", "deepscaler", "zebra_puzzles", "color_cube"):
        rows = list(rows)
        rng.shuffle(rows)

    out: list[dict] = []
    for i, row in enumerate(rows):
        if max_samples is not None and len(out) >= max_samples:
            break
        question = _compose_question(row, spec)
        if not question:
            continue
        ref = _reference_answer(row, spec)
        uid = _record_id(row, spec, i)
        out.append(
            {
                "id": uid,
                "dataset": spec.id,
                "question": question,
                "reference": ref,
                "a0": "",
                "is_correct": False,
                "label_wrong": 1,
                "entry_point": row.get("entry_point", ""),
                "test": row.get("test", ""),
            }
        )
    return out


def _load_minerva_gsm8k_math500(tfttcl_root: Path, spec: DatasetSpec, max_samples: int | None) -> list[dict]:
    from prs.token_qaac.data import load_tfttcl_originals

    by_id = load_tfttcl_originals(spec.id, tfttcl_root)
    rows = list(by_id.values())
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def load_dataset_records(
    dataset: str,
    *,
    tokur_data_root: Path | None = None,
    tfttcl_root: Path | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load questions + references for one dataset."""
    spec = get_dataset_spec(dataset)
    tokur_root = tokur_data_root or DEFAULT_TOKUR_DATA
    tfttcl = tfttcl_root or DEFAULT_TFTTCL

    if spec.id in ("minerva", "gsm8k", "math500"):
        return _load_minerva_gsm8k_math500(tfttcl, spec, max_samples)

    if spec.id == "deepscaler":
        path = tokur_root / spec.jsonl_name
        return load_tokur_jsonl_rows(path, spec, max_samples=max_samples, seed=seed)

    path = tokur_root / spec.jsonl_name
    if not path.exists():
        alt = tokur_root / spec.jsonl_name.replace("_", "-")
        if alt.exists():
            path = alt
    return load_tokur_jsonl_rows(path, spec, max_samples=max_samples, seed=seed)
