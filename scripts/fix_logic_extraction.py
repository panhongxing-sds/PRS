#!/usr/bin/env python
"""Offline repair for logic/code datasets whose answers were extracted with the
math extractor at generation time (bug: a plain word like "indigo" -> "").

For each raw record it re-extracts answer_raw/answer_normalized for the base
generation and every perturbation run from the stored full_response using the
dataset-appropriate extractor, recomputes is_correct via the dataset grader,
then regenerates summary_metrics (ASE/PANDA + clean label) and rebuilds
summary.jsonl / features.jsonl.

No GPU / model calls: full_response is already saved in every raw json.
Run only on shards whose generation has finished (avoid racing the live run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from panda.core.metrics import metrics_from_record
from panda.core.record import rebuild_summary_jsonl
from panda.grading.answer_canonicalizer import grade_answer
from panda.grading.extract import extract_answer_for_dataset

RUN_SECTIONS = ("text_rephrase_runs", "weight_perturb_runs", "high_temp_sample_runs")


def _reextract_run(run: dict, dataset: str) -> bool:
    """Re-extract a single run's answer from its full_response. Returns True if changed."""
    fr = run.get("full_response") or run.get("response_text") or ""
    if not fr:
        return False  # nothing to recover (e.g. high_temp stores answer only)
    new = extract_answer_for_dataset(fr, dataset) or fr.strip()
    if new != run.get("answer_normalized"):
        run["answer_raw"] = new
        run["answer_normalized"] = new
        return True
    return False


def fix_record(record: dict, dataset: str) -> bool:
    changed = False
    base = record.get("base_generation") or {}
    if base:
        changed |= _reextract_run(base, dataset)
    for sect in RUN_SECTIONS:
        for run in record.get(sect) or []:
            changed |= _reextract_run(run, dataset)

    a0 = base.get("answer_normalized", "")
    ref = str(record.get("reference", ""))
    g = grade_answer(a0, ref, record_id=record.get("id"), dataset=dataset)
    ok = bool(g["is_correct_clean"])  # mode-aware strict verdict, not bogus math_equal
    if record.get("is_correct") != ok or record.get("label_wrong") != (0 if ok else 1):
        changed = True
    record["is_correct"] = ok
    record["label_wrong"] = 0 if ok else 1
    record["is_correct_clean"] = ok
    record["label_wrong_clean"] = g["label_wrong_clean"]
    record["label_drop"] = g["label_drop"]
    record["relabeled"] = g["relabeled"]
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True, help="e.g. .../maintable_qwen25_3b/seed41")
    ap.add_argument("--datasets", default="color_cube")
    ap.add_argument("--atu-top-pct", type=float, default=0.10)
    args = ap.parse_args()

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        raw_dir = args.out_dir / ds / "raw_runs"
        if not raw_dir.exists():
            print(f"skip {ds}: no {raw_dir}")
            continue
        paths = sorted(
            p for p in raw_dir.glob("*.json")
            if not p.name.endswith(".error.json") and not p.name.endswith(".partial.json")
        )
        rows = []
        n_changed = 0
        for p in tqdm(paths, desc=f"fix {ds}"):
            text = p.read_text(encoding="utf-8").strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            changed = fix_record(record, ds)
            row = metrics_from_record(record, top_pct=args.atu_top_pct)
            record["summary_metrics"] = row
            rows.append(row)
            if changed:
                n_changed += 1
            p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        feat = args.out_dir / ds / "features.jsonl"
        with feat.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n = rebuild_summary_jsonl(args.out_dir, ds)
        acc = sum(1 for r in rows if not r.get("label_wrong", 1)) / len(rows) if rows else 0.0
        print(f"{ds}: records={len(rows)} changed={n_changed} summary={n} acc={acc*100:.1f}%")


if __name__ == "__main__":
    main()
