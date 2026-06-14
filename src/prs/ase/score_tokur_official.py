#!/usr/bin/env python3
"""Strict TokUR baseline: official vLLM + bayesian_transformer greedy EU only.

This is **not** the post-hoc ``LowRankWeightPerturbation`` path in ``score_tokur_baseline.py``.
EU comes from per-token ``epistemic_uncertainty`` during **greedy generation** in
``TokUR/run/greedy_unc_single_batch_refine.py`` (TFB checkpoint, native TFB layers).

Writes ``tokur_baseline.jsonl`` rows: ``id``, ``tokur_eu_sum``, ``is_correct`` (TokUR grader),
optional ``is_correct_ase`` from ASE ``raw_runs`` when ``--out-dir`` is set.
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

from prs.grading.tokur_records import (
    _extract_answer_text,
    extract_tokur_unc_from_output,
    label_from_answer,
)
from prs.grading.export_tokur_pkl import _StubUnpickler
from prs.paths import DEFAULT_OUT, TOKUR_ROOT


def _read_raw_header(p: Path) -> dict | None:
    try:
        data = p.read_bytes()
    except OSError:
        return None
    if len(data) < 1000:
        return None
    for marker in (b'"base_generation"', b'"clean_gen"'):
        idx = data.find(marker)
        if idx >= 0:
            try:
                return json.loads(data[:idx].rstrip().rstrip(b",") + b"}")
            except json.JSONDecodeError:
                return None
    return None


def _load_ase_labels(out_dir: Path, dataset: str) -> dict[str, bool | None]:
    raw_dir = out_dir / dataset / "raw_runs"
    labels: dict[str, bool | None] = {}
    if not raw_dir.is_dir():
        return labels
    for p in raw_dir.glob("*.json"):
        if p.name.endswith((".error.json", ".partial.json")):
            continue
        header = _read_raw_header(p)
        if header:
            labels[header.get("id", p.stem)] = header.get("is_correct")
    return labels


def pkl_records_to_baseline_rows(
    pkl_glob: str,
    dataset: str,
    *,
    ase_labels: dict[str, bool | None] | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Load official pkl batches → tokur_baseline.jsonl-shaped dicts."""
    rows: list[dict] = []
    for fp in sorted(glob(pkl_glob)):
        try:
            with open(fp, "rb") as f:
                batch = _StubUnpickler(f).load()
        except Exception:
            continue
        if not isinstance(batch, list):
            batch = [batch]
        for sample in batch:
            try:
                result = sample["result"][0] if isinstance(sample["result"], list) else sample["result"]
                out = result.outputs[0]
                text = out.text
                unc = extract_tokur_unc_from_output(out)
                if unc is None:
                    continue
                gt = sample["answer"]
                uid = str(sample["unique_id"]).replace("/", "_").replace(".json", "")
                pred = _extract_answer_text(text, dataset)
                tokur_correct = label_from_answer(pred, gt, dataset)
                row = {
                    "id": uid,
                    "dataset": dataset,
                    "tokur_eu_sum": unc["eu_sum"],
                    "tokur_tu_sum": unc["tu_sum"],
                    "tokur_au_sum": unc["au_sum"],
                    "tokur_eu_mean": unc["eu_sum"] / max(len(getattr(out, "uncertainties", []) or []), 1),
                    "tokur_tu_mean": unc["tu_sum"] / max(len(getattr(out, "uncertainties", []) or []), 1),
                    "n_response_tokens": len(getattr(out, "uncertainties", []) or []),
                    "is_correct": tokur_correct,
                    "scoring_mode": "official_vllm",
                    "tokur_seed": seed,
                    "error": "",
                }
                if ase_labels and uid in ase_labels:
                    row["is_correct_ase"] = ase_labels[uid]
                rows.append(row)
            except Exception:
                continue
    return rows


def write_baseline_jsonl(rows: list[dict], out_path: Path) -> int:
    by_id: dict[str, dict] = {}
    for r in rows:
        by_id[r["id"]] = r
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rid in sorted(by_id):
            f.write(json.dumps(by_id[rid], ensure_ascii=False) + "\n")
    return len(by_id)


def default_pkl_glob(
    tokur_root: Path,
    model_tag: str,
    dataset: str,
    seed: int,
    *,
    legacy_ds: bool = False,
) -> str:
    # Legacy qwen3b/ase_full: ase_{dataset}; other models: ase_{model_tag}_{dataset}
    tokur_ds = f"ase_{dataset}" if legacy_ds else f"ase_{model_tag}_{dataset}"
    return str(
        tokur_root
        / "results"
        / f"{model_tag}_results_vllm_pg"
        / tokur_ds
        / f"seed{seed}"
        / "greedy_unc"
        / "*.pkl"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Official TokUR pkl → tokur_baseline.jsonl")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pkl-glob", default=None, help="Override pkl glob (default: TokUR results/...)")
    ap.add_argument("--tokur-root", type=Path, default=TOKUR_ROOT)
    ap.add_argument("--model-tag", default="qwen3b", help="TokUR results folder tag (qwen3b, llama8b, ...)")
    ap.add_argument(
        "--legacy-tokur-ds",
        action="store_true",
        help="PKL under ase_{dataset}/ (qwen3b + ase_full only)",
    )
    ap.add_argument("--seed", type=int, default=96)
    ap.add_argument(
        "--merge-seeds",
        default="",
        help="Comma-separated seeds; average tokur_eu_sum per id (e.g. 96,89,64)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: {out_dir}/{dataset}/tokur_baseline.jsonl",
    )
    args = ap.parse_args()

    ase_labels = _load_ase_labels(args.out_dir, args.dataset)
    seeds = [args.seed]
    if args.merge_seeds.strip():
        seeds = [int(s.strip()) for s in args.merge_seeds.split(",") if s.strip()]

    all_rows: list[dict] = []
    for seed in seeds:
        pkl_glob = args.pkl_glob or default_pkl_glob(
            args.tokur_root,
            args.model_tag,
            args.dataset,
            seed,
            legacy_ds=args.legacy_tokur_ds,
        )
        rows = pkl_records_to_baseline_rows(
            pkl_glob, args.dataset, ase_labels=ase_labels, seed=seed
        )
        all_rows.extend(rows)
        print(f"seed={seed}: {len(rows)} rows from {pkl_glob}")

    if len(seeds) > 1 and all_rows:
        by_id: dict[str, list[dict]] = {}
        for r in all_rows:
            by_id.setdefault(r["id"], []).append(r)
        merged: list[dict] = []
        for rid, group in by_id.items():
            eu_mean = sum(g["tokur_eu_sum"] for g in group) / len(group)
            base = dict(group[0])
            base["tokur_eu_sum"] = eu_mean
            base["tokur_seed"] = seeds
            base["scoring_mode"] = "official_vllm_multi_seed_mean"
            merged.append(base)
        all_rows = merged

    out_path = args.output or (args.out_dir / args.dataset / "tokur_baseline.jsonl")
    n = write_baseline_jsonl(all_rows, out_path)
    print(f"Wrote {n} rows → {out_path}")


if __name__ == "__main__":
    main()