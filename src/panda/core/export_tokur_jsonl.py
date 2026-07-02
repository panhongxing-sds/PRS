#!/usr/bin/env python3
"""Export ASE ``raw_runs`` subset to TokUR ``greedy_unc`` jsonl (problem/answer/unique_id)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from panda.paths import DEFAULT_OUT, TOKUR_ROOT


def _read_raw_header(p: Path) -> dict | None:
    """Parse only metadata before ``base_generation`` (raw_runs files can be 50MB+)."""
    try:
        data = p.read_bytes()
    except OSError:
        return None
    if len(data) < 1000:
        return None
    marker = b'"base_generation"'
    idx = data.find(marker)
    if idx < 0:
        marker = b'"clean_gen"'
        idx = data.find(marker)
    if idx < 0:
        return None
    try:
        return json.loads(data[:idx].rstrip().rstrip(b",") + b"}")
    except json.JSONDecodeError:
        return None


def _load_raw_rows(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(raw_dir.glob("*.json")):
        if p.name.endswith((".error.json", ".partial.json")):
            continue
        header = _read_raw_header(p)
        if not header:
            continue
        rid = header.get("id", p.stem)
        rows.append(
            {
                "id": rid,
                "question": header.get("question", ""),
                "reference": header.get("reference", ""),
            }
        )
    return rows


def export_ase_tokur_jsonl(
    out_dir: Path,
    dataset: str,
    jsonl_path: Path,
) -> int:
    raw_dir = out_dir / dataset / "raw_runs"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing ASE raw_runs: {raw_dir}")

    rows = _load_raw_rows(raw_dir)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in rows:
            line = {
                "unique_id": rec["id"],
                "problem": rec["question"],
                "answer": rec["reference"],
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ASE raw_runs → TokUR greedy_unc jsonl")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Output path (default: $TOKUR_ROOT/datasets/panda_{dataset}.jsonl)",
    )
    ap.add_argument("--tokur-root", type=Path, default=TOKUR_ROOT)
    args = ap.parse_args()

    jsonl = args.jsonl or (args.tokur_root / "datasets" / f"ase_{args.dataset}.jsonl")
    n = export_ase_tokur_jsonl(args.out_dir, args.dataset, jsonl)
    print(f"Wrote {n} rows → {jsonl}")


if __name__ == "__main__":
    main()