#!/usr/bin/env python3
"""Recompute features + summary from raw_runs/ (no GPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from prs.ase.metrics import metrics_from_record
from prs.ase.record import rebuild_summary_jsonl
from prs.paths import DEFAULT_OUT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/home/phx/PRS/outputs/ase_full"))
    ap.add_argument("--datasets", default="minerva,math500")
    ap.add_argument("--atu-top-pct", type=float, default=0.10)
    args = ap.parse_args()

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        raw_dir = args.out_dir / ds / "raw_runs"
        if not raw_dir.exists():
            print(f"skip {ds}: no {raw_dir}")
            continue
        paths = sorted(
            p
            for p in raw_dir.glob("*.json")
            if not p.name.endswith(".error.json") and not p.name.endswith(".partial.json")
        )
        rows = []
        skipped = []
        for p in tqdm(paths, desc=f"recompute {ds}"):
            text = p.read_text(encoding="utf-8").strip()
            if not text:
                skipped.append(p.name)
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                skipped.append(p.name)
                continue
            row = metrics_from_record(record, top_pct=args.atu_top_pct)
            rows.append(row)
            # refresh embedded summary_metrics
            record["summary_metrics"] = row
            p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        if skipped:
            print(f"{ds}: skipped {len(skipped)} empty/corrupt raw files: {', '.join(skipped)}")

        feat = args.out_dir / ds / "features.jsonl"
        with feat.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n = rebuild_summary_jsonl(args.out_dir, ds)
        print(f"{ds}: features={len(rows)} summary={n} top_pct={args.atu_top_pct}")


if __name__ == "__main__":
    main()