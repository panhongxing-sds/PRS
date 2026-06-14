#!/usr/bin/env python3
"""Recompute features + summary from raw_runs/ (no GPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from prs.ase.metrics import enrich_row_with_clean_label, metrics_from_record
from prs.ase.record import rebuild_summary_jsonl
from prs.baselines.from_record import official_nli_se_from_record
from prs.paths import DEFAULT_OUT

_SE_FIELDS = (
    "baseline_SE_H",
    "baseline_SE_H_norm",
    "baseline_SE_num_clusters",
    "baseline_SE_cluster_mode",
    "baseline_SE_status",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/home/phx/PRS/outputs/ase_full"))
    ap.add_argument("--datasets", default="minerva,math500")
    ap.add_argument("--atu-top-pct", type=float, default=0.10)
    ap.add_argument(
        "--from-cache",
        action="store_true",
        help="Fast path: read the summary_metrics already embedded in each raw json "
        "(computed at generation time) instead of recomputing metrics_from_record. "
        "~15-25x faster, no raw rewrite. Use when metric code is unchanged since generation.",
    )
    ap.add_argument(
        "--recompute-se",
        action="store_true",
        help="Re-score official SE (NLI clustering on high_temp_sample_runs) and refresh "
        "baseline_SE_* fields in summary_metrics + features.jsonl. Compatible with --from-cache.",
    )
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
        missing_cache = 0
        mode = "recompute"
        if args.recompute_se and not args.from_cache:
            mode = "recompute-se"
        elif args.from_cache and not args.recompute_se:
            mode = "extract"
        elif args.from_cache and args.recompute_se:
            mode = "recompute-se"
        for p in tqdm(paths, desc=f"{mode} {ds}"):
            text = p.read_text(encoding="utf-8").strip()
            if not text:
                skipped.append(p.name)
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                skipped.append(p.name)
                continue
            if args.from_cache or args.recompute_se:
                row = record.get("summary_metrics")
                if not row:
                    # Fall back to a one-off recompute for this record only.
                    row = metrics_from_record(record, top_pct=args.atu_top_pct)
                    missing_cache += 1
                    persist = True
                else:
                    # Always refresh the clean label with the strict grader so that
                    # label_wrong_clean / label_drop reflect the current grading code
                    # (fixes the legacy sci-notation misjudgment; adds label_drop).
                    prev = (row.get("label_wrong_clean"), row.get("label_drop"))
                    enrich_row_with_clean_label(row, model=(record.get("model_info") or {}).get("model_name"))
                    persist = prev != (row.get("label_wrong_clean"), row.get("label_drop"))
                if args.recompute_se:
                    se = official_nli_se_from_record(record)
                    for key in _SE_FIELDS:
                        if key in se:
                            row[key] = se[key]
                    persist = True
                if persist:
                    # Persist into raw json so rebuild_summary_jsonl picks up strict labels.
                    record["summary_metrics"] = row
                    p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(row)
                continue
            row = metrics_from_record(record, top_pct=args.atu_top_pct)
            rows.append(row)
            # refresh embedded summary_metrics
            record["summary_metrics"] = row
            p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        if skipped:
            print(f"{ds}: skipped {len(skipped)} empty/corrupt raw files: {', '.join(skipped)}")
        if args.from_cache and missing_cache:
            print(f"{ds}: {missing_cache} records lacked cached summary_metrics (recomputed those)")

        feat = args.out_dir / ds / "features.jsonl"
        with feat.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n = rebuild_summary_jsonl(args.out_dir, ds)
        print(f"{ds}: features={len(rows)} summary={n} top_pct={args.atu_top_pct}")


if __name__ == "__main__":
    main()