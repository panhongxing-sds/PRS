#!/usr/bin/env python3
"""Generate ISO baseline comparison table (AUROC / AUPRC / ACC*)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from panda.core.analyze_main_tables import load_enriched, merge_tokur_scores, relabel
from panda.baselines.from_record import enrich_row_with_baselines
from panda.baselines.registry import BASELINE_REGISTRY, BaselineSpec
from panda.metrics_tokur import compute_detection_metrics
from panda.paths import DEFAULT_OUT


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}" if x == x else "—"


def eval_baseline(rows: list[dict], spec: BaselineSpec) -> dict[str, float]:
    if spec.tier == "diagnostic":
        acc = sum(1.0 - r["label_wrong_clean"] for r in rows) / len(rows) if rows else float("nan")
        return {"accuracy": acc, "auroc": 0.5, "auprc": float("nan"), "acc_star": acc, "n": len(rows)}

    labels = [r["label_wrong_clean"] == 0 for r in rows]
    scores = [float(r.get(spec.key, float("nan"))) for r in rows]
    if spec.invert:
        scores = [-s if math.isfinite(s) else float("nan") for s in scores]
    if all(math.isnan(s) for s in scores):
        return {"auroc": float("nan"), "auprc": float("nan"), "acc_star": float("nan"), "n": len(rows)}
    m = compute_detection_metrics(labels, scores)
    return {"auroc": m.auroc, "auprc": m.auprc, "acc_star": m.acc_star, "n": m.n}


def _status(spec: BaselineSpec, rows: list[dict]) -> str:
    vals = [r.get(spec.key) for r in rows]
    finite = sum(1 for v in vals if v is not None and math.isfinite(float(v)))
    if finite == 0:
        return "缺失"
    if finite < len(rows):
        return "部分"
    return "就绪"


def build_markdown(rows: list[dict], dataset: str) -> str:
    n = len(rows)
    wrong = sum(r["label_wrong_clean"] for r in rows)
    lines = [
        f"# {dataset} ISO Baseline 对比表",
        "",
        f"N={n}, wrong={wrong}, acc={1 - wrong / n:.1%}",
        "",
        "| # | 方法 | AUROC | AUPRC | ACC* | 状态 | 说明 |",
        "|---:|------|------:|------:|-----:|------|------|",
    ]
    for i, spec in enumerate(BASELINE_REGISTRY, 1):
        m = eval_baseline(rows, spec)
        status = _status(spec, rows)
        if spec.tier == "diagnostic":
            lines.append(
                f"| {i} | **{spec.name}** | — | — | {_fmt(m.get('acc_star', float('nan')))} | {status} | "
                f"Greedy ACC 下界（非不确定性分数） |"
            )
        else:
            lines.append(
                f"| {i} | **{spec.name}** | {_fmt(m['auroc'])} | {_fmt(m['auprc'])} | "
                f"{_fmt(m['acc_star'])} | {status} | {spec.description} |"
            )
    lines.extend(
        [
            "",
            "> PANDA 主方法请见 `analyze_main_tables` / `analyze_paper_tables`。",
            "> GPU baselines（P(True)、INSIDE）需先运行 `python -m panda.baselines.score_gpu_baselines`。",
        ]
    )
    return "\n".join(lines) + "\n"


def merge_gpu_baselines(out_dir: Path, dataset: str, rows: list[dict]) -> list[dict]:
    gpu_path = out_dir / dataset / "baselines_gpu.jsonl"
    if not gpu_path.exists():
        return rows
    by_id: dict[str, dict] = {}
    for line in gpu_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            by_id[rec["id"]] = rec
    for r in rows:
        if r["id"] in by_id:
            for k, v in by_id[r["id"]].items():
                if k != "id":
                    r[k] = v
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--features-only", action="store_true")
    args = ap.parse_args()

    if args.features_only:
        feat_path = args.out_dir / args.dataset / "features.jsonl"
        rows = [json.loads(ln) for ln in feat_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows = relabel(rows)
        rows = merge_tokur_scores(args.out_dir, args.dataset, rows)
    else:
        rows = load_enriched(args.out_dir, args.dataset)

    rows = merge_gpu_baselines(args.out_dir, args.dataset, rows)
    for i, r in enumerate(rows):
        rows[i] = enrich_row_with_baselines(r)

    report = build_markdown(rows, args.dataset)
    out = args.out_dir / f"BASELINES_{args.dataset}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
