#!/usr/bin/env python3
"""Report 4-model × 4-dataset readiness for PANDA / TW-ASE / F_resp / D_ans."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "Qwen2.5-3B": "panda_full",
    "Llama-3.2-1B": "ase_llama32_1b",
    "Llama-3.1-8B": "ase_llama31_8b",
    "Qwen3-8B": "ase_qwen3_8b",
}
DATASETS = ("minerva", "math500", "gsm8k", "deepscaler")
METRIC_KEYS = {
    "PANDA": "PANDA",
    "TW-ASE/F_resp": ("TW_ASE", "F_resp"),
    "D_ans": "D_ans",
}


def _finite_frac(rows: list[dict], key: str) -> tuple[int, int]:
    ok = 0
    for r in rows:
        v = r.get(key)
        if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
            ok += 1
    return ok, len(rows)


def _metric_status(rows: list[dict], keys: str | tuple[str, ...]) -> str:
    if not rows:
        return "—"
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        ok, n = _finite_frac(rows, key)
        if ok == n and n > 0:
            return f"{ok}/{n}"
    # prefer first key for partial reporting
    ok, n = _finite_frac(rows, keys[0])
    return f"{ok}/{n}" if n else "—"


def _status(n_raw: int, n_feat: int, panda_ok: int, n: int) -> str:
    if n_raw == 0 and n_feat == 0:
        return "missing"
    if n == 0:
        return "no_features"
    if panda_ok < n:
        return "partial_metrics"
    if n_raw < 50:
        return "low_n"
    return "ready"


def load_rows(out_dir: Path, dataset: str) -> tuple[int, list[dict]]:
    raw_dir = out_dir / dataset / "raw_runs"
    n_raw = 0
    if raw_dir.is_dir():
        n_raw = sum(
            1
            for p in raw_dir.glob("*.json")
            if not p.name.endswith((".error.json", ".partial.json"))
        )
    feat = out_dir / dataset / "features.jsonl"
    rows: list[dict] = []
    if feat.is_file():
        for line in feat.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return n_raw, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(os.environ.get("PANDA_OUTPUTS", ROOT / "outputs")),
    )
    args = ap.parse_args()
    root = args.outputs_root

    print("| Model | Dataset | N rows | PANDA | TW-ASE/F_resp | D_ans | Status |")
    print("| --- | --- | ---: | --- | --- | --- | --- |")

    for model, subdir in MODELS.items():
        out_dir = root / subdir
        for ds in DATASETS:
            n_raw, rows = load_rows(out_dir, ds)
            n = len(rows)
            panda_ok, _ = _finite_frac(rows, "PANDA")
            panda_s = _metric_status(rows, "PANDA")
            tw_s = _metric_status(rows, METRIC_KEYS["TW-ASE/F_resp"])
            d_s = _metric_status(rows, "D_ans")
            st = _status(n_raw, n, panda_ok, n)
            n_disp = n if n else n_raw
            print(f"| {model} | {ds} | {n_disp} | {panda_s} | {tw_s} | {d_s} | {st} |")


if __name__ == "__main__":
    main()
