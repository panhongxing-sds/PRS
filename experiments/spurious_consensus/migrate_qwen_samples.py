#!/usr/bin/env python3
"""从 aul-study 合并 shard 样本 → spurious-consensus/data/samples/。"""
from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUL_DATA = Path("/root/aul-study/data")
OUT_DIR = ROOT / "data" / "samples"
TAG = "qwen25_3b"
SEED = 41

# aul-study 旧 benchmark 名 → sc 核心集名
BENCH_MAP = {
    "competition_math_l5_500": "competition_math_l5_500",
    "math_level4plus_300": "math_level4plus_300",
    "minerva": "minerva",
    "gpqa_diamond": "gpqa_diamond",
    "aime_2024": "aime_2024",
}

DEEPSCALER_PARTS = ["deepscaler_1500", "deepscaler_500", "deepscaler_200"]


def load_shards(pattern: str) -> list[dict]:
    rows: dict[str, dict] = {}
    for path in sorted(glob(pattern)):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["id"]] = r
    return list(rows.values())


def normalize_row(r: dict, benchmark: str) -> dict:
    return {
        "id": r["id"],
        "dataset": benchmark,
        "benchmark": benchmark,
        "seed": r.get("seed", SEED),
        "gold": r["gold"],
        "grading": r.get("grading", "math"),
        "label_drop": int(r.get("label_drop", 0)),
        "answers": r["answers"],
        "correct": r["correct"],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def migrate_benchmark(aul_name: str, sc_name: str, aul_dir: Path) -> int:
    single = aul_dir / f"samples_{TAG}_seed{SEED}_{aul_name}.jsonl"
    if single.exists():
        rows = [normalize_row(json.loads(l), sc_name) for l in single.read_text().splitlines() if l.strip()]
    else:
        pattern = str(aul_dir / f"samples_{TAG}_seed{SEED}_{aul_name}.shard*.jsonl")
        raw = load_shards(pattern)
        if not raw:
            print(f"[skip] 无样本: {aul_name}")
            return 0
        rows = [normalize_row(r, sc_name) for r in raw]
    out = OUT_DIR / f"samples_{TAG}_seed{SEED}_{sc_name}.jsonl"
    write_jsonl(out, rows)
    print(f"[ok] {sc_name}: {len(rows)} → {out}")
    return len(rows)


def migrate_deepscaler(aul_dir: Path) -> int:
    rows: dict[str, dict] = {}
    for part in DEEPSCALER_PARTS:
        pattern = str(aul_dir / f"samples_{TAG}_seed{SEED}_{part}.shard*.jsonl")
        for r in load_shards(pattern):
            rows[r["id"]] = normalize_row(r, "deepscaler")
    if not rows:
        print("[skip] deepscaler: 无 shard 样本")
        return 0
    out = OUT_DIR / f"samples_{TAG}_seed{SEED}_deepscaler.jsonl"
    write_jsonl(out, list(rows.values()))
    print(f"[ok] deepscaler: {len(rows)} (缺 3000-{len(rows)} 题为 supp_800，需 --resume 补采)")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aul-dir", default=str(AUL_DATA))
    args = ap.parse_args()
    aul_dir = Path(args.aul_dir)
    if not aul_dir.exists():
        raise SystemExit(f"找不到 {aul_dir}")

    total = 0
    for aul_name, sc_name in BENCH_MAP.items():
        total += migrate_benchmark(aul_name, sc_name, aul_dir)
    total += migrate_deepscaler(aul_dir)
    print(f"\n合计迁入 {total} 条；运行 python check_samples.py --tag {TAG} 查看缺口")


if __name__ == "__main__":
    main()
