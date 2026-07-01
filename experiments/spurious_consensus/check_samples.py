#!/usr/bin/env python3
"""对账采样完整性：题数、K、空答案率。"""
from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "data" / "questions"
SAMPLES = ROOT / "data" / "samples"


def load_questions(benchmark: str) -> set[str]:
    path = QUESTIONS / f"{benchmark}.jsonl"
    return {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}


def load_sample_file(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def audit_file(path: str, expected_ids: set[str], k_expected: int) -> dict:
    rows = load_sample_file(path)
    have = set(rows)
    missing = expected_ids - have
    extra = have - expected_ids
    bad_k, empty_ans = 0, 0
    for r in rows.values():
        ans = r.get("answers", [])
        if len(ans) != k_expected:
            bad_k += 1
        if sum(1 for a in ans if a == "") > len(ans) * 0.1:
            empty_ans += 1
    return {
        "path": path,
        "n": len(rows),
        "expected": len(expected_ids),
        "missing": len(missing),
        "extra": len(extra),
        "bad_k": bad_k,
        "high_empty": empty_ans,
        "missing_ids_sample": sorted(missing)[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="模型 tag，如 qwen25_3b")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    args = ap.parse_args()

    benches = args.benchmarks or [
        "deepscaler", "gpqa_diamond", "aime_2024",
    ]
    pattern = str(SAMPLES / f"samples_{args.tag}_seed{args.seed}_*.jsonl")
    files = {Path(p).name.split(f"_seed{args.seed}_", 1)[-1].replace(".jsonl", ""): p
             for p in glob(pattern)}

    print(f"{'benchmark':<28} {'have':>6} {'need':>6} {'miss':>6} {'bad_k':>6} {'empty%':>7}")
    print("-" * 70)
    total_miss = 0
    for b in benches:
        exp = load_questions(b)
        path = files.get(b)
        if not path:
            print(f"{b:<28} {0:>6} {len(exp):>6} {len(exp):>6} {'-':>6} {'-':>7}")
            total_miss += len(exp)
            continue
        rep = audit_file(path, exp, args.k)
        total_miss += rep["missing"]
        print(f"{b:<28} {rep['n']:>6} {rep['expected']:>6} {rep['missing']:>6} "
              f"{rep['bad_k']:>6} {rep['high_empty']:>7}")
        if rep["missing"] and rep["missing_ids_sample"]:
            print(f"  缺例: {rep['missing_ids_sample']} ...")

    print("-" * 70)
    if total_miss == 0:
        print("✓ 全部 benchmark 题数对齐")
    else:
        print(f"✗ 共缺 {total_miss} 题 → 对缺 benchmark 跑 run_sampling.sh --resume")


if __name__ == "__main__":
    main()
