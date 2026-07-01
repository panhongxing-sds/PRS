"""构建 Spurious Consensus 实验题库 → data/questions/"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH_DIR = ROOT / "data" / "benchmarks"
QUESTIONS_DIR = ROOT / "data" / "questions"

DEEPSCALER_TARGET = 3000
DEEPSCALER_BASE_PARTS = ["deepscaler_1500", "deepscaler_500", "deepscaler_200"]
DEEPSCALER_POOL = BENCH_DIR / "deepscaler_5000.jsonl"
DEEPSCALER_SUPP_SEED = 43

BENCHMARKS: list[dict] = [
    {"name": "deepscaler", "source": "deepscaler", "grading": "math", "max_new_tokens": 2048},
    {"name": "competition_math_l5_500", "source": "benchmarks", "grading": "math", "max_new_tokens": 2048},
    {"name": "math_level4plus_300", "source": "benchmarks", "grading": "math", "max_new_tokens": 2048},
    {"name": "minerva", "source": "existing", "grading": "math", "max_new_tokens": 2048},
    {"name": "gpqa_diamond", "source": "benchmarks", "grading": "mcq", "max_new_tokens": 2048},
    {"name": "aime_2024", "source": "benchmarks", "grading": "math", "max_new_tokens": 4096},
]


def _norm_row(row: dict, benchmark: str, grading: str, source_file: str) -> dict:
    qid = row.get("id") or row.get("unique_id")
    if not qid:
        raise ValueError(f"missing id in {source_file}")
    return {
        "id": str(qid),
        "benchmark": benchmark,
        "dataset": benchmark,
        "problem": row["problem"],
        "answer": row["answer"],
        "grading": grading,
        "source_file": source_file,
    }


def load_benchmark_file(name: str) -> list[dict]:
    path = BENCH_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    grading = rows[0].get("grading", "math") if rows else "math"
    return [_norm_row(r, name, grading, str(path.relative_to(ROOT))) for r in rows]


def load_existing(name: str) -> list[dict]:
    path = QUESTIONS_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_merged_benchmark(name: str, parts: list[str], grading: str) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for part in parts:
        for r in load_benchmark_file(part):
            r = {**r, "benchmark": name, "dataset": name, "source_subset": part}
            if r["id"] in seen:
                raise ValueError(f"重复 id {r['id']} from {part}")
            seen.add(r["id"])
            rows.append(r)
    return rows


def load_deepscaler(name: str, grading: str, target: int) -> list[dict]:
    rows = load_merged_benchmark(name, DEEPSCALER_BASE_PARTS, grading)
    if len(rows) >= target:
        return rows[:target]
    if not DEEPSCALER_POOL.exists():
        raise FileNotFoundError(f"需要 {DEEPSCALER_POOL}")
    need = target - len(rows)
    seen = {r["id"] for r in rows}
    pool = [json.loads(l) for l in DEEPSCALER_POOL.read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l)["id"] not in seen]
    rng = random.Random(DEEPSCALER_SUPP_SEED)
    rng.shuffle(pool)
    supp_path = BENCH_DIR / "deepscaler_supp_800.jsonl"
    with supp_path.open("w", encoding="utf-8") as f:
        for r in pool[:need]:
            f.write(json.dumps({**r, "dataset": "deepscaler_supp_800"}, ensure_ascii=False) + "\n")
    for r in pool[:need]:
        row = _norm_row(r, name, grading, str(supp_path.relative_to(ROOT)))
        rows.append({**row, "source_subset": "deepscaler_supp_800"})
    return rows


def main() -> None:
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    global_idx = 0
    manifest: dict = {}

    for spec in BENCHMARKS:
        name = spec["name"]
        src = spec["source"]
        if src == "existing":
            rows = load_existing(name)
        elif src == "deepscaler":
            rows = load_deepscaler(name, spec["grading"], DEEPSCALER_TARGET)
        else:
            rows = load_benchmark_file(name)

        out = QUESTIONS_DIR / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                r_out = {**r, "global_idx": global_idx}
                f.write(json.dumps(r_out, ensure_ascii=False) + "\n")
                all_rows.append(r_out)
                global_idx += 1
        manifest[name] = {"n": len(rows), "grading": spec["grading"],
                          "max_new_tokens": spec["max_new_tokens"]}
        print(f"  {name:<28} {len(rows):5d}")

    with (QUESTIONS_DIR / "all_questions.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"benchmarks": manifest, "total": len(all_rows)},
              open(QUESTIONS_DIR / "manifest.json", "w"), indent=2)
    print(f"\n合计 {len(all_rows)} 题 → data/questions/")


if __name__ == "__main__":
    print("构建题库...")
    main()
