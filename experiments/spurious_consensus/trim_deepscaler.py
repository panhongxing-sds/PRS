#!/usr/bin/env python3
"""将 deepscaler 题库裁至 2000 题，并过滤已有采样中超出范围的行。"""
from __future__ import annotations

import json
import shutil
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QDIR = ROOT / "data" / "questions"
TARGET = 2000


def main() -> None:
    src = QDIR / "deepscaler.jsonl"
    bak = QDIR / "deepscaler_3000.jsonl.bak"
    if not bak.exists():
        shutil.copy2(src, bak)
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = rows[:TARGET]
    valid = {r["id"] for r in keep}
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    print(f"题库 deepscaler: {len(rows)} → {len(keep)}")

    for path in glob(str(ROOT / "data/samples/samples_*_seed41_deepscaler.jsonl")):
        kept_lines = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in valid:
                kept_lines.append(line)
        Path(path).write_text("".join(kept_lines), encoding="utf-8")
        print(f"  过滤 {Path(path).name}: {len(kept_lines)} 行")

    summary = json.loads((ROOT / "data/questions/summary.json").read_text())
    summary["benchmarks"]["deepscaler"] = TARGET
    summary["total_questions"] = sum(summary["benchmarks"].values())
    (ROOT / "data/questions/summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = json.loads((ROOT / "data/questions/manifest.json").read_text())
    manifest["benchmarks"]["deepscaler"]["n"] = TARGET
    (ROOT / "data/questions/manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"总题量: {summary['total_questions']}")


if __name__ == "__main__":
    main()
