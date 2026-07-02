#!/usr/bin/env python3
"""逐行清洗单个 jsonl，避免大文件 OOM。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__import__("os").environ.get("PANDA_ROOT", "/root/PANDA")) / "src"))

from clean_samples import clean_row


def main() -> None:
    path = Path(sys.argv[1])
    tag = sys.argv[2]
    tmp = path.with_suffix(".jsonl.tmp")
    n = 0
    with open(path, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = clean_row(json.loads(line), tag)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    print(f"{path.name}: {n} rows")


if __name__ == "__main__":
    main()
