"""Build substitute public datasets for minerva / math500 / gsm8k (TF-TTCL JSON)
and zebra_puzzles (TokUR jsonl). color_cube handled separately via reasoning-gym.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset

TFTTCL = Path("/home/phx/TF-TTCL/data/MATH")
TOKUR_DS = Path("/root/autodl-tmp/PANDA/third_party/TokUR/datasets")
TFTTCL.mkdir(parents=True, exist_ok=True)
TOKUR_DS.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, rows: list):
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"  wrote {path} ({len(rows)} rows)")


def write_jsonl(path: Path, rows: list):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def load_local_jsonl(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# 1) minerva  (math-ai/minervamath: question/answer) -> instruction/output
print(">>> minerva")
mv = load_dataset("math-ai/minervamath", split="test")
write_json(TFTTCL / "minerva.json",
           [{"instruction": r["question"], "output": str(r["answer"])} for r in mv])

# 2) gsm8k  (downloaded jsonl: problem/answer) -> instruction/input/output
print(">>> gsm8k")
g = load_local_jsonl(TOKUR_DS / "gsm8k.jsonl")
write_json(TFTTCL / "gsm8k.json",
           [{"instruction": r["problem"], "input": "", "output": str(r["answer"])} for r in g])

# 3) math500 (downloaded jsonl: problem/answer/unique_id) -> keep keys
print(">>> math500")
m5 = load_local_jsonl(TOKUR_DS / "math500.jsonl")
write_json(TFTTCL / "math500.json",
           [{"problem": r["problem"], "answer": r["answer"],
             "unique_id": r.get("unique_id", f"math500_{i}")} for i, r in enumerate(m5)])

# 4) zebra_puzzles (WildEval/ZebraLogic grid_mode: puzzle/solution) -> question/solution
print(">>> zebra_puzzles")
z = load_dataset("WildEval/ZebraLogic", "grid_mode", split="test")
zrows = [{"id": r["id"], "question": r["puzzle"], "solution": r["solution"]} for r in z]
write_jsonl(TOKUR_DS / "zebra_puzzles.jsonl", zrows)

print("DONE")
