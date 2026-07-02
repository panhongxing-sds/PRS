"""Download TokUR / PANDA benchmark JSONL files from HuggingFace."""

from __future__ import annotations

import sys
from pathlib import Path

REPOS = [
    ("Tyrion279/math500", "math500.jsonl"),
    ("Tyrion279/gsm8k", "gsm8k.jsonl"),
    ("Tyrion279/deepscaler", "deepscaler.jsonl"),
    ("Tyrion279/leg-counting", "leg-counting.jsonl"),
    ("Tyrion279/zebra-puzzles", "zebra_puzzles.jsonl"),
    ("Tyrion279/color-cube", "color_cube.jsonl"),
    ("openai/openai_humaneval", "humaneval.jsonl"),
]

# Alternate repo names if primary fails
FALLBACK_REPOS = {
    "zebra_puzzles.jsonl": ["Tyrion279/zebra_puzzles"],
    "color_cube.jsonl": ["Tyrion279/color_cube", "Tyrion279/acre"],
    "humaneval.jsonl": ["Tyrion279/humaneval"],
}


def _save_dataset(repo: str, out_path: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    ds.to_json(out_path, orient="records", lines=True)
    return len(ds)


def download_all(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for repo, fname in REPOS:
        out_path = output_dir / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            status[fname] = "exists"
            print(f"Skip {fname} (already present)")
            continue
        repos = [repo, *FALLBACK_REPOS.get(fname, [])]
        ok = False
        for r in repos:
            try:
                n = _save_dataset(r, out_path)
                status[fname] = f"ok:{n}"
                print(f"Saved {fname} from {r}: {n} examples")
                ok = True
                break
            except Exception as exc:
                print(f"WARN {r} -> {fname}: {exc}", file=sys.stderr)
        if not ok:
            status[fname] = "missing"
            print(f"FAILED {fname} — download manually or check HF access", file=sys.stderr)
    return status


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    download_all(out)
