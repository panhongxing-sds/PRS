import os

# Default HF Hub endpoint (e.g. mirror); override with HF_ENDPOINT.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from pathlib import Path

REPOS = [
    "Tyrion279/math500",
    "Tyrion279/gsm8k", 
    "Tyrion279/deepscaler",
    "Tyrion279/leg-counting",
]

output_dir = Path("./")
output_dir.mkdir(exist_ok=True)

for repo in REPOS:
    name = repo.split("/")[-1]
    ds = load_dataset(repo, split="train")
    out_path = output_dir / f"{name}.jsonl"
    ds.to_json(out_path, orient="records", lines=True)
    print(f"Saved {name}: {len(ds)} examples")
    # Scripts reference gsm8k_test.jsonl
    if name == "gsm8k":
        link = output_dir / "gsm8k_test.jsonl"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to("gsm8k.jsonl")
        print("Linked gsm8k_test.jsonl -> gsm8k.jsonl")