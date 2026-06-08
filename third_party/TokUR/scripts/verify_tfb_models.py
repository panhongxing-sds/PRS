#!/usr/bin/env python3
"""Quick smoke test: load TFB checkpoints for teacher-forcing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tprd" / "src"))

from atokur.tfb_load import load_tfb_for_teacher_force


MODELS = {
    "llama-3.2-1b": "/HDDDATA/phx/model/TFB-Llama-3.2-1B-Instruct",
    "llama-3.1-8b": "/HDDDATA/phx/model/TFB-Llama-3.1-8B-Instruct",
    "mistral-7b": "/HDDDATA/phx/model/TFB-Mistral-7B-Instruct",
    "qwen3-8b": "/HDDDATA/phx/model/TFB-Qwen3-8B",
    "phi35-mini": "/HDDDATA/phx/model/TFB-Phi-3.5-mini-instruct",
    "opt-6.7b": "/HDDDATA/phx/model/TFB-OPT-6.7B",
    "tfb-qwen25-3b": "/home/phx/TokUR/models/TFB-Qwen2.5-3B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--models", nargs="*", default=list(MODELS.keys()))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping load test")
        return

    for name in args.models:
        path = MODELS[name]
        if not Path(path, "config.json").exists():
            print(f"[skip] {name}: missing {path}")
            continue
        print(f"[load] {name} ...")
        model, tok = load_tfb_for_teacher_force(path, device=args.device, dtype=args.dtype)
        ids = tok("Hello", return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            out = model(input_ids=ids)
        print(f"[ok] {name} logits={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
