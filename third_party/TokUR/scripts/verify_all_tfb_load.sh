#!/usr/bin/env python3
"""Load-test every local TFB checkpoint (one model per process)."""
from __future__ import annotations

import gc
import subprocess
import sys
from pathlib import Path

MODELS = {
    "tfb-qwen25-3b": ("/home/phx/TokUR/models/TFB-Qwen2.5-3B-Instruct", "bfloat16"),
    "qwen3-8b": ("/HDDDATA/phx/model/TFB-Qwen3-8B", "bfloat16"),
    "mistral-7b": ("/HDDDATA/phx/model/TFB-Mistral-7B-Instruct", "bfloat16"),
    "llama-3.2-1b": ("/HDDDATA/phx/model/TFB-Llama-3.2-1B-Instruct", "bfloat16"),
    "llama-3.1-8b": ("/HDDDATA/phx/model/TFB-Llama-3.1-8B-Instruct", "bfloat16"),
    "phi35-mini": ("/HDDDATA/phx/model/TFB-Phi-3.5-mini-instruct", "bfloat16"),
    "opt-6.7b": ("/HDDDATA/phx/model/TFB-OPT-6.7B", "float16"),
}

SNIPPET = """
import gc, torch
from atokur.tfb_load import load_tfb_for_teacher_force
path, dtype = {path!r}, {dtype!r}
model, tok = load_tfb_for_teacher_force(path, device='cuda:0', dtype=dtype)
ids = tok('Test prompt', return_tensors='pt').input_ids.cuda()
with torch.no_grad():
    out = model(input_ids=ids)
print('OK', tuple(out.logits.shape))
del model, tok
gc.collect(); torch.cuda.empty_cache()
"""


def main() -> int:
    root = Path(__file__).resolve().parents[2] / "tprd" / "src"
    failed = []
    for name, (path, dtype) in MODELS.items():
        if not Path(path, "config.json").exists():
            failed.append((name, "missing checkpoint"))
            print(f"[FAIL] {name}: missing {path}")
            continue
        code = SNIPPET.format(path=path, dtype=dtype)
        print(f"[run] {name} ...", flush=True)
        r = subprocess.run(
            [sys.executable, "-c", code],
            env={**dict(__import__("os").environ), "PYTHONPATH": str(root)},
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and "OK" in r.stdout:
            print(f"[pass] {name} {r.stdout.strip()}")
        else:
            err = (r.stderr or r.stdout).strip().splitlines()[-3:]
            failed.append((name, "\n".join(err)))
            print(f"[FAIL] {name}:\n" + "\n".join(err))
    print(f"\nSummary: {len(MODELS)-len(failed)}/{len(MODELS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
