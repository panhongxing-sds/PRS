#!/usr/bin/env python3
"""Export TokUR vLLM pkl batches to jsonl (no fork vLLM import required)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from glob import glob
from pathlib import Path


class _StubUnpickler(pickle.Unpickler):
    """Load fork-vLLM pickles using placeholder classes."""

    def find_class(self, module: str, name: str):
        if module.startswith("vllm") or module.startswith("bayesian"):
            return type(name, (), {})
        return super().find_class(module, name)


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_record(sample: dict, dataset: str) -> dict | None:
    try:
        result = sample["result"][0] if isinstance(sample["result"], list) else sample["result"]
        outputs = _get(result, "outputs") or []
        if not outputs:
            return None
        out = outputs[0]
        text = _get(out, "text") or ""

        eu = tu = au = 0.0
        ll_sum = 0.0
        unc_list = _get(out, "uncertainties") or []
        for d in unc_list:
            if not d:
                continue
            u = list(d.values())[0]
            eu += float(_get(u, "epistemic_uncertainty", 0))
            tu += float(_get(u, "total_uncertainty", 0))
            au += float(_get(u, "aleatoric_uncertainty", 0))

        ll_list = _get(out, "logprobs") or []
        for d in ll_list:
            if not d:
                continue
            lp = list(d.values())[0]
            ll_sum += float(_get(lp, "logprob", 0))

        uid = str(sample.get("unique_id", "")).replace("/", "_").replace(".json", "")
        prompt = sample.get("formatted_prompt") or sample.get("problem", "")
        return {
            "id": uid,
            "dataset": dataset,
            "prompt": prompt,
            "response": text,
            "reference": sample.get("answer", ""),
            "tokur_eu": eu,
            "tokur_au": au,
            "tokur_tu": tu,
            "tokur_ll": ll_sum,
            "tokur_neg_eu": -eu,
            "tokur_neg_au": -au,
            "tokur_neg_tu": -tu,
        }
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="math500")
    args = ap.parse_args()

    records = []
    for fp in sorted(glob(args.pkl_glob)):
        with open(fp, "rb") as f:
            batch = _StubUnpickler(f).load()
        if not isinstance(batch, list):
            batch = [batch]
        for sample in batch:
            if not isinstance(sample, dict):
                continue
            rec = extract_record(sample, args.dataset)
            if rec:
                records.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Exported {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
