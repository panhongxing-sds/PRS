#!/usr/bin/env python3
"""Fast pooled LODO ablation macro from cached feature rows."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate_panda_v2 as agg  # noqa: E402

NEAR_CACHE = Path("/root/autodl-tmp/panda-outputs/.proc_near_answer_final_cache.pkl")
BD_CACHE = Path("/root/autodl-tmp/panda-outputs/.bd_variant_cache.pkl")
OUT = ROOT / "paper" / "maintable" / "ablation_macro.json"


def merge_rows() -> list[dict]:
    near = pickle.loads(NEAR_CACHE.read_bytes())
    bdrows = pickle.loads(BD_CACHE.read_bytes())
    bd_by_key = {(r["model"], r["seed"], r["ds"], r["y"], r["D_base"]): r for r in bdrows}
    # stable join via model/seed/ds/y/bd (bd is full 8-run dissent)
    out: list[dict] = []
    miss = 0
    for r in near:
        key = (r["model"], r["seed"], r["ds"], r["y"], r["bd"])
        b = bd_by_key.get(key)
        if b is None:
            # fallback: scan same model/seed/ds with closest bd
            cands = [x for x in bdrows if x["model"] == r["model"] and x["seed"] == r["seed"] and x["ds"] == r["ds"]]
            b = min(cands, key=lambda x: abs(x["D_base"] - r["bd"])) if cands else None
        if b is None:
            miss += 1
            continue
        out.append(
            {
                "model": r["model"],
                "seed": r["seed"],
                "ds": r["ds"],
                "y": r["y"],
                "bd": r["bd"],
                "bd_text": b["D_text"],
                "bd_weight": b["D_weight"],
                "T_ent_prox_lin": r["T_ent_prox_lin"],
                "T_ent_prox_text": r["T_ent_prox_text"],
                "T_ent_prox_weight": r["T_ent_prox_weight"],
                "T_ent_uniform": r["TW_ent_sum"] / 8.0 if np.isfinite(r.get("TW_ent_sum", float("nan"))) else float("nan"),
            }
        )
    if miss:
        print(f"warning: {miss} rows missing bd split join")
    return out


def main() -> None:
    rows = merge_rows()
    print(f"N={len(rows)}")
    ab = agg.compute_ablation_macro(rows, agg.MATH_DS)
    payload = {"N": len(rows), "ablation_macro": ab}
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}\n")
    for name, m in ab.items():
        print(f"  {name:22s}  AUROC={m['auroc']:.3f}  AUPRC={m['auprc']:.3f}")


if __name__ == "__main__":
    main()
