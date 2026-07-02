#!/usr/bin/env python3
"""Fast CPU paper analyses — pickle cache only, ~10s, no network, no math_equal."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate_panda_v2 as agg  # noqa: E402

NEAR_CACHE = Path("/root/autodl-tmp/panda-outputs/.proc_near_answer_final_cache.pkl")
BD_CACHE = Path("/root/autodl-tmp/panda-outputs/.bd_variant_cache.pkl")
OUT_JSON = ROOT / "paper" / "analysis" / "cpu_results.json"
OUT_MD = ROOT / "paper" / "analysis" / "cpu_results.md"


def _auroc(y, s):
    m = np.isfinite(s)
    y, s = y[m], s[m]
    return float(roc_auc_score(y, s)) if len(y) >= 10 and len(set(y)) > 1 else float("nan")


def _fast_bd(a0: str, answers: list[str]) -> float:
    a0 = str(a0).strip()
    if not answers:
        return 0.0
    return sum(1 for a in answers if str(a).strip() != a0) / len(answers)


def load_rows_from_cache() -> list[dict]:
    near = pickle.loads(NEAR_CACHE.read_bytes())
    bdrows = pickle.loads(BD_CACHE.read_bytes())
    bd_by = {(r["model"], r["seed"], r["ds"], r["y"], r["D_base"]): r for r in bdrows}
    rows = []
    for r in near:
        b = bd_by.get((r["model"], r["seed"], r["ds"], r["y"], r["bd"]))
        if not b:
            continue
        rows.append({
            "model": r["model"], "seed": r["seed"], "ds": r["ds"], "y": int(r["y"]),
            "bd": float(r["bd"]), "bd_text": float(b["D_text"]), "bd_weight": float(b["D_weight"]),
            "T_ent_prox_lin": float(r["T_ent_prox_lin"]),
            "T_ent_prox_text": float(r.get("T_ent_prox_text", float("nan"))),
            "T_ent_prox_weight": float(r.get("T_ent_prox_weight", float("nan"))),
            "T_ent_uniform": float(r.get("TW_ent_sum", float("nan"))) / 8.0
            if np.isfinite(float(r.get("TW_ent_sum", float("nan")))) else float("nan"),
        })
    return rows


def attach_summary_fast(rows: list[dict], out_root: Path) -> int:
    """One-pass summary scan; join on (model,seed,ds,y,bd_round) using fast_bd."""
    pool: dict[tuple, dict[float, dict]] = defaultdict(dict)
    for mk, folder in agg.MODELS.items():
        for seed in agg.SEEDS:
            for ds in agg.MATH_DS:
                p = out_root / folder / f"seed{seed}" / ds / "summary.jsonl"
                if not p.exists():
                    continue
                for ln in p.read_text().splitlines():
                    if not ln.strip():
                        continue
                    s = json.loads(ln)
                    if s.get("label_drop") or s.get("label_wrong_clean") is None:
                        continue
                    a0 = str(s.get("a0", ""))
                    ans = [str(x) for x in (s.get("text_answers") or []) + (s.get("weight_answers") or [])]
                    bd = round(_fast_bd(a0, ans), 4)
                    ref = str(s.get("reference", "")).strip()
                    wt = [str(x).strip() for x in (s.get("weight_answers") or [])]
                    pool[(mk, seed, ds, int(s["label_wrong_clean"]))][bd] = {
                        "TW_ASE_H_norm": s.get("TW_ASE_H_norm"),
                        "baseline_SE_H": s.get("baseline_SE_H"),
                        "base_ok": int(s.get("is_correct_clean", s.get("is_correct", 0))),
                        "wt_acc": sum(1 for w in wt if w == ref or w == a0.strip()) / max(len(wt), 1),
                    }
    hit = 0
    for r in rows:
        g = pool.get((r["model"], r["seed"], r["ds"], r["y"]), {})
        bd = round(r["bd"], 4)
        extra = g.get(bd)
        if extra is None and g:
            extra = g[min(g, key=lambda b: abs(b - bd))]
        if extra:
            r.update(extra)
            hit += 1
    return hit


def weight_stats(rows: list[dict]) -> dict:
    by = defaultdict(lambda: {"b": 0, "bo": 0, "ws": 0.0, "wn": 0})
    for r in rows:
        if "wt_acc" not in r:
            continue
        m = r["model"]
        by[m]["b"] += 1
        by[m]["bo"] += r["base_ok"]
        by[m]["ws"] += r["wt_acc"]
        by[m]["wn"] += 1
    tot_b = sum(v["b"] for v in by.values())
    tot_bo = sum(v["bo"] for v in by.values())
    tot_ws = sum(v["ws"] for v in by.values())
    tot_wn = sum(v["wn"] for v in by.values())
    return {
        "base_acc": tot_bo / tot_b if tot_b else float("nan"),
        "weight_acc_proxy": tot_ws / tot_wn if tot_wn else float("nan"),
        "per_model": {
            m: {"base": v["bo"] / v["b"], "weight": v["ws"] / v["wn"]}
            for m, v in by.items() if v["b"]
        },
    }


def subset(rows, name, fn):
    sub = [r for r in rows if fn(r)]
    if len(sub) < 10:
        return {"name": name, "n": len(sub)}
    y = np.array([r["y"] for r in sub])
    out = {"name": name, "n": len(sub), "wrong_rate": float(y.mean())}
    for lab, k in [("bd", "bd"), ("hes", "T_ent_prox_lin"), ("se", "baseline_SE_H")]:
        out[f"auroc_{lab}"] = _auroc(y, np.array([float(r.get(k, float("nan"))) for r in sub]))
    X = np.array([[r["bd"], r["T_ent_prox_lin"]] for r in sub], float)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
    out["auroc_panda_uw"] = _auroc(y, ((X - mu) / sd).sum(1))
    return out


def scan_summary_collapse(out_root: Path) -> dict:
    """Summary-only: wrong & TW_ASE_H_norm==0, string-fast bd (no cache join)."""
    n = 0
    bd_vals = []
    for mk, folder in agg.MODELS.items():
        for seed in agg.SEEDS:
            for ds in agg.MATH_DS:
                p = out_root / folder / f"seed{seed}" / ds / "summary.jsonl"
                if not p.exists():
                    continue
                for ln in p.read_text().splitlines():
                    if not ln.strip():
                        continue
                    s = json.loads(ln)
                    if s.get("label_drop") or int(s.get("label_wrong_clean", 0)) != 1:
                        continue
                    if float(s.get("TW_ASE_H_norm") or 1) != 0.0:
                        continue
                    a0 = str(s.get("a0", ""))
                    ans = [str(x) for x in (s.get("text_answers") or []) + (s.get("weight_answers") or [])]
                    n += 1
                    bd_vals.append(_fast_bd(a0, ans))
    arr = np.array(bd_vals) if bd_vals else np.array([])
    return {
        "n": n,
        "mean_bd_fast": float(arr.mean()) if len(arr) else float("nan"),
        "p_bd_gt0": float((arr > 0).mean()) if len(arr) else float("nan"),
        "note": "string-fast bd; cache-exact bd0_wrong n≈96 in collapse_lodo",
    }


def collapse_lodo(rows):
    out = {}
    for name, fn in {
        "all": lambda r: True,
        "bd0": lambda r: r["bd"] == 0,
        "bd0_wrong": lambda r: r["bd"] == 0 and r["y"] == 1,
        "se_cons_wrong": lambda r: r["y"] == 1 and float(r.get("TW_ASE_H_norm") or 1) == 0,
    }.items():
        sub = [r for r in rows if fn(r)]
        if len(sub) < 20:
            out[name] = {"n": len(sub)}
            continue
        se = [{**r, "se": float(r["baseline_SE_H"])} for r in sub
              if math.isfinite(float(r.get("baseline_SE_H", float("nan"))))]
        out[name] = {
            "n": len(sub),
            "panda": agg.lodo_macro(sub, ["bd", "T_ent_prox_lin"], agg.MATH_DS),
            "bd": agg.lodo_macro(sub, ["bd"], agg.MATH_DS),
            "hes": agg.lodo_macro(sub, ["T_ent_prox_lin"], agg.MATH_DS),
            "se": agg.lodo_macro(se, ["se"], agg.MATH_DS) if len(se) > 50 else float("nan"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, default=Path("/root/autodl-tmp/panda-outputs"))
    args = ap.parse_args()

    rows = load_rows_from_cache()
    hit = attach_summary_fast(rows, args.outputs_root)
    ab = agg.compute_ablation_macro(rows, agg.MATH_DS)

    bd = np.array([r["bd"] for r in rows])
    hes = np.array([r["T_ent_prox_lin"] for r in rows])
    rho, _ = spearmanr(bd, hes, nan_policy="omit")

    sc_scan = scan_summary_collapse(args.outputs_root)
    sub_sc = [r for r in rows if r["y"] == 1 and float(r.get("TW_ASE_H_norm") or 1) == 0]
    bd_sc = np.array([r["bd"] for r in sub_sc]) if sub_sc else np.array([])

    payload = {
        "N": len(rows), "summary_join": hit,
        "weight": weight_stats(rows),
        "ablation_macro": ab,
        "spearman_bd_hes": float(rho),
        "se_consistent_wrong_summary_scan": sc_scan,
        "se_consistent_wrong": sc_scan,
        "subsets": [
            subset(rows, "all", lambda r: True),
            subset(rows, "bd0", lambda r: r["bd"] == 0),
            subset(rows, "bd0_wrong", lambda r: r["bd"] == 0 and r["y"] == 1),
            subset(rows, "se_cons_wrong", lambda r: r["y"] == 1 and float(r.get("TW_ASE_H_norm") or 1) == 0),
        ],
        "collapse_lodo": collapse_lodo(rows),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    (ROOT / "paper/maintable/ablation_macro.json").write_text(
        json.dumps({"N": len(rows), "ablation_macro": ab}, indent=2)
    )

    md = [f"# CPU results (N={len(rows)}, join={hit})", "",
          f"Weight: base={payload['weight']['base_acc']:.3f} wt={payload['weight']['weight_acc_proxy']:.3f}",
          f"Spearman bd~hes: {rho:.3f}",
          f"SE-consistent wrong n={sc_scan['n']} P(bd>0)={sc_scan.get('p_bd_gt0', float('nan')):.1%} (summary scan)",
          "## Ablation"]
    for k, v in ab.items():
        md.append(f"- {k}: AUROC={v['auroc']:.3f} AUPRC={v['auprc']:.3f}")
    md += ["", "## Collapse LODO"]
    for k, v in payload["collapse_lodo"].items():
        if "panda" in v:
            md.append(f"- {k} n={v['n']}: PANDA={v['panda']:.3f} bd={v['bd']:.3f} hes={v['hes']:.3f} SE={v['se']:.3f}")
        else:
            md.append(f"- {k} n={v.get('n',0)}")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"N={len(rows)} join={hit} → {OUT_JSON}")


if __name__ == "__main__":
    main()
