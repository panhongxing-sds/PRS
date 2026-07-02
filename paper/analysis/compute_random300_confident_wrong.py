#!/usr/bin/env python3
"""Fair SC@9 vs PANDA@9 confident-wrong rates on DeepScaler random300 (Qwen2.5-3B)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_panda_v2 import eq  # noqa: E402
from panda.grading.math_grader import math_equal  # noqa: E402

DEFAULT_K9 = (
    ROOT
    / "experiments/spurious_consensus/data/samples"
    / "samples_qwen25_3b_seed41_deepscaler_random300_k9.jsonl"
)
DEFAULT_SUMM = Path(
    "/root/autodl-tmp/panda-outputs/maintable_qwen25_3b_deepscaler_random300"
    "/seed41/deepscaler/summary.jsonl"
)
DEFAULT_META = ROOT / "paper/analysis/deepscaler_random300_meta.json"
DEFAULT_OUT = ROOT / "paper/analysis/random300_confident_wrong.json"
DEFAULT_JOIN = ROOT / "paper/analysis/random300_sc9_panda_join.jsonl"

TAU_SPECS = [
    {"name": "tau_0.9", "tau": 0.9},
    {"name": "tau_8_over_9", "tau": 8 / 9},
    {"name": "tau_7_over_9", "tau": 7 / 9},
]


def _grade(a: str, gold: str) -> bool:
    a, gold = str(a).strip(), str(gold).strip()
    if a == gold:
        return True
    try:
        return bool(math_equal(a, gold))
    except Exception:
        return False


def sc_vote_stats(rec: dict) -> dict | None:
    pairs = [(a, c) for a, c in zip(rec.get("answers") or [], rec.get("correct") or []) if a]
    if len(pairs) < 2:
        return None
    answers, correct = zip(*pairs)
    cnt = Counter(answers)
    maj, top = cnt.most_common(1)[0]
    p_top = top / len(answers)
    cmap: dict[str, int] = {}
    for a, c in zip(answers, correct):
        cmap.setdefault(a, int(c))
    maj_correct = cmap.get(maj, 0) == 1
    return {
        "maj": maj,
        "p_top": p_top,
        "maj_wrong": not maj_correct,
        "maj_correct": maj_correct,
        "n_votes": len(answers),
    }


def panda_nine_answers(rec: dict) -> list[str]:
    a0 = str(rec.get("a0") or "").strip()
    text = [str(x).strip() for x in (rec.get("text_answers") or [])]
    weight = [str(x).strip() for x in (rec.get("weight_answers") or [])]
    return [a for a in [a0, *text, *weight] if a]


def panda_vote_stats(rec: dict) -> dict | None:
    gold = str(rec.get("reference") or rec.get("reference_normalized") or "").strip()
    answers = panda_nine_answers(rec)
    if len(answers) < 2:
        return None
    cnt = Counter(answers)
    maj, top = cnt.most_common(1)[0]
    p_top = top / len(answers)
    maj_correct = _grade(maj, gold)
    a0 = str(rec.get("a0") or "").strip()
    a0_correct = bool(rec.get("is_correct_clean"))
    if gold and a0:
        a0_correct = _grade(a0, gold)
    pert = [a for a in answers if a != a0][:8]
    if len(pert) < 8:
        pert = answers[1:9]
    bd = float(rec.get("answer_drift") if rec.get("answer_drift") is not None else 0.0)
    if rec.get("answer_drift") is None and pert:
        bd = sum(0.0 if eq(a0, p) else 1.0 for p in pert) / len(pert)
    return {
        "maj": maj,
        "p_top": p_top,
        "maj_wrong": not maj_correct,
        "maj_correct": maj_correct,
        "a0_correct": a0_correct,
        "a0_wrong": not a0_correct,
        "bd": bd,
        "n_votes": len(answers),
    }


def rate_confident_wrong(rows: list[dict], key: str, tau: float) -> dict:
    n = len(rows)
    maj_wrong = [r for r in rows if r[f"{key}_maj_wrong"]]
    cw_all = sum(1 for r in rows if r[f"{key}_maj_wrong"] and r[f"{key}_p_top"] >= tau - 1e-12)
    cw_given_wrong = sum(
        1 for r in maj_wrong if r[f"{key}_p_top"] >= tau - 1e-12
    )
    return {
        "tau": tau,
        "confident_wrong_rate_all": cw_all / n if n else float("nan"),
        "confident_wrong_count_all": cw_all,
        "confident_wrong_rate_given_majority_wrong": cw_given_wrong / len(maj_wrong)
        if maj_wrong
        else float("nan"),
        "confident_wrong_count_given_majority_wrong": cw_given_wrong,
        "majority_wrong_count": len(maj_wrong),
        "n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k9", type=Path, default=DEFAULT_K9)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMM)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--join-out", type=Path, default=DEFAULT_JOIN)
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text())
    ids = [str(x) for x in meta["ids"]]

    sc_by_id: dict[str, dict] = {}
    for ln in args.k9.read_text().splitlines():
        if ln.strip():
            o = json.loads(ln)
            sc_by_id[str(o["id"])] = o

    summ_by_id: dict[str, dict] = {}
    for ln in args.summary.read_text().splitlines():
        if not ln.strip():
            continue
        p = json.loads(ln)
        num = str(p["id"]).split("_", 1)[-1]
        summ_by_id[num] = p

    per_question: list[dict] = []
    for qid in ids:
        if qid not in sc_by_id or qid not in summ_by_id:
            continue
        scs = sc_vote_stats(sc_by_id[qid])
        pvs = panda_vote_stats(summ_by_id[qid])
        if not scs or not pvs:
            continue
        summ = summ_by_id[qid]
        per_question.append(
            {
                "id": qid,
                "gold": summ.get("reference"),
                "label_drop": int(summ.get("label_drop") or 0),
                "sc_maj": scs["maj"],
                "sc_p_top": scs["p_top"],
                "sc_maj_wrong": scs["maj_wrong"],
                "sc_maj_correct": scs["maj_correct"],
                "panda_maj": pvs["maj"],
                "panda_p_top": pvs["p_top"],
                "panda_maj_wrong": pvs["maj_wrong"],
                "panda_maj_correct": pvs["maj_correct"],
                "panda_a0_correct": pvs["a0_correct"],
                "panda_bd": pvs["bd"],
            }
        )

    def subset(rows: list[dict], eval_only: bool) -> list[dict]:
        if not eval_only:
            return rows
        return [r for r in rows if r["label_drop"] == 0]

    out_doc: dict = {
        "cohort": str(args.meta),
        "n_questions": len(per_question),
        "definitions": {
            "sc@9": "First 9 samples from K=64 pool; majority vote on extracted answers; "
            "p_top = max vote share / 9; confident wrong = majority wrong AND p_top >= tau.",
            "panda@9": "1 greedy base + 4 text rephrases + 4 weight perturbations; "
            "majority vote on math-equivalence classes of normalized answers; "
            "p_top = largest class size / 9; confident wrong = majority wrong AND p_top >= tau.",
            "panda_collapse_bd0": "Greedy base (a0) wrong AND answer_drift (bd) = 0 "
            "(all 8 perturbations match a0; spurious consensus / collapse proxy).",
        },
        "sources": {
            "sc_k9": str(args.k9),
            "panda_summary": str(args.summary),
        },
        "tau_thresholds": TAU_SPECS,
        "metrics": {},
    }

    for eval_only, label in [(False, "n300"), (True, "n228_eval_no_label_drop")]:
        rows = subset(per_question, eval_only)
        flat = []
        for r in rows:
            flat.append(
                {
                    **r,
                    "sc_maj_wrong": r["sc_maj_wrong"],
                    "sc_p_top": r["sc_p_top"],
                    "panda_maj_wrong": r["panda_maj_wrong"],
                    "panda_p_top": r["panda_p_top"],
                }
            )
        sc_acc = sum(1 for r in rows if r["sc_maj_correct"]) / len(rows) if rows else float("nan")
        panda_vote_acc = (
            sum(1 for r in rows if r["panda_maj_correct"]) / len(rows) if rows else float("nan")
        )
        panda_a0_acc = (
            sum(1 for r in rows if r["panda_a0_correct"]) / len(rows) if rows else float("nan")
        )
        sc_taus = {
            spec["name"]: rate_confident_wrong(flat, "sc", spec["tau"]) for spec in TAU_SPECS
        }
        panda_taus = {
            spec["name"]: rate_confident_wrong(flat, "panda", spec["tau"]) for spec in TAU_SPECS
        }
        a0_wrong = [r for r in rows if not r["panda_a0_correct"]]
        collapse = [r for r in a0_wrong if abs(r["panda_bd"]) < 1e-12]
        out_doc["metrics"][label] = {
            "n": len(rows),
            "accuracy": {
                "sc9_majority_vote": sc_acc,
                "panda9_majority_vote": panda_vote_acc,
                "panda_greedy_a0": panda_a0_acc,
                "delta_panda_vote_minus_sc_pp": (panda_vote_acc - sc_acc) * 100
                if rows
                else float("nan"),
            },
            "confident_wrong_sc@9": sc_taus,
            "confident_wrong_panda@9_vote": panda_taus,
            "panda_collapse_a0_wrong_bd0": {
                "rate_all": len(collapse) / len(rows) if rows else float("nan"),
                "count_all": len(collapse),
                "rate_given_a0_wrong": len(collapse) / len(a0_wrong) if a0_wrong else float("nan"),
                "count_given_a0_wrong": len(collapse),
                "a0_wrong_count": len(a0_wrong),
            },
            "reduction_panda_vs_sc_confident_wrong_pp": {
                spec["name"]: (
                    sc_taus[spec["name"]]["confident_wrong_rate_all"]
                    - panda_taus[spec["name"]]["confident_wrong_rate_all"]
                )
                * 100
                for spec in TAU_SPECS
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")

    args.join_out.parent.mkdir(parents=True, exist_ok=True)
    with args.join_out.open("w") as f:
        for r in per_question:
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "gold": r["gold"],
                        "panda_correct": r["panda_a0_correct"],
                        "panda_maj_correct": r["panda_maj_correct"],
                        "sc_maj_correct": r["sc_maj_correct"],
                        "y_wrong_panda": int(not r["panda_a0_correct"]),
                        "y_wrong_sc": int(r["sc_maj_wrong"]),
                        "p_top": r["sc_p_top"],
                        "panda_p_top": r["panda_p_top"],
                        "sc_uptop": 1.0 - r["sc_p_top"],
                        "bd": r["panda_bd"],
                        "a0": summ_by_id[r["id"]].get("a0"),
                        "sc_maj": r["sc_maj"],
                        "panda_maj": r["panda_maj"],
                        "label_drop": r["label_drop"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {args.join_out} ({len(per_question)} rows)")


if __name__ == "__main__":
    main()
