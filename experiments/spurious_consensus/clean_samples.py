#!/usr/bin/env python3
"""清洗采样结果：剔除垃圾提取、重判 correct、合并格式等价答案。

针对小模型输出不稳定、extract_answer 误提取整段推理/截断 LaTeX 等问题。
原始文件备份为 *.jsonl.bak；清洗后写回原路径。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_PRS = Path(os.environ.get("PRS_ROOT", "/root/PRS")) / "src"
if str(_PRS) not in sys.path:
    sys.path.insert(0, str(_PRS))

from prs.grading.answer_canonicalizer import canonicalize_answer, math_equal_clean

from grading import extract_mcq_letter, extract_answer

# 已是提取结果（非 raw text）时的垃圾模式
_JUNK_RE = [
    re.compile(r"#{2,}\s*Step", re.I),
    re.compile(r"^(so,|therefore,|let me|we have|the answer is)\b", re.I),
    re.compile(r"\\frac\{[^}]*$"),
    re.compile(r"\\boxed\{[^}]*$"),
    re.compile(r"\\\[[^\]]*$"),
]
_MCQ_OK = re.compile(r"^[A-D]$")


def _is_junk_extracted(ans: str, grading: str) -> bool:
    if not ans or not str(ans).strip():
        return True
    s = str(ans).strip()
    if grading == "mcq":
        return _MCQ_OK.match(s.upper()) is None
    if len(s) > 180:
        return True
    if s.count("\\") > 20:
        return True
    for pat in _JUNK_RE:
        if pat.search(s):
            return True
    return False


def _normalize_extracted(ans: str, grading: str, dataset: str) -> str:
    """把已提取答案规范化；无法挽救的垃圾返回空串。"""
    if not ans or not str(ans).strip():
        return ""
    s = str(ans).strip()
    if grading == "mcq":
        if _MCQ_OK.match(s.upper()):
            return s.upper()
        letter = extract_mcq_letter(s)
        return letter if letter else ""
    if _is_junk_extracted(s, grading):
        return ""
    # 轻量再提取：有时答案字段里仍含 boxed 外壳
    s2 = extract_answer(s, grading, dataset)
    if s2 and not _is_junk_extracted(s2, grading):
        s = s2
    elif _is_junk_extracted(s, grading):
        return ""
    # 数学：canonicalize 合并格式变体（供共识统计）
    c = canonicalize_answer(s)
    return c if c else s.strip()


def _grade_clean(pred: str, gold: str, grading: str) -> int:
    if not pred:
        return 0
    if grading == "mcq":
        return int(pred.strip().upper() == gold.strip().upper())
    return int(math_equal_clean(pred, gold))


def clean_row(row: dict, model_tag: str, min_valid: int = 32) -> dict:
    grading = row.get("grading", "math")
    dataset = row.get("benchmark") or row.get("dataset", "")
    gold = row["gold"]
    qid = row["id"]
    orig_answers = row.get("answers", [])
    K = len(orig_answers)

    answers_clean: list[str] = []
    correct_clean: list[int] = []
    n_junk = 0

    for a in orig_answers:
        norm = _normalize_extracted(a, grading, dataset)
        if not norm:
            n_junk += 1
            answers_clean.append("")
            correct_clean.append(0)
            continue
        cc = _grade_clean(norm, gold, grading)
        answers_clean.append(norm)
        correct_clean.append(cc)

    n_valid = sum(1 for a in answers_clean if a)
    label_drop = 0
    drop_reason = ""
    if n_valid < min_valid:
        label_drop = 1
        drop_reason = f"valid<{min_valid}"
    elif n_junk > K * 0.35:
        label_drop = 1
        drop_reason = f"junk>{0.35:.0%}"

    out = dict(row)
    out["answers_orig"] = orig_answers
    out["correct_orig"] = row.get("correct", [])
    out["answers"] = answers_clean
    out["correct"] = correct_clean
    out["label_drop"] = label_drop
    out["clean_meta"] = {
        "n_valid": n_valid,
        "n_junk": n_junk,
        "drop_reason": drop_reason,
        "model_tag": model_tag,
    }
    return out


def summarize(paths: list[str], label: str) -> None:
    total = dropped = relabeled = 0
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            if r.get("label_drop"):
                dropped += 1
            oc, cc = r.get("correct_orig", r["correct"]), r["correct"]
            if oc != cc:
                relabeled += 1
    print(f"  {label}: {total} 题, label_drop={dropped}, 重标样本行={relabeled}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--min-valid", type=int, default=32, help="少于此有效答案则 label_drop")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pattern = str(ROOT / "data" / "samples" / f"samples_{args.tag}_seed{args.seed}_*.jsonl")
    paths = sorted(glob(pattern))
    if not paths:
        raise SystemExit(f"未找到: {pattern}")

    print(f"清洗 {args.tag} seed{args.seed}，{len(paths)} 个文件")
    if not args.dry_run:
        for p in paths:
            bak = p + ".bak"
            if not Path(bak).exists():
                shutil.copy2(p, bak)
                print(f"  备份 → {bak}")

    cleaned_paths = []
    stats = {"rows": 0, "label_drop": 0, "junk_slots": 0, "valid_slots": 0}
    for path in paths:
        out_rows = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = clean_row(json.loads(line), args.tag, args.min_valid)
            stats["rows"] += 1
            stats["label_drop"] += int(row["label_drop"])
            stats["junk_slots"] += row["clean_meta"]["n_junk"]
            stats["valid_slots"] += row["clean_meta"]["n_valid"]
            out_rows.append(row)
        if args.dry_run:
            tmp = path + ".dryrun"
            with open(tmp, "w", encoding="utf-8") as f:
                for r in out_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            cleaned_paths.append(tmp)
        else:
            with open(path, "w", encoding="utf-8") as f:
                for r in out_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            cleaned_paths.append(path)

    print(f"\n统计: {stats['rows']} 题, label_drop={stats['label_drop']}, "
          f"垃圾槽位={stats['junk_slots']}, 有效槽位={stats['valid_slots']}")
    print("\n--- 清洗效果 (maj@64, SCR@0.9) ---")
    for p in paths:
        bak = p + ".bak"
        if Path(bak).exists():
            _scr_report(bak, Path(p).name + " (原始)")
        cp = p + ".dryrun" if args.dry_run else p
        if Path(cp).exists():
            _scr_report(cp, Path(p).name + " (清洗后)")


def _scr_report(path: str, label: str) -> None:
    from collections import Counter as C

    wrong = scr = 0
    drop = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("label_drop"):
            drop += 1
            continue
        pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a]
        if len(pairs) < 2:
            continue
        a, _ = zip(*pairs)
        cnt = C(a)
        maj, top = cnt.most_common(1)[0]
        p = top / len(a)
        cmap = {}
        for ai, ci in zip(r["answers"], r["correct"]):
            if ai:
                cmap.setdefault(ai, ci)
        if cmap.get(maj, 0) == 0:
            wrong += 1
            if p >= 0.9:
                scr += 1
    denom = wrong
    rate = scr / denom * 100 if denom else 0
    print(f"  {label}: drop={drop}, wrong={wrong}, SCR@0.9={scr} ({rate:.2f}%)")


if __name__ == "__main__":
    main()
