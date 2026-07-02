"""Cluster-aware token trace: causal/structural features tied to answer clusters."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from panda.core.cluster import cluster_answers
from panda.core.reasoning_token_features import (
    _classify_token,
    _extract_last_equation,
    _js_divergence,
    _reasoning_text,
    _span_slice,
    _topk_probs,
)
from panda.grading.answer_canonicalizer import math_equal_clean
from panda.grading.math_grader import extract_math_answer, math_equal

_LATEX_CMD = re.compile(r"\\([a-zA-Z]+)")


def _entropy_discrete(counts: Counter) -> float:
    if not counts:
        return float("nan")
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    p = np.array([c / total for c in counts.values()], dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _math_token_sequence(run: dict, *, reasoning_only: bool = True) -> list[tuple[float, str, int, str]]:
    """Relative-position math tokens: (rel_pos, text, token_id, kind)."""
    trace = run.get("token_trace") or []
    if reasoning_only and run.get("answer_span"):
        s = int(run["answer_span"].get("start_token", len(trace)))
        trace = trace[: max(0, s)]
    seq = []
    math_items = [(i, t) for i, t in enumerate(trace) if _classify_token(t.get("token", "")) != "other"]
    n = len(math_items)
    if n == 0:
        return seq
    for j, (i, t) in enumerate(math_items):
        rp = j / max(n - 1, 1)
        tok = t.get("token", "").strip()
        seq.append((round(rp, 2), tok, int(t.get("token_id", 0)), _classify_token(tok)))
    return seq


def _token_dist(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    c = Counter(tokens)
    total = sum(c.values())
    return {k: v / total for k, v in c.items()}


def _js_categorical(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    pa = np.array([p.get(k, 0.0) for k in keys])
    pb = np.array([q.get(k, 0.0) for k in keys])
    return _js_divergence(pa, pb)


def _skeletonize(text: str) -> str:
    s = text.strip()
    s = re.sub(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", "NUM", s)
    s = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", "VAR", s)
    s = _LATEX_CMD.sub(lambda m: m.group(1).upper(), s)
    s = re.sub(r"\s+", "", s.lower())
    return s


def _formula_skeleton(run: dict) -> str:
    reasoning = _reasoning_text(run)
    if not reasoning:
        return ""
    # last few equation rhs + math tokens
    parts = []
    for line in reasoning.splitlines():
        if "=" in line:
            parts.append(_skeletonize(line.split("=")[-1]))
    seq = _math_token_sequence(run, reasoning_only=True)
    tail = " ".join(tok for _, tok, _, _ in seq[-12:])
    if tail:
        parts.append(_skeletonize(tail))
    return "|".join(parts[-3:]) if parts else _skeletonize(reasoning[-200:])


def _cluster_runs_by_answer(runs: list[dict]) -> tuple[list[int], list[str], int]:
    answers = [r.get("answer_normalized", "") for r in runs]
    labels, sizes = cluster_answers(answers)
    return labels, answers, len(sizes)


def cluster_aware_token_divergence(runs: list[dict], prefix: str) -> dict:
    """1. cluster_math_token_JS_max + 2. earliest_cluster_branch_pos."""
    nan = float("nan")
    if len(runs) < 2:
        return {
            f"{prefix}_cluster_math_token_js_max": nan,
            f"{prefix}_cluster_token_js_max": nan,
            f"{prefix}_earliest_cluster_branch_pos": nan,
            f"{prefix}_earliest_cluster_branch_ratio": nan,
            f"{prefix}_cluster_conditioned_flip_max": nan,
        }
    labels, _, n_clusters = _cluster_runs_by_answer(runs)
    if n_clusters < 2:
        return {
            f"{prefix}_cluster_math_token_js_max": 0.0,
            f"{prefix}_cluster_token_js_max": 0.0,
            f"{prefix}_earliest_cluster_branch_pos": nan,
            f"{prefix}_earliest_cluster_branch_ratio": nan,
            f"{prefix}_cluster_conditioned_flip_max": 0.0,
        }

    # position -> cluster -> list of token texts (math only)
    pos_cluster_tokens: dict[float, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    pos_cluster_all: dict[float, list[tuple[int, str]]] = defaultdict(list)
    max_math_len = 0

    for run, lab in zip(runs, labels):
        seq = _math_token_sequence(run, reasoning_only=True)
        max_math_len = max(max_math_len, len(seq))
        for rp, tok, _tid, kind in seq:
            if kind in ("numeric", "symbol", "variable"):
                pos_cluster_tokens[rp][lab].append(tok)
            pos_cluster_all[rp].append((lab, tok))

    js_max = 0.0
    math_js_max = 0.0
    branch_positions = []
    flip_max = 0.0

    for rp, clust_map in pos_cluster_tokens.items():
        if len(clust_map) < 2:
            continue
        dists = {c: _token_dist(toks) for c, toks in clust_map.items() if toks}
        if len(dists) < 2:
            continue
        clusters = list(dists.keys())
        pair_js = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pair_js.append(_js_categorical(dists[clusters[i]], dists[clusters[j]]))
        if not pair_js:
            continue
        pos_js = max(pair_js)
        math_js_max = max(math_js_max, pos_js)
        if pos_js > 0.05:
            branch_positions.append(rp)

        # conditioned flip: fraction of pairs from different clusters with different token
        items = pos_cluster_all.get(rp, [])
        diff = 0
        total = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] != items[j][0]:
                    total += 1
                    if items[i][1] != items[j][1]:
                        diff += 1
        if total:
            flip_max = max(flip_max, diff / total)

    for rp, items in pos_cluster_all.items():
        if len(set(lab for lab, _ in items)) < 2:
            continue
        dists_all = _token_dist([t for _, t in items])
        # split by cluster and JS
        clust_map = pos_cluster_tokens.get(rp, {})
        dists = {c: _token_dist(toks) for c, toks in clust_map.items() if toks}
        if len(dists) >= 2:
            clusters = list(dists.keys())
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    js_max = max(js_max, _js_categorical(dists[clusters[i]], dists[clusters[j]]))

    earliest = min(branch_positions) if branch_positions else nan
    return {
        f"{prefix}_cluster_math_token_js_max": float(math_js_max),
        f"{prefix}_cluster_token_js_max": float(js_max),
        f"{prefix}_earliest_cluster_branch_pos": float(earliest) if earliest == earliest else nan,
        f"{prefix}_earliest_cluster_branch_ratio": float(earliest) if earliest == earliest else nan,
        f"{prefix}_cluster_conditioned_flip_max": float(flip_max),
    }


def formula_skeleton_metrics(runs: list[dict], prefix: str) -> dict:
    """3. formula skeleton stability across perturbations."""
    nan = float("nan")
    skels = [_formula_skeleton(r) for r in runs]
    skels = [s for s in skels if s]
    if not skels:
        return {
            f"{prefix}_formula_skeleton_entropy": nan,
            f"{prefix}_formula_skeleton_num_clusters": nan,
            f"{prefix}_operator_flip_rate": nan,
            f"{prefix}_missing_symbol_rate": nan,
        }
    counts = Counter(skels)
    # operator flip: / vs * patterns differ
    ops = []
    for s in skels:
        ops.append(tuple(re.findall(r"[+\-*/=]", s)))
    op_flip = 1.0 - (len(set(ops)) / max(len(ops), 1))
    # missing delta/pi/2 pattern in skeleton vs most common
    mode = counts.most_common(1)[0][0]
    missing = []
    for s in skels:
        miss = 0
        for sym in ("delta", "pi", "sqrt", "sin", "cos"):
            if sym in mode and sym not in s:
                miss += 1
        missing.append(miss / max(len(re.findall(r"[a-z]+", mode)), 1))
    return {
        f"{prefix}_formula_skeleton_entropy": _entropy_discrete(counts),
        f"{prefix}_formula_skeleton_num_clusters": float(len(counts)),
        f"{prefix}_operator_flip_rate": float(op_flip),
        f"{prefix}_missing_symbol_rate": float(np.mean(missing)) if missing else 0.0,
    }


def final_answer_equiv_last_equation(run: dict) -> dict:
    """4. canonical final answer vs last k equations."""
    reasoning = _reasoning_text(run)
    final = run.get("answer_normalized") or extract_math_answer(run.get("full_response", ""))
    if not final or not reasoning:
        return {
            "final_answer_equiv_last_equation": float("nan"),
            "final_answer_in_last_k_equations": float("nan"),
            "final_number_newly_introduced": float("nan"),
        }
    eqs = []
    for line in reasoning.splitlines():
        line = line.strip()
        if "=" in line and len(line) > 2:
            eqs.append(line.split("=")[-1].strip())
    if not eqs:
        eqs = [_extract_last_equation(reasoning)]
    last_k = eqs[-5:]
    equiv = any(math_equal_clean(final, e) or math_equal(final, e) for e in last_k if e)
    in_last = float(equiv)
    # number newly introduced: final numeric not in reasoning body
    from panda.core.numeric_trajectory import extract_numbers

    fn = extract_numbers(str(final))
    rn = extract_numbers(reasoning)
    newly = 0.0
    if fn:
        newly = float(np.mean([not any(math_equal(str(a), str(r)) for r in rn) for a in fn]))
    return {
        "final_answer_equiv_last_equation": float(equiv),
        "final_answer_in_last_k_equations": in_last,
        "final_number_newly_introduced": newly,
    }


def alternative_answer_mass_topk(runs: list[dict], prefix: str) -> dict:
    """5. top-k mass for tokens from OTHER answer clusters at answer span."""
    nan = float("nan")
    if len(runs) < 2:
        return {
            f"{prefix}_alternative_answer_mass_topk": nan,
            f"{prefix}_alternative_answer_mass_max": nan,
            f"{prefix}_base_answer_mass_under_perturb": nan,
        }
    labels, answers, n_clusters = _cluster_runs_by_answer(runs)
    if n_clusters < 2:
        return {
            f"{prefix}_alternative_answer_mass_topk": 0.0,
            f"{prefix}_alternative_answer_mass_max": 0.0,
            f"{prefix}_base_answer_mass_under_perturb": 1.0,
        }

    # cluster -> set of answer span token strings
    cluster_ans_tokens: dict[int, set[str]] = defaultdict(set)
    for run, lab in zip(runs, labels):
        ans = _span_slice(run.get("token_trace") or [], run.get("answer_span"))
        for t in ans:
            tok = t.get("token", "").strip()
            if tok:
                cluster_ans_tokens[lab].add(tok)

    alt_masses = []
    alt_max = []
    for run, lab in zip(runs, labels):
        ans = _span_slice(run.get("token_trace") or [], run.get("answer_span"))
        other_tokens = set()
        for c, toks in cluster_ans_tokens.items():
            if c != lab:
                other_tokens |= toks
        if not other_tokens or not ans:
            continue
        pos_masses = []
        for t in ans:
            topk = t.get("topk") or []
            if not topk:
                continue
            total_p = 0.0
            alt_p = 0.0
            for tok_s, lp in topk:
                p = math.exp(float(lp))
                total_p += p
                if tok_s.strip() in other_tokens:
                    alt_p += p
            if total_p > 0:
                pos_masses.append(alt_p / total_p)
        if pos_masses:
            alt_masses.append(float(np.mean(pos_masses)))
            alt_max.append(float(np.max(pos_masses)))

    return {
        f"{prefix}_alternative_answer_mass_topk": float(np.mean(alt_masses)) if alt_masses else nan,
        f"{prefix}_alternative_answer_mass_max": float(np.max(alt_max)) if alt_max else nan,
        f"{prefix}_base_answer_mass_under_perturb": float(1.0 - np.mean(alt_masses)) if alt_masses else nan,
    }


def confident_fragmentation(run_labels_scores: tuple[float, float]) -> dict:
    """PANDA vs avg token entropy ratio (single scalar inputs)."""
    ase, avg_ent = run_labels_scores
    if avg_ent <= 1e-9:
        return {"confident_fragmentation": float("nan")}
    return {"confident_fragmentation": float(ase / avg_ent)}


def merge_cluster_token_trace(base: dict, text_runs: list[dict], weight_runs: list[dict]) -> dict:
    out: dict[str, float] = {}
    for prefix, runs in (("T", text_runs), ("W", weight_runs)):
        out.update(cluster_aware_token_divergence(runs, prefix))
        out.update(formula_skeleton_metrics(runs, prefix))
        out.update(alternative_answer_mass_topk(runs, prefix))
    out.update(final_answer_equiv_last_equation(base))
    # confident fragmentation on weight branch
    if weight_runs:
        answers = [r.get("answer_normalized", "") for r in weight_runs]
        _, sizes = cluster_answers(answers)
        n_c = max(len(sizes), 1)
        ase_u = 1.0 - max(sizes.values()) / max(len(answers), 1) if sizes else 0.0
        ents = []
        for r in weight_runs:
            ent = r.get("token_entropies") or [t.get("entropy", 0) for t in r.get("token_trace", [])]
            if ent:
                ents.append(float(np.mean(ent)))
        avg_ent = float(np.mean(ents)) if ents else float("nan")
        out.update(confident_fragmentation((ase_u, avg_ent)))
        out["W_confident_fragmentation"] = out.get("confident_fragmentation", float("nan"))
    return out
