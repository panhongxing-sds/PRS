"""Perturbation Reliability Score (PRS) and component field resolution.

PRS(x) = F_resp(x) + lambda_a * D_ans(x) + lambda_r * D_reason(x)

Components (scoring layer; perturbation pipeline still uses T/W/TW internally):
  F_resp  — response fragmentation (4T+4W ASE, legacy TW_ASE)
  D_ans   — answer drift at commitment (AltMass_final / W_alternative_answer_mass_topk)
  D_reason — reasoning drift (AltMass_local_spread_reason)
"""

from __future__ import annotations

import math
from typing import Any

LAMBDA_A = 0.05
LAMBDA_R = 0.03

# New primary keys written by the pipeline
KEY_F_RESP = "F_resp"
KEY_D_ANS = "D_ans"
KEY_D_REASON = "D_reason"
KEY_PRS = "PRS"

# Human-readable aliases (same values as components)
KEY_RESPONSE_FRAGMENTATION = "response_fragmentation"
KEY_ANSWER_DRIFT = "answer_drift"
KEY_REASONING_DRIFT = "reasoning_drift"

# Legacy read order (first finite value wins)
_F_RESP_ALIASES = (
    KEY_F_RESP,
    KEY_RESPONSE_FRAGMENTATION,
    "TW_ASE",
)
_D_ANS_ALIASES = (
    KEY_D_ANS,
    KEY_ANSWER_DRIFT,
    "AltMass_final",
    "W_alternative_answer_mass_topk",
)
_D_REASON_ALIASES = (
    KEY_D_REASON,
    KEY_REASONING_DRIFT,
    # B (primary): domain-agnostic top-k%% local spread; falls back to legacy math-token mean.
    "AltMass_local_spread_topk",
    "AltMass_local_spread_reason",
)


def _first_finite(row: dict[str, Any], keys: tuple[str, ...]) -> float:
    for k in keys:
        if k not in row:
            continue
        v = row[k]
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return float("nan")


def resolve_f_resp(row: dict[str, Any]) -> float:
    return _first_finite(row, _F_RESP_ALIASES)


def resolve_d_ans(row: dict[str, Any]) -> float:
    return _first_finite(row, _D_ANS_ALIASES)


def resolve_d_reason(row: dict[str, Any]) -> float:
    return _first_finite(row, _D_REASON_ALIASES)


def resolve_feature(row: dict[str, Any], key: str) -> float:
    """Read a feature by PRS or legacy key (finite value or nan)."""
    if key in (KEY_F_RESP, KEY_RESPONSE_FRAGMENTATION):
        return resolve_f_resp(row)
    if key in (KEY_D_ANS, KEY_ANSWER_DRIFT):
        return resolve_d_ans(row)
    if key in (KEY_D_REASON, KEY_REASONING_DRIFT):
        return resolve_d_reason(row)
    if key == KEY_PRS:
        v = row.get(KEY_PRS)
        if v is not None:
            try:
                f = float(v)
                if math.isfinite(f):
                    return f
            except (TypeError, ValueError):
                pass
        return compute_prs_from_row(row)[KEY_PRS]
    if key not in row:
        return float("nan")
    try:
        f = float(row[key])
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def compute_prs(
    f_resp: float,
    d_ans: float,
    d_reason: float,
    *,
    lambda_a: float = LAMBDA_A,
    lambda_r: float = LAMBDA_R,
) -> float:
    """Linear PRS from three components; nan if any input is non-finite."""
    if not all(math.isfinite(x) for x in (f_resp, d_ans, d_reason)):
        return float("nan")
    return float(f_resp + lambda_a * d_ans + lambda_r * d_reason)


def compute_prs_from_row(
    row: dict[str, Any],
    *,
    lambda_a: float = LAMBDA_A,
    lambda_r: float = LAMBDA_R,
) -> dict[str, float]:
    """Resolve components from *row* and return PRS bundle (no mutation)."""
    f_resp = resolve_f_resp(row)
    d_ans = resolve_d_ans(row)
    d_reason = resolve_d_reason(row)
    prs = compute_prs(f_resp, d_ans, d_reason, lambda_a=lambda_a, lambda_r=lambda_r)
    return {
        KEY_F_RESP: f_resp,
        KEY_D_ANS: d_ans,
        KEY_D_REASON: d_reason,
        KEY_PRS: prs,
        KEY_RESPONSE_FRAGMENTATION: f_resp,
        KEY_ANSWER_DRIFT: d_ans,
        KEY_REASONING_DRIFT: d_reason,
        "lambda_a": lambda_a,
        "lambda_r": lambda_r,
    }


def enrich_row_with_prs(
    row: dict[str, Any],
    *,
    lambda_a: float = LAMBDA_A,
    lambda_r: float = LAMBDA_R,
    write_legacy: bool = True,
) -> dict[str, Any]:
    """Add PRS fields to *row*; keep legacy keys when *write_legacy* is True."""
    bundle = compute_prs_from_row(row, lambda_a=lambda_a, lambda_r=lambda_r)
    row[KEY_F_RESP] = bundle[KEY_F_RESP]
    row[KEY_D_ANS] = bundle[KEY_D_ANS]
    row[KEY_D_REASON] = bundle[KEY_D_REASON]
    row[KEY_PRS] = bundle[KEY_PRS]
    row[KEY_RESPONSE_FRAGMENTATION] = bundle[KEY_RESPONSE_FRAGMENTATION]
    row[KEY_ANSWER_DRIFT] = bundle[KEY_ANSWER_DRIFT]
    row[KEY_REASONING_DRIFT] = bundle[KEY_REASONING_DRIFT]
    row["lambda_a"] = lambda_a
    row["lambda_r"] = lambda_r
    if write_legacy:
        if math.isfinite(bundle[KEY_F_RESP]):
            row.setdefault("TW_ASE", bundle[KEY_F_RESP])
        if math.isfinite(bundle[KEY_D_ANS]):
            row.setdefault("AltMass_final", bundle[KEY_D_ANS])
            row.setdefault("W_alternative_answer_mass_topk", bundle[KEY_D_ANS])
        if math.isfinite(bundle[KEY_D_REASON]):
            row.setdefault("AltMass_local_spread_reason", bundle[KEY_D_REASON])
    return row
