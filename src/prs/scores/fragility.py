"""Adversarial drop (AD) and margin collapse (MC)."""

from __future__ import annotations

import numpy as np


def adversarial_drop(logprob_clean: float, logprob_adv: float) -> float:
    """AD_t = max(0, log p_0(y_t) - log p_adv(y_t))."""
    return float(max(0.0, logprob_clean - logprob_adv))


def log_margin(logits: np.ndarray, token_id: int) -> float:
    """log p(y_t) - max_{v != y_t} log p(v)."""
    log_probs = logits - np.logaddexp.reduce(logits)
    tid = int(token_id)
    others = np.delete(log_probs, tid) if tid < len(log_probs) else log_probs
    best_other = float(np.max(others)) if len(others) else -np.inf
    return float(log_probs[tid] - best_other) if tid < len(log_probs) else 0.0


def margin_collapse(margin_clean: float, margin_adv: float) -> float:
    """MC_t = M^0 - M^adv (larger => boundary collapsed)."""
    return float(max(0.0, margin_clean - margin_adv))
