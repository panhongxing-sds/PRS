import numpy as np
import pytest

from prs.scores.fragility import adversarial_drop, log_margin, margin_collapse
from prs.scores.uncertainty import epistemic_uncertainty
from prs.scores.composite import compute_token_scores, zscore
from prs.types import TokenFeatures


def test_adversarial_drop():
    assert adversarial_drop(-0.1, -0.5) == pytest.approx(0.4)
    assert adversarial_drop(-0.1, -0.05) == 0.0


def test_epistemic_disagreement():
    v = 50
    p1, p2 = np.zeros(v), np.zeros(v)
    p1[1], p2[2] = 1.0, 1.0
    assert epistemic_uncertainty([p1, p2]) > 0


def test_margin_collapse():
    logits = np.array([2.0, 1.0, 0.5])
    m0 = log_margin(logits, 0)
    logits_adv = np.array([0.5, 2.0, 0.1])
    m1 = log_margin(logits_adv, 0)
    assert margin_collapse(m0, m1) > 0


def test_zscore_composite():
    tokens = [
        TokenFeatures(0, 0, "a", 0, 0, 0, eu_weight=0.1, ad_embedding=0.2, ad_hidden=0.3, mc_hidden=0.4),
        TokenFeatures(1, 1, "b", 0, 0, 0, eu_weight=0.5, ad_embedding=0.6, ad_hidden=0.7, mc_hidden=0.8),
    ]
    compute_token_scores(tokens, zscore_within_response=True)
    assert tokens[1].score > tokens[0].score
