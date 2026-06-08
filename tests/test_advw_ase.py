"""Unit tests for AdvW-ASE multi-start PGD and clustering metrics."""

from __future__ import annotations

import torch
import torch.nn as nn

from prs.ase.advw_metrics import advw_metrics_from_record
from prs.perturbations.weight_attack import (
    WeightAttackConfig,
    multi_start_weight_pgd,
    weight_pgd_attack,
)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 8, bias=False)

    def forward(self, input_ids=None, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        h = self.q_proj(self.embed(input_ids)) + self.k_proj(self.embed(input_ids))
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits})()


def test_multi_start_returns_m_deltas():
    model = TinyLM()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = input_ids.clone()
    response_mask = torch.tensor([[0, 1, 1, 1]])
    cfg = WeightAttackConfig(
        epsilon=0.05, steps=2, step_size=0.02, rank=2, targets=("q_proj", "k_proj"), objective="margin"
    )
    seeds = [42, 43, 44]
    results = multi_start_weight_pgd(
        model, input_ids, labels, response_mask, None, cfg, seeds=seeds, init_sigma=0.01
    )
    assert len(results) == 3
    for r, seed in zip(results, seeds):
        assert r.seed == seed
        assert "q_proj" in r.deltas or "k_proj" in r.deltas
        assert r.deltas
        assert r.attack_loss == r.attack_loss  # finite


def test_weight_pgd_accepts_initial_coefficients():
    model = TinyLM()
    input_ids = torch.tensor([[1, 2, 3]])
    labels = input_ids.clone()
    response_mask = torch.tensor([[0, 1, 1]])
    cfg = WeightAttackConfig(epsilon=0.05, steps=1, step_size=0.02, rank=2, targets=("q_proj",))
    initial = {"q_proj": torch.tensor([0.01, -0.01])}
    pgd = weight_pgd_attack(
        model, input_ids, labels, response_mask, None, cfg, initial_coefficients=initial
    )
    assert pgd.deltas
    assert pgd.attack_loss == pgd.attack_loss


def test_advw_ase_clustering_metric_synthetic():
    """Uniform answers → max_mass=1 → AdvW_ASE=0; split answers → higher ASE."""
    uniform = {
        "advw_perturb_runs": [
            {"answer_normalized": "42", "perturb_config": {"attack_loss_final": 0.5}},
            {"answer_normalized": "42", "perturb_config": {"attack_loss_final": 0.6}},
        ]
    }
    split = {
        "advw_perturb_runs": [
            {"answer_normalized": "42", "perturb_config": {"attack_loss_final": 0.5}},
            {"answer_normalized": "7", "perturb_config": {"attack_loss_final": 0.9}},
        ]
    }
    m_u = advw_metrics_from_record(uniform)
    m_s = advw_metrics_from_record(split)
    assert m_u["AdvW_ASE"] == 0.0
    assert m_s["AdvW_ASE"] == 0.5
    assert m_u["AdvW_severity"] == 0.55
    assert m_s["AdvW_worst"] == 0.9
