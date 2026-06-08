"""Adversarial perturbation in weight space using PGD on attention projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torch >= 2.0
    from torch.func import functional_call
except Exception:  # pragma: no cover - older torch fallback
    from torch.nn.utils.stateless import functional_call


@dataclass
class WeightAttackConfig:
    epsilon: float = 0.1
    steps: int = 5
    step_size: float = 0.02
    rank: int = 8
    targets: tuple[str, ...] = ("q_proj", "k_proj")
    objective: Literal["ce_loss", "margin", "entropy"] = "margin"
    use_svd_basis: bool = True
    max_modules: int | None = 4  # attack last N matched modules (memory)


def _resolve_target_modules(
    model: nn.Module,
    target_suffixes: tuple[str, ...],
) -> list[tuple[str, nn.Linear]]:
    """Find all Linear modules matching target suffixes."""
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(s) for s in target_suffixes):
                targets.append((name, module))
    return targets


def _resolve_attack_targets(model: nn.Module, config: WeightAttackConfig) -> list[tuple[str, nn.Module]]:
    targets = _resolve_target_modules(model, config.targets)
    if config.max_modules is not None and len(targets) > config.max_modules:
        targets = targets[-config.max_modules :]
    return targets


def _compute_svd_basis(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute low-rank SVD basis for efficient perturbation."""
    w = weight.float()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    r = min(rank, u.shape[1], vh.shape[0])
    return u[:, :r].to(weight.dtype), vh[:r, :].to(weight.dtype)


class WeightPGDResult(NamedTuple):
    deltas: dict[str, torch.Tensor]
    attack_loss: float


def apply_weight_deltas(model: nn.Module, deltas: dict[str, torch.Tensor]) -> None:
    """Add low-rank weight deltas in-place."""
    for name, module in model.named_modules():
        if name in deltas and isinstance(module, nn.Linear):
            module.weight.data.add_(deltas[name])


def remove_weight_deltas(model: nn.Module, deltas: dict[str, torch.Tensor]) -> None:
    """Remove previously applied weight deltas."""
    for name, module in model.named_modules():
        if name in deltas and isinstance(module, nn.Linear):
            module.weight.data.sub_(deltas[name])


def weight_pgd_attack(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: WeightAttackConfig,
    *,
    initial_coefficients: dict[str, torch.Tensor] | None = None,
) -> WeightPGDResult:
    """
    PGD attack on model weights.

    Perturbs weights in low-rank SVD subspace to maximize attack objective.
    Optional ``initial_coefficients`` maps module name → rank-r diagonal coeffs in SVD basis.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    targets = _resolve_attack_targets(model, config)
    if not targets:
        return WeightPGDResult({}, float("nan"))

    bases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    deltas: dict[str, torch.Tensor] = {}

    for name, module in targets:
        u, vh = _compute_svd_basis(module.weight.data, config.rank)
        bases[name] = (u, vh)
        r = u.shape[1]
        if initial_coefficients and name in initial_coefficients:
            init = initial_coefficients[name].detach().to(
                device=module.weight.device, dtype=module.weight.dtype
            )
            deltas[name] = init.clone().requires_grad_(True)
        else:
            deltas[name] = torch.zeros(
                r, device=module.weight.device, dtype=module.weight.dtype, requires_grad=True
            )
    
    was_training = model.training
    model.eval()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    
    final_loss = float("nan")
    try:
        for _step in range(config.steps):
            for name, d in deltas.items():
                if d.grad is not None:
                    d.grad.zero_()

            patched_params = dict(params)
            for name, _module in targets:
                u, vh = bases[name]
                param_name = f"{name}.weight"
                if param_name not in patched_params:
                    continue
                delta_w = u @ torch.diag(deltas[name]) @ vh
                patched_params[param_name] = patched_params[param_name] + delta_w

            outputs = functional_call(
                model,
                {**patched_params, **buffers},
                args=(),
                kwargs={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                },
            )
            logits = outputs.logits
            loss = _compute_weight_objective(
                logits, labels, response_mask, config.objective
            )
            final_loss = float(loss.item())

            # Gradient ascent on the configured attack objective.
            loss.backward()

            with torch.no_grad():
                for name, d in deltas.items():
                    if d.grad is None:
                        continue
                    d.data = d.data + config.step_size * d.grad.sign()
                    d.data = torch.clamp(d.data, -config.epsilon, config.epsilon)

    finally:
        model.train(was_training)

    result: dict[str, torch.Tensor] = {}
    for name, d in deltas.items():
        u, vh = bases[name]
        delta_w = u @ torch.diag(d.detach()) @ vh
        result[name] = delta_w

    return WeightPGDResult(result, final_loss)


@dataclass
class MultiStartPGDResult:
    deltas: dict[str, torch.Tensor]
    attack_loss: float
    seed: int


def multi_start_weight_pgd(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: WeightAttackConfig,
    *,
    seeds: list[int],
    init_sigma: float = 0.01,
) -> list[MultiStartPGDResult]:
    """
    Multi-start PGD: random init ΔW_0 ~ N(0, σ²) in SVD subspace, then T-step ascent.

    Returns one result per seed with converged deltas and final attack loss.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    targets = _resolve_attack_targets(model, config)
    if not targets:
        return []

    bases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, module in targets:
        u, vh = _compute_svd_basis(module.weight.data, config.rank)
        bases[name] = (u, vh)

    results: list[MultiStartPGDResult] = []
    for seed in seeds:
        initial: dict[str, torch.Tensor] = {}
        gen = torch.Generator(device=targets[0][1].weight.device)
        gen.manual_seed(seed)
        for name, module in targets:
            u, _vh = bases[name]
            r = u.shape[1]
            initial[name] = (
                torch.randn(r, generator=gen, device=module.weight.device, dtype=module.weight.dtype)
                * init_sigma
            )

        pgd = weight_pgd_attack(
            model,
            input_ids,
            labels,
            response_mask,
            attention_mask,
            config,
            initial_coefficients=initial,
        )
        results.append(MultiStartPGDResult(pgd.deltas, pgd.attack_loss, seed))

    return results


def _compute_weight_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    """Compute attack objective to maximize."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = response_mask[..., 1:].contiguous().float()
    
    if objective == "ce_loss":
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        return (loss * shift_mask.view(-1)).sum() / shift_mask.sum().clamp(min=1)
    
    elif objective == "margin":
        sorted_logits = shift_logits.sort(dim=-1, descending=True).values
        margin = sorted_logits[..., 0] - sorted_logits[..., 1]
        return -(margin * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    
    elif objective == "entropy":
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return (entropy * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    
    raise ValueError(f"Unknown objective: {objective}")


def score_weight_fragility(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: WeightAttackConfig,
) -> dict[str, float]:
    """
    Score weight-space fragility: how much do adversarial weight perturbations
    change the model's predictions?
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        orig_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        orig_logits = orig_outputs.logits
        orig_logprobs = F.log_softmax(orig_logits, dim=-1)
    
    pgd = weight_pgd_attack(
        model, input_ids, labels, response_mask, attention_mask, config
    )
    deltas = pgd.deltas

    if not deltas:
        return {"weight_ad_sum": 0.0, "weight_ad_mean": 0.0, "n_tokens": 0, "status": "no_targets"}

    targets = _resolve_attack_targets(model, config)
    
    for name, module in targets:
        if name in deltas:
            module.weight.data.add_(deltas[name])
    
    try:
        with torch.no_grad():
            pert_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            pert_logits = pert_outputs.logits
            pert_logprobs = F.log_softmax(pert_logits, dim=-1)
    finally:
        for name, module in targets:
            if name in deltas:
                module.weight.data.sub_(deltas[name])
    
    response_positions = response_mask[0].nonzero(as_tuple=True)[0].tolist()
    
    ad_tokens = []
    margin_collapse_tokens = []
    
    for pos in response_positions:
        if pos == 0:
            continue
        pred_pos = pos - 1
        target_id = int(labels[0, pos].item())
        if target_id < 0:
            continue
        
        orig_lp = float(orig_logprobs[0, pred_pos, target_id].item())
        pert_lp = float(pert_logprobs[0, pred_pos, target_id].item())
        ad_tokens.append(max(0, orig_lp - pert_lp))
        
        orig_sorted = orig_logits[0, pred_pos].sort(descending=True).values
        pert_sorted = pert_logits[0, pred_pos].sort(descending=True).values
        orig_margin = float((orig_sorted[0] - orig_sorted[1]).item())
        pert_margin = float((pert_sorted[0] - pert_sorted[1]).item())
        margin_collapse_tokens.append(max(0, orig_margin - pert_margin))
    
    n = len(ad_tokens) or 1
    
    scores = {
        "weight_ad_sum": sum(ad_tokens),
        "weight_ad_mean": sum(ad_tokens) / n,
        "weight_ad_max": max(ad_tokens) if ad_tokens else 0.0,
        "weight_mc_sum": sum(margin_collapse_tokens),
        "weight_mc_mean": sum(margin_collapse_tokens) / n,
        "weight_mc_max": max(margin_collapse_tokens) if margin_collapse_tokens else 0.0,
        "n_tokens": n,
    }
    
    if ad_tokens:
        ad_sorted = sorted(ad_tokens, reverse=True)
        top_10_pct = max(1, int(n * 0.1))
        scores["weight_ad_top10pct"] = sum(ad_sorted[:top_10_pct]) / top_10_pct
        
        import numpy as np
        ad_arr = np.array(ad_tokens)
        if ad_arr.sum() > 0:
            sorted_ad = np.sort(ad_arr)
            cumsum = sorted_ad.cumsum()
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_ad))) / (n * cumsum[-1]) - (n + 1) / n
            scores["weight_ad_gini"] = float(gini)
        else:
            scores["weight_ad_gini"] = 0.0
    
    return scores
