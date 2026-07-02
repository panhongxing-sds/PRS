"""Global sequence-level PGD on hidden states at layer ℓ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F


PGDObjective = Literal["ce_loss", "entropy", "top1_prob", "margin", "kl_div"]


@dataclass
class HiddenAttackConfig:
    epsilon: float = 0.01
    steps: int = 5
    step_size: float = 0.002
    objective: PGDObjective = "ce_loss"  # PGD attack objective


def resolve_layer_modules(model) -> list[torch.nn.Module]:
    """Qwen2 / Llama / Phi-3 / OPT style layer stacks."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Unsupported architecture: expected model.model.layers or model.model.decoder.layers")


def layer_indices_from_fractions(model, fractions: list[float]) -> list[int]:
    layers = resolve_layer_modules(model)
    L = len(layers)
    return sorted({min(L - 1, max(0, int(f * L) - 1)) for f in fractions})


def _compute_pgd_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    objective: PGDObjective,
    clean_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the objective to maximize for PGD attack."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = response_mask[..., 1:].contiguous().float()
    
    if objective == "ce_loss":
        # Maximize cross-entropy loss (original)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        return (loss * shift_mask.view(-1)).sum() / shift_mask.sum().clamp(min=1)
    
    elif objective == "entropy":
        # Maximize entropy of predictions
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, T-1]
        return (entropy * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    
    elif objective == "top1_prob":
        # Minimize top-1 probability (maximize -top1_prob)
        probs = F.softmax(shift_logits, dim=-1)
        top1_prob = probs.max(dim=-1).values  # [B, T-1]
        return -(top1_prob * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    
    elif objective == "margin":
        # Maximize margin collapse: reduce gap between top-1 and top-2 logits
        sorted_logits = shift_logits.sort(dim=-1, descending=True).values
        margin = sorted_logits[..., 0] - sorted_logits[..., 1]  # [B, T-1]
        return -(margin * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    
    elif objective == "kl_div":
        # Maximize KL divergence from clean distribution
        if clean_logits is None:
            raise ValueError("kl_div objective requires clean_logits")
        shift_clean = clean_logits[..., :-1, :].contiguous()
        clean_log_probs = F.log_softmax(shift_clean, dim=-1)
        adv_probs = F.softmax(shift_logits, dim=-1)
        # KL(adv || clean) = sum(adv * (log_adv - log_clean))
        adv_log_probs = F.log_softmax(shift_logits, dim=-1)
        kl = (adv_probs * (adv_log_probs - clean_log_probs)).sum(dim=-1)  # [B, T-1]
        return (kl * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    
    else:
        raise ValueError(f"Unknown objective: {objective}")


def global_hidden_pgd(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    layer_idx: int,
    config: HiddenAttackConfig,
    clean_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Add learnable δ to hidden states at layer_idx output, PGD to maximize objective on response tokens.
    Returns δ used (shape of hidden at that layer).
    
    Supported objectives:
    - ce_loss: maximize cross-entropy loss (original)
    - entropy: maximize entropy of predictions
    - top1_prob: minimize top-1 probability
    - margin: maximize margin collapse
    - kl_div: maximize KL divergence from clean (requires clean_logits)
    """
    layers = resolve_layer_modules(model)
    target_layer = layers[layer_idx]
    delta_holder: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        delta_holder["_dtype"] = h.dtype
        if "delta" not in delta_holder:
            delta_holder["delta"] = torch.zeros_like(h, requires_grad=True)
        d = delta_holder["delta"].to(dtype=h.dtype)
        return (h + d,) if isinstance(out, tuple) else h + d

    handle = target_layer.register_forward_hook(hook)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    try:
        for _ in range(config.steps):
            model.zero_grad(set_to_none=True)
            if "delta" in delta_holder and delta_holder["delta"].grad is not None:
                delta_holder["delta"].grad.zero_()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits
            
            objective_val = _compute_pgd_objective(
                logits, labels, response_mask, config.objective, clean_logits
            )
            
            # Maximize objective (negative for gradient ascent)
            (-objective_val).backward()

            with torch.no_grad():
                d = delta_holder["delta"]
                d.data = d.data + config.step_size * d.grad.sign()
                d.data = torch.clamp(d.data, -config.epsilon, config.epsilon)
                d.data = d.data.to(dtype=delta_holder["_dtype"])

        return delta_holder.get("delta", torch.zeros(1)).detach()
    finally:
        handle.remove()


def forward_with_hidden_delta(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    layer_idx: int,
    delta: torch.Tensor,
):
    """Single forward pass with fixed hidden perturbation."""
    layers = resolve_layer_modules(model)
    target_layer = layers[layer_idx]

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = delta.to(device=h.device, dtype=h.dtype)
        return (h + d,) if isinstance(out, tuple) else h + d

    handle = target_layer.register_forward_hook(hook)
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()


def _apply_position_delta(h: torch.Tensor, position: int, delta: torch.Tensor) -> torch.Tensor:
    """Add delta (hidden_dim,) at a single sequence position; delta may require grad."""
    mask = torch.zeros(
        h.shape[0], h.shape[1], 1,
        device=h.device,
        dtype=h.dtype,
    )
    mask[:, position : position + 1, :] = 1.0
    d = delta.view(1, 1, -1).to(device=h.device, dtype=h.dtype)
    return h + mask * d


def forward_with_position_delta(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    layer_idx: int,
    position: int,
    delta: torch.Tensor,
):
    """Forward with perturbation applied only at one hidden position."""
    layers = resolve_layer_modules(model)
    target_layer = layers[layer_idx]

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        pert = _apply_position_delta(h, position, delta)
        return (pert,) if isinstance(out, tuple) else pert

    handle = target_layer.register_forward_hook(hook)
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()


def _token_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pred_pos: int,
) -> torch.Tensor:
    """Cross-entropy for a single token prediction at pred_pos."""
    target_id = labels[0, pred_pos + 1]
    return F.cross_entropy(
        logits[0, pred_pos : pred_pos + 1, :],
        target_id.view(1),
    )


def _relative_radius(hidden_vec: torch.Tensor, epsilon: float, min_norm: float = 1e-6) -> float:
    """L2 radius scaled by hidden norm: epsilon * ||h||."""
    norm = float(hidden_vec.detach().float().norm().item())
    return epsilon * max(norm, min_norm)


def _project_delta(delta: torch.Tensor, radius: float) -> torch.Tensor:
    """Project delta onto L2 ball of given radius."""
    norm = delta.norm()
    if norm <= radius or norm < 1e-12:
        return delta
    return delta * (radius / norm)


def local_hidden_attack_one_token(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    layer_idx: int,
    pred_pos: int,
    config: HiddenAttackConfig,
    attack: Literal["fgsm", "pgd"] = "fgsm",
) -> tuple[float, float]:
    """
    Token-local hidden attack at pred_pos.

    Returns (orig_logprob, pert_logprob) for the target token.
    Perturbation radius: epsilon * ||h_{layer, pred_pos}||.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    target_id = int(labels[0, pred_pos + 1].item())
    if target_id < 0:
        return 0.0, 0.0

    layers = resolve_layer_modules(model)
    target_layer = layers[layer_idx]

    clean_hidden: dict[str, torch.Tensor] = {}

    def capture_hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        clean_hidden["h"] = h[0, pred_pos].detach().clone()
        return out

    capture = target_layer.register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            clean_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            orig_logprob = float(
                F.log_softmax(clean_out.logits, dim=-1)[0, pred_pos, target_id].item()
            )
    finally:
        capture.remove()

    if "h" not in clean_hidden:
        return orig_logprob, orig_logprob

    device = clean_hidden["h"].device
    dtype = clean_hidden["h"].dtype
    hidden_dim = clean_hidden["h"].shape[-1]

    radius = _relative_radius(clean_hidden["h"], config.epsilon)
    delta = torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True)

    steps = 1 if attack == "fgsm" else max(config.steps, 1)
    step_size = config.step_size if attack == "pgd" else radius

    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        if delta.grad is not None:
            delta.grad.zero_()

        outputs = forward_with_position_delta(
            model,
            input_ids,
            attention_mask,
            layer_idx,
            pred_pos,
            delta,
        )
        loss = _token_ce_loss(outputs.logits, labels, pred_pos)
        (-loss).backward()

        with torch.no_grad():
            grad = delta.grad
            if grad is None:
                break
            if attack == "fgsm":
                delta.data = _project_delta(
                    radius * grad / grad.norm().clamp(min=1e-12),
                    radius,
                )
            else:
                delta.data = delta.data + step_size * grad.sign()
                delta.data = _project_delta(delta.data, radius)

    with torch.no_grad():
        pert_out = forward_with_position_delta(
            model,
            input_ids,
            attention_mask,
            layer_idx,
            pred_pos,
            delta.detach(),
        )
        pert_logprob = float(
            F.log_softmax(pert_out.logits, dim=-1)[0, pred_pos, target_id].item()
        )

    return orig_logprob, pert_logprob


def compute_local_hidden_ad(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    layer_idx: int,
    config: HiddenAttackConfig,
    attack: Literal["fgsm", "pgd"] = "fgsm",
    max_tokens: int | None = None,
) -> list[float]:
    """
    Compute per-token adversarial drop with token-local hidden attack.

    Each token uses its own perturbation at pred_pos = pos - 1 with
    radius epsilon * ||h_{pred_pos}||.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    shift_mask = response_mask[0, 1:].bool()
    pred_positions = torch.nonzero(shift_mask, as_tuple=False).view(-1).tolist()

    if max_tokens is not None and len(pred_positions) > max_tokens:
        pred_positions = pred_positions[:max_tokens]

    ad_tokens: list[float] = []
    for pred_pos in pred_positions:
        orig_lp, pert_lp = local_hidden_attack_one_token(
            model,
            input_ids,
            labels,
            attention_mask,
            layer_idx,
            pred_pos,
            config,
            attack=attack,
        )
        ad_tokens.append(max(0.0, orig_lp - pert_lp))

    return ad_tokens
