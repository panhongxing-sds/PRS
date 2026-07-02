"""Semantic-guided adversarial perturbation using negative/contrastive tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F


SemanticDirection = Literal["negation", "contrast", "antonym", "random"]


@dataclass
class SemanticAttackConfig:
    direction: SemanticDirection = "contrast"
    strength: float = 0.5
    top_k_contrast: int = 100
    use_gradient_guidance: bool = True
    objective: Literal["ce_loss", "margin", "entropy"] = "margin"


def find_contrast_tokens(
    model,
    token_ids: torch.Tensor,
    top_k: int = 100,
) -> torch.Tensor:
    """
    For each token, find semantically contrasting tokens.
    Uses embedding cosine similarity to find furthest tokens.
    
    Returns: [batch, seq_len, top_k] tensor of contrasting token IDs
    """
    embed = model.get_input_embeddings()
    vocab_size = embed.num_embeddings
    
    token_embeds = embed(token_ids)
    all_embeds = embed.weight
    
    token_embeds_norm = F.normalize(token_embeds, p=2, dim=-1)
    all_embeds_norm = F.normalize(all_embeds, p=2, dim=-1)
    
    similarity = torch.matmul(token_embeds_norm, all_embeds_norm.T)
    _, contrast_ids = similarity.topk(top_k, dim=-1, largest=False)
    
    return contrast_ids


def compute_semantic_direction(
    model,
    token_ids: torch.Tensor,
    contrast_ids: torch.Tensor,
    aggregation: Literal["mean", "max", "weighted"] = "mean",
) -> torch.Tensor:
    """
    Compute perturbation direction from original to contrasting embeddings.
    
    Returns: [batch, seq_len, embed_dim] tensor of direction vectors
    """
    embed = model.get_input_embeddings()
    
    orig_embeds = embed(token_ids)
    contrast_embeds = embed(contrast_ids)
    
    if aggregation == "mean":
        contrast_mean = contrast_embeds.mean(dim=-2)
    elif aggregation == "max":
        norms = (contrast_embeds - orig_embeds.unsqueeze(-2)).norm(dim=-1)
        max_idx = norms.argmax(dim=-1, keepdim=True).unsqueeze(-1)
        max_idx = max_idx.expand(-1, -1, -1, contrast_embeds.size(-1))
        contrast_mean = contrast_embeds.gather(-2, max_idx).squeeze(-2)
    else:
        contrast_mean = contrast_embeds.mean(dim=-2)
    
    direction = contrast_mean - orig_embeds
    direction = F.normalize(direction, p=2, dim=-1)
    
    return direction


def semantic_guided_perturbation(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: SemanticAttackConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Perturb embeddings in semantically meaningful directions.
    
    Returns:
        - perturbed_embeds: [batch, seq_len, embed_dim]
        - info: dict with intermediate values for analysis
    """
    embed_layer = model.get_input_embeddings()
    orig_embeds = embed_layer(input_ids)
    dtype = orig_embeds.dtype
    device = orig_embeds.device
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    contrast_ids = find_contrast_tokens(model, input_ids, config.top_k_contrast)
    direction = compute_semantic_direction(model, input_ids, contrast_ids)
    
    if config.use_gradient_guidance:
        direction = direction.detach().requires_grad_(False)
        delta = torch.zeros_like(orig_embeds, requires_grad=True)
        
        optimizer_steps = 5
        step_size = config.strength / optimizer_steps
        
        for _ in range(optimizer_steps):
            perturbed = orig_embeds + delta
            outputs = model(
                inputs_embeds=perturbed.to(dtype),
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            
            loss = _compute_objective(
                logits, labels, response_mask, config.objective
            )
            
            loss.backward()
            
            with torch.no_grad():
                if delta.grad is not None:
                    grad_proj = (delta.grad * direction).sum(dim=-1, keepdim=True)
                    guided_grad = grad_proj * direction
                    delta.data = delta.data + step_size * guided_grad.sign()
                    delta.data = torch.clamp(
                        delta.data, 
                        -config.strength * direction.abs(), 
                        config.strength * direction.abs()
                    )
                    delta.grad = None
        
        perturbed_embeds = (orig_embeds + delta).detach()
    else:
        delta = config.strength * direction
        perturbed_embeds = orig_embeds + delta
    
    delta_final = perturbed_embeds - orig_embeds
    delta_norm = delta_final.norm(dim=-1)
    delta_alignment = (F.normalize(delta_final, dim=-1) * direction).sum(dim=-1)
    
    info = {
        "contrast_ids": contrast_ids,
        "direction": direction,
        "delta_norm": delta_norm,
        "delta_alignment": delta_alignment,
    }
    
    return perturbed_embeds, info


def _compute_objective(
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


def score_semantic_fragility(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: SemanticAttackConfig,
) -> dict[str, float]:
    """
    Score semantic fragility: how much does perturbing toward negative semantics
    change the model's predictions?
    
    Returns dict with various fragility scores.
    """
    embed_layer = model.get_input_embeddings()
    orig_embeds = embed_layer(input_ids)
    dtype = orig_embeds.dtype
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        orig_outputs = model(
            inputs_embeds=orig_embeds.to(dtype),
            attention_mask=attention_mask,
            use_cache=False,
        )
        orig_logits = orig_outputs.logits
        orig_logprobs = F.log_softmax(orig_logits, dim=-1)
    
    perturbed_embeds, info = semantic_guided_perturbation(
        model, input_ids, labels, response_mask, attention_mask, config
    )
    
    with torch.no_grad():
        pert_outputs = model(
            inputs_embeds=perturbed_embeds.to(dtype),
            attention_mask=attention_mask,
            use_cache=False,
        )
        pert_logits = pert_outputs.logits
        pert_logprobs = F.log_softmax(pert_logits, dim=-1)
    
    response_positions = response_mask[0].nonzero(as_tuple=True)[0].tolist()
    
    ad_tokens = []
    margin_collapse_tokens = []
    kl_tokens = []
    
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
        
        kl = F.kl_div(
            pert_logprobs[0, pred_pos],
            orig_logprobs[0, pred_pos].exp(),
            reduction="sum",
        ).item()
        kl_tokens.append(max(0, kl))
    
    n = len(ad_tokens) or 1
    
    scores = {
        "semantic_ad_sum": sum(ad_tokens),
        "semantic_ad_mean": sum(ad_tokens) / n,
        "semantic_ad_max": max(ad_tokens) if ad_tokens else 0.0,
        "semantic_mc_sum": sum(margin_collapse_tokens),
        "semantic_mc_mean": sum(margin_collapse_tokens) / n,
        "semantic_mc_max": max(margin_collapse_tokens) if margin_collapse_tokens else 0.0,
        "semantic_kl_sum": sum(kl_tokens),
        "semantic_kl_mean": sum(kl_tokens) / n,
        "n_tokens": n,
    }
    
    if ad_tokens:
        ad_sorted = sorted(ad_tokens, reverse=True)
        top_10_pct = max(1, int(n * 0.1))
        scores["semantic_ad_top10pct"] = sum(ad_sorted[:top_10_pct]) / top_10_pct
        
        import numpy as np
        ad_arr = np.array(ad_tokens)
        if ad_arr.sum() > 0:
            sorted_ad = np.sort(ad_arr)
            cumsum = sorted_ad.cumsum()
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_ad))) / (n * cumsum[-1]) - (n + 1) / n
            scores["semantic_ad_gini"] = float(gini)
        else:
            scores["semantic_ad_gini"] = 0.0
    
    return scores
