"""Global sequence-level adversarial perturbation on input embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EmbeddingAttackConfig:
    epsilon: float = 0.01
    steps: int = 5
    step_size: float = 0.002
    mask: str = "answer"  # all | answer | query


def global_embedding_pgd(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: EmbeddingAttackConfig,
) -> torch.Tensor:
    """
    δ* = argmax_{||δ||≤ε} Σ_t -log p(y_t | E+δ) over response positions.

    Returns delta (same shape as input embeddings).
    """
    embed = model.get_input_embeddings()
    input_embeds = embed(input_ids).detach()
    dtype = input_embeds.dtype
    delta = torch.zeros_like(input_embeds, requires_grad=True, dtype=dtype)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    if config.mask == "answer":
        perturb_mask = response_mask.unsqueeze(-1).float()
    elif config.mask == "query":
        perturb_mask = (1 - response_mask).unsqueeze(-1).float()
    else:
        perturb_mask = torch.ones_like(input_embeds[..., :1])

    was_training = model.training
    model.eval()
    try:
        for _ in range(config.steps):
            perturbed = (input_embeds + delta * perturb_mask).to(dtype=dtype)
            with torch.enable_grad():
                outputs = model(
                    inputs_embeds=perturbed,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_mask = response_mask[..., 1:].contiguous().float()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                )
                loss = (loss * shift_mask.view(-1)).sum() / shift_mask.sum().clamp(min=1)
                (-loss).backward()

            with torch.no_grad():
                if delta.grad is None:
                    continue
                grad = delta.grad * perturb_mask
                delta.data = delta.data + config.step_size * grad.sign()
                delta.data = torch.clamp(delta.data, -config.epsilon, config.epsilon)
                delta.data = (delta.data * perturb_mask.squeeze(-1).unsqueeze(-1)).to(dtype)
                delta.grad = None
    finally:
        model.train(was_training)

    return delta.detach()
