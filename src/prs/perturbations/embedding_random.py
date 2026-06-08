"""Random embedding noise fragility (no backward, memory-friendly)."""

from __future__ import annotations

import torch

from prs.teacher_force import TeacherForceBatch, extract_logprobs_from_forward


@torch.no_grad()
def random_embedding_fragility(
    model,
    batch: TeacherForceBatch,
    epsilon: float = 0.005,
    num_samples: int = 3,
    mask: str = "answer",
) -> list[float]:
    """AD^E proxy: average logprob drop under random ||δ||≤ε."""
    embed_layer = model.get_input_embeddings()
    base = embed_layer(batch.input_ids).detach()
    dtype = base.dtype

    if mask == "answer":
        perturb_mask = batch.response_mask.unsqueeze(-1).float()
    elif mask == "query":
        perturb_mask = (1 - batch.response_mask).unsqueeze(-1).float()
    else:
        perturb_mask = torch.ones_like(base[..., :1])

    clean = extract_logprobs_from_forward(model, batch)
    drops: list[float] = []
    for _ in range(num_samples):
        delta = torch.randn_like(base) * epsilon * perturb_mask
        perturbed = (base + delta).to(dtype=dtype)
        noisy = extract_logprobs_from_forward(model, batch, input_embeds=perturbed)
        drops.append([max(0.0, c - n) for c, n in zip(clean, noisy)])

    # mean drop per position
    if not drops:
        return [0.0] * len(clean)
    stacked = torch.tensor(drops)
    return stacked.mean(dim=0).tolist()
