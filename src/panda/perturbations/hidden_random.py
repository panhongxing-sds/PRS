"""Random hidden-state noise fragility (no backward)."""

from __future__ import annotations

import torch

from panda.perturbations.hidden_attack import forward_with_hidden_delta
from panda.teacher_force import TeacherForceBatch, extract_logprobs_from_forward


@torch.no_grad()
def random_hidden_fragility(
    model,
    batch: TeacherForceBatch,
    layer_idx: int,
    epsilon: float = 0.01,
    num_samples: int = 3,
) -> list[float]:
    """AD^H proxy: mean logprob drop under random hidden noise at layer ℓ."""
    clean = extract_logprobs_from_forward(model, batch)
    T = len(clean)
    if T == 0:
        return []

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "n_embd", 4096)
    seq_len = batch.input_ids.shape[1]
    device = batch.input_ids.device
    dtype = next(model.parameters()).dtype

    accum = torch.zeros(T, device="cpu")
    for _ in range(num_samples):
        delta = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype) * epsilon
        logprobs_h = extract_logprobs_from_forward(
            model,
            batch,
            hidden_delta_fn=lambda d=delta: forward_with_hidden_delta(
                model,
                batch.input_ids,
                batch.attention_mask,
                layer_idx,
                d,
            ),
        )
        for t in range(T):
            accum[t] += max(0.0, clean[t] - logprobs_h[t])

    return (accum / max(num_samples, 1)).tolist()
