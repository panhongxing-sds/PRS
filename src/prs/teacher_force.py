"""Teacher-forced scoring on a fixed (prompt, response) pair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TeacherForceBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor  # 1 on response token positions in input_ids
    prompt_len: int
    response_positions: list[int]  # indices in input_ids for each response token


def build_teacher_force_batch(
    tokenizer,
    prompt: str,
    response: str,
    device: torch.device,
    max_length: int = 2048,
) -> TeacherForceBatch:
    """Tokenize [prompt + response]; label only response tokens."""
    enc = tokenizer(
        prompt + response,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    enc_prompt = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    full = enc["input_ids"][0].tolist()
    prompt_ids = enc_prompt["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    if prompt_len >= len(full):
        prompt_len = max(1, len(full) - 1)

    input_ids = torch.tensor([full], device=device)
    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    response_mask = torch.zeros_like(input_ids)
    response_mask[0, prompt_len:] = 1

    response_positions = list(range(prompt_len, len(full)))

    return TeacherForceBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        response_mask=response_mask,
        prompt_len=prompt_len,
        response_positions=response_positions,
    )


@torch.no_grad()
def extract_clean_stats(
    model,
    batch: TeacherForceBatch,
) -> tuple[list[float], list[np.ndarray], list[int], list[str], list[float]]:
    """
    Returns per response token:
      logprobs, full logits (as numpy), token_ids, token_texts, entropies
    """
    out = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    )
    logits = out.logits[0]
    logprobs: list[float] = []
    distributions: list[np.ndarray] = []
    token_ids: list[int] = []
    entropies: list[float] = []

    for pos in batch.response_positions:
        pred_pos = pos - 1
        if pred_pos < 0:
            continue
        log_p = F.log_softmax(logits[pred_pos], dim=-1)
        tid = int(batch.input_ids[0, pos].item())
        token_ids.append(tid)
        logprobs.append(float(log_p[tid].item()))
        probs = torch.softmax(logits[pred_pos], dim=-1).cpu().float().numpy()
        distributions.append(probs)
        entropies.append(float(-(probs * np.log(probs + 1e-12)).sum()))

    return logprobs, distributions, token_ids, [], entropies


@torch.no_grad()
def extract_logprobs_from_forward(
    model,
    batch: TeacherForceBatch,
    input_embeds: torch.Tensor | None = None,
    hidden_delta_fn=None,
) -> list[float]:
    """Log p(y_t) for each response token after optional embed / hidden perturbation."""
    if hidden_delta_fn is not None:
        out = hidden_delta_fn()
    elif input_embeds is not None:
        out = model(
            inputs_embeds=input_embeds,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )
    else:
        out = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )

    logits = out.logits[0]
    logprobs: list[float] = []
    for pos in batch.response_positions:
        pred_pos = pos - 1
        log_p = F.log_softmax(logits[pred_pos], dim=-1)
        tid = int(batch.input_ids[0, pos].item())
        logprobs.append(float(log_p[tid].item()))
    return logprobs


@torch.no_grad()
def extract_logits_clean(model, batch: TeacherForceBatch) -> list[np.ndarray]:
    out = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    )
    logits = out.logits[0]
    result: list[np.ndarray] = []
    for pos in batch.response_positions:
        result.append(logits[pos - 1].cpu().float().numpy())
    return result
