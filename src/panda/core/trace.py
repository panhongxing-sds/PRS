"""Per-token generation traces (top-k logprobs, entropy, margins)."""

from __future__ import annotations

import math
import re

import numpy as np
import torch


def entropy_from_logprobs(log_probs: torch.Tensor) -> float:
    p = log_probs.exp()
    return float(-(p * log_probs).sum().item())


def entropy_topk_renorm(log_probs: torch.Tensor, k: int) -> float:
    topv, _ = torch.topk(log_probs, k=min(k, log_probs.numel()))
    p = topv.exp()
    p = p / p.sum()
    return float(-(p * torch.log(p + 1e-12)).sum().item())


def topk_pairs(log_probs: torch.Tensor, token_ids: torch.Tensor, tokenizer, k: int) -> list[list]:
    topv, topi = torch.topk(log_probs, k=min(k, log_probs.numel()))
    out = []
    for lp, tid in zip(topv.tolist(), topi.tolist()):
        tok = tokenizer.decode([int(tid)])
        out.append([tok, float(lp)])
    return out


def find_answer_span(token_texts: list[str], response_text: str) -> dict:
    """Heuristic: locate \\boxed{...} or last numeric line in token stream."""
    boxed = re.search(r"\\boxed\{", response_text)
    if boxed:
        # map char offset to token indices (approximate via cumulative decode lengths)
        start_char = boxed.start()
        end_marker = response_text.find("}", boxed.end())
        if end_marker < 0:
            end_char = len(response_text)
        else:
            end_char = end_marker + 1
        pos = 0
        start_tok = 0
        end_tok = len(token_texts) - 1
        for i, t in enumerate(token_texts):
            pos += len(t)
            if start_tok == 0 and pos >= start_char:
                start_tok = i
            if pos >= end_char:
                end_tok = i
                break
        return {"start_token": start_tok, "end_token": end_tok, "method": "boxed"}
    return {"start_token": max(0, len(token_texts) - 8), "end_token": len(token_texts) - 1, "method": "tail8"}


def build_token_trace(
    gen_ids: torch.Tensor,
    scores: tuple,
    tokenizer,
    *,
    topk_save: int = 20,
) -> list[dict]:
    trace: list[dict] = []
    token_texts: list[str] = []
    for pos, (tid, score) in enumerate(zip(gen_ids.tolist(), scores)):
        logits = score[0].float() if score.dim() > 1 else score.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        tid = int(tid)
        tok = tokenizer.decode([tid])
        token_texts.append(tok)

        if log_probs.numel() >= 2:
            top2 = torch.topk(log_probs, k=2)
            margin = float(top2.values[0] - top2.values[1])
        else:
            margin = 0.0

        trace.append(
            {
                "pos": pos,
                "token": tok,
                "token_id": tid,
                "logprob": float(log_probs[tid].item()),
                "entropy": entropy_from_logprobs(log_probs),
                "entropy_topk": entropy_topk_renorm(log_probs, topk_save),
                "margin_top2": margin,
                "rank": int((log_probs >= log_probs[tid]).sum().item()),
                "topk": topk_pairs(log_probs, torch.arange(log_probs.numel()), tokenizer, topk_save),
            }
        )
    return trace, token_texts
