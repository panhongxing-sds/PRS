"""Length-normalized teacher-forced token scores for final answers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from prs.teacher_force import build_teacher_force_batch, extract_clean_stats

ANSWER_PREFIX = "\n\nFinal answer:\n"


@dataclass
class AnswerTokenScore:
    len_norm_logprob: float
    num_tokens: int
    token_logprobs: list[float]
    token_entropies: list[float]
    token_margins: list[float]


def build_scoring_prompt(question: str) -> str:
    return f"Question:\n{question.strip()}{ANSWER_PREFIX}"


def _token_margins(logits_list: list[np.ndarray], token_ids: list[int]) -> list[float]:
    margins: list[float] = []
    for dist, tid in zip(logits_list, token_ids):
        # dist is probs; convert to log probs for margin
        logp = np.log(dist + 1e-12)
        sorted_lp = np.sort(logp)[::-1]
        if len(sorted_lp) < 2:
            margins.append(float(sorted_lp[0]))
        else:
            margins.append(float(sorted_lp[0] - sorted_lp[1]))
    return margins


def score_answer_tokens(model, tokenizer, question: str, answer: str) -> AnswerTokenScore:
    """Teacher-force score only the answer tokens under the scoring prompt."""
    answer = (answer or "").strip()
    if not answer:
        return AnswerTokenScore(float("nan"), 0, [], [], [])

    prompt = build_scoring_prompt(question)
    batch = build_teacher_force_batch(tokenizer, prompt, answer, model.device)
    logprobs, distributions, token_ids, _, entropies = extract_clean_stats(model, batch)
    margins = _token_margins(distributions, token_ids)

    if not logprobs:
        return AnswerTokenScore(float("nan"), 0, [], [], [])

    return AnswerTokenScore(
        len_norm_logprob=float(np.mean(logprobs)),
        num_tokens=len(logprobs),
        token_logprobs=[float(x) for x in logprobs],
        token_entropies=[float(x) for x in entropies],
        token_margins=margins,
    )
