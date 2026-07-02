"""Bidirectional NLI clustering for official Semantic Entropy (Kuhn et al.).

Aligned with jlko/semantic_uncertainty:
  - DeBERTa MNLI: microsoft/deberta-v2-xlarge-mnli (override via PANDA_NLI_MODEL)
  - Non-strict bidirectional equivalence: no contradiction either way, not both neutral
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

DEFAULT_NLI_MODEL = "microsoft/deberta-v2-xlarge-mnli"
# DeBERTa MNLI label ids: 0=contradiction, 1=neutral, 2=entailment
CONTRADICTION = 0
NEUTRAL = 1
ENTAILMENT = 2


class EntailmentModel(Protocol):
    def check_implication(self, text1: str, text2: str) -> int: ...


class _DebertaEntailment:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device

    def _probs(self, text1: str, text2: str):
        inputs = self.tokenizer(text1, text2, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
            return self._torch.softmax(logits, dim=-1)[0]

    def check_implication(self, text1: str, text2: str) -> int:
        return int(self._probs(text1, text2).argmax().item())

    def entailment_probability(self, text1: str, text2: str) -> float:
        """P(entailment) for Lin et al. a_NLI,entail similarity."""
        return float(self._probs(text1, text2)[ENTAILMENT].item())


@lru_cache(maxsize=1)
def get_entailment_model() -> EntailmentModel:
    import torch

    model_id = os.environ.get("PANDA_NLI_MODEL", DEFAULT_NLI_MODEL)
    device = os.environ.get("PANDA_NLI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    return _DebertaEntailment(model_id, device)


def _are_semantically_equivalent(
    text1: str,
    text2: str,
    model: EntailmentModel,
    *,
    strict_entailment: bool = False,
    cache: dict[tuple[str, str], int] | None = None,
) -> bool:
    def _impl(a: str, b: str) -> int:
        key = (a, b)
        if cache is not None and key in cache:
            return cache[key]
        v = model.check_implication(a, b)
        if cache is not None:
            cache[key] = v
        return v

    i1 = _impl(text1, text2)
    i2 = _impl(text2, text1)
    if strict_entailment:
        return i1 == ENTAILMENT and i2 == ENTAILMENT
    implications = [i1, i2]
    return (CONTRADICTION not in implications) and (implications != [NEUTRAL, NEUTRAL])


def cluster_sequences_nli(
    sequences: list[str],
    *,
    strict_entailment: bool = False,
    model: EntailmentModel | None = None,
) -> tuple[list[int], dict[int, int]]:
    """Official bidirectional-entailment clustering (Algorithm 1, Kuhn et al.)."""
    n = len(sequences)
    if n == 0:
        return [], {}

    texts = [s.strip() for s in sequences]
    model = model or get_entailment_model()
    cache: dict[tuple[str, str], int] = {}

    semantic_ids = [-1] * n
    next_id = 0
    for i, s1 in enumerate(texts):
        if semantic_ids[i] != -1:
            continue
        semantic_ids[i] = next_id
        for j in range(i + 1, n):
            if semantic_ids[j] != -1:
                continue
            if _are_semantically_equivalent(
                s1, texts[j], model, strict_entailment=strict_entailment, cache=cache
            ):
                semantic_ids[j] = next_id
        next_id += 1

    sizes: dict[int, int] = {}
    for lab in semantic_ids:
        sizes[lab] = sizes.get(lab, 0) + 1
    return semantic_ids, sizes


def nli_entail_similarity_matrix(
    sequences: list[str],
    *,
    model: EntailmentModel | None = None,
) -> "np.ndarray":
    """Symmetric NLI entailment affinity W (Lin et al., UQ-NLG Eq. 4 style).

    W_ij = (a_NLI,entail(s_i, s_j) + a_NLI,entail(s_j, s_i)) / 2
    """
    import numpy as np

    texts = [s.strip() for s in sequences if (s or "").strip()]
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    model = model or get_entailment_model()
    w = np.eye(n, dtype=float)
    cache: dict[tuple[str, str], float] = {}

    def _p(a: str, b: str) -> float:
        key = (a, b)
        if key not in cache:
            cache[key] = float(getattr(model, "entailment_probability")(a, b))
        return cache[key]

    for i in range(n):
        for j in range(i + 1, n):
            sim = 0.5 * (_p(texts[i], texts[j]) + _p(texts[j], texts[i]))
            w[i, j] = w[j, i] = sim
    return w
