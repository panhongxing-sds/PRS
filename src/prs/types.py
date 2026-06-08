from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenFeatures:
    """Per-token feature vector φ_t."""

    index: int  # position in response (0 .. T-1)
    token_id: int
    token_text: str
    logprob_clean: float
    nll: float
    entropy_clean: float
    eu_weight: float = 0.0
    ad_embedding: float = 0.0
    ad_hidden: float = 0.0
    mc_embedding: float = 0.0
    mc_hidden: float = 0.0
    score: float = 0.0
    is_corrupted: bool | None = None  # synthetic eval label


@dataclass
class ScoreResult:
    id: str
    prompt: str
    response: str
    tokens: list[TokenFeatures] = field(default_factory=list)
    response_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
