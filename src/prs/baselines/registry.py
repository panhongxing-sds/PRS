"""Registry of ISO uncertainty baselines for PRS comparison tables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    """One row in the baseline comparison table."""

    id: str
    name: str
    key: str
    description: str
    source: str
    tier: str  # "token" | "sample" | "gpu" | "diagnostic"
    invert: bool = False  # True when higher score => more likely correct
    official: bool = False


# Keys written into features.jsonl / enriched rows by enrich_row_with_baselines().
BASELINE_REGISTRY: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        id="cot",
        name="CoT (Lower-Bound)",
        key="cot_greedy_acc",
        description="Greedy accuracy lower bound (not an uncertainty score; AUROC≈0.5)",
        source="—",
        tier="diagnostic",
        invert=True,
    ),
    BaselineSpec(
        id="se",
        name="SE (Semantic Entropy)",
        key="baseline_SE_H",
        description="Shannon entropy over math_equal clusters (high-temp samples)",
        source="jlko/semantic_uncertainty",
        tier="sample",
        official=True,
    ),
    BaselineSpec(
        id="sar",
        name="SAR",
        key="baseline_SAR",
        description="Token relevance–weighted entropy (approx Token-SAR)",
        source="jinhaoduan/SAR",
        tier="token",
    ),
    BaselineSpec(
        id="u_ecc",
        name="U_Ecc",
        key="baseline_U_Ecc",
        description="Eccentricity from graph Laplacian (math_equal similarity)",
        source="zlin7/UQ-NLG",
        tier="sample",
        official=True,
    ),
    BaselineSpec(
        id="u_deg",
        name="U_Deg",
        key="baseline_U_Deg",
        description="Degree-matrix trace uncertainty (math_equal similarity)",
        source="zlin7/UQ-NLG",
        tier="sample",
        official=True,
    ),
    BaselineSpec(
        id="p_true",
        name="P(True)",
        key="baseline_P_True",
        description="P(answer is True) from prompted follow-up (GPU)",
        source="jlko/semantic_uncertainty",
        tier="gpu",
        official=True,
        invert=True,
    ),
    BaselineSpec(
        id="inside",
        name="INSIDE",
        key="baseline_INSIDE",
        description="EigenScore from hidden-state covariance (GPU)",
        source="D2I-ai/eigenscore",
        tier="gpu",
        official=True,
    ),
    BaselineSpec(
        id="pe",
        name="PE (Predictive Entropy)",
        key="baseline_PE_mean",
        description="Mean token entropy on base greedy response",
        source="lm-polygraph MeanTokenEntropy",
        tier="token",
        official=True,
    ),
    BaselineSpec(
        id="ll",
        name="LL (Log-Likelihood)",
        key="baseline_LL_nll",
        description="Mean NLL = -mean(logprob) on base response",
        source="lm-polygraph Perplexity",
        tier="token",
        official=True,
    ),
    BaselineSpec(
        id="self_certainty",
        name="Self-Certainty",
        key="baseline_SC_mean",
        description="TokUR self-certainty from saved top-k (approx)",
        source="backprop07/Self-Certainty",
        tier="token",
        invert=True,
    ),
    BaselineSpec(
        id="deepconf",
        name="DeepConf",
        key="baseline_DC_mean",
        description="TokUR token confidence k=40 from saved top-k",
        source="facebookresearch/deepconf",
        tier="token",
        invert=True,
    ),
)

BASELINE_INVERT_KEYS = frozenset(b.key for b in BASELINE_REGISTRY if b.invert)

BASELINE_BY_KEY = {b.key: b for b in BASELINE_REGISTRY}
BASELINE_TABLE_ROWS = [
    (b.name, b.key, b.description) for b in BASELINE_REGISTRY if b.tier != "diagnostic"
]
