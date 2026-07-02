"""Response-level feature extraction from token scores."""

from __future__ import annotations

from dataclasses import dataclass, field

from atokur.types import TokenFeatures


@dataclass
class ResponseFeatures:
    id: str
    dataset: str
    is_correct: bool
    num_tokens: int = 0
    # sum aggregates
    ll_sum: float = 0.0
    entropy_sum: float = 0.0
    eu_weight_sum: float = 0.0
    ad_emb_sum: float = 0.0
    ad_hidden_sum: float = 0.0
    mc_hidden_sum: float = 0.0
    # mean aggregates
    ll_mean: float = 0.0
    entropy_mean: float = 0.0
    eu_weight_mean: float = 0.0
    ad_emb_mean: float = 0.0
    ad_hidden_mean: float = 0.0
    mc_hidden_mean: float = 0.0
    # per-layer hidden (ablation)
    ad_hidden_by_layer: dict[str, float] = field(default_factory=dict)
    mc_hidden_by_layer: dict[str, float] = field(default_factory=dict)
    # raw for calibration
    eu_weight_raw: float = 0.0
    ad_hidden_raw: float = 0.0
    mc_hidden_raw: float = 0.0

    def to_flat_dict(self) -> dict[str, float]:
        d = {
            "ll_sum": self.ll_sum,
            "ll_mean": self.ll_mean,
            "entropy_sum": self.entropy_sum,
            "entropy_mean": self.entropy_mean,
            "tokur_eu_sum": self.eu_weight_sum,
            "tokur_eu_mean": self.eu_weight_mean,
            "ad_emb_sum": self.ad_emb_sum,
            "ad_emb_mean": self.ad_emb_mean,
            "ad_hidden_sum": self.ad_hidden_sum,
            "ad_hidden_mean": self.ad_hidden_mean,
            "mc_hidden_sum": self.mc_hidden_sum,
            "mc_hidden_mean": self.mc_hidden_mean,
            "eu_weight": self.eu_weight_raw,
            "ad_hidden": self.ad_hidden_raw,
            "mc_hidden": self.mc_hidden_raw,
        }
        for k, v in self.ad_hidden_by_layer.items():
            d[f"ad_hidden_{k}"] = v
        for k, v in self.mc_hidden_by_layer.items():
            d[f"mc_hidden_{k}"] = v
        return d


def tokens_to_response_features(
    record_id: str,
    dataset: str,
    is_correct: bool,
    tokens: list[TokenFeatures],
    hidden_layer_scores: dict[str, list[float]] | None = None,
) -> ResponseFeatures:
    """Aggregate token-level signals to response level (sum + mean)."""
    n = max(len(tokens), 1)
    rf = ResponseFeatures(id=record_id, dataset=dataset, is_correct=is_correct, num_tokens=len(tokens))

    if not tokens:
        return rf

    rf.ll_sum = sum(-t.logprob_clean for t in tokens)
    rf.entropy_sum = sum(t.entropy_clean for t in tokens)
    rf.eu_weight_sum = sum(t.eu_weight for t in tokens)
    rf.ad_emb_sum = sum(t.ad_embedding for t in tokens)
    rf.ad_hidden_sum = sum(t.ad_hidden for t in tokens)
    rf.mc_hidden_sum = sum(t.mc_hidden for t in tokens)

    rf.ll_mean = rf.ll_sum / n
    rf.entropy_mean = rf.entropy_sum / n
    rf.eu_weight_mean = rf.eu_weight_sum / n
    rf.ad_emb_mean = rf.ad_emb_sum / n
    rf.ad_hidden_mean = rf.ad_hidden_sum / n
    rf.mc_hidden_mean = rf.mc_hidden_sum / n

    rf.eu_weight_raw = rf.eu_weight_sum
    rf.ad_hidden_raw = rf.ad_hidden_sum
    rf.mc_hidden_raw = rf.mc_hidden_sum

    if hidden_layer_scores:
        for layer_name, ad_vals in hidden_layer_scores.items():
            s = float(sum(ad_vals)) if ad_vals else 0.0
            rf.ad_hidden_by_layer[layer_name] = s
        # max/mean across layers for ablation reporting
        if rf.ad_hidden_by_layer:
            vals = list(rf.ad_hidden_by_layer.values())
            rf.ad_hidden_by_layer["hidden_max"] = max(vals)
            rf.ad_hidden_by_layer["hidden_mean"] = sum(vals) / len(vals)

    return rf
