"""ATokUR main pipeline: teacher forcing + weight EU + embedding/hidden adversarial fragility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from prs.aggregate import aggregate_response_score
from prs.model import CausalLMWrapper, ModelConfig
from prs.perturbations.embedding_attack import EmbeddingAttackConfig, global_embedding_pgd
from prs.perturbations.embedding_random import random_embedding_fragility
from prs.perturbations.hidden_attack import (
    HiddenAttackConfig,
    forward_with_hidden_delta,
    global_hidden_pgd,
    layer_indices_from_fractions,
    resolve_layer_modules,
)
from prs.perturbations.hidden_random import random_hidden_fragility
from prs.perturbations.weight import LowRankWeightPerturbation, WeightPerturbConfig
from prs.scores import (
    adversarial_drop,
    compute_token_scores,
    epistemic_per_position,
    log_margin,
    margin_collapse,
)
from prs.teacher_force import (
    build_teacher_force_batch,
    extract_clean_stats,
    extract_logprobs_from_forward,
    extract_logits_clean,
)
from prs.types import ScoreResult, TokenFeatures


@dataclass
class PipelineConfig:
    weight: WeightPerturbConfig = field(default_factory=WeightPerturbConfig)
    embedding: EmbeddingAttackConfig = field(default_factory=EmbeddingAttackConfig)
    hidden: HiddenAttackConfig = field(default_factory=HiddenAttackConfig)
    use_weight: bool = True
    use_embedding: bool = True
    use_hidden: bool = True
    use_margin: bool = True
    embedding_use_pgd: bool = True
    hidden_use_pgd: bool = True
    hidden_layer_fractions: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    hidden_aggregate: str = "max"  # max | mean | L4 | L2 | 3L4
    zscore_within_response: bool = False
    aggregation: str = "mean"
    topk: int = 5
    teacher_force_max_length: int = 2048

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "PipelineConfig":
        w, e, h = cfg.get("weight", {}), cfg.get("embedding_attack", {}), cfg.get("hidden_attack", {})
        sc = cfg.get("scoring", {})
        ag = cfg.get("aggregation", {})
        return cls(
            weight=WeightPerturbConfig(
                sigma=float(w.get("sigma", 0.1)),
                rank=int(w.get("rank", 8)),
                num_samples=int(w.get("num_samples", 8)),
                target_suffixes=tuple(w.get("target_suffixes", ["q_proj", "k_proj"])),
            ),
            embedding=EmbeddingAttackConfig(
                epsilon=float(e.get("epsilon", 0.01)),
                steps=int(e.get("steps", 5)),
                step_size=float(e.get("step_size", 0.002)),
                mask=str(e.get("mask", "answer")),
            ),
            hidden=HiddenAttackConfig(
                epsilon=float(h.get("epsilon", 0.01)),
                steps=int(h.get("steps", 5)),
                step_size=float(h.get("step_size", 0.002)),
            ),
            use_weight=bool(w.get("enabled", True)),
            use_embedding=bool(e.get("enabled", True)),
            use_hidden=bool(h.get("enabled", True)),
            use_margin=bool(sc.get("use_margin_collapse", True)),
            embedding_use_pgd=bool(e.get("use_pgd", False)),
            hidden_use_pgd=bool(h.get("use_pgd", True)),
            hidden_layer_fractions=list(h.get("layer_fractions", [0.25, 0.5, 0.75])),
            hidden_aggregate=str(h.get("aggregate", "max")),
            zscore_within_response=bool(sc.get("zscore_within_response", False)),
            aggregation=str(ag.get("response", "mean")),
            topk=int(ag.get("topk", 5)),
            teacher_force_max_length=int(cfg.get("teacher_force_max_length", 2048)),
        )


class ATokURPipeline:
    def __init__(self, model: CausalLMWrapper | None = None, config: PipelineConfig | None = None):
        self.model = model
        self.config = config or PipelineConfig()
        self._weight_perturb: LowRankWeightPerturbation | None = None

    def score(
        self,
        record_id: str,
        prompt: str,
        response: str,
        token_labels: list[bool] | None = None,
    ) -> ScoreResult:
        if self.model is None:
            return self._score_synthetic(record_id, prompt, response, token_labels)

        return self._score_with_model(record_id, prompt, response, token_labels)

    def _score_with_model(
        self,
        record_id: str,
        prompt: str,
        response: str,
        token_labels: list[bool] | None,
    ) -> ScoreResult:
        m = self.model
        batch = build_teacher_force_batch(
            m.tokenizer,
            prompt,
            response,
            m.device,
            max_length=self.config.teacher_force_max_length,
        )

        logprobs_clean, dists_clean, token_ids, _, entropies = extract_clean_stats(m.model, batch)
        logits_clean = extract_logits_clean(m.model, batch)
        T = len(logprobs_clean)

        # --- Weight perturbation: EU^W per position ---
        eu_w = [0.0] * T
        if self.config.use_weight:
            wp = self._get_weight_perturb()
            dists_per_sample: list[list[np.ndarray]] = [[] for _ in range(T)]
            for i in wp.iterate():
                with wp.sample(seed=i):
                    _, dists, _, _, _ = extract_clean_stats(m.model, batch)
                    for t in range(T):
                        dists_per_sample[t].append(dists[t])
            eu_w = epistemic_per_position(dists_per_sample)

        # --- Embedding global PGD ---
        ad_e = [0.0] * T
        mc_e = [0.0] * T
        if self.config.use_embedding:
            if self.config.embedding_use_pgd:
                for p in m.model.parameters():
                    p.requires_grad_(False)
                m.model.zero_grad(set_to_none=True)
                delta = global_embedding_pgd(
                    m.model,
                    batch.input_ids,
                    batch.labels,
                    batch.response_mask,
                    batch.attention_mask,
                    self.config.embedding,
                )
                embed = m.model.get_input_embeddings()(batch.input_ids)
                perturbed_embeds = embed + delta
                logprobs_e = extract_logprobs_from_forward(m.model, batch, input_embeds=perturbed_embeds)
                for t in range(T):
                    ad_e[t] = adversarial_drop(logprobs_clean[t], logprobs_e[t])
                if self.config.use_margin:
                    out_adv = m.model(
                        inputs_embeds=perturbed_embeds,
                        attention_mask=batch.attention_mask,
                        use_cache=False,
                    )
                    for t, pos in enumerate(batch.response_positions):
                        adv_logits = out_adv.logits[0, pos - 1].cpu().float().numpy()
                        mc_e[t] = margin_collapse(
                            log_margin(logits_clean[t], token_ids[t]),
                            log_margin(adv_logits, token_ids[t]),
                        )
                for p in m.model.parameters():
                    p.requires_grad_(False)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                ad_e = random_embedding_fragility(
                    m.model,
                    batch,
                    epsilon=self.config.embedding.epsilon,
                    num_samples=3,
                    mask=self.config.embedding.mask,
                )

        # --- Hidden attack: per-layer AD/MC + aggregate (max/mean/L4/L2/3L4) ---
        ad_h = [0.0] * T
        mc_h = [0.0] * T
        hidden_layer_ad: dict[str, list[float]] = {}
        hidden_layer_mc: dict[str, list[float]] = {}
        if self.config.use_hidden:
            fractions = self.config.hidden_layer_fractions
            layer_ids = layer_indices_from_fractions(m.model, fractions)
            n_layers = len(resolve_layer_modules(m.model))
            labels = []
            for f, ell in zip(fractions, layer_ids):
                if abs(f - 0.25) < 1e-6:
                    labels.append("L4")
                elif abs(f - 0.5) < 1e-6:
                    labels.append("L2")
                elif abs(f - 0.75) < 1e-6:
                    labels.append("3L4")
                else:
                    labels.append(f"L{ell}")

            ad_per_layer: list[list[float]] = []
            mc_per_layer: list[list[float]] = []
            for ell, lname in zip(layer_ids, labels):
                if self.config.hidden_use_pgd:
                    for p in m.model.parameters():
                        p.requires_grad_(False)
                    m.model.zero_grad(set_to_none=True)
                    delta_h = global_hidden_pgd(
                        m.model,
                        batch.input_ids,
                        batch.labels,
                        batch.response_mask,
                        batch.attention_mask,
                        ell,
                        self.config.hidden,
                    )

                    def _fwd(d=delta_h, layer=ell):
                        return forward_with_hidden_delta(
                            m.model,
                            batch.input_ids,
                            batch.attention_mask,
                            layer,
                            d,
                        )

                    logprobs_h = extract_logprobs_from_forward(
                        m.model, batch, hidden_delta_fn=_fwd
                    )
                else:
                    ad_rand = random_hidden_fragility(
                        m.model,
                        batch,
                        layer_idx=ell,
                        epsilon=self.config.hidden.epsilon,
                        num_samples=3,
                    )
                    logprobs_h = [
                        logprobs_clean[t] - ad_rand[t] for t in range(T)
                    ]
                    _fwd = None

                ad_layer = [
                    adversarial_drop(logprobs_clean[t], logprobs_h[t]) for t in range(T)
                ]
                ad_per_layer.append(ad_layer)
                hidden_layer_ad[lname] = ad_layer

                if self.config.use_margin and self.config.hidden_use_pgd:
                    out_h = _fwd()
                    mc_layer = []
                    for t, pos in enumerate(batch.response_positions):
                        adv_logits = out_h.logits[0, pos - 1].cpu().float().numpy()
                        mc_layer.append(
                            margin_collapse(
                                log_margin(logits_clean[t], token_ids[t]),
                                log_margin(adv_logits, token_ids[t]),
                            )
                        )
                    mc_per_layer.append(mc_layer)
                    hidden_layer_mc[lname] = mc_layer

            agg = self.config.hidden_aggregate.lower()
            if agg == "mean" and ad_per_layer:
                ad_h = [
                    sum(ad_per_layer[L][t] for L in range(len(ad_per_layer)))
                    / len(ad_per_layer)
                    for t in range(T)
                ]
            else:
                key_map = {"l4": "L4", "l2": "L2", "3l4": "3L4"}
                k = key_map.get(agg, agg)
                if k in hidden_layer_ad:
                    ad_h = list(hidden_layer_ad[k])
                elif ad_per_layer:
                    ad_h = [
                        max(ad_per_layer[L][t] for L in range(len(ad_per_layer)))
                        for t in range(T)
                    ]

            if mc_per_layer:
                if agg == "mean":
                    mc_h = [
                        sum(mc_per_layer[L][t] for L in range(len(mc_per_layer)))
                        / len(mc_per_layer)
                        for t in range(T)
                    ]
                elif agg in ("l4", "l2", "3l4", "L4", "L2", "3L4"):
                    key_map = {"l4": "L4", "l2": "L2", "3l4": "3L4"}
                    k = key_map.get(agg.lower(), agg)
                    if k in hidden_layer_mc:
                        mc_h = list(hidden_layer_mc[k])
                else:
                    mc_h = [
                        max(mc_per_layer[L][t] for L in range(len(mc_per_layer)))
                        for t in range(T)
                    ]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tokens: list[TokenFeatures] = []
        for t in range(T):
            label = token_labels[t] if token_labels and t < len(token_labels) else None
            tokens.append(
                TokenFeatures(
                    index=t,
                    token_id=token_ids[t],
                    token_text=m.decode_token(token_ids[t]),
                    logprob_clean=logprobs_clean[t],
                    nll=-logprobs_clean[t],
                    entropy_clean=entropies[t],
                    eu_weight=eu_w[t],
                    ad_embedding=ad_e[t],
                    ad_hidden=ad_h[t],
                    mc_embedding=mc_e[t],
                    mc_hidden=mc_h[t],
                    is_corrupted=label,
                )
            )

        compute_token_scores(tokens, zscore_within_response=self.config.zscore_within_response)
        resp_score = aggregate_response_score(
            tokens, mode=self.config.aggregation, topk=self.config.topk
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ScoreResult(
            id=record_id,
            prompt=prompt,
            response=response,
            tokens=tokens,
            response_score=resp_score,
            metadata={
                "hidden_layer_ad_tokens": hidden_layer_ad,
                "hidden_layer_mc_tokens": hidden_layer_mc,
            },
        )

    def _get_weight_perturb(self) -> LowRankWeightPerturbation:
        if self._weight_perturb is None:
            self._weight_perturb = LowRankWeightPerturbation(self.model.model, self.config.weight)
        return self._weight_perturb

    def _score_synthetic(
        self,
        record_id: str,
        prompt: str,
        response: str,
        token_labels: list[bool] | None,
    ) -> ScoreResult:
        """Demo without model: corrupted tokens get higher fragility."""
        ids = list(range(max(1, len(response.split()))))
        tokens: list[TokenFeatures] = []
        for i, _ in enumerate(ids):
            corrupt = token_labels[i] if token_labels and i < len(token_labels) else False
            tokens.append(
                TokenFeatures(
                    index=i,
                    token_id=i,
                    token_text=response.split()[i] if i < len(response.split()) else "?",
                    logprob_clean=-0.1 if not corrupt else -1.5,
                    nll=0.1 if not corrupt else 1.5,
                    entropy_clean=0.5 if not corrupt else 1.2,
                    eu_weight=0.05 if not corrupt else 0.25,
                    ad_embedding=0.02 if not corrupt else 0.35,
                    ad_hidden=0.03 if not corrupt else 0.40,
                    mc_embedding=0.01 if not corrupt else 0.20,
                    mc_hidden=0.02 if not corrupt else 0.30,
                    is_corrupted=corrupt,
                )
            )
        compute_token_scores(tokens, zscore_within_response=True)
        return ScoreResult(
            id=record_id,
            prompt=prompt,
            response=response,
            tokens=tokens,
            response_score=aggregate_response_score(tokens),
        )
