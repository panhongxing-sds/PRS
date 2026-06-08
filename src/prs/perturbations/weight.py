"""Low-rank weight perturbation on attention projections (TokUR-style)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Iterator

import torch
import torch.nn as nn


@dataclass
class WeightPerturbConfig:
    sigma: float = 0.1
    rank: int = 8
    num_samples: int = 8
    target_suffixes: tuple[str, ...] = ("q_proj", "k_proj")


class LowRankWeightPerturbation:
    """W' = W + U @ diag(ε) with U from SVD(W), ε ~ N(0, σ²)."""

    def __init__(self, model: nn.Module, config: WeightPerturbConfig):
        self.model = model
        self.config = config
        self._bases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._register_bases(model, config.target_suffixes)
        if not self._bases:
            from prs.model_load import infer_weight_target_suffixes

            auto = infer_weight_target_suffixes(model)
            if auto != config.target_suffixes:
                self._register_bases(model, auto)

    def _register_bases(self, model: nn.Module, suffixes: tuple[str, ...]) -> None:
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.endswith(s) for s in suffixes):
                continue
            w = module.weight.data.float()
            u, _, vh = torch.linalg.svd(w, full_matrices=False)
            r = min(self.config.rank, u.shape[1], vh.shape[0])
            self._bases[name] = (u[:, :r].to(module.weight.dtype), vh[:r, :].to(module.weight.dtype))

    @contextmanager
    def sample(self, seed: int | None = None) -> Generator[None, None, None]:
        if seed is not None:
            torch.manual_seed(seed)
        deltas: list[tuple[nn.Module, torch.Tensor]] = []
        sigma = self.config.sigma
        for name, module in self.model.named_modules():
            if name not in self._bases:
                continue
            u, vh = self._bases[name]
            r = u.shape[1]
            scale = sigma / (module.in_features**0.5)
            eps = torch.randn(r, device=module.weight.device, dtype=module.weight.dtype) * scale
            # Low-rank update without materializing full (out x in) when large
            delta_w = u @ torch.diag(eps) @ vh
            if delta_w.numel() > 4_000_000:
                for j in range(eps.shape[0]):
                    module.weight.data.add_(eps[j] * torch.outer(u[:, j], vh[j, :]))
            else:
                module.weight.data.add_(delta_w)
            deltas.append((module, delta_w))
        try:
            yield
        finally:
            for module, delta_w in reversed(deltas):
                module.weight.data.sub_(delta_w)

    def iterate(self) -> Iterator[int]:
        for i in range(self.config.num_samples):
            yield i
