"""Model loading helpers (Qwen3 trust_remote_code, weight-target auto-detect)."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def infer_weight_target_suffixes(model: nn.Module) -> tuple[str, ...]:
    """
    Pick attention projection suffixes for low-rank weight perturbation.
    Phi-3: qkv_proj; Qwen/Llama/Mistral: q_proj, k_proj, v_proj, o_proj.
    """
    candidates = ("qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj")
    counts = {s: 0 for s in candidates}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for s in candidates:
            if name.endswith(s):
                counts[s] += 1

    if counts["qkv_proj"] > 0:
        out = ["qkv_proj"]
        if counts["o_proj"] > 0:
            out.append("o_proj")
        logger.info("Weight perturb targets (Phi-style): %s (%d qkv layers)", out, counts["qkv_proj"])
        return tuple(out)

    qwen_style = tuple(s for s in ("q_proj", "k_proj", "v_proj", "o_proj") if counts[s] > 0)
    if qwen_style:
        logger.info("Weight perturb targets (Qwen/Llama-style): %s", qwen_style)
        return qwen_style

    logger.warning("No known attention projections found; falling back to q_proj, k_proj")
    return ("q_proj", "k_proj")


def count_matched_linear_layers(model: nn.Module, suffixes: tuple[str, ...]) -> int:
    n = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(s) for s in suffixes):
            n += 1
    return n


def load_tokenizer(model_path: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    load_in_4bit: bool = False,
) -> AutoModelForCausalLM:
    """Load HF causal LM; requires transformers>=4.51 for Qwen3 (model_type=qwen3)."""
    torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
    gpu_id = int(device.split(":")[-1]) if "cuda" in device else 0
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        kwargs["device_map"] = {"": gpu_id}
        kwargs["load_in_4bit"] = True
    else:
        kwargs["dtype"] = torch_dtype
        kwargs["device_map"] = {"": gpu_id}

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model


def resolve_weight_suffixes(
    model: nn.Module,
    configured: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Use config if it matches layers; otherwise auto-detect (fixes Qwen + qkv_proj bug)."""
    if configured:
        cfg = tuple(configured)
        if count_matched_linear_layers(model, cfg) > 0:
            return cfg
        logger.warning(
            "Configured weight targets %s match 0 layers — auto-detecting.",
            cfg,
        )
    return infer_weight_target_suffixes(model)
