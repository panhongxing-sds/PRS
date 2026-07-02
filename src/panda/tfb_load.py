"""Load TFB checkpoint weights for deterministic teacher-forcing."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

_TFB_KEYS = (
    "bayes_sigma",
    "bayes_noise",
    "basis_idx",
    "sample",
    "num_samples",
    "lowrank",
    "init_basis_vectors",
    "use_tfb_for_generation",
    "base_lm_path",
)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _read_config(path: Path) -> dict:
    return json.loads((path / "config.json").read_text(encoding="utf-8"))


def _strip_tfb_fields(cfg_dict: dict) -> dict:
    out = dict(cfg_dict)
    for k in _TFB_KEYS:
        out.pop(k, None)
    return out


def _infer_backbone(cfg_dict: dict, model_path: str) -> str:
    """Return one of: qwen3, qwen2, llama, mistral, phi3, opt, auto."""
    lowered = model_path.lower()
    if "phi" in lowered or "phi3" in lowered:
        return "phi3"
    if "opt" in lowered:
        return "opt"
    if "qwen3" in lowered:
        return "qwen3"
    if "qwen2" in lowered or "qwen25" in lowered or "qwen2.5" in lowered:
        return "qwen2"
    if "mistral" in lowered:
        return "mistral"
    if "llama" in lowered:
        return "llama"

    model_type = str(cfg_dict.get("model_type", "")).lower()
    if "phi3" in model_type or "phi" in model_type:
        return "phi3"
    if model_type == "opt":
        return "opt"
    if "qwen3" in model_type:
        return "qwen3"
    if "qwen2" in model_type:
        return "qwen2"
    if "mistral" in model_type:
        return "mistral"
    if "llama" in model_type:
        return "llama"

    hidden = cfg_dict.get("hidden_size")
    n_kv = cfg_dict.get("num_key_value_heads")
    if hidden == 4096 and n_kv == 8 and cfg_dict.get("head_dim") == 128:
        return "qwen3"
    return "auto"


def _build_backbone_config(cfg_dict: dict, backbone: str):
    if backbone == "phi3":
        cfg_dict["model_type"] = "phi3"
        cfg_dict["architectures"] = ["Phi3ForCausalLM"]
        return Phi3Config.from_dict(cfg_dict), Phi3ForCausalLM
    if backbone == "opt":
        cfg_dict["model_type"] = "opt"
        cfg_dict["architectures"] = ["OPTForCausalLM"]
        return OPTConfig.from_dict(cfg_dict), OPTForCausalLM
    if backbone == "qwen3":
        cfg_dict["model_type"] = "qwen3"
        cfg_dict["architectures"] = ["Qwen3ForCausalLM"]
        return Qwen3Config.from_dict(cfg_dict), Qwen3ForCausalLM
    if backbone == "qwen2":
        cfg_dict["model_type"] = "qwen2"
        cfg_dict["architectures"] = ["Qwen2ForCausalLM"]
        return Qwen2Config.from_dict(cfg_dict), Qwen2ForCausalLM
    if backbone in ("llama", "mistral"):
        cfg_dict["model_type"] = "llama"
        cfg_dict["architectures"] = ["LlamaForCausalLM"]
        return LlamaConfig.from_dict(cfg_dict), LlamaForCausalLM
    return None, AutoModelForCausalLM


def _safetensors_checkpoint(path: Path) -> Path | None:
    for name in ("model.safetensors",):
        candidate = path / name
        if candidate.is_file():
            return candidate
    shards = sorted(path.glob("model-*.safetensors"))
    return shards[0] if len(shards) == 1 else None


def _needs_manual_safetensors_load(path: Path) -> bool:
    checkpoint = _safetensors_checkpoint(path)
    if checkpoint is None:
        return False
    with safe_open(str(checkpoint), framework="pt") as handle:
        metadata = handle.metadata() or {}
    return metadata.get("format") not in (None, "pt", "tf", "flax", "mlx")


def load_tfb_for_teacher_force(
    model_path: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
):
    """
    Load a TFB checkpoint as a standard causal LM for deterministic forward.
    Extra ``basis_vectors`` tensors are ignored.
    """
    path = Path(model_path)
    cfg_dict = _strip_tfb_fields(_read_config(path))
    backbone = _infer_backbone(cfg_dict, model_path)
    config, model_cls = _build_backbone_config(cfg_dict, backbone)
    torch_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)
    gpu_id = int(device.split(":")[-1]) if "cuda" in device else 0

    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        config=config,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        dtype=torch_dtype,
        device_map={"": gpu_id},
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    if _needs_manual_safetensors_load(path):
        checkpoint = _safetensors_checkpoint(path)
        if checkpoint is None:
            raise FileNotFoundError(f"No safetensors checkpoint under {path}")
        model = model_cls(config)
        state_dict = load_file(str(checkpoint), device=device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(dtype=torch_dtype, device=device)
    elif model_cls is AutoModelForCausalLM:
        model = model_cls.from_pretrained(str(path), **load_kwargs)
    else:
        model = model_cls.from_pretrained(str(path), config=config, **load_kwargs)
    model.eval()
    return model, tokenizer


def load_tfb_qwen2_for_teacher_force(
    model_path: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
):
    """Backward-compatible alias; delegates to :func:`load_tfb_for_teacher_force`."""
    return load_tfb_for_teacher_force(model_path, device=device, dtype=dtype)
