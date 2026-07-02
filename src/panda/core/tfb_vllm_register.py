"""Register TokUR TFB architectures with vLLM + Transformers (stock vLLM 0.10.x).

Avoids importing the full ``bayesian_transformer`` package, which pulls in HF
model code incompatible with newer ``transformers`` releases.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

_REGISTERED = False
_VLLM_MODELS = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "TokUR"
    / "bayesian_transformer"
    / "bayesian_transformer"
    / "vllm_models"
)


def _needs_tfb_registration(model_path: str | Path) -> bool:
    cfg = Path(model_path) / "config.json"
    if not cfg.is_file():
        return False
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    mt = str(data.get("model_type", ""))
    archs = data.get("architectures") or []
    return mt.startswith("tfb_") or any("TFB" in str(a) for a in archs)


def register_tfb_vllm_models() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from transformers import AutoConfig, AutoTokenizer, TOKENIZER_MAPPING, Qwen2Config
    from vllm import ModelRegistry

    base = _VLLM_MODELS
    for name in ("bayesian_transformer", "bayesian_transformer.vllm_models"):
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)

    def _load(fullname: str, path: Path):
        spec = importlib.util.spec_from_file_location(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = mod
        spec.loader.exec_module(mod)
        return mod

    llama = _load("bayesian_transformer.vllm_models.tfb_llama_vllm", base / "tfb_llama_vllm.py")
    qwen = _load("bayesian_transformer.vllm_models.tfb_qwen2_vllm", base / "tfb_qwen2_vllm.py")

    from transformers import LlamaConfig

    AutoConfig.register(llama.TFBLlamaConfig.model_type, llama.TFBLlamaConfig)
    AutoTokenizer.register(llama.TFBLlamaConfig, *TOKENIZER_MAPPING[LlamaConfig])
    ModelRegistry.register_model("VllmTFBLLamaForCausalLM", llama.LlamaForCausalLM)

    AutoConfig.register(qwen.TFBQwen2Config.model_type, qwen.TFBQwen2Config)
    AutoTokenizer.register(qwen.TFBQwen2Config, *TOKENIZER_MAPPING[Qwen2Config])
    ModelRegistry.register_model("VllmTFBQwen2ForCausalLM", qwen.Qwen2ForCausalLM)
    _REGISTERED = True


def ensure_tfb_vllm_registered(model_path: str | Path) -> None:
    if _needs_tfb_registration(model_path):
        register_tfb_vllm_models()
