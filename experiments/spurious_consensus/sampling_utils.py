"""模型加载与 prompt 构建。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from grading import extract_answer, is_correct  # noqa: F401 — re-export for sample.py

_PRS = Path(os.environ.get("PANDA_ROOT", "/root/PANDA")) / "src"
if str(_PRS) not in sys.path:
    sys.path.insert(0, str(_PRS))

_TFB_CFG_KEYS = (
    "basis_idx", "bayes_noise", "bayes_sigma", "sample", "num_samples",
    "lowrank", "init_basis_vectors", "use_tfb_for_generation", "auto_map",
    "base_lm_path",
)


def _read_config(model_path: str) -> dict:
    return json.load(open(os.path.join(model_path, "config.json"), encoding="utf-8"))


def clean_config_dict(cfgd: dict, model_path: str) -> dict:
    """剥离 TFB 字段，改回标准 architecture（供 vLLM / HF 共用）。

    仅对 TFB 魔改模型改写 architectures；非 TFB 原版模型保留原 config。
    """
    orig_arch = cfgd.get("architectures")
    orig_mt = str(cfgd.get("model_type", ""))
    is_tfb = "tfb" in orig_mt.lower() or any(k in cfgd for k in _TFB_CFG_KEYS)

    out = dict(cfgd)
    for k in _TFB_CFG_KEYS:
        out.pop(k, None)

    if not is_tfb:
        return out

    out.pop("architectures", None)
    out.pop("transformers_version", None)
    mt = orig_mt.lower()
    lp = model_path.lower()
    if "llama" in mt or "llama" in lp:
        out["model_type"] = "llama"
        out["architectures"] = ["LlamaForCausalLM"]
    elif "qwen" in mt or "qwen" in lp:
        is_qwen3 = (
            "qwen3" in lp
            or str(cfgd.get("head_dim")) == "128"
            and cfgd.get("hidden_size") == 4096
            and cfgd.get("num_key_value_heads") == 8
        )
        if is_qwen3:
            out["model_type"] = "qwen3"
            out["architectures"] = ["Qwen3ForCausalLM"]
        else:
            out["model_type"] = "qwen2"
            out["architectures"] = ["Qwen2ForCausalLM"]
    else:
        out["architectures"] = orig_arch
    return out


_TFB_WEIGHT_MARKERS = ("basis_vectors", "basis_idx", "tfb_")


def _is_tfb_weight_key(key: str) -> bool:
    return any(m in key for m in _TFB_WEIGHT_MARKERS)


def _clean_weights(src: Path, dst: Path) -> bool:
    """剔除 TFB 专属权重（如 basis_vectors），重写标准 safetensors。

    返回 True 表示已重写权重；False 表示无 TFB 权重，可直接软链原文件。
    """
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        return False
    from safetensors import safe_open
    from safetensors.torch import save_file

    has_tfb = False
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            if any(_is_tfb_weight_key(k) for k in f.keys()):
                has_tfb = True
                break
    if not has_tfb:
        return False

    for shard in shards:
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                if _is_tfb_weight_key(k):
                    continue
                tensors[k] = f.get_tensor(k)
        save_file(tensors, str(dst / shard.name), metadata={"format": "pt", **meta})
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        data = json.load(open(idx, encoding="utf-8"))
        wmap = data.get("weight_map", {})
        data["weight_map"] = {k: v for k, v in wmap.items() if not _is_tfb_weight_key(k)}
        json.dump(data, open(dst / idx.name, "w", encoding="utf-8"))
    return True


def prepare_vllm_model_path(model_path: str) -> str:
    """为 vLLM 准备可加载目录（洗 config + 剔除 TFB 权重）。"""
    src = Path(model_path).resolve()
    cfg_probe = clean_config_dict(_read_config(str(src)), str(src))
    arch_tag = (cfg_probe.get("architectures") or [""])[0]
    dst = src.parent / ".vllm_ready" / f"{src.name}.{arch_tag}"
    cfg_out = dst / "config.json"
    if cfg_out.exists():
        return str(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = clean_config_dict(_read_config(str(src)), str(src))
    rewrote = _clean_weights(src, dst)
    json.dump(cfg, open(cfg_out, "w", encoding="utf-8"), indent=2)
    skip = {"config.json"}
    if rewrote:
        skip |= {p.name for p in src.glob("*.safetensors")}
        skip.add("model.safetensors.index.json")
    for item in src.iterdir():
        if item.name in skip:
            continue
        link = dst / item.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(item.resolve())
    return str(dst)


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    trust = "phi" not in model_path.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def use_hf_backend(model_path: str) -> bool:
    """Gemma-3 多模态在 vLLM 上偶发兼容问题，默认走 HF；其余走 vLLM。"""
    if os.environ.get("FORCE_HF", "").strip() in ("1", "true", "yes"):
        return True
    if os.environ.get("FORCE_VLLM", "").strip() in ("1", "true", "yes"):
        return False
    mt = str(_read_config(model_path).get("model_type", "")).lower()
    return mt == "gemma3"


def resolve_model_path(model_path: str) -> str:
    """TFB checkpoint 洗 config 后供 HF 加载。"""
    mt = str(_read_config(model_path).get("model_type", "")).lower()
    if "tfb" in mt or "tfb" in model_path.lower():
        return prepare_vllm_model_path(model_path)
    return model_path


def build_std_config(model_path: str):
    from transformers import LlamaConfig, Phi3Config, Qwen2Config

    cfgd = _read_config(model_path)
    mt = str(cfgd.get("model_type", "")).lower()
    for k in ["architectures", "transformers_version", *_TFB_CFG_KEYS]:
        cfgd.pop(k, None)
    lp = model_path.lower()
    if "llama" in mt or "llama" in lp:
        cfgd["model_type"] = "llama"
        return LlamaConfig(**cfgd)
    if "qwen" in mt or "qwen" in lp:
        cfgd["model_type"] = "qwen2"
        return Qwen2Config(**cfgd)
    if "phi3" in mt or "phi" in lp:
        cfgd["model_type"] = "phi3"
        return Phi3Config(**cfgd)
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(resolve_model_path(model_path))


def load_model(model_path: str, device: str):
    """加载模型与 tokenizer；Gemma-3 多模态权重按纯文本推理。"""
    import torch
    from transformers import AutoModelForCausalLM

    mt = str(_read_config(model_path).get("model_type", "")).lower()
    resolved = resolve_model_path(model_path)

    if mt == "gemma3":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).to(device).eval()
        return model, tokenizer

    if "phi3" in mt or "phi" in model_path.lower():
        from transformers import Phi3ForCausalLM

        tokenizer = load_tokenizer(model_path)
        model = Phi3ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).to(device).eval()
        return model, tokenizer

    tokenizer = load_tokenizer(resolved)
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).to(device).eval()
    return model, tokenizer


def build_prompt(problem: str, tokenizer, model_path: str, grading: str) -> str:
    from panda.grading.tokur_records import build_prompt_tfb, _is_qwen_model

    if grading == "mcq":
        tail = (
            "\n\nReason step by step, then give your final answer as a single letter (A, B, C, or D) "
            "on the last line in the form: Answer: X"
        )
        prob = problem + tail
    else:
        prob = problem
    if _is_qwen_model(model_path, tokenizer):
        return build_prompt_tfb(prob, tokenizer, model_path)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prob}], tokenize=False, add_generation_prompt=True
    )
