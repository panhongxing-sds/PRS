"""HuggingFace causal LM wrapper for ATokUR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelConfig:
    name_or_path: str
    device: str = "cuda"
    dtype: str = "bfloat16"


class CausalLMWrapper:
    def __init__(
        self,
        config: ModelConfig,
        model: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
    ):
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)
        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            from prs.model_load import load_causal_lm, load_tokenizer

            self.tokenizer = load_tokenizer(config.name_or_path)
            if config.device == "cpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.name_or_path,
                    dtype=torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).to("cpu")
            else:
                self.model = load_causal_lm(
                    config.name_or_path,
                    device=config.device,
                    dtype=config.dtype,
                    trust_remote_code=True,
                )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])
