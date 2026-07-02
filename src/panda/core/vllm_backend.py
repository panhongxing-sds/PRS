"""Optional vLLM batched backend for the plain-decoding routes (clean + text + SE).

Only the routes that are *pure decoding* are accelerated here:

* ``clean``  – greedy base generation (feeds PE/LL/SC/DeepConf/SAR baselines).
* ``text``   – text-rephrase perturbations R (feed F_resp / U_Ecc / T-branch ATU).
* ``SE``     – high-temperature i.i.d. samples (answers only).

The weight-perturbation branch (W), TokUR EU, and INSIDE stay on Hugging Face:
vLLM cannot inject per-sample low-rank weight noise nor expose hidden states.

Fidelity note
-------------
vLLM returns the top-``logprobs`` token logprobs per position, not the full vocab
distribution. We therefore approximate the full-vocab predictive entropy with a
top-k renormalised entropy. Exact quantities are preserved (chosen-token logprob,
top-k pairs, top-2 margin, rank), so LL / DeepConf / Self-Certainty match HF
closely; PE / SAR / T-branch ATU use the top-k entropy approximation.
"""

from __future__ import annotations

import math
from typing import Any

from panda.core.trace import find_answer_span
from panda.grading.math_grader import extract_math_answer

# vLLM caps per-token logprobs; keep requests within a safe bound.
_VLLM_MAX_LOGPROBS = 20

_GREEDY_DECODING = {"temperature": 0.0, "top_p": 1.0, "do_sample": False}


def _topk_renorm_entropy(logprobs: list[float]) -> float:
    """Shannon entropy over the renormalised top-k probabilities (nats)."""
    if not logprobs:
        return 0.0
    m = max(logprobs)
    probs = [math.exp(lp - m) for lp in logprobs]
    s = sum(probs)
    if s <= 0:
        return 0.0
    probs = [p / s for p in probs]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def _trace_from_vllm_logprobs(token_ids: list[int], logprobs_per_pos: list, topk_save: int) -> tuple[list[dict], list[str]]:
    """Build a token_trace compatible with ``panda.core.trace.build_token_trace``.

    ``logprobs_per_pos[i]`` is the vLLM mapping ``{token_id: Logprob}`` for the
    i-th generated token (always includes the sampled token).
    """
    trace: list[dict] = []
    token_texts: list[str] = []
    for pos, (tid, lp_map) in enumerate(zip(token_ids, logprobs_per_pos)):
        if not lp_map:
            tok = ""
            chosen_lp = 0.0
            pairs: list[list] = []
            margin = 0.0
            rank = 1
        else:
            # Sort candidates by logprob desc for top-k pairs / margin.
            ranked = sorted(lp_map.items(), key=lambda kv: kv[1].logprob, reverse=True)
            pairs = [[lo.decoded_token if lo.decoded_token is not None else "", float(lo.logprob)] for _, lo in ranked[:topk_save]]
            chosen = lp_map.get(tid)
            chosen_lp = float(chosen.logprob) if chosen is not None else float(ranked[0][1].logprob)
            tok = (chosen.decoded_token if (chosen is not None and chosen.decoded_token is not None) else (ranked[0][1].decoded_token or ""))
            if len(ranked) >= 2:
                margin = float(ranked[0][1].logprob - ranked[1][1].logprob)
            else:
                margin = 0.0
            # vLLM Logprob.rank is 1-based rank in the full vocab when available.
            rank = int(getattr(chosen, "rank", None) or (1 + sum(1 for _, lo in ranked if lo.logprob > chosen_lp)))
        token_texts.append(tok)
        ent = _topk_renorm_entropy([p[1] for p in pairs])
        trace.append(
            {
                "pos": pos,
                "token": tok,
                "token_id": int(tid),
                "logprob": chosen_lp,
                "entropy": ent,            # top-k approximation of full-vocab entropy
                "entropy_topk": ent,
                "margin_top2": margin,
                "rank": rank,
                "topk": pairs,
            }
        )
    return trace, token_texts


class VLLMGenerator:
    """Thin wrapper over ``vllm.LLM`` exposing the two batched routes we need."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        seed: int = 0,
    ) -> None:
        try:
            from vllm import LLM  # noqa: WPS433 (optional dependency)
        except ImportError as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "vLLM is not installed. Install it (pip install vllm) to use the "
                "vLLM backend, or run the HF pipeline without --engine vllm."
            ) from exc

        self.model_path = model_path
        kwargs: dict[str, Any] = {
            "model": model_path,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "seed": seed,
            "trust_remote_code": True,
            "enforce_eager": True,
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        vllm_model = model_path
        lp = str(model_path).lower()
        if "qwen3" in lp:
            import sys
            from pathlib import Path as _Path

            sc_root = _Path(__file__).resolve().parents[3] / "experiments" / "spurious_consensus"
            if str(sc_root) not in sys.path:
                sys.path.insert(0, str(sc_root))
            from sampling_utils import prepare_vllm_model_path

            vllm_model = prepare_vllm_model_path(model_path)
        else:
            from panda.core.tfb_vllm_register import ensure_tfb_vllm_registered

            ensure_tfb_vllm_registered(model_path)
        kwargs["model"] = vllm_model
        self.llm = LLM(**kwargs)

    def generate_with_stats_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        topk_save: int = 10,
    ) -> list[dict]:
        """Greedy decode a batch of prompts; return ``generate_with_stats``-shaped dicts."""
        from vllm import SamplingParams

        logprobs = min(max(topk_save, 1), _VLLM_MAX_LOGPROBS)
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            logprobs=logprobs,
        )
        outputs = self.llm.generate(prompts, params)
        # vLLM preserves input order in returned list.
        decoding = {**_GREEDY_DECODING, "topk_save": topk_save, "max_new_tokens": max_new_tokens}
        results: list[dict] = []
        for out in outputs:
            comp = out.outputs[0]
            response_text = comp.text
            token_ids = list(comp.token_ids)
            lp_per_pos = comp.logprobs or [{} for _ in token_ids]
            trace, token_texts = _trace_from_vllm_logprobs(token_ids, lp_per_pos, topk_save)
            ans = extract_math_answer(response_text)
            results.append(
                {
                    "response_text": response_text,
                    "final_answer": ans,
                    "answer_raw": ans,
                    "answer_normalized": ans,
                    "parse_success": bool(response_text.strip()),
                    "token_entropies": [t["entropy"] for t in trace],
                    "token_margins": [-t["margin_top2"] for t in trace],
                    "token_trace": trace,
                    "answer_span": find_answer_span(token_texts, response_text),
                    "n_tokens": len(trace),
                    "decoding": decoding,
                }
            )
        return results

    def generate_answers_batch(
        self,
        prompts: list[str],
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> list[list[str]]:
        """Sample ``num_samples`` answers per prompt (SE route); answers only, no trace."""
        from vllm import SamplingParams

        params = SamplingParams(
            n=num_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(prompts, params)
        results: list[list[str]] = []
        for out in outputs:
            answers: list[str] = []
            for comp in out.outputs:
                text = comp.text
                answers.append(extract_math_answer(text) or text.strip())
            results.append(answers)
        return results

    def generate_full_responses_batch(
        self,
        prompts: list[str],
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> list[list[str]]:
        """Sample ``num_samples`` full generations per prompt (official SE / NLI clustering)."""
        from vllm import SamplingParams

        params = SamplingParams(
            n=num_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(prompts, params)
        results: list[list[str]] = []
        for out in outputs:
            results.append([comp.text for comp in out.outputs])
        return results
