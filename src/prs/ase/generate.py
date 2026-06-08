"""Generation with full per-token traces for offline metric recomputation."""

from __future__ import annotations

import torch

from prs.ase.trace import build_token_trace, find_answer_span
from prs.grading.math_grader import extract_math_answer


@torch.no_grad()
def generate_requery_answers(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device,
    *,
    num_samples: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> list[str]:
    """Fast prefix re-query: batched sampling, no token trace (answers only)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id,
        "use_cache": True,
        "num_return_sequences": num_samples,
    }
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=inputs.get("attention_mask"),
        **gen_kwargs,
    )

    sequences = out.sequences if hasattr(out, "sequences") else out
    answers: list[str] = []
    for i in range(num_samples):
        gen_ids = sequences[i, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        ans = extract_math_answer(text) or text.strip()
        answers.append(ans)
    return answers


@torch.no_grad()
def generate_with_stats(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device,
    *,
    topk_save: int = 20,
    decoding: dict | None = None,
) -> dict:
    """
    Greedy decode with KV cache; save full token_trace (top-k logprobs, entropy, margin).

    Returns legacy fields (token_entropies, token_margins) plus token_trace, answer_span.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    attention_mask = inputs.get("attention_mask")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
    )

    gen_ids = out.sequences[0, prompt_len:]
    response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    trace, token_texts = build_token_trace(gen_ids, out.scores, tokenizer, topk_save=topk_save)
    answer_span = find_answer_span(token_texts, response_text)

    token_entropies = [t["entropy"] for t in trace]
    token_margins = [-t["margin_top2"] for t in trace]

    row = {
        "response_text": response_text,
        "final_answer": extract_math_answer(response_text),
        "answer_raw": extract_math_answer(response_text),
        "answer_normalized": extract_math_answer(response_text),
        "parse_success": bool(response_text.strip()),
        "token_entropies": token_entropies,
        "token_margins": token_margins,
        "token_trace": trace,
        "answer_span": answer_span,
        "n_tokens": len(trace),
        "decoding": decoding or {"temperature": 0.0, "top_p": 1.0, "do_sample": False},
    }
    return row
