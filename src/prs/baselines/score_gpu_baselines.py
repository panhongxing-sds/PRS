#!/usr/bin/env python3
"""Score GPU baselines (P(True), INSIDE) and merge into baselines.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from prs.paths import DEFAULT_OUT


P_TRUE_TEMPLATE = (
    "Question: {question}\n"
    "Proposed answer: {answer}\n\n"
    "Is the proposed answer correct? Reply with exactly one word: True or False."
)


def _load_raw_records(out_dir: Path, dataset: str) -> list[dict]:
    raw_dir = out_dir / dataset / "raw_runs"
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        if p.name.endswith((".error.json", ".partial.json")):
            continue
        try:
            if p.stat().st_size < 100:
                continue
        except OSError:
            continue
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def score_p_true(
    model,
    tokenizer,
    question: str,
    answer: str,
    device,
    *,
    max_new_tokens: int = 8,
) -> float:
    """
    P(True): probability mass on 'True' token at first generated step.

    Follows semantic_uncertainty prompting style (binary correctness probe).
    """
    prompt = P_TRUE_TEMPLATE.format(question=question.strip(), answer=answer.strip())
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    if not out.scores:
        return float("nan")
    logits = out.scores[0][0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    true_ids: list[int] = []
    for frag in ("True", " true", "True"):
        ids = tokenizer.encode(frag, add_special_tokens=False)
        if ids:
            true_ids.append(int(ids[0]))
    if not true_ids:
        return float("nan")
    probs = [float(torch.exp(log_probs[tid]).item()) for tid in true_ids]
    return float(max(probs))


def _hidden_matrix(model, tokenizer, text: str, device) -> torch.Tensor | None:
    """Last-layer hidden states for ``text`` (prompt + response), shape [T, H]."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    if not out.hidden_states:
        return None
    return out.hidden_states[-1][0]


def score_inside(model, tokenizer, prompt: str, response: str, device) -> float:
    """
    INSIDE EigenScore (approx): log determinant of response-token covariance.

    Uses last-layer hidden states on the generated response tokens only.
    """
    full = (prompt or "") + (response or "")
    h = _hidden_matrix(model, tokenizer, full, device)
    if h is None or h.shape[0] < 2:
        return float("nan")
    prompt_len = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)[
        "input_ids"
    ].shape[1]
    resp = h[prompt_len:]
    if resp.shape[0] < 2:
        resp = h
    x = resp.float().cpu().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    sign, logdet = torch.linalg.slogdet(torch.tensor(cov, dtype=torch.float64))
    if sign.item() <= 0:
        return float("nan")
    # Higher eigen-spectrum spread => more uncertain
    return float(logdet.item())


def main() -> None:
    ap = argparse.ArgumentParser(description="Score GPU baselines P(True) and INSIDE")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--methods", default="p_true,inside", help="Comma-separated: p_true, inside")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    out_path = args.out_dir / args.dataset / "baselines_gpu.jsonl"
    done: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    records = _load_raw_records(args.out_dir, args.dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and out_path.exists() else "w"
    n_new = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for rec in tqdm(records, desc=f"gpu baselines {args.dataset}"):
            rid = rec["id"]
            if rid in done:
                continue
            base = rec.get("base_generation") or {}
            question = rec.get("question", "")
            answer = base.get("answer_normalized") or base.get("final_answer") or ""
            prompt = base.get("input_prompt") or ""
            response = base.get("full_response") or base.get("response_text") or ""
            row = {"id": rid, "baseline_P_True": float("nan"), "baseline_INSIDE": float("nan")}
            if "p_true" in methods and question and answer:
                try:
                    row["baseline_P_True"] = score_p_true(
                        model, tokenizer, question, answer, device
                    )
                except Exception:
                    pass
            if "inside" in methods and response:
                try:
                    row["baseline_INSIDE"] = score_inside(
                        model, tokenizer, prompt, response, device
                    )
                except Exception:
                    pass
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_new += 1

    print(f"Wrote {n_new} rows → {out_path}")


if __name__ == "__main__":
    main()
