"""Load TokUR official generations (pkl / json) for paper-aligned eval."""

from __future__ import annotations

import json
import pickle
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
from panda.paths import TOKUR_ROOT

_TOKUR_ROOT = TOKUR_ROOT
if str(_TOKUR_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOKUR_ROOT))


def _normalize_id(raw: str) -> str:
    return str(raw).replace("/", "_").replace(".json", "")


def extract_tokur_unc_from_output(output: Any) -> dict[str, float] | None:
    """Same aggregation as TokUR eval ``extract_unc``."""
    try:
        unc_list = output.uncertainties
        ll_list = output.logprobs
        au_list = [list(d.values())[0].aleatoric_uncertainty for d in unc_list]
        tu_list = [list(d.values())[0].total_uncertainty for d in unc_list]
        eu_list = [list(d.values())[0].epistemic_uncertainty for d in unc_list]
        ll_tok = [list(d.values())[0].logprob for d in ll_list]
        return {
            "au_sum": float(np.sum(au_list)),
            "tu_sum": float(np.sum(tu_list)),
            "eu_sum": float(np.sum(eu_list)),
            "ll_mean": float(np.mean(ll_tok)),
            "ll_sum": float(np.sum(ll_tok)),
        }
    except Exception:
        return None


def extract_tokur_unc_from_json_blob(blob: dict) -> dict[str, float] | None:
    u = blob.get("uncertainties") or {}
    lp = blob.get("logprobs") or {}
    if not u:
        return None
    return {
        "au_sum": float(u.get("au", 0)),
        "tu_sum": float(u.get("tu", 0)),
        "eu_sum": float(u.get("eu", 0)),
        "ll_mean": float(lp.get("nll", 0)),
        "ll_sum": float(lp.get("ll", 0)),
    }


def label_from_answer(pred_text: str, ground_truth: str, dataset: str) -> bool:
    from panda.grading.answer_canonicalizer import grade_answer
    from panda.grading.math_grader import math_equal
    from panda.datasets.registry import normalize_dataset_id

    ds = dataset.replace("-", "_")
    try:
        ds = normalize_dataset_id(ds)
        g = grade_answer(pred_text, ground_truth, dataset=ds)
        return bool(g["is_correct_clean"])
    except ValueError:
        pass
    if ds in ("math500", "deepscaler", "gsm8k", "gsm8k_test", "minerva"):
        return math_equal(pred_text, ground_truth)
    return pred_text.strip().lower() == ground_truth.strip().lower()


def _extract_answer_text(text: str, dataset: str) -> str:
    from panda.datasets.registry import get_dataset_spec, normalize_dataset_id
    from panda.grading.code_grader import extract_code_block
    from panda.grading.logic_grader import extract_logic_answer

    try:
        spec = get_dataset_spec(normalize_dataset_id(dataset.replace("-", "_")))
        if spec.grading == "code":
            return extract_code_block(text)
        if spec.grading == "string":
            return extract_logic_answer(text)
    except ValueError:
        pass
    if dataset in ("math500", "deepscaler", "gsm8k", "gsm8k_test", "minerva"):
        try:
            from run.utils.qwen_math_parser import extract_answer, strip_string

            return strip_string(extract_answer(text, "math"))
        except Exception:
            pass
    return text.strip()


def load_from_pkl_glob(pkl_glob: str, dataset: str) -> list[dict]:
    from panda.grading.export_tokur_pkl import _StubUnpickler

    records: list[dict] = []
    for fp in sorted(glob(pkl_glob)):
        try:
            with open(fp, "rb") as f:
                data = _StubUnpickler(f).load()
        except Exception:
            continue
        if not isinstance(data, list):
            data = [data]
        for sample in data:
            try:
                result = sample["result"][0] if isinstance(sample["result"], list) else sample["result"]
                out = result.outputs[0]
                text = out.text
                unc = extract_tokur_unc_from_output(out)
                if unc is None:
                    continue
                gt = sample["answer"]
                uid = _normalize_id(sample["unique_id"])
                prompt = sample.get("formatted_prompt") or sample.get("problem", "")
                records.append(
                    {
                        "id": uid,
                        "dataset": dataset,
                        "prompt": prompt,
                        "response": text,
                        "reference": gt,
                        "is_correct": label_from_answer(
                            _extract_answer_text(text, dataset), gt, dataset
                        ),
                        "tokur_eu": unc["eu_sum"],
                        "tokur_au": unc["au_sum"],
                        "tokur_tu": unc["tu_sum"],
                        "tokur_ll": unc["ll_sum"],
                        "tokur_neg_eu": -unc["eu_sum"],
                        "tokur_neg_au": -unc["au_sum"],
                        "tokur_neg_tu": -unc["tu_sum"],
                    }
                )
            except Exception:
                continue
    return records


def load_from_tokur_json_dir(json_dir: str, dataset: str) -> list[dict]:
    """Load per-question JSON from TokUR greedy output (essential uncertainties)."""
    records: list[dict] = []
    root = Path(json_dir)
    for fp in sorted(root.glob("*.json")):
        if fp.name.endswith("_error.json"):
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        outputs = data.get("outputs") or []
        if not outputs:
            continue
        out0 = outputs[0]
        unc = extract_tokur_unc_from_json_blob(out0)
        if unc is None:
            continue
        text = out0.get("text", "")
        gt = data.get("answer", "")
        uid = fp.stem
        prompt = data.get("formatted_prompt", data.get("problem", ""))
        records.append(
            {
                "id": uid,
                "dataset": dataset,
                "prompt": prompt,
                "response": text,
                "reference": gt,
                "is_correct": label_from_answer(
                    _extract_answer_text(text, dataset), gt, dataset
                ),
                "tokur_eu": unc["eu_sum"],
                "tokur_au": unc["au_sum"],
                "tokur_tu": unc["tu_sum"],
                "tokur_ll": unc["ll_sum"],
                "tokur_neg_eu": -unc["eu_sum"],
                "tokur_neg_au": -unc["au_sum"],
                "tokur_neg_tu": -unc["tu_sum"],
            }
        )
    return records


def load_from_jsonl(path: str, dataset: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            r["dataset"] = dataset
            if "is_correct" not in r:
                r["is_correct"] = label_from_answer(
                    _extract_answer_text(r.get("response", ""), dataset),
                    r.get("reference", ""),
                    dataset,
                )
            records.append(r)
    return records


_TOKUR_MATH_SYSTEM = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "- For simple problems (2 steps or fewer):\n"
    "Provide a concise solution with minimal explanation.\n\n"
    "- For complex problems (3 steps or more):\n"
    "Use this step-by-step format:\n\n"
    "## Step 1: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "...\n\n"
    "Regardless of the approach, always conclude with:\n\n"
    "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
    "Where [answer] is just the final number or expression that solves the problem."
)


def _is_qwen_model(model_path: str, tokenizer) -> bool:
    hay = f"{model_path} {getattr(tokenizer, 'name_or_path', '')}".lower()
    return "qwen" in hay


def build_prompt_for_dataset(problem: str, tokenizer, model_path: str = "", dataset: str = "math500") -> str:
    """Chat prompt: math uses TokUR template; logic/code use plain user turn."""
    from panda.datasets.registry import get_dataset_spec, normalize_dataset_id

    try:
        spec = get_dataset_spec(normalize_dataset_id(dataset))
        if spec.domain in ("logic", "code"):
            system = [{"role": "system", "content": "You are a helpful assistant."}]
            return tokenizer.apply_chat_template(
                system + [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except ValueError:
        pass
    return build_prompt_tfb(problem, tokenizer, model_path)


def build_prompt_tfb(problem: str, tokenizer, model_path: str = "") -> str:
    """Chat prompt aligned with TokUR greedy_unc (Qwen vs Llama/others)."""
    if _is_qwen_model(model_path, tokenizer):
        system = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            }
        ]
        user_content = (
            f"{problem} Let's think step by step and output the final answer within \\boxed{{}}."
        )
    else:
        system = [{"role": "system", "content": _TOKUR_MATH_SYSTEM}]
        user_content = problem
    return tokenizer.apply_chat_template(
        system + [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def prompt_template_version(model_path: str, tokenizer) -> str:
    return "qwen_boxed_v1" if _is_qwen_model(model_path, tokenizer) else "tokur_math_system_v1"


def build_prompt_qwen(problem: str, tokenizer) -> str:
    """Backward-compatible alias (assumes Qwen). Prefer ``build_prompt_tfb``."""
    return build_prompt_tfb(problem, tokenizer, model_path="qwen")


def load_from_dataset_plus_json(
    dataset_path: str,
    json_dir: str,
    dataset_name: str,
    tokenizer,
) -> list[dict]:
    """Match json outputs to dataset rows by unique_id."""
    by_id = {r["id"]: r for r in load_from_tokur_json_dir(json_dir, dataset_name)}
    records: list[dict] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = _normalize_id(row.get("unique_id", ""))
            if uid not in by_id:
                continue
            rec = dict(by_id[uid])
            if not rec.get("prompt"):
                prob = row.get("problem") or row.get("question", "")
                rec["prompt"] = build_prompt_qwen(prob, tokenizer)
            records.append(rec)
    return records