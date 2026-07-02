#!/usr/bin/env python3
"""vLLM batched backfill of official SE high-temp samples into existing raw_runs.

Use when Phase B (HF) finished with ``--se-samples 0`` but you want official SE:
  1. This script: vLLM ``n=8`` samples → ``high_temp_sample_runs`` (full_response kept for NLI)
  2. ``recompute_metrics --from-cache --recompute-se``: DeBERTa NLI → ``baseline_SE_*`` + ``baseline_U_Ecc/U_Deg``

Does **not** regenerate clean / R / W branches. Resumable: skips records that already
have ``>= se_samples`` high-temp runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from panda.core.metrics import metrics_from_record
from panda.core.record import _atomic_write_json
from panda.core.vllm_backend import VLLMGenerator
from panda.grading.extract import extract_answer_for_dataset
from panda.grading.math_grader import math_equal
from panda.grading.tokur_records import build_prompt_for_dataset, prompt_template_version
from panda.paths import DEFAULT_MODEL, DEFAULT_OUT

DEFAULT_SE_TEMPERATURE = 0.5
DEFAULT_SE_TOP_P = 0.95


def _needs_se(record: dict, se_samples: int, *, force: bool = False) -> bool:
    runs = record.get("high_temp_sample_runs") or []
    if se_samples <= 0:
        return False
    if len(runs) < se_samples:
        return True
    if force:
        return True
    # Re-backfill placeholder rows (answers only, no full_response for NLI).
    for run in runs[:se_samples]:
        if not (run.get("full_response") or run.get("response_text") or "").strip():
            return True
    return False


def _prompt_for_record(record: dict, tokenizer, model_path: str) -> str:
    base = record.get("base_generation") or {}
    prompt = base.get("input_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    ds = record.get("dataset", "math500")
    question = record.get("question") or ""
    return build_prompt_for_dataset(question, tokenizer, model_path, dataset=ds)


def runs_from_high_temp_responses(
    responses: list[str],
    *,
    temperature: float,
    top_p: float,
    reference: str = "",
    dataset: str | None = None,
) -> list[dict]:
    """SE run rows with ``full_response`` for official NLI clustering."""
    decoding = {"temperature": temperature, "top_p": top_p, "do_sample": True}
    ref = reference.strip()
    runs: list[dict] = []
    for i, resp in enumerate(responses):
        text = (resp or "").strip()
        ans = extract_answer_for_dataset(text, dataset) or text
        correct = math_equal(ans, ref) if ref and ans else False
        runs.append(
            {
                "run_id": f"SE_{i}",
                "source": "high_temp_sample",
                "input_prompt": None,
                "rephrase_text": None,
                "perturb_config": None,
                "answer_raw": ans,
                "answer_normalized": ans,
                "full_response": text,
                "response_text": text,
                "parse_success": bool(text),
                "correctness": correct,
                "token_trace": [],
                "answer_span": {},
                "n_tokens": 0,
                "decoding": decoding,
                "token_entropies": [],
                "token_margins": [],
            }
        )
    return runs


def _list_raw_paths(out_dir: Path, dataset: str) -> list[Path]:
    raw_dir = out_dir / dataset / "raw_runs"
    if not raw_dir.is_dir():
        return []
    return sorted(
        p
        for p in raw_dir.glob("*.json")
        if not p.name.endswith((".error.json", ".partial.json"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM SE backfill into existing raw_runs")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--se-samples", type=int, default=8)
    ap.add_argument("--se-temperature", type=float, default=DEFAULT_SE_TEMPERATURE)
    ap.add_argument("--se-top-p", type=float, default=DEFAULT_SE_TOP_P)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=32, help="Prompts per vLLM batch")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Replace existing high_temp_sample_runs even when count >= se_samples",
    )
    args = ap.parse_args()

    paths = _list_raw_paths(args.out_dir, args.dataset)
    paths = paths[args.shard_id :: args.num_shards]
    pending: list[tuple[Path, dict]] = []
    n_skip = 0
    for p in paths:
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if args.resume and not _needs_se(record, args.se_samples, force=args.force):
            n_skip += 1
            continue
        pending.append((p, record))

    print(
        f"[vLLM SE backfill] {args.dataset} shard {args.shard_id}/{args.num_shards}: "
        f"pending={len(pending)} skipped={n_skip}",
        flush=True,
    )
    if not pending:
        return

    gen = VLLMGenerator(
        args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tokenizer = gen.llm.get_tokenizer()
    ds = args.dataset
    chunk = max(1, args.chunk_size)
    n_written = 0

    for start in range(0, len(pending), chunk):
        batch = pending[start : start + chunk]
        prompts = [_prompt_for_record(rec, tokenizer, args.model_path) for _, rec in batch]
        all_responses = gen.generate_full_responses_batch(
            prompts,
            num_samples=args.se_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.se_temperature,
            top_p=args.se_top_p,
        )
        for (path, record), responses in zip(batch, all_responses):
            rid = record["id"]
            ref = str(record.get("reference", "")).strip()
            record["high_temp_sample_runs"] = runs_from_high_temp_responses(
                responses,
                temperature=args.se_temperature,
                top_p=args.se_top_p,
                reference=ref,
                dataset=record.get("dataset", ds),
            )
            exp = record.setdefault("experiment_config", {})
            exp["se_samples"] = args.se_samples
            exp["se_temperature"] = args.se_temperature
            exp["se_top_p"] = args.se_top_p
            exp["se_engine"] = "vllm_backfill"
            mi = record.setdefault("model_info", {})
            mi.setdefault("prompt_template_version", prompt_template_version(args.model_path, tokenizer))

            row = record.get("summary_metrics") or metrics_from_record(record)
            record["summary_metrics"] = row
            _atomic_write_json(path, record)
            print(f"[SE backfill] {ds}/{rid} → {path}", flush=True)
            n_written += 1

    print(f"[vLLM SE backfill] {args.dataset}: written={n_written}", flush=True)
    print(
        "  Next: python -m panda.core.recompute_metrics --from-cache --recompute-se "
        f"--out-dir {args.out_dir} --datasets {args.dataset}",
        flush=True,
    )


if __name__ == "__main__":
    main()
