#!/usr/bin/env python3
"""Phase A of the hybrid pipeline: batched vLLM generation of clean + text + SE.

This writes *partial* raw records (clean_gen + text_runs + high_temp_runs, with an
empty weight_runs list). Phase B is the existing HF pipeline run with ``--resume``:

    python -m prs.ase.run_vllm_phase  --dataset math500 ...      # vLLM: clean+R+SE
    python -m prs.ase.run_ase_experiment --mode all --resume ... # HF:   weight + metrics

``run_ase_experiment`` already skips any clean/text/SE present in a partial record,
so Phase B only adds the weight-perturbation branch (which vLLM cannot do) and then
computes metrics + the final raw_runs/{id}.json.

Resumable at chunk granularity: a partial that already has clean + n_rephrases text
runs + se_samples high-temp runs is skipped, so re-running continues where it left off.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prs.ase.record import (
    _run_from_gen,
    load_partial_record,
    perturb_config_dict,
    record_exists,
    runs_from_high_temp_answers,
    save_partial_record,
)
from prs.ase.vllm_backend import VLLMGenerator
from prs.datasets.registry import normalize_dataset_id
from prs.grading.tokur_records import build_prompt_for_dataset, prompt_template_version
from prs.paths import DEFAULT_BENCH, DEFAULT_MODEL, DEFAULT_OUT
from prs.token_qaac.data import build_records

DEFAULT_TFTTCL = Path("/home/phx/TF-TTCL/data/MATH")
DEFAULT_WEIGHT_SEEDS = [42, 43, 44, 45]
DEFAULT_SE_TEMPERATURE = 0.7
DEFAULT_SE_TOP_P = 0.95


def _phase_a_done(partial: dict | None, n_rephrases: int, se_samples: int) -> bool:
    if not partial:
        return False
    if partial.get("clean_gen") is None:
        return False
    if len(partial.get("text_runs") or []) < n_rephrases:
        return False
    if se_samples > 0 and len(partial.get("high_temp_runs") or []) < se_samples:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM Phase A: clean + text-rephrase + SE")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variants-path", type=Path, default=None)
    ap.add_argument("--tfttcl-root", type=Path, default=DEFAULT_TFTTCL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--n-rephrases", type=int, default=4)
    ap.add_argument("--topk-save", type=int, default=10)
    ap.add_argument("--se-samples", type=int, default=8)
    ap.add_argument("--se-temperature", type=float, default=DEFAULT_SE_TEMPERATURE)
    ap.add_argument("--se-top-p", type=float, default=DEFAULT_SE_TOP_P)
    # weight config is only recorded (Phase B regenerates it) for provenance consistency.
    ap.add_argument("--weight-seeds", type=str, default=",".join(map(str, DEFAULT_WEIGHT_SEEDS)))
    ap.add_argument("--weight-sigma", type=float, default=0.03)
    ap.add_argument("--weight-rank", type=int, default=4)
    # vLLM engine knobs.
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=256, help="Records per vLLM batch / checkpoint.")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dataset = args.dataset
    weight_seeds = [int(s.strip()) for s in args.weight_seeds.split(",") if s.strip()]

    variants_path = args.variants_path or (DEFAULT_BENCH / dataset / "variants.jsonl")
    records = build_records(
        dataset=dataset,
        variants_path=variants_path,
        tfttcl_root=args.tfttcl_root,
        max_samples=args.max_samples,
    )
    records = sorted(records, key=lambda r: r["id"])
    records = records[args.shard_id :: args.num_shards]
    print(f"[vLLM phase A] {dataset}: {len(records)} records (shard {args.shard_id}/{args.num_shards})")

    gen = VLLMGenerator(
        args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tokenizer = gen.llm.get_tokenizer()

    decoding = {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "topk_save": args.topk_save, "max_new_tokens": args.max_new_tokens}
    model_info = {
        "model_name": args.model_path,
        "checkpoint": args.model_path,
        "tokenizer": getattr(tokenizer, "name_or_path", args.model_path),
        "prompt_template_version": prompt_template_version(args.model_path, tokenizer),
        "decoding": decoding,
        "engine": "vllm",
    }
    experiment_config = {
        "n_rephrases": args.n_rephrases,
        "weight_seeds": weight_seeds,
        "weight_sigma": args.weight_sigma,
        "weight_rank": args.weight_rank,
        "max_new_tokens": args.max_new_tokens,
        "topk_save": args.topk_save,
        "se_samples": args.se_samples,
        "se_temperature": args.se_temperature,
        "se_top_p": args.se_top_p,
    }

    n_done = 0
    n_skipped = 0
    chunk = max(1, args.chunk_size)
    for start in range(0, len(records), chunk):
        batch = records[start : start + chunk]
        pending: list[dict] = []
        for rec in batch:
            if args.resume and record_exists(args.out_dir, dataset, rec["id"]):
                n_skipped += 1
                continue
            partial = load_partial_record(args.out_dir, dataset, rec["id"]) if args.resume else None
            if _phase_a_done(partial, args.n_rephrases, args.se_samples):
                n_skipped += 1
                continue
            pending.append(rec)
        if not pending:
            continue

        # Build clean + rephrase prompts (greedy, with logprobs) in one vLLM call.
        flat_prompts: list[str] = []
        layout: list[tuple[int, str, str | None]] = []  # (rec_index, kind, rephrase_text)
        for ri, rec in enumerate(pending):
            ds = rec.get("dataset", dataset)
            q = rec.get("question") or ""
            prompt_clean = build_prompt_for_dataset(q, tokenizer, args.model_path, dataset=ds)
            flat_prompts.append(prompt_clean)
            layout.append((ri, "clean", None))
            for rq in (rec.get("rephrases") or [])[: args.n_rephrases]:
                pr = build_prompt_for_dataset(rq, tokenizer, args.model_path, dataset=ds)
                flat_prompts.append(pr)
                layout.append((ri, "text", rq))

        stats = gen.generate_with_stats_batch(
            flat_prompts, max_new_tokens=args.max_new_tokens, topk_save=args.topk_save
        )

        # SE: high-temperature samples on the clean prompt only.
        se_answers: list[list[str]] = [[] for _ in pending]
        if args.se_samples > 0:
            clean_prompts = [flat_prompts[i] for i, (_, kind, _) in enumerate(layout) if kind == "clean"]
            se_answers = gen.generate_answers_batch(
                clean_prompts,
                num_samples=args.se_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.se_temperature,
                top_p=args.se_top_p,
            )

        # Reassemble per-record partials (layout / flat_prompts / stats are aligned 1:1).
        per_rec: list[dict] = [{"clean": None, "text": []} for _ in pending]
        for idx, ((ri, kind, rq), gd) in enumerate(zip(layout, stats)):
            gd["input_prompt"] = flat_prompts[idx]
            if kind == "clean":
                per_rec[ri]["clean"] = gd
            else:
                per_rec[ri]["text"].append((rq, gd))

        for ri, rec in enumerate(pending):
            ref = str(rec.get("reference", "")).strip()
            clean_gen = per_rec[ri]["clean"]
            text_runs = [
                _run_from_gen(f"T_{i}", g, source="text", rephrase_text=rq, reference=ref)
                for i, (rq, g) in enumerate(per_rec[ri]["text"])
            ]
            high_temp_runs = runs_from_high_temp_answers(
                se_answers[ri],
                temperature=args.se_temperature,
                top_p=args.se_top_p,
                reference=ref,
            )
            save_partial_record(
                args.out_dir,
                {
                    "id": rec["id"],
                    "dataset": dataset,
                    "question": rec.get("question", ""),
                    "reference": ref,
                    "model_info": model_info,
                    "experiment_config": experiment_config,
                    "clean_gen": clean_gen,
                    "text_runs": text_runs,
                    "weight_runs": [],
                    "high_temp_runs": high_temp_runs,
                },
            )
            n_done += 1
        print(f"[vLLM phase A] {dataset}: chunk {start}-{start + len(batch)} done (total written={n_done}, skipped={n_skipped})", flush=True)

    print(f"[vLLM phase A] {dataset}: complete. written={n_done}, skipped={n_skipped}")
    print("  Next: python -m prs.ase.run_ase_experiment --mode all --resume "
          f"--dataset {dataset} --out-dir {args.out_dir} --model-path {args.model_path} "
          f"--n-rephrases {args.n_rephrases} --se-samples {args.se_samples} ...")


if __name__ == "__main__":
    main()
