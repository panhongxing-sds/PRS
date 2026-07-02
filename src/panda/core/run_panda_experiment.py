#!/usr/bin/env python3
"""
ASE experiment with full raw_runs/{id}.json per sample + summary.jsonl.

Modes:
  all      — GPU generate + save raw + summary metrics
  generate — only save raw_runs/*.json
  metrics  — recompute summary from raw_runs (CPU)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm


def _try_claim(claim_dir: Path, rec_id: str, stale_sec: float = 1800.0) -> bool:
    """Atomically claim a record for dynamic work distribution across processes.

    Returns True if this process won the claim (and should process the record).
    A claim file older than ``stale_sec`` is assumed orphaned (crashed worker)
    and may be re-claimed. Double-processing on rare stale races is harmless
    (idempotent: last save_record wins).
    """
    claim_dir.mkdir(parents=True, exist_ok=True)
    cp = claim_dir / f"{rec_id}.claim"
    try:
        fd = os.open(str(cp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            age = time.time() - cp.stat().st_mtime
        except FileNotFoundError:
            return False
        if age > stale_sec:
            try:
                os.utime(cp, None)
                cp.write_text(str(os.getpid()))
                return True
            except OSError:
                return False
        return False

from panda.core.generate import generate_requery_answers, generate_with_stats
from panda.core.metrics import metrics_from_record
from panda.core.record import (
    _run_from_gen,
    build_full_record,
    delete_partial_record,
    load_partial_record,
    load_record,
    perturb_config_dict,
    record_exists,
    runs_from_high_temp_answers,
    save_partial_record,
    save_record,
)
from panda.datasets.registry import DATASET_IDS, get_dataset_spec, normalize_dataset_id
from panda.grading.tokur_records import build_prompt_for_dataset, prompt_template_version
from panda.perturbations.weight import LowRankWeightPerturbation, WeightPerturbConfig
from panda.token_qaac.data import build_records

from panda.paths import DEFAULT_BENCH, DEFAULT_MODEL, DEFAULT_OUT

DEFAULT_TFTTCL = Path("/home/phx/TF-TTCL/data/MATH")
DEFAULT_WEIGHT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
DECODING = {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "topk_save": 20}
DEFAULT_SE_SAMPLES = 10
DEFAULT_SE_TEMPERATURE = 0.5
DEFAULT_SE_TOP_P = 0.95


def load_model(model_path: str, device: str, attn_implementation: str = "sdpa"):
    from panda.tfb_load import load_tfb_for_teacher_force

    dtype = "bfloat16" if device != "cpu" else "float32"
    model, tokenizer = load_tfb_for_teacher_force(
        model_path, device=device, dtype=dtype, attn_implementation=attn_implementation
    )
    model.eval()
    return model, tokenizer


def generate_full_record(
    rec: dict,
    model,
    tokenizer,
    device: str,
    wp: LowRankWeightPerturbation,
    *,
    max_new_tokens: int,
    n_rephrases: int,
    weight_seeds: list[int],
    model_path: str,
    weight_sigma: float,
    weight_rank: int,
    topk_save: int,
    se_samples: int = 0,
    se_temperature: float = DEFAULT_SE_TEMPERATURE,
    se_top_p: float = DEFAULT_SE_TOP_P,
    out_dir: Path | None = None,
    checkpoint_every: int = 1,
) -> dict:
    question = rec.get("question") or ""
    rephrases = (rec.get("rephrases") or [])[:n_rephrases]
    ds_name = rec.get("dataset", "math500")
    prompt_clean = build_prompt_for_dataset(question, tokenizer, model_path, dataset=ds_name)
    reference = rec.get("reference", "")

    decoding = {**DECODING, "max_new_tokens": max_new_tokens, "topk_save": topk_save}
    model_info = {
        "model_name": model_path,
        "checkpoint": model_path,
        "tokenizer": getattr(tokenizer, "name_or_path", model_path),
        "prompt_template_version": prompt_template_version(model_path, tokenizer),
        "decoding": decoding,
    }
    experiment_config = {
        "n_rephrases": n_rephrases,
        "weight_seeds": weight_seeds,
        "weight_sigma": weight_sigma,
        "weight_rank": weight_rank,
        "max_new_tokens": max_new_tokens,
        "topk_save": topk_save,
        "se_samples": se_samples,
        "se_temperature": se_temperature,
        "se_top_p": se_top_p,
    }

    partial = load_partial_record(out_dir, rec["dataset"], rec["id"]) if out_dir else None
    clean_gen = (partial or {}).get("clean_gen")
    text_runs = list((partial or {}).get("text_runs") or [])
    weight_runs = list((partial or {}).get("weight_runs") or [])
    high_temp_runs = list((partial or {}).get("high_temp_runs") or [])
    _step = 0

    def _checkpoint() -> None:
        nonlocal _step
        _step += 1
        if not out_dir or (_step % checkpoint_every != 0):
            return
        save_partial_record(
            out_dir,
            {
                "id": rec["id"],
                "dataset": rec["dataset"],
                "question": question,
                "reference": reference,
                "model_info": model_info,
                "experiment_config": experiment_config,
                "clean_gen": clean_gen,
                "text_runs": text_runs,
                "weight_runs": weight_runs,
                "high_temp_runs": high_temp_runs,
            },
        )

    if clean_gen is None:
        clean_gen = generate_with_stats(
            model, tokenizer, prompt_clean, max_new_tokens, device, topk_save=topk_save, decoding=decoding, dataset=ds_name
        )
        clean_gen["input_prompt"] = prompt_clean
        _checkpoint()

    start_text = len(text_runs)
    for i, rq in enumerate(rephrases[start_text:], start=start_text):
        pr = build_prompt_for_dataset(rq, tokenizer, model_path, dataset=ds_name)
        g = generate_with_stats(model, tokenizer, pr, max_new_tokens, device, topk_save=topk_save, decoding=decoding, dataset=ds_name)
        g["input_prompt"] = pr
        text_runs.append(_run_from_gen(f"T_{i}", g, source="text", rephrase_text=rq, reference=reference))
        _checkpoint()

    done_seeds = {
        r["perturb_config"]["perturb_seed"]
        for r in weight_runs
        if r.get("perturb_config") and r["perturb_config"].get("perturb_seed") is not None
    }
    target_modules = list(wp._bases.keys())
    for seed in weight_seeds:
        if seed in done_seeds:
            continue
        with wp.sample(seed=seed):
            g = generate_with_stats(
                model, tokenizer, prompt_clean, max_new_tokens, device, topk_save=topk_save, decoding=decoding, dataset=ds_name
            )
        g["input_prompt"] = prompt_clean
        pcfg = perturb_config_dict(
            seed=seed,
            sigma=weight_sigma,
            rank=weight_rank,
            target_suffixes=wp.config.target_suffixes,
            target_modules=target_modules,
            noise_norm=None,
        )
        weight_runs.append(_run_from_gen(f"W_{seed}", g, source="weight", perturb_config=pcfg, reference=reference))
        _checkpoint()

    if se_samples > 0 and len(high_temp_runs) < se_samples:
        answers = generate_requery_answers(
            model,
            tokenizer,
            prompt_clean,
            max_new_tokens,
            device,
            num_samples=se_samples,
            temperature=se_temperature,
            top_p=se_top_p,
            dataset=ds_name,
        )
        high_temp_runs = runs_from_high_temp_answers(
            answers,
            temperature=se_temperature,
            top_p=se_top_p,
            reference=reference,
        )
        _checkpoint()

    if out_dir:
        _step = checkpoint_every - 1
        _checkpoint()

    return build_full_record(
        rec,
        clean_gen=clean_gen,
        text_runs=text_runs,
        weight_runs=weight_runs,
        high_temp_runs=high_temp_runs,
        model_info=model_info,
        experiment_config=experiment_config,
    )


def backfill_se_samples(
    record: dict,
    model,
    tokenizer,
    device: str,
    *,
    max_new_tokens: int,
    se_samples: int,
    se_temperature: float,
    se_top_p: float,
    model_path: str,
) -> dict | None:
    """Add high_temp_sample_runs to an existing raw record (SE baseline only)."""
    if se_samples <= 0 or (record.get("high_temp_sample_runs") or []):
        return None
    question = record.get("question") or ""
    reference = record.get("reference", "")
    prompt_clean = build_prompt_for_dataset(question, tokenizer, model_path, dataset=record.get("dataset", "math500"))
    answers = generate_requery_answers(
        model,
        tokenizer,
        prompt_clean,
        max_new_tokens,
        device,
        num_samples=se_samples,
        temperature=se_temperature,
        top_p=se_top_p,
        dataset=record.get("dataset", "math500"),
    )
    record["high_temp_sample_runs"] = runs_from_high_temp_answers(
        answers,
        temperature=se_temperature,
        top_p=se_top_p,
        reference=reference,
    )
    exp = record.setdefault("experiment_config", {})
    exp["se_samples"] = se_samples
    exp["se_temperature"] = se_temperature
    exp["se_top_p"] = se_top_p
    return record


def recompute_dataset_summary(out_dir: Path, dataset: str, top_pct: float) -> int:
    raw_dir = out_dir / dataset / "raw_runs"
    feat_path = out_dir / dataset / "features.jsonl"
    if not raw_dir.exists():
        return 0
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        record = json.loads(p.read_text(encoding="utf-8"))
        rows.append(metrics_from_record(record, top_pct=top_pct))
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    with feat_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="PANDA with full raw_runs storage")
    ap.add_argument("--mode", choices=("all", "generate", "metrics"), default="all")
    ap.add_argument(
        "--dataset",
        choices=DATASET_IDS,
        required=True,
    )
    ap.add_argument("--variants-path", type=Path, default=None)
    ap.add_argument("--tfttcl-root", type=Path, default=DEFAULT_TFTTCL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--n-rephrases", type=int, default=8)
    ap.add_argument("--weight-sigma", type=float, default=0.03)
    ap.add_argument("--weight-rank", type=int, default=4)
    ap.add_argument("--weight-seeds", type=str, default=",".join(map(str, DEFAULT_WEIGHT_SEEDS)))
    ap.add_argument("--atu-top-pct", type=float, default=0.10)
    ap.add_argument("--topk-save", type=int, default=10)
    ap.add_argument("--attn-implementation", type=str, default="sdpa", choices=("sdpa", "eager", "flash_attention_2"))
    ap.add_argument("--checkpoint-every", type=int, default=1, help="Save partial record every N generate steps")
    ap.add_argument(
        "--se-samples",
        type=int,
        default=DEFAULT_SE_SAMPLES,
        help="High-temp samples for official SE baseline (0=skip; stored in high_temp_sample_runs)",
    )
    ap.add_argument("--se-temperature", type=float, default=DEFAULT_SE_TEMPERATURE)
    ap.add_argument("--se-top-p", type=float, default=DEFAULT_SE_TOP_P)
    ap.add_argument("--fast", action="store_true", help="Fast profile: sparse checkpoints, sdpa (topk=10, 8 weight seeds)")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--dynamic-claim",
        action="store_true",
        help="Dynamic work queue: every shard scans the full record list and atomically "
        "claims the next undone record (eliminates static-shard straggler tail). "
        "Ignores shard-id/num-shards slicing.",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.fast:
        if args.checkpoint_every == 1:
            args.checkpoint_every = 99
        if args.topk_save > 10:
            args.topk_save = 10

    out_dir = args.out_dir

    if args.mode == "metrics":
        n = recompute_dataset_summary(out_dir, args.dataset, args.atu_top_pct)
        print(f"metrics-only: wrote {n} rows → {out_dir / args.dataset / 'features.jsonl'}")
        return

    variants_path = args.variants_path or (DEFAULT_BENCH / args.dataset / "variants.jsonl")
    records = build_records(
        dataset=args.dataset,
        variants_path=variants_path,
        tfttcl_root=args.tfttcl_root,
        max_samples=args.max_samples,
    )
    records = sorted(records, key=lambda r: r["id"])
    if not args.dynamic_claim:
        records = records[args.shard_id :: args.num_shards]
    claim_dir = out_dir / args.dataset / "claims"

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    weight_seeds = [int(s.strip()) for s in args.weight_seeds.split(",") if s.strip()]

    mode_desc = "dynamic-claim" if args.dynamic_claim else f"shard {args.shard_id}/{args.num_shards}"
    print(
        f"[{args.mode}] {mode_desc} {args.dataset}: "
        f"{len(records)} records, device={device}"
    )

    model = tokenizer = wp = None
    if args.mode in ("all", "generate"):
        model, tokenizer = load_model(args.model_path, device, args.attn_implementation)
        wp_cfg = WeightPerturbConfig(
            sigma=args.weight_sigma,
            rank=args.weight_rank,
            num_samples=len(weight_seeds),
            target_suffixes=("q_proj", "k_proj"),
        )
        wp = LowRankWeightPerturbation(model, wp_cfg)

    def _process_one(rec) -> bool:
        """Process a single record. Returns True if generation was attempted."""
        if args.resume and record_exists(out_dir, args.dataset, rec["id"]):
            if not args.dynamic_claim and args.se_samples > 0 and args.mode in ("all", "generate"):
                try:
                    existing = load_record(out_dir, args.dataset, rec["id"])
                    patched = backfill_se_samples(
                        existing,
                        model,
                        tokenizer,
                        device,
                        max_new_tokens=args.max_new_tokens,
                        se_samples=args.se_samples,
                        se_temperature=args.se_temperature,
                        se_top_p=args.se_top_p,
                        model_path=args.model_path,
                    )
                    if patched is not None:
                        metrics = metrics_from_record(patched, top_pct=args.atu_top_pct)
                        save_record(out_dir, patched, metrics)
                except Exception as exc:
                    print(f"SE backfill ERROR {rec['id']}: {exc}")
            return False
        if args.dynamic_claim and not _try_claim(claim_dir, rec["id"]):
            return False
        try:
            full = generate_full_record(
                rec,
                model,
                tokenizer,
                device,
                wp,
                max_new_tokens=args.max_new_tokens,
                n_rephrases=args.n_rephrases,
                weight_seeds=weight_seeds,
                model_path=args.model_path,
                weight_sigma=args.weight_sigma,
                weight_rank=args.weight_rank,
                topk_save=args.topk_save,
                se_samples=args.se_samples,
                se_temperature=args.se_temperature,
                se_top_p=args.se_top_p,
                out_dir=out_dir,
                checkpoint_every=args.checkpoint_every,
            )
            metrics = metrics_from_record(full, top_pct=args.atu_top_pct)
            save_record(out_dir, full, metrics)
        except Exception as exc:
            err = out_dir / args.dataset / "raw_runs" / f"{rec['id']}.error.json"
            err.parent.mkdir(parents=True, exist_ok=True)
            err.write_text(json.dumps({"id": rec["id"], "error": str(exc)}), encoding="utf-8")
            print(f"ERROR {rec['id']}: {exc}")
        return True

    if args.dynamic_claim:
        # Repeat full passes until a pass does no work: catches records orphaned
        # by crashed workers (stale claims become re-claimable after stale_sec).
        while True:
            did_work = False
            for rec in tqdm(records, desc=f"PANDA [{args.dataset}] dyn"):
                if _process_one(rec):
                    did_work = True
            if not did_work:
                break
    else:
        for rec in tqdm(records, desc=f"PANDA [{args.dataset}] shard {args.shard_id}"):
            _process_one(rec)

    print(f"raw_runs → {out_dir / args.dataset / 'raw_runs/'}")


if __name__ == "__main__":
    main()
