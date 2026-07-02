#!/usr/bin/env python3
"""
AdvW-PANDA experiment: multi-start PGD weight attacks + answer clustering.

Separate from run_panda_experiment (random W-ASE). Saves to outputs/advw_ase/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from panda.core.advw_metrics import advw_metrics_from_record
from panda.core.generate import generate_with_stats
from panda.core.record import (
    _run_from_gen,
    adversarial_pgd_config_dict,
    delete_partial_record,
    load_partial_record,
    record_exists,
    save_partial_record,
    save_record,
)
from panda.grading.math_grader import math_equal
from panda.grading.tokur_records import build_prompt_tfb, prompt_template_version
from panda.metrics_tokur import compute_detection_metrics
from panda.perturbations.weight_attack import (
    WeightAttackConfig,
    apply_weight_deltas,
    multi_start_weight_pgd,
    remove_weight_deltas,
)
from panda.teacher_force import build_teacher_force_batch
from panda.token_qaac.data import build_records
from panda.paths import DEFAULT_OUT, DEFAULT_BENCH, DEFAULT_MODEL


DEFAULT_OUT = OUTPUTS / "advw_ase"
DEFAULT_TFTTCL = Path("/home/phx/TF-TTCL/data/MATH")
DEFAULT_BENCH = DEFAULT_BENCH
DEFAULT_WEIGHT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
DECODING = {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "topk_save": 20}


def load_model(model_path: str, device: str, attn_implementation: str = "sdpa"):
    from panda.tfb_load import load_tfb_for_teacher_force

    dtype = "bfloat16" if device != "cpu" else "float32"
    model, tokenizer = load_tfb_for_teacher_force(
        model_path, device=device, dtype=dtype, attn_implementation=attn_implementation
    )
    model.eval()
    return model, tokenizer


def build_advw_record(
    rec: dict,
    *,
    clean_gen: dict,
    advw_runs: list[dict],
    model_info: dict,
    experiment_config: dict,
) -> dict:
    ref = str(rec.get("reference", "")).strip()
    a0 = clean_gen.get("answer_normalized", "")
    ok = math_equal(a0, ref) if ref and a0 else False

    return {
        "id": rec["id"],
        "dataset": rec["dataset"],
        "question": rec.get("question", ""),
        "reference": ref,
        "reference_normalized": ref,
        "is_correct": ok,
        "label_wrong": 0 if ok else 1,
        "model_info": model_info,
        "experiment_config": experiment_config,
        "base_generation": _run_from_gen("base", clean_gen, source="base", reference=ref),
        "advw_perturb_runs": advw_runs,
    }


def generate_advw_record(
    rec: dict,
    model,
    tokenizer,
    device: str,
    attack_cfg: WeightAttackConfig,
    *,
    max_new_tokens: int,
    weight_seeds: list[int],
    init_sigma: float,
    model_path: str,
    topk_save: int,
    out_dir: Path | None = None,
    checkpoint_every: int = 1,
) -> dict:
    question = rec.get("question") or ""
    prompt_clean = build_prompt_tfb(question, tokenizer, model_path)
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
        "weight_seeds": weight_seeds,
        "init_sigma": init_sigma,
        "perturb_rank": attack_cfg.rank,
        "perturb_epsilon": attack_cfg.epsilon,
        "pgd_steps": attack_cfg.steps,
        "attack_objective": attack_cfg.objective,
        "target_suffixes": list(attack_cfg.targets),
        "max_new_tokens": max_new_tokens,
        "topk_save": topk_save,
    }

    partial = load_partial_record(out_dir, rec["dataset"], rec["id"]) if out_dir else None
    clean_gen = (partial or {}).get("clean_gen")
    advw_runs = list((partial or {}).get("advw_runs") or [])
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
                "advw_runs": advw_runs,
            },
        )

    if clean_gen is None:
        clean_gen = generate_with_stats(
            model, tokenizer, prompt_clean, max_new_tokens, device, topk_save=topk_save, decoding=decoding
        )
        clean_gen["input_prompt"] = prompt_clean
        _checkpoint()

    done_seeds = {
        r["perturb_config"]["perturb_seed"]
        for r in advw_runs
        if r.get("perturb_config") and r["perturb_config"].get("perturb_seed") is not None
    }
    remaining_seeds = [s for s in weight_seeds if s not in done_seeds]

    if remaining_seeds:
        full_response = clean_gen.get("response_text") or clean_gen.get("final_answer", "")
        batch = build_teacher_force_batch(tokenizer, prompt_clean, full_response, model.device)
        pgd_results = multi_start_weight_pgd(
            model,
            batch.input_ids,
            batch.labels,
            batch.response_mask,
            batch.attention_mask,
            attack_cfg,
            seeds=remaining_seeds,
            init_sigma=init_sigma,
        )
    else:
        pgd_results = []

    target_modules = sorted({name for r in pgd_results for name in r.deltas})
    for pgd in pgd_results:
        seed = pgd.seed
        apply_weight_deltas(model, pgd.deltas)
        try:
            g = generate_with_stats(
                model, tokenizer, prompt_clean, max_new_tokens, device, topk_save=topk_save, decoding=decoding
            )
        finally:
            remove_weight_deltas(model, pgd.deltas)
        g["input_prompt"] = prompt_clean
        pcfg = adversarial_pgd_config_dict(
            seed=seed,
            rank=attack_cfg.rank,
            epsilon=attack_cfg.epsilon,
            pgd_steps=attack_cfg.steps,
            attack_objective=attack_cfg.objective,
            attack_loss_final=pgd.attack_loss,
            target_modules=target_modules or list(pgd.deltas.keys()),
        )
        advw_runs.append(
            _run_from_gen(f"AdvW_{seed}", g, source="advw", perturb_config=pcfg, reference=reference)
        )
        _checkpoint()

    if out_dir:
        _step = checkpoint_every - 1
        _checkpoint()

    return build_advw_record(
        rec,
        clean_gen=clean_gen,
        advw_runs=advw_runs,
        model_info=model_info,
        experiment_config=experiment_config,
    )


def recompute_dataset_summary(out_dir: Path, dataset: str) -> int:
    raw_dir = out_dir / dataset / "raw_runs"
    feat_path = out_dir / dataset / "features.jsonl"
    if not raw_dir.exists():
        return 0
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        if p.name.endswith(".error.json") or p.name.endswith(".partial.json"):
            continue
        record = json.loads(p.read_text(encoding="utf-8"))
        sm = record.get("summary_metrics") or advw_metrics_from_record(record)
        rows.append({"id": record["id"], "dataset": dataset, "is_correct": record.get("is_correct"), "label_wrong": record.get("label_wrong"), **sm})
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    with feat_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def print_summary_table(out_dir: Path, dataset: str) -> None:
    feat_path = out_dir / dataset / "features.jsonl"
    if not feat_path.exists():
        print("No features.jsonl found.")
        return
    rows = [json.loads(l) for l in feat_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("Empty features.jsonl.")
        return

    print(f"\n=== AdvW-PANDA summary ({dataset}, n={len(rows)}) ===")
    print(f"{'id':<20} {'correct':>8} {'AdvW_ASE':>10} {'severity':>10} {'worst':>10} {'clusters':>9}")
    for r in rows:
        print(
            f"{str(r.get('id', '')):<20} "
            f"{str(r.get('is_correct', '')):>8} "
            f"{r.get('AdvW_ASE', float('nan')):>10.4f} "
            f"{r.get('AdvW_severity', float('nan')):>10.4f} "
            f"{r.get('AdvW_worst', float('nan')):>10.4f} "
            f"{r.get('AdvW_num_clusters', 0):>9}"
        )

    labels = [bool(x.get("is_correct", False)) for x in rows]
    scores = [float(x.get("AdvW_ASE", float("nan"))) for x in rows]
    finite = [(l, s) for l, s in zip(labels, scores) if s == s]
    if len(finite) >= 4 and len(set(l for l, _ in finite)) > 1:
        ys = [l for l, _ in finite]
        ss = [s for _, s in finite]
        m = compute_detection_metrics(ys, ss)
        print(f"\nAdvW_ASE AUROC (label_wrong): {m.auroc:.4f}  AUPRC: {m.auprc:.4f}  n={m.n}")
    else:
        print("\n(AUROC skipped: need ≥4 samples with both classes)")


def main() -> None:
    ap = argparse.ArgumentParser(description="AdvW-PANDA multi-start PGD experiment")
    ap.add_argument("--mode", choices=("all", "generate", "metrics"), default="all")
    ap.add_argument(
        "--dataset",
        choices=("minerva", "math500", "gsm8k", "deepscaler"),
        required=True,
    )
    ap.add_argument("--variants-path", type=Path, default=None)
    ap.add_argument("--tfttcl-root", type=Path, default=DEFAULT_TFTTCL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--weight-seeds", type=str, default=",".join(map(str, DEFAULT_WEIGHT_SEEDS)))
    ap.add_argument("--perturb-rank", type=int, default=4)
    ap.add_argument("--perturb-epsilon", type=float, default=0.05)
    ap.add_argument("--pgd-steps", type=int, default=3)
    ap.add_argument("--pgd-step-size", type=float, default=0.02)
    ap.add_argument("--init-sigma", type=float, default=0.01)
    ap.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj",
        help="Comma-separated module suffixes (q_proj,k_proj,v_proj,o_proj)",
    )
    ap.add_argument("--attack-objective", type=str, default="margin", choices=("ce_loss", "margin", "entropy"))
    ap.add_argument("--topk-save", type=int, default=10)
    ap.add_argument("--attn-implementation", type=str, default="sdpa", choices=("sdpa", "eager", "flash_attention_2"))
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir

    if args.mode == "metrics":
        n = recompute_dataset_summary(out_dir, args.dataset)
        print(f"metrics-only: wrote {n} rows → {out_dir / args.dataset / 'features.jsonl'}")
        print_summary_table(out_dir, args.dataset)
        return

    variants_path = args.variants_path or (DEFAULT_BENCH / args.dataset / "variants.jsonl")
    records = build_records(
        dataset=args.dataset,
        variants_path=variants_path,
        tfttcl_root=args.tfttcl_root,
        max_samples=args.max_samples,
    )
    records = sorted(records, key=lambda r: r["id"])
    records = records[args.shard_id :: args.num_shards]

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    weight_seeds = [int(s.strip()) for s in args.weight_seeds.split(",") if s.strip()]
    target_suffixes = tuple(s.strip() for s in args.target_modules.split(",") if s.strip())

    attack_cfg = WeightAttackConfig(
        epsilon=args.perturb_epsilon,
        steps=args.pgd_steps,
        step_size=args.pgd_step_size,
        rank=args.perturb_rank,
        targets=target_suffixes,
        objective=args.attack_objective,
        max_modules=4,
    )

    print(
        f"[{args.mode}] shard {args.shard_id}/{args.num_shards} {args.dataset}: "
        f"{len(records)} records, device={device}, M={len(weight_seeds)} restarts"
    )

    model = tokenizer = None
    if args.mode in ("all", "generate"):
        model, tokenizer = load_model(args.model_path, device, args.attn_implementation)

    for rec in tqdm(records, desc=f"AdvW-PANDA [{args.dataset}] shard {args.shard_id}"):
        if args.resume and record_exists(out_dir, args.dataset, rec["id"]):
            continue
        try:
            full = generate_advw_record(
                rec,
                model,
                tokenizer,
                device,
                attack_cfg,
                max_new_tokens=args.max_new_tokens,
                weight_seeds=weight_seeds,
                init_sigma=args.init_sigma,
                model_path=args.model_path,
                topk_save=args.topk_save,
                out_dir=out_dir,
                checkpoint_every=args.checkpoint_every,
            )
            metrics = advw_metrics_from_record(full)
            save_record(out_dir, full, metrics)
        except Exception as exc:
            err = out_dir / args.dataset / "raw_runs" / f"{rec['id']}.error.json"
            err.parent.mkdir(parents=True, exist_ok=True)
            err.write_text(json.dumps({"id": rec["id"], "error": str(exc)}), encoding="utf-8")
            print(f"ERROR {rec['id']}: {exc}")

    recompute_dataset_summary(out_dir, args.dataset)
    print(f"raw_runs → {out_dir / args.dataset / 'raw_runs/'}")
    print_summary_table(out_dir, args.dataset)


if __name__ == "__main__":
    main()