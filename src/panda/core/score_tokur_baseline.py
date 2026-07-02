#!/usr/bin/env python3
"""**APPROXIMATE** TokUR EU baseline (post-hoc teacher forcing on ASE greedy text).

This reimplements weight EU via tprd ``LowRankWeightPerturbation`` + ``ATokURPipeline`` on
responses already produced by the ASE experiment — **not** official TokUR vLLM generation.

For paper-aligned strict TokUR (native TFB, EU during greedy decode), use::

    python3 -m panda.core.score_tokur_official ...
    bash scripts/run_tokur_strict_baseline.sh

Approx vs official gaps:
  - Generation: ASE greedy (HF) vs TokUR greedy (vLLM + bayesian_transformer).
  - Perturbation: post-hoc Monte Carlo on q/k/v/o_proj vs TFB low-rank on q_proj+v_proj.
  - ``num_samples``: 8 (tprd config) vs TFB ``config.num_samples`` (5 for Qwen2.5-3B).
  - EU is summed over teacher-forced tokens of the **ASE** full response, not TokUR's.

Previous bug (fixed here): scoring only ``answer_normalized`` (~1--15 tokens) → AUROC ~0.52.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from panda.grading.tokur_records import build_prompt_tfb
from panda.features import tokens_to_response_features
from panda.model import CausalLMWrapper, ModelConfig
from panda.model_load import resolve_weight_suffixes
from panda.pipeline import ATokURPipeline, PipelineConfig
from panda.tfb_load import load_tfb_for_teacher_force
from panda.paths import DEFAULT_OUT


DEFAULT_OUT = DEFAULT_OUT

# Matches configs/tokur_benchmark.yaml weight block (EU-only subset).
TOKUR_PIPE_CFG = {
    "weight": {
        "enabled": True,
        "num_samples": 8,
        "rank": 8,
        "sigma": 0.1,
        "target_suffixes": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "embedding_attack": {"enabled": False},
    "hidden_attack": {"enabled": False},
    "scoring": {"use_margin_collapse": False},
    "aggregation": {"response": "mean", "topk": 5},
}


def shard_path(out_dir: Path, dataset: str, shard_id: int) -> Path:
    return out_dir / dataset / "tokur_baseline_shards" / f"shard_{shard_id}.jsonl"


def merged_path(out_dir: Path, dataset: str) -> Path:
    return out_dir / dataset / "tokur_baseline.jsonl"


def merge_tokur_shards(out_dir: Path, dataset: str) -> int:
    """Merge per-shard outputs into ``tokur_baseline.jsonl`` (dedupe by id)."""
    shard_dir = out_dir / dataset / "tokur_baseline_shards"
    out_path = merged_path(out_dir, dataset)
    by_id: dict[str, dict] = {}
    if shard_dir.exists():
        for p in sorted(shard_dir.glob("shard_*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    by_id[rec["id"]] = rec
    # Keep rows from an in-progress single-GPU merged file not yet in shards.
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                by_id.setdefault(rec["id"], rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rid in sorted(by_id):
            f.write(json.dumps(by_id[rid], ensure_ascii=False) + "\n")
    return len(by_id)


def _read_raw_header(p: Path) -> dict | None:
    try:
        data = p.read_bytes()
    except OSError:
        return None
    if len(data) < 1000:
        return None
    idx = data.find(b'"base_generation"')
    if idx < 0:
        return None
    try:
        return json.loads(data[:idx].rstrip().rstrip(b",") + b"}")
    except json.JSONDecodeError:
        return None


def _load_base_generation(p: Path, max_bytes: int = 4_000_000) -> dict:
    """Load ``base_generation`` without parsing full token traces (multi-MB files)."""
    try:
        with p.open("rb") as f:
            data = f.read(max_bytes)
    except OSError:
        return {}
    idx = data.find(b'"base_generation"')
    if idx < 0:
        return {}
    end = -1
    for marker in (b'"text_rephrase_runs"', b'"weight_perturb_runs"', b'"token_trace"'):
        end = data.find(marker, idx)
        if end >= 0:
            break
    if end < 0:
        return {}
    try:
        obj = json.loads(b"{" + data[idx:end].rstrip().rstrip(b",") + b"}")
    except json.JSONDecodeError:
        return {}
    return obj.get("base_generation") or {}


def load_raw_records(out_dir: Path, dataset: str) -> list[dict]:
    """Load fields needed for approx TokUR EU; header via fast prefix parse."""
    raw_dir = out_dir / dataset / "raw_runs"
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        if p.name.endswith((".error.json", ".partial.json")):
            continue
        header = _read_raw_header(p)
        if not header:
            continue
        base = _load_base_generation(p)
        rows.append(
            {
                "id": header.get("id", p.stem),
                "dataset": header.get("dataset", dataset),
                "question": header.get("question", ""),
                "reference": header.get("reference", ""),
                "is_correct": header.get("is_correct"),
                "base_generation": {
                    "input_prompt": base.get("input_prompt"),
                    "full_response": base.get("full_response") or base.get("response_text"),
                    "answer_normalized": base.get("answer_normalized"),
                    "final_answer": base.get("final_answer"),
                },
            }
        )
    return rows


def _teacher_force_pair(rec: dict, tokenizer, model_path: str = "") -> tuple[str, str]:
    """Return (prompt, response) for official TokUR EU teacher forcing."""
    base = rec.get("base_generation") or {}
    question = rec.get("question") or ""
    prompt = (base.get("input_prompt") or "").strip()
    if not prompt:
        prompt = build_prompt_tfb(question, tokenizer, model_path)
    response = (
        base.get("full_response")
        or base.get("response_text")
        or rec.get("full_response")
        or ""
    ).strip()
    if not response:
        # Last resort: short extracted answer (legacy; poor AUROC)
        response = (
            base.get("answer_normalized")
            or base.get("final_answer")
            or ""
        ).strip()
    return prompt, response


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("approx", "official"),
        default="approx",
        help="approx: this script (post-hoc). official: delegate to score_tokur_official.",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge tokur_baseline_shards/*.jsonl → tokur_baseline.jsonl and exit.",
    )
    args, passthrough = ap.parse_known_args()

    if args.mode == "official":
        import sys

        from panda.core import score_tokur_official

        sys.argv = [
            "score_tokur_official",
            "--out-dir",
            str(args.out_dir),
            "--dataset",
            args.dataset,
            *passthrough,
        ]
        score_tokur_official.main()
        return

    if args.merge_only:
        n = merge_tokur_shards(args.out_dir, args.dataset)
        print(f"Merged {n} rows → {merged_path(args.out_dir, args.dataset)}")
        return

    records = load_raw_records(args.out_dir, args.dataset)
    records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard_id]
    merged = merged_path(args.out_dir, args.dataset)
    out_path = (
        shard_path(args.out_dir, args.dataset, args.shard_id)
        if args.num_shards > 1
        else merged
    )

    assigned_ids = {r["id"] for r in records}
    done: set[str] = set()
    if args.resume:
        for path in (out_path, merged if args.num_shards > 1 else None):
            if path is None or not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec["id"] in assigned_ids:
                        done.add(rec["id"])

    device = args.device if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_tfb_for_teacher_force(args.model_path, device=device, dtype="bfloat16")
    model.eval()
    pipe_cfg = PipelineConfig.from_yaml(TOKUR_PIPE_CFG)
    pipe_cfg.use_weight = True
    pipe_cfg.use_embedding = False
    pipe_cfg.use_hidden = False
    pipe_cfg.use_margin = False
    pipe_cfg.teacher_force_max_length = 4096
    pipe_cfg.weight.target_suffixes = resolve_weight_suffixes(
        model, pipe_cfg.weight.target_suffixes
    )
    wrapper = CausalLMWrapper(
        ModelConfig(args.model_path, device, "bfloat16"), model=model, tokenizer=tokenizer
    )
    pipeline = ATokURPipeline(model=wrapper, config=pipe_cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and out_path.exists() else "w"
    n_new = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for rec in tqdm(records, desc=f"tokur {args.dataset} shard {args.shard_id}"):
            rid = rec["id"]
            if rid in done:
                continue
            prompt, response = _teacher_force_pair(rec, tokenizer, args.model_path)
            row = {
                "id": rid,
                "dataset": args.dataset,
                "tokur_eu_sum": float("nan"),
                "tokur_eu_mean": float("nan"),
                "n_response_tokens": 0,
                "response_chars": len(response),
                "scoring_mode": "approx_posthoc_full_response",
                "error": "",
            }
            if not response:
                row["error"] = "empty_response"
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                continue
            try:
                result = pipeline.score(rid, prompt, response)
                rf = tokens_to_response_features(
                    rid,
                    args.dataset,
                    bool(rec.get("is_correct")),
                    result.tokens,
                )
                row["tokur_eu_sum"] = rf.eu_weight_sum
                row["tokur_eu_mean"] = rf.eu_weight_mean
                row["n_response_tokens"] = rf.num_tokens
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1

    print(f"Wrote {n_new} new rows → {out_path}")
    if args.num_shards > 1 and args.shard_id == 0:
        print("Run --merge-only after all shards finish to build tokur_baseline.jsonl")


if __name__ == "__main__":
    main()