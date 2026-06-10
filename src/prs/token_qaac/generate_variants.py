#!/usr/bin/env python3
"""Generate API rephrases → outputs/qaac_api_bench/{dataset}/variants.jsonl."""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from prs.datasets.loaders import load_dataset_records
from prs.datasets.registry import DATASET_IDS, get_dataset_spec, normalize_dataset_id
from prs.token_qaac.api_rephrase import build_api_client, rephrase_one
from prs.paths import DEFAULT_BENCH, TOKUR_ROOT

DEFAULT_OUT = DEFAULT_BENCH
DEFAULT_TFTTCL = Path("/home/phx/TF-TTCL/data/MATH")
DEFAULT_DEEPSCALER = TOKUR_ROOT / "datasets" / "deepscaler.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_questions(
    dataset: str,
    *,
    tfttcl_root: Path,
    deepscaler_path: Path,
    max_samples: int,
    seed: int,
) -> list[dict]:
    if dataset == "minerva":
        rows = json.loads((tfttcl_root / "minerva.json").read_text(encoding="utf-8"))
        out = []
        for i, r in enumerate(rows[:max_samples]):
            q = str(r.get("instruction", "")).strip()
            out.append({"idx": i, "unique_id": f"minerva_{i}", "question": q})
        return out

    if dataset == "gsm8k":
        rows = json.loads((tfttcl_root / "gsm8k.json").read_text(encoding="utf-8"))
        out = []
        for i, r in enumerate(rows[:max_samples]):
            parts = [str(r.get("instruction", "")).strip(), str(r.get("input", "")).strip()]
            q = "\n".join(p for p in parts if p)
            out.append({"idx": i, "unique_id": f"gsm8k_{i}", "question": q})
        return out

    if dataset == "math500":
        path = tfttcl_root / "math500.json"
        text = path.read_text(encoding="utf-8").strip()
        rows = json.loads(text) if text.startswith("[") else _load_jsonl(path)
        rng = random.Random(seed)
        rows = list(rows)
        rng.shuffle(rows)
        out = []
        for i, r in enumerate(rows):
            q = str(r.get("problem") or r.get("instruction") or "").strip()
            if r.get("input"):
                q = "\n".join(p for p in [q, str(r.get("input")).strip()] if p)
            # Keep raw path ids (e.g. test/precalculus/807.json) to match existing bench.
            uid = str(r.get("unique_id", i))
            out.append({"idx": i, "unique_id": uid, "question": q})
        return out

    if dataset == "aime24":
        rows = json.loads((tfttcl_root / "aime24.json").read_text(encoding="utf-8"))
        out = []
        for i, r in enumerate(rows[:max_samples]):
            q = str(r.get("instruction", "")).strip()
            out.append({"idx": i, "unique_id": f"aime24_{i}", "question": q})
        return out

    if dataset == "deepscaler":
        rows = _load_jsonl(deepscaler_path)
        rng = random.Random(seed)
        rng.shuffle(rows)
        out = []
        for i, r in enumerate(rows[:max_samples]):
            q = str(r.get("problem") or r.get("question") or "").strip()
            uid = str(r.get("unique_id", i)).replace("/", "_").replace(".json", "")
            out.append({"idx": i, "unique_id": f"deepscaler_{uid}", "question": q})
        return out

    try:
        spec = normalize_dataset_id(dataset)
        get_dataset_spec(spec)
        rows = load_dataset_records(
            spec,
            tfttcl_root=tfttcl_root,
            max_samples=max_samples,
            seed=seed,
        )
        return [
            {"idx": i, "unique_id": r["id"], "question": r["question"]}
            for i, r in enumerate(rows)
        ]
    except ValueError:
        pass

    raise ValueError(f"Unsupported dataset: {dataset}")


def load_done_by_id(path: Path, *, n_rephrases: int = 8) -> dict[str, dict]:
    need = n_rephrases + 1
    done: dict[str, dict] = {}
    for row in _load_jsonl(path):
        uid = row.get("unique_id", "")
        variants = row.get("variants") or []
        if uid and len(variants) >= need:
            done[uid] = row
    return done


def plan_pending(
    pool: list[dict],
    complete: dict[str, dict],
    *,
    max_samples: int,
) -> list[dict]:
    """Keep existing rows; add new questions from pool until max_samples."""
    if len(complete) >= max_samples:
        return []
    existing_ids = set(complete.keys())
    next_idx = max((int(r.get("idx", 0)) for r in complete.values()), default=-1) + 1
    pending: list[dict] = []
    for q in pool:
        if q["unique_id"] in existing_ids:
            continue
        pending.append({**q, "idx": next_idx})
        next_idx += 1
        if len(complete) + len(pending) >= max_samples:
            break
    return pending


def generate_variants_for_question(
    client,
    model: str,
    *,
    idx: int,
    unique_id: str,
    question: str,
    n_rephrases: int,
    sleep_s: float,
) -> dict:
    variants = [question]
    seen = {question.strip()}
    for _ in range(n_rephrases):
        for _attempt in range(3):
            try:
                text = rephrase_one(client, model, question)
            except Exception:
                text = ""
            if text and text not in seen and len(text) >= 15:
                seen.add(text)
                variants.append(text)
                break
            time.sleep(sleep_s)
        time.sleep(sleep_s)

    return {
        "idx": idx,
        "unique_id": unique_id,
        "original": question,
        "variants": variants,
        "augment_mode": "api",
    }


def run_dataset(
    dataset: str,
    *,
    out_root: Path,
    tfttcl_root: Path,
    deepscaler_path: Path,
    max_samples: int,
    n_rephrases: int,
    model: str,
    workers: int,
    sleep_s: float,
    seed: int,
    resume: bool,
) -> None:
    ds_dir = out_root / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    out_path = ds_dir / "variants.jsonl"

    pool = load_questions(
        dataset,
        tfttcl_root=tfttcl_root,
        deepscaler_path=deepscaler_path,
        max_samples=max_samples,
        seed=seed,
    )
    done = load_done_by_id(out_path, n_rephrases=n_rephrases) if resume else {}
    pending = plan_pending(pool, done, max_samples=max_samples)

    print(
        f"[{dataset}] target={max_samples} pool={len(pool)} "
        f"done={len(done)} pending={len(pending)}"
    )

    if not pending:
        print(f"[{dataset}] already complete → {out_path}")
        return

    client = build_api_client()

    def _one(q: dict) -> dict:
        # Each thread gets its own client to avoid shared-state issues.
        local_client = build_api_client()
        return generate_variants_for_question(
            local_client,
            model,
            idx=q["idx"],
            unique_id=q["unique_id"],
            question=q["question"],
            n_rephrases=n_rephrases,
            sleep_s=sleep_s,
        )

    new_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_one, q): q for q in pending}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"API rephrase [{dataset}]"):
            q = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                print(f"ERROR {q['unique_id']}: {exc}")
                continue
            new_rows.append(row)
            if len(row.get("variants") or []) < n_rephrases + 1:
                print(f"WARN {q['unique_id']}: only {len(row.get('variants', [])) - 1} rephrases")

    # Merge: keep idx order, prefer newly generated rows on conflict.
    merged: dict[str, dict] = dict(done)
    for row in new_rows:
        merged[row["unique_id"]] = row

    ordered = sorted(merged.values(), key=lambda r: int(r.get("idx", 0)))[:max_samples]

    with out_path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    complete = sum(1 for r in ordered if len(r.get("variants") or []) >= n_rephrases + 1)
    print(f"[{dataset}] wrote {len(ordered)} rows ({complete} with >={n_rephrases} rephrases) → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="API rephrase → qaac_api_bench variants.jsonl")
    ap.add_argument("--datasets", default="deepscaler")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tfttcl-root", type=Path, default=DEFAULT_TFTTCL)
    ap.add_argument("--deepscaler-path", type=Path, default=DEFAULT_DEEPSCALER)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--n-rephrases", type=int, default=8)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sleep-s", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    for ds in args.datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        max_n = args.max_samples
        if ds == "aime24":
            max_n = min(max_n, 30)
        run_dataset(
            ds,
            out_root=args.out_dir,
            tfttcl_root=args.tfttcl_root,
            deepscaler_path=args.deepscaler_path,
            max_samples=max_n,
            n_rephrases=args.n_rephrases,
            model=args.model,
            workers=args.workers,
            sleep_s=args.sleep_s,
            seed=args.seed,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()