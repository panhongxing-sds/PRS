"""Answer-level 采样：从 data/questions/ 读题，写入 data/samples/。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from grading import extract_answer, is_correct
from sampling_utils import build_prompt, load_model

ROOT = Path(__file__).resolve().parent
QUESTIONS_DIR = ROOT / "data" / "questions"


def load_benchmark(name: str) -> tuple[list[dict], str]:
    path = QUESTIONS_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows, rows[0]["grading"]


def generate_answers(
    model, tokenizer, batch: list[dict], model_path: str, grading: str,
    device: str, k: int, k_chunk: int, max_tokens: int, temp: float, top_p: float,
    seed: int, bi: int,
) -> list[list[str]]:
    prompts = [build_prompt(q["problem"], tokenizer, model_path, grading) for q in batch]
    enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    plen = enc["input_ids"].shape[1]
    texts_batch: list[list[str]] = [[] for _ in batch]
    for chunk_i, start in enumerate(range(0, k, k_chunk)):
        n_seq = min(k_chunk, k - start)
        torch.manual_seed(seed * 100003 + bi * 10 + chunk_i)
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=True, temperature=temp, top_p=top_p,
                num_return_sequences=n_seq, max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)
        for j in range(len(batch)):
            texts_batch[j].extend(decoded[j * n_seq: (j + 1) * n_seq])
    torch.cuda.empty_cache()
    return texts_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--k-chunk", type=int, default=0)
    ap.add_argument("--prompt-batch", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    model_tag = cfg["model"]["tag"]
    model_path = os.path.expandvars(cfg["model"]["path"])
    samp = cfg["sampling"]
    k = args.k or samp["k"]
    k_chunk = args.k_chunk or int(os.environ.get("K_CHUNK", 0)) or int(samp.get("k_chunk", 4))
    prompt_batch = args.prompt_batch or int(os.environ.get("PROMPT_BATCH", 0)) or int(samp.get("prompt_batch", 1))
    temp = args.temp or samp["temperature"]
    top_p = args.top_p or samp["top_p"]
    seed = args.seed or cfg["seeds"][0]
    max_tokens = args.max_tokens or cfg["max_new_tokens"].get(args.benchmark, 2048)
    if max_tokens >= 4096:
        if "1b" in model_tag.lower():
            prompt_batch = min(prompt_batch, 2)
        else:
            prompt_batch = max(1, prompt_batch // 2)
            k_chunk = max(2, k_chunk // 2)
    print(f"[batch] k_chunk={k_chunk} prompt_batch={prompt_batch} max_tokens={max_tokens}", flush=True)

    if args.benchmark not in cfg["benchmarks"]:
        raise SystemExit(f"未知 benchmark: {args.benchmark}")

    rows, grading = load_benchmark(args.benchmark)
    rows = rows[args.shard_id::args.num_shards]
    dataset = args.benchmark

    model, tokenizer = load_model(model_path, args.device)

    out_dir = ROOT / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.num_shards == 1 else f".shard{args.shard_id}"
    out_path = out_dir / f"samples_{model_tag}_seed{seed}_{args.benchmark}{suffix}.jsonl"

    done: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    todo = [q for q in rows if q["id"] not in done]
    if not todo:
        print(f"[skip] {args.benchmark} 已完成")
        return

    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        bi = 0
        while bi < len(todo):
            attempt_bs = min(prompt_batch, len(todo) - bi)
            kc = k_chunk
            while True:
                sub = todo[bi: bi + attempt_bs]
                try:
                    texts_batch = generate_answers(
                        model, tokenizer, sub, model_path, grading, args.device,
                        k, kc, max_tokens, temp, top_p, seed, bi,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if attempt_bs > 1:
                        attempt_bs = max(1, attempt_bs // 2)
                        print(f"[OOM] prompt_batch → {attempt_bs}", flush=True)
                    elif kc > 1:
                        kc = max(1, kc // 2)
                        print(f"[OOM] k_chunk → {kc}", flush=True)
                    else:
                        raise
            for q, texts in zip(sub, texts_batch):
                gold = q["answer"]
                answers = [extract_answer(t, grading, dataset) for t in texts]
                correct = [int(is_correct(a, gold, grading)) for a in answers]
                f.write(json.dumps({
                    "id": q["id"], "dataset": dataset, "benchmark": args.benchmark,
                    "seed": seed, "gold": gold, "grading": grading,
                    "label_drop": 0, "answers": answers, "correct": correct,
                }, ensure_ascii=False) + "\n")
            bi += len(sub)
            if bi % 5 == 0 or bi == len(todo):
                print(f"[{args.benchmark}] {bi}/{len(todo)}", flush=True)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
