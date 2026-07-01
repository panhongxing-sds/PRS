"""vLLM 批采样：多题并行 + 每题 n=K（分块避免 OOM）。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from grading import extract_answer, is_correct
from sampling_utils import build_prompt, load_tokenizer, prepare_vllm_model_path, use_hf_backend

ROOT = Path(__file__).resolve().parent
QUESTIONS_DIR = ROOT / "data" / "questions"


def load_benchmark(name: str) -> tuple[list[dict], str]:
    path = QUESTIONS_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows, rows[0]["grading"]


def run_hf_fallback(argv: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "sample.py"), *argv]
    print(f"[fallback HF] {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def generate_vllm_batch(llm, SamplingParams, prompts, batch_qs, k, k_chunk, temp, top_p,
                        max_tokens, seed, bi, grading, dataset, model_path, tokenizer):
    # 一次性提交整个 batch（每个 prompt n=K），交给 vLLM 连续批处理调度，
    # 这样所有题 × K 个序列同时解码，吞吐远高于逐题/逐 chunk 串行。
    sp = SamplingParams(
        n=k, temperature=temp, top_p=top_p,
        max_tokens=max_tokens, seed=seed * 100003 + bi * 10,
    )
    outs = llm.generate(prompts, sp, use_tqdm=False)
    texts_batch = [[c.text for c in o.outputs] for o in outs]
    results = []
    for q, texts in zip(batch_qs, texts_batch):
        gold = q["answer"]
        answers = [extract_answer(t, grading, dataset) for t in texts]
        correct = [int(is_correct(a, gold, grading)) for a in answers]
        results.append({
            "id": q["id"], "dataset": dataset, "benchmark": dataset,
            "seed": seed, "gold": gold, "grading": grading,
            "label_drop": 0, "answers": answers, "correct": correct,
        })
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--k-chunk", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-suffix", default="", help="自定义输出文件后缀，避免与其它分片撞名")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    model_path = os.path.expandvars(cfg["model"]["path"])
    if use_hf_backend(model_path):
        run_hf_fallback([a for a in sys.argv[1:] if a != __file__])
        return

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[warn] vLLM 未安装，回退 HF sample.py", flush=True)
        run_hf_fallback([a for a in sys.argv[1:] if a != __file__])
        return

    model_tag = cfg["model"]["tag"]
    samp = cfg["sampling"]
    vllm_cfg = cfg.get("vllm", {})
    k = args.k or samp["k"]
    k_chunk = args.k_chunk or int(os.environ.get("K_CHUNK", 0)) or int(samp.get("k_chunk", 16))
    temp = args.temp or samp["temperature"]
    top_p = args.top_p or samp["top_p"]
    seed = args.seed or cfg["seeds"][0]
    max_tokens = args.max_tokens or cfg["max_new_tokens"].get(args.benchmark, 2048)
    batch_size = args.batch_size or int(os.environ.get("PROMPT_BATCH", 0)) or vllm_cfg.get("batch_size", 8)

    if args.benchmark not in cfg["benchmarks"]:
        raise SystemExit(f"未知 benchmark: {args.benchmark}")

    rows, grading = load_benchmark(args.benchmark)
    rows = rows[args.shard_id::args.num_shards]
    dataset = args.benchmark

    tokenizer = load_tokenizer(model_path)
    vllm_path = prepare_vllm_model_path(model_path)
    # 上下文按需收紧：prompt 短，留 max_tokens + 余量即可，避免 KV cache 过大 OOM；
    # 但必须 >= 该 benchmark 的 max_tokens + headroom（如 aime 需 4096 输出）。
    headroom = int(vllm_cfg.get("prompt_headroom", 1536))
    need_mml = max_tokens + headroom
    cfg_mml = int(os.environ.get("MAX_MODEL_LEN", 0)) or int(vllm_cfg.get("max_model_len", 0))
    max_model_len = max(cfg_mml, need_mml) if cfg_mml else need_mml
    # 限制同时解码的序列数，vLLM 自动排队 → 满载且不 OOM
    max_num_seqs = int(os.environ.get("MAX_NUM_SEQS", 0)) or int(vllm_cfg.get("max_num_seqs", 128))
    print(f"[vLLM] {model_tag} batch={batch_size} k={k} max_tokens={max_tokens} "
          f"max_model_len={max_model_len} max_num_seqs={max_num_seqs}", flush=True)
    llm = LLM(
        model=vllm_path,
        dtype=vllm_cfg.get("dtype", "bfloat16"),
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.9)),
        trust_remote_code=True,
        enforce_eager=bool(vllm_cfg.get("enforce_eager", False)),
        tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
    )

    out_dir = ROOT / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_suffix:
        suffix = args.out_suffix
    elif args.num_shards == 1:
        suffix = ""
    else:
        suffix = f".shard{args.shard_id}"
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
        for bi in range(0, len(todo), batch_size):
            batch = todo[bi: bi + batch_size]
            prompts = [build_prompt(q["problem"], tokenizer, model_path, grading) for q in batch]
            rows_out = generate_vllm_batch(
                llm, SamplingParams, prompts, batch, k, k_chunk, temp, top_p,
                max_tokens, seed, bi, grading, dataset, model_path, tokenizer,
            )
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done_n = bi + len(batch)
            if done_n % 10 == 0 or done_n == len(todo):
                print(f"[{args.benchmark}] {done_n}/{len(todo)}", flush=True)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
