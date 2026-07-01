#!/usr/bin/env python3
"""保存 Qwen 7B SCR 题（清洗后，可调 tau）的完整 reasoning + token 级信息。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("PRS_ROOT", "/root/PRS") + "/src")

import yaml
from vllm import LLM, SamplingParams

from grading import extract_answer
from sampling_utils import build_prompt, load_tokenizer, prepare_vllm_model_path

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/root/autodl-tmp/prs-models/Qwen2.5-7B-Instruct"
)
OUT_DIR = ROOT / "data" / "scr_reasoning" / "qwen25_7b"
SCR_LIST = ROOT / "figures" / "qwen25_7b_scr_questions_clean.md"

MAX_TOKENS = {
    "deepscaler": 2048,
    "gpqa_diamond": 2048,
    "aime_2024": 4096,
}


def load_scr_ids(tau: float = 0.95) -> list[dict]:
    """清洗后 SCR 题（majority 错且 p_top >= tau）。"""
    cache = Path(f"/root/autodl-tmp/tmp/scr7b_clean_t{tau:.2f}.json")
    if cache.exists():
        return json.load(open(cache, encoding="utf-8"))
    from collections import Counter

    samples = {}
    for f in (ROOT / "data" / "samples").glob("samples_qwen25_7b_seed41_*.jsonl"):
        if str(f).endswith(".bak"):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                samples[r["id"]] = r
    questions = {}
    for f in (ROOT / "data" / "questions").glob("*.jsonl"):
        if f.name == "all_questions.jsonl":
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                questions[r["id"]] = r
    scr = []
    for qid, r in samples.items():
        if int(r.get("label_drop", 0)) == 1:
            continue
        pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a]
        if len(pairs) < 2:
            continue
        a, c = zip(*pairs)
        cnt = Counter(a)
        maj, top = cnt.most_common(1)[0]
        cmap = {ai: ci for ai, ci in zip(a, c)}
        if cmap.get(maj, 0) == 0 and top / len(a) >= tau:
            q = questions.get(qid, {})
            scr.append({
                "id": qid,
                "bench": r.get("benchmark", r.get("dataset", "")),
                "gold": r["gold"],
                "maj": maj,
                "p_top": top / len(a),
                "grading": r.get("grading", "math"),
                "problem": q.get("problem", ""),
            })
    scr.sort(key=lambda x: (x["bench"], -x["p_top"], x["id"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(scr, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return scr


def token_records(tokenizer, token_ids: list[int], logprobs_list) -> list[dict]:
    """逐 token 序列化：id、decoded、logprob。"""
    recs = []
    for i, tid in enumerate(token_ids):
        entry = {
            "pos": i,
            "id": int(tid),
            "token": tokenizer.convert_ids_to_tokens(int(tid)),
            "text": tokenizer.decode([int(tid)], skip_special_tokens=False),
        }
        if logprobs_list and i < len(logprobs_list) and logprobs_list[i]:
            lp_dict = logprobs_list[i]
            if tid in lp_dict:
                lp = lp_dict[tid]
                entry["logprob"] = float(getattr(lp, "logprob", lp))
            else:
                for k, v in lp_dict.items():
                    entry["logprob"] = float(getattr(v, "logprob", v))
                    break
            entry["top_logprobs"] = {
                str(k): float(getattr(v, "logprob", v))
                for k, v in list(lp_dict.items())[:5]
            }
        recs.append(entry)
    return recs


def serialize_sample(
    qmeta: dict,
    sample_idx: int,
    tokenizer,
    req_out,
    comp,
) -> dict:
    prompt_ids = list(req_out.prompt_token_ids or [])
    comp_ids = list(comp.token_ids)
    extracted = extract_answer(comp.text, qmeta["grading"], qmeta["bench"])
    row = {
        "id": qmeta["id"],
        "benchmark": qmeta["bench"],
        "sample_idx": sample_idx,
        "seed": 41,
        "gold": qmeta["gold"],
        "stored_scr_maj_wrong": qmeta["maj"],
        "stored_p_top": qmeta["p_top"],
        "text": comp.text,
        "extracted_answer": extracted,
        "finish_reason": comp.finish_reason,
        "stop_reason": comp.stop_reason,
        "cumulative_logprob": comp.cumulative_logprob,
        "n_prompt_tokens": len(prompt_ids),
        "n_completion_tokens": len(comp_ids),
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": comp_ids,
        "prompt_tokens": token_records(
            tokenizer, prompt_ids, req_out.prompt_logprobs
        ),
        "completion_tokens": token_records(
            tokenizer, comp_ids, comp.logprobs
        ),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=1.0, help="SCR 阈值 p_top>=")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    scr_all = load_scr_ids(args.tau)
    scr = scr_all[args.shard_id :: args.num_shards]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tau_tag = f"t{args.tau:.2f}".replace(".", "")
    out_sub = OUT_DIR / tau_tag
    out_sub.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    vllm_cfg = cfg.get("vllm", {})
    tokenizer = load_tokenizer(MODEL_PATH)
    vllm_path = prepare_vllm_model_path(MODEL_PATH)

    max_mml = int(vllm_cfg.get("max_model_len", 5120))
    max_mml = max(max_mml, 5120)

    print(
        f"[scr-reasoning] tau>={args.tau} shard {args.shard_id}/{args.num_shards} "
        f"{len(scr)}/{len(scr_all)} 题, K={args.k}",
        flush=True,
    )
    llm = LLM(
        model=vllm_path,
        dtype=vllm_cfg.get("dtype", "bfloat16"),
        max_model_len=max_mml,
        max_num_seqs=min(64, args.k),
        gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.92)),
        trust_remote_code=True,
    )

    manifest = {
        "model": "qwen25_7b",
        "model_path": MODEL_PATH,
        "k": args.k,
        "temperature": 0.5,
        "top_p": 0.95,
        "seed": 41,
        "tau": args.tau,
        "out_dir": str(out_sub.relative_to(ROOT)),
        "n_questions_total": len(scr_all),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "questions": [],
    }

    for qi, qmeta in enumerate(scr):
        qid = qmeta["id"]
        out_path = out_sub / f"{qid}.jsonl"
        if args.resume and out_path.exists():
            n_done = sum(1 for _ in open(out_path, encoding="utf-8") if _.strip())
            if n_done >= args.k:
                print(f"[{qi+1}/{len(scr)}] skip {qid} ({n_done} samples)", flush=True)
                manifest["questions"].append({"id": qid, "path": str(out_path.relative_to(ROOT)), "n_samples": n_done})
                continue

        bench = qmeta["bench"]
        max_tokens = MAX_TOKENS.get(bench, 2048)
        prompt = build_prompt(qmeta["problem"], tokenizer, MODEL_PATH, qmeta["grading"])
        sp = SamplingParams(
            n=args.k,
            temperature=0.5,
            top_p=0.95,
            max_tokens=max_tokens,
            seed=41,
            logprobs=1,
            prompt_logprobs=1,
        )
        req_outs = llm.generate([prompt], sp, use_tqdm=False)
        req = req_outs[0]
        with open(out_path, "w", encoding="utf-8") as f:
            for si, comp in enumerate(req.outputs):
                row = serialize_sample(qmeta, si, tokenizer, req, comp)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{qi+1}/{len(scr)}] {qid} → {out_path.name} ({len(req.outputs)} samples)", flush=True)
        manifest["questions"].append({
            "id": qid,
            "benchmark": bench,
            "path": str(out_path.relative_to(ROOT)),
            "n_samples": len(req.outputs),
            "stored_p_top": qmeta["p_top"],
        })

    man_path = out_sub / f"manifest.shard{args.shard_id}.json"
    json.dump(manifest, open(man_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"→ {man_path}", flush=True)


if __name__ == "__main__":
    main()
