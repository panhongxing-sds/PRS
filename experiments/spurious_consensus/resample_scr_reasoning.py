"""对 SCR 题重采样完整输出，比较 reasoning 文本相似度。"""
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("/root/spurious-consensus")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/PANDA/src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sampling_utils import build_prompt

MODEL = "/root/autodl-tmp/panda-models/Qwen2.5-7B-Instruct"
OUT = ROOT / "results" / "scr7b_reasoning_resample.json"
N = 8
MAX_NEW = 1536
SEED = 42

scr = json.load(open("/root/autodl-tmp/tmp/scr7b_clean.json"))
# 分层抽 10 题：p_top=1 且全同 / p_top=1 多样 / p_top~0.9
div = json.load(open("/root/autodl-tmp/tmp/scr7b_diversity.json"))
by_id = {d["id"]: d for d in div}
pick = []
for pred in [
    lambda d: d["p_top"] == 1.0 and d["all_same_clean"],
    lambda d: d["p_top"] == 1.0 and not d["all_same_clean"],
    lambda d: 0.9 <= d["p_top"] < 1.0,
]:
    for d in sorted(div, key=lambda x: -x["p_top"]):
        if pred(d) and d["id"] not in pick:
            pick.append(d["id"])
            break
# 再补 deepscaler 高 p_top
for d in sorted(div, key=lambda x: (-x["p_top"], x["n_unique_clean"])):
    if len(pick) >= 10:
        break
    if d["id"] not in pick:
        pick.append(d["id"])

questions = {}
for f in (ROOT / "data" / "questions").glob("*.jsonl"):
    if f.name == "all_questions.jsonl":
        continue
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            questions[r["id"]] = r

BOXED = re.compile(r"\\boxed\{", re.I)

def strip_answer(t: str) -> str:
    m = BOXED.search(t)
    return t[: m.start()].strip() if m else t.strip()

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def analyze_texts(texts: list[str]) -> dict:
    rs = [strip_answer(t) for t in texts]
    n = len(rs)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(sim(rs[i], rs[j]))
    avg_pair = sum(pairs) / len(pairs) if pairs else 0.0
    # 与第一条的相似度
    ref_sims = [sim(rs[0], rs[k]) for k in range(1, n)] if n > 1 else []
    uniq = len(set(re.sub(r"\s+", " ", x) for x in rs))
    return {
        "n": n,
        "avg_pairwise_sim": avg_pair,
        "sim_to_first_mean": sum(ref_sims) / len(ref_sims) if ref_sims else 1.0,
        "sim_to_first_min": min(ref_sims) if ref_sims else 1.0,
        "unique_reasoning_bodies": uniq,
        "exact_reasoning_match_rate": Counter(re.sub(r"\s+", " ", x) for x in rs).most_common(1)[0][1] / n,
        "reasoning_prefix200_same": len(set(x[:200] for x in rs)) == 1,
    }

print(f"重采样 {len(pick)} 题 × N={N} (CPU)...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
)
model.eval()

results = []
for qid in pick:
    q = questions[qid]
    prompt = build_prompt(q["problem"], tok, MODEL, q.get("grading", "math"))
    texts = []
    for k in range(N):
        inp = tok(prompt, return_tensors="pt")
        g = torch.Generator().manual_seed(SEED + hash(qid) % 10000 + k)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=0.5,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
                generator=g,
            )
        texts.append(tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))
    meta = analyze_texts(texts)
    meta.update({
        "id": qid,
        "bench": q["benchmark"],
        "p_top_stored": by_id[qid]["p_top"],
        "n_unique_clean_stored": by_id[qid]["n_unique_clean"],
        "gold": q["answer"],
        "sample0_reasoning_head": strip_answer(texts[0])[:600],
        "sample1_reasoning_head": strip_answer(texts[1])[:600] if len(texts) > 1 else "",
    })
    results.append(meta)
    print(f"  {qid}: avg_sim={meta['avg_pairwise_sim']:.3f} uniq={meta['unique_reasoning_bodies']} exact={meta['exact_reasoning_match_rate']:.2f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
print(f"→ {OUT}")
