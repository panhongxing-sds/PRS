"""Optional API rephrase via OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

REPHRASE_SYSTEM = """Rewrite the following math problem into a semantically equivalent version.

Requirements:
1. Preserve all mathematical conditions exactly.
2. Do not solve the problem.
3. Do not add hints.
4. Do not change the final answer.
5. Change the wording, sentence structure, or order of conditions when possible.
6. Output only the rewritten problem."""


def _parse_rephrase(content: str) -> str:
    content = content.strip()
    if not content:
        return ""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and obj.get("rewritten_problem"):
            return str(obj["rewritten_problem"]).strip()
    except json.JSONDecodeError:
        pass
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def rephrase_one(client, model: str, question: str, *, max_tokens: int = 1024) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REPHRASE_SYSTEM},
            {"role": "user", "content": f"Problem:\n{question}"},
        ],
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        content = str(getattr(msg, "reasoning_content", "") or "").strip()
    return _parse_rephrase(content)


def build_api_client():
    from openai import OpenAI

    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("QAAC_API_KEY")
    )
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or OPENAI_API_KEY for API rephrase")
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com/v1"
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_rephrases(
    *,
    qid: str,
    question: str,
    n: int,
    out_path: Path,
    model: str,
    sleep_s: float = 0.2,
) -> list[str]:
    client = build_api_client()
    rows: list[dict] = []
    seen: set[str] = {question.strip()}
    rephrases: list[str] = []

    for rewrite_id in range(1, n + 1):
        for attempt in range(3):
            try:
                text = rephrase_one(client, model, question)
            except Exception:
                text = ""
            if text and text not in seen and len(text) >= 15:
                seen.add(text)
                rephrases.append(text)
                rows.append(
                    {
                        "qid": qid,
                        "rewrite_id": rewrite_id,
                        "rephrased_question": text,
                    }
                )
                break
            time.sleep(sleep_s)
        time.sleep(sleep_s)

    if rows:
        with out_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rephrases
