"""Persist raw ASE generations; recompute metrics without re-running the model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def cache_path(out_dir: Path, dataset: str, shard_id: int) -> Path:
    return out_dir / dataset / "cache" / f"shard_{shard_id}.jsonl"


def features_path(out_dir: Path, dataset: str, shard_id: int) -> Path:
    return out_dir / dataset / "shards" / f"shard_{shard_id}.jsonl"


def load_jsonl_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["id"])
    return done


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cache_by_id(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if not path.exists():
        return by_id
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_id[row["id"]] = row
    return by_id


def build_config_blob(
    *,
    n_rephrases: int,
    weight_seeds: list[int],
    max_new_tokens: int,
    weight_sigma: float,
    weight_rank: int,
    model_path: str,
) -> dict[str, Any]:
    return {
        "n_rephrases": n_rephrases,
        "weight_seeds": weight_seeds,
        "max_new_tokens": max_new_tokens,
        "weight_sigma": weight_sigma,
        "weight_rank": weight_rank,
        "model_path": model_path,
    }


def pack_generation(gen: dict, extra: dict | None = None) -> dict:
    """Store one generation trajectory (token lists included for metric recomputation)."""
    row = {
        "response_text": gen.get("response_text", ""),
        "final_answer": gen.get("final_answer", ""),
        "token_entropies": list(gen.get("token_entropies") or []),
        "token_margins": list(gen.get("token_margins") or []),
    }
    if extra:
        row.update(extra)
    return row
