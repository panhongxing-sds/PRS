"""Central registry for math, logic, and code benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Domain = Literal["math", "logic", "code"]
GradingMode = Literal["math", "string", "code"]


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    label: str
    domain: Domain
    grading: GradingMode
    jsonl_name: str
    hf_repo: str | None = None
    default_max_samples: int = 200
    aliases: tuple[str, ...] = ()


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("minerva", "Minerva", "math", "math", "minerva.json", default_max_samples=272),
    DatasetSpec("math500", "MATH-500", "math", "math", "math500.json", hf_repo="Tyrion279/math500", default_max_samples=500),
    DatasetSpec("gsm8k", "GSM8K", "math", "math", "gsm8k.json", hf_repo="Tyrion279/gsm8k", default_max_samples=500),
    DatasetSpec(
        "deepscaler",
        "DeepScaler",
        "math",
        "math",
        "deepscaler.jsonl",
        hf_repo="Tyrion279/deepscaler",
        default_max_samples=500,
    ),
    DatasetSpec(
        "leg_counting",
        "Leg Counting",
        "logic",
        "string",
        "leg-counting.jsonl",
        hf_repo="Tyrion279/leg-counting",
        default_max_samples=100,
        aliases=("leg-counting", "leg_counting"),
    ),
    DatasetSpec(
        "zebra_puzzles",
        "Zebra Puzzles",
        "logic",
        "string",
        "zebra_puzzles.jsonl",
        hf_repo="Tyrion279/zebra-puzzles",
        default_max_samples=200,
        aliases=("zebra-puzzles", "zebra_puzzles"),
    ),
    DatasetSpec(
        "color_cube",
        "Color Cube",
        "logic",
        "string",
        "color_cube.jsonl",
        hf_repo="Tyrion279/color-cube",
        default_max_samples=200,
        aliases=("color-cube", "color_cube"),
    ),
    DatasetSpec(
        "humaneval",
        "HumanEval",
        "code",
        "code",
        "humaneval.jsonl",
        hf_repo="openai/openai_humaneval",
        default_max_samples=164,
        aliases=("HumanEval",),
    ),
)

_BY_ID: dict[str, DatasetSpec] = {}
_BY_ALIAS: dict[str, DatasetSpec] = {}
for _spec in DATASETS:
    _BY_ID[_spec.id] = _spec
    for _alias in _spec.aliases:
        _BY_ALIAS[_alias] = _spec

DATASET_IDS = tuple(_spec.id for _spec in DATASETS)
MATH_DATASET_IDS = tuple(s.id for s in DATASETS if s.domain == "math")
LOGIC_DATASET_IDS = tuple(s.id for s in DATASETS if s.domain == "logic")
CODE_DATASET_IDS = tuple(s.id for s in DATASETS if s.domain == "code")

# Paper main-table presets
MAINTABLE_MATH = MATH_DATASET_IDS
MAINTABLE_LOGIC_CODE = LOGIC_DATASET_IDS + CODE_DATASET_IDS
MAINTABLE_ALL = DATASET_IDS

# Decoding / generation seeds for 3-seed mean±std tables (distinct from weight-perturb seeds)
DEFAULT_EXPERIMENT_SEEDS: tuple[int, ...] = (41, 42, 43)


def normalize_dataset_id(name: str) -> str:
    key = name.strip().replace("-", "_")
    if key in _BY_ID:
        return key
    if key in _BY_ALIAS:
        return _BY_ALIAS[key].id
    alt = name.strip()
    if alt in _BY_ALIAS:
        return _BY_ALIAS[alt].id
    raise ValueError(f"Unknown dataset: {name!r}. Known: {list_dataset_ids()}")


def get_dataset_spec(name: str) -> DatasetSpec:
    return _BY_ID[normalize_dataset_id(name)]


def list_dataset_ids(*, domain: Domain | None = None) -> list[str]:
    if domain is None:
        return list(DATASET_IDS)
    return [s.id for s in DATASETS if s.domain == domain]
