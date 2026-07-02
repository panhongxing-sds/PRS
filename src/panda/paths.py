"""Repository-root paths (clone-and-run friendly)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("PANDA_ROOT", Path(__file__).resolve().parents[2]))
OUTPUTS = Path(os.environ.get("PANDA_OUTPUTS", REPO_ROOT / "outputs"))
MODELS = Path(os.environ.get("PANDA_MODELS", REPO_ROOT / "models"))
TOKUR_ROOT = Path(os.environ.get("TOKUR_ROOT", REPO_ROOT / "third_party" / "TokUR"))
PAPER = REPO_ROOT / "paper"

DEFAULT_OUT = OUTPUTS / "panda_full"
DEFAULT_BENCH = OUTPUTS / "qaac_api_bench"
DEFAULT_MODEL = MODELS / "TFB-Qwen2.5-3B-Instruct"
