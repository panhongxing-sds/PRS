#!/usr/bin/env python3
"""One-shot ASE -> PANDA/core rebrand. Run from repo root."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".vllm_venv",
    ".tokur_venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}

SKIP_PATH_PARTS = (
    "/third_party/",
    "/.vllm_venv/",
    "/.tokur_venv/",
    "/experiments/spurious_consensus/data/questions/",
    "/experiments/spurious_consensus/data/benchmarks/",
    "/experiments/spurious_consensus/data/samples/",
    "/paper/analysis/logs/",
)

TEXT_EXTENSIONS = {
    ".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".tex", ".json",
    ".txt", ".cfg", ".ini", ".bat", ".fish", ".csh",
}

FILE_RENAMES = [
    ("src/panda/ase", "src/panda/core"),
    ("src/panda/core/run_ase_experiment.py", "src/panda/core/run_panda_experiment.py"),
    ("src/panda/core/run_advw_ase_experiment.py", "src/panda/core/run_advw_panda_experiment.py"),
    ("tests/test_advw_ase.py", "tests/test_advw_panda.py"),
    ("scripts/run_ase_model_pipeline.sh", "scripts/run_panda_model_pipeline.sh"),
    ("scripts/recompute_ase_metrics.sh", "scripts/recompute_panda_metrics.sh"),
    ("scripts/launch_ase_full_8gpu.sh", "scripts/launch_panda_full_8gpu.sh"),
    ("scripts/queue_ase_extra_models.sh", "scripts/queue_panda_extra_models.sh"),
    ("scripts/ase_gpu_lock.sh", "scripts/panda_gpu_lock.sh"),
]

# Order matters: longer / more specific first
REPLACEMENTS = [
    ("python -m panda.ase.", "python -m panda.core."),
    ("from panda.ase.", "from panda.core."),
    ("import panda.ase.", "import panda.core."),
    ("panda/ase/", "panda/core/"),
    ("panda.ase.", "panda.core."),
    ("run_advw_ase_experiment", "run_advw_panda_experiment"),
    ("run_ase_experiment", "run_panda_experiment"),
    ("run_ase_model_pipeline", "run_panda_model_pipeline"),
    ("recompute_ase_metrics", "recompute_panda_metrics"),
    ("launch_ase_full_8gpu", "launch_panda_full_8gpu"),
    ("queue_ase_extra_models", "queue_panda_extra_models"),
    ("ase_gpu_lock", "panda_gpu_lock"),
    ("test_advw_ase", "test_advw_panda"),
    ("AdvW-ASE", "AdvW-PANDA"),
    ("ASE pipeline", "PANDA pipeline"),
    ("ASE jsonl", "PANDA jsonl"),
    ("ASE FULL", "PANDA FULL"),
    ("ASE extra-model", "PANDA extra-model"),
    ('desc=f"ASE [', 'desc=f"PANDA ['),
    ("ASE with full raw_runs", "PANDA with full raw_runs"),
    ("Adversarial Semantic Entropy (ASE)", "PANDA core method"),
    ("Compute ASE/ATU", "Compute PANDA/ATU"),
    ("ASE H_norm", "PANDA H_norm"),
    ("ASE vs avg", "PANDA vs avg"),
    ("ASE_ATTN_IMPLEMENTATION", "PANDA_ATTN_IMPLEMENTATION"),
    ("ASE_DYNAMIC_CLAIM", "PANDA_DYNAMIC_CLAIM"),
    ("ASE_SKIP_TOKUR", "PANDA_SKIP_TOKUR"),
    ("ASE_SKIP_GENERATE", "PANDA_SKIP_GENERATE"),
    ("ASE_PARALLEL_RECOMPUTE", "PANDA_PARALLEL_RECOMPUTE"),
    ("ASE_TOKUR_PARALLEL_SHARDS", "PANDA_TOKUR_PARALLEL_SHARDS"),
    ("ASE_TOKUR_STRICT", "PANDA_TOKUR_STRICT"),
    ("ASE_GPU_MIN_FREE_MIB", "PANDA_GPU_MIN_FREE_MIB"),
    ("ASE_MAX_TOKENS_GSM8K", "PANDA_MAX_TOKENS_GSM8K"),
    ("ASE_MAX_TOKENS", "PANDA_MAX_TOKENS"),
    ("ASE_8B_SEQUENTIAL", "PANDA_8B_SEQUENTIAL"),
    ("ASE_FAST", "PANDA_FAST"),
    ("ASE_JSONL", "PANDA_JSONL"),
    ("ASE_ATTN", "PANDA_ATTN"),
    ("/outputs/ase_full", "/outputs/panda_full"),
    ("outputs/ase_full", "outputs/panda_full"),
    ("ase_full", "panda_full"),
    ("/tmp/prs_tmp", "/tmp/panda_tmp"),
    (".prs_v2_aggregated_complete", ".panda_v2_aggregated_complete"),
    ("prs_v2_aggregated", "panda_v2_aggregated"),
    ("prs_download_models", "panda_download_models"),
    ("prs_full_20260615", "panda_full_20260615"),
    ("prs_no_bd", "panda_no_bd"),
    ("prs_no_T", "panda_no_T"),
    ("prs_F_bd_T", "panda_F_bd_T"),
    ("prs_F_bd", "panda_F_bd"),
    ("prs_cols", "panda_cols"),
    ("prs_m", "panda_m"),
    ("prs_ok", "panda_ok"),
    ("prs_s", "panda_s"),
    ("prs_auroc", "panda_auroc"),
    ("prs_auprc", "panda_auprc"),
    ("prs_root", "panda_root"),
    ("delta_prs_minus", "delta_panda_minus"),
    ("macro[\"prs_full\"]", 'macro["panda_full"]'),
    ('("PANDA (Ours)", "prs_full")', '("PANDA (Ours)", "panda_full")'),
    ("run_advw_ase_full", "run_advw_panda_full"),
    ("pgrep -f \"run_advw_ase", 'pgrep -f "run_advw_panda'),
    ("Legacy qwen3b/ase_full", "Legacy qwen3b/panda_full"),
    ("qwen3b/ase_full", "qwen3b/panda_full"),
    ("ase_{dataset}", "panda_{dataset}"),
    ("ase_{model_tag}_{dataset}", "panda_{model_tag}_{dataset}"),
    ("ase_{ds}", "panda_{ds}"),
    ("# Legacy qwen3b/ase_full: ase_", "# Legacy qwen3b/panda_full: panda_"),
    ("abl_PRS_full", "abl_PANDA_full"),
    ("prs_iclr2026", "panda_iclr2026"),
    ("compute_prs_from_row", "compute_panda_from_row"),
    ("compute_prs", "compute_panda"),
    ("enrich_row_with_prs", "enrich_row_with_panda"),
    ("KEY_PRS", "KEY_PANDA"),
    ("PRS_VLLM_VENV", "PANDA_VLLM_VENV"),
    ("PRS_OUTPUTS", "PANDA_OUTPUTS"),
    ("PRS_MODELS", "PANDA_MODELS"),
    ("PRS_ROOT", "PANDA_ROOT"),
    ("PRS_MODEL_TAG", "PANDA_MODEL_TAG"),
    ("PRS_SE_CLUSTER", "PANDA_SE_CLUSTER"),
    ("PRS_NLI_MODEL", "PANDA_NLI_MODEL"),
    ("PRS_NLI_DEVICE", "PANDA_NLI_DEVICE"),
    ("/root/autodl-tmp/PRS", "/root/autodl-tmp/PANDA"),
    ("/root/autodl-tmp/prs-models", "/root/autodl-tmp/panda-models"),
    ("/root/autodl-tmp/prs-outputs", "/root/autodl-tmp/panda-outputs"),
    ("prs-models", "panda-models"),
    ("prs-outputs", "panda-outputs"),
    ("from prs.", "from panda."),
    ("import prs", "import panda"),
    ("python -m prs.", "python -m panda."),
    ("panda.ase.prs import", "panda.core.panda import"),
    ("panda/ase/prs.py", "panda/core/panda.py"),
    ("aggregate_prs_v2", "aggregate_panda_v2"),
    ("update_experiment_plan_prs_v2", "update_experiment_plan_panda_v2"),
    ("wait_and_aggregate_prs_v2", "wait_and_aggregate_panda_v2"),
    ("setup_vllm_prs_venv", "setup_vllm_panda_venv"),
    ("download_prs_models", "download_panda_models"),
    ("prs_v2_results", "panda_v2_results"),
    ("ase_models.yaml", "panda_models.yaml"),
    ('name = "prs"', 'name = "panda"'),
    ('"""PRS:', '"""PANDA:'),
    (r"\bPRS\b", "PANDA"),
]


def should_skip(path: Path) -> bool:
    parts = path.parts
    if any(p in SKIP_DIRS for p in parts):
        return True
    s = str(path)
    return any(part in s for part in SKIP_PATH_PARTS)


def git_mv(src: Path, dst: Path) -> None:
    if not src.exists():
        if dst.exists():
            return
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    r = subprocess.run(
        ["git", "mv", str(src), str(dst)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        shutil.move(str(src), str(dst))


def rename_files() -> list[tuple[str, str]]:
    done: list[tuple[str, str]] = []
    for old, new in FILE_RENAMES:
        src, dst = REPO / old, REPO / new
        if src.exists() or (not dst.exists() and old.startswith("src/panda/ase")):
            git_mv(src, dst)
            done.append((old, new))
    return done


def transform_text(text: str) -> str:
    for old, new in REPLACEMENTS:
        if old.startswith(r"\b") or "\\b" in old:
            text = re.sub(old, new, text)
        else:
            text = text.replace(old, new)
    # Clean break: stop reading legacy PRS JSON key
    text = text.replace(
        'v = row.get(KEY_PANDA) or row.get("PRS")',
        "v = row.get(KEY_PANDA)",
    )
    text = text.replace(
        'v = row.get(KEY_PANDA) or row.get("PANDA")',
        "v = row.get(KEY_PANDA)",
    )
    return text


def replace_in_files() -> list[str]:
    changed: list[str] = []
    skip_names = {"_rename_to_panda.py", "_rename_ase_to_core.py"}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = Path(root) / name
            if should_skip(path):
                continue
            if path.suffix not in TEXT_EXTENSIONS and path.name not in {"activate", "pyvenv.cfg"}:
                continue
            if path.name in skip_names:
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            new = transform_text(raw)
            if new != raw:
                path.write_text(new, encoding="utf-8")
                changed.append(str(path.relative_to(REPO)))
    return changed


def main() -> None:
    renames = rename_files()
    changed = replace_in_files()
    print(f"Renamed {len(renames)} paths")
    for a, b in renames:
        print(f"  {a} -> {b}")
    print(f"Updated {len(changed)} files")


if __name__ == "__main__":
    main()
