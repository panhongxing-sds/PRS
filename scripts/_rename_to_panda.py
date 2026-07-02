#!/usr/bin/env python3
"""One-shot PRS -> PANDA rebrand. Run from repo root before final dir rename."""
from __future__ import annotations

import os
import re
import shutil
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
)

TEXT_EXTENSIONS = {
    ".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".tex", ".json",
    ".txt", ".cfg", ".ini", ".bat", ".fish", ".csh",
}

FILE_RENAMES = [
    ("src/prs", "src/panda"),
    ("src/panda/ase/prs.py", "src/panda/ase/panda.py"),
    ("tests/test_prs.py", "tests/test_panda.py"),
    ("scripts/aggregate_prs_v2.py", "scripts/aggregate_panda_v2.py"),
    ("scripts/update_experiment_plan_prs_v2.py", "scripts/update_experiment_plan_panda_v2.py"),
    ("scripts/wait_and_aggregate_prs_v2.sh", "scripts/wait_and_aggregate_panda_v2.sh"),
    ("scripts/setup_vllm_prs_venv.sh", "scripts/setup_vllm_panda_venv.sh"),
    ("scripts/download_prs_models.sh", "scripts/download_panda_models.sh"),
    ("paper/maintable/prs_v2_results.json", "paper/maintable/panda_v2_results.json"),
    ("paper/iclr2026/prs_iclr2026.tex", "paper/iclr2026/panda_iclr2026.tex"),
    ("paper/iclr2026/prs_iclr2026_panda.tex", "paper/iclr2026/panda_iclr2026_draft.tex"),
    ("paper/report/ASE_experiment_report.tex", "paper/report/PANDA_experiment_report.tex"),
    ("configs/ase_models.yaml", "configs/panda_models.yaml"),
]

# Order matters: longer / more specific first
REPLACEMENTS = [
    (r"/root/autodl-tmp/PRS", "/root/autodl-tmp/PANDA"),
    (r"/root/autodl-tmp/prs-models", "/root/autodl-tmp/panda-models"),
    (r"/root/autodl-tmp/prs-outputs", "/root/autodl-tmp/panda-outputs"),
    ("from prs.", "from panda."),
    ("import prs", "import panda"),
    ("python -m prs.", "python -m panda."),
    ("panda.ase.prs import", "panda.ase.panda import"),
    ("panda/ase/prs.py", "panda/ase/panda.py"),
    ("enrich_row_with_prs", "enrich_row_with_panda"),
    ("compute_prs_from_row", "compute_panda_from_row"),
    ("compute_prs", "compute_panda"),
    ("KEY_PRS", "KEY_PANDA"),
    ("PRS_VLLM_VENV", "PANDA_VLLM_VENV"),
    ("PRS_OUTPUTS", "PANDA_OUTPUTS"),
    ("PRS_MODELS", "PANDA_MODELS"),
    ("PRS_ROOT", "PANDA_ROOT"),
    ("prs-outputs", "panda-outputs"),
    ("prs-models", "panda-models"),
    ("prs_v2_results", "panda_v2_results"),
    ("aggregate_prs_v2", "aggregate_panda_v2"),
    ("update_experiment_plan_prs_v2", "update_experiment_plan_panda_v2"),
    ("wait_and_aggregate_prs_v2", "wait_and_aggregate_panda_v2"),
    ("setup_vllm_prs_venv", "setup_vllm_panda_venv"),
    ("download_prs_models", "download_panda_models"),
    ("prs_iclr2026_panda", "panda_iclr2026_draft"),
    ("prs_iclr2026", "panda_iclr2026"),
    ("ase_models.yaml", "panda_models.yaml"),
    ("Perturbation Reliability Score (PRS)", "PANDA uncertainty score"),
    ("Perturbation Reliability Score", "PANDA"),
    ('name = "prs"', 'name = "panda"'),
    ('"""PRS:', '"""PANDA:'),
    (r"\bPRS\b", "PANDA"),
    (r"\bprs\b", "panda"),
]

# Keep legacy JSON field reads for existing experiment artifacts
LEGACY_READ_PATCHES = [
    (
        'if key == KEY_PANDA:\n        v = row.get(KEY_PANDA)',
        'if key == KEY_PANDA:\n        v = row.get(KEY_PANDA) or row.get("PRS")',
    ),
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
    import subprocess
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
        if src.exists() or (not dst.exists() and old.startswith("src/prs")):
            git_mv(src, dst)
            done.append((old, new))
    return done


def transform_text(text: str) -> str:
    for old, new in REPLACEMENTS:
        if old.startswith(r"\b") or "\\b" in old:
            text = re.sub(old, new, text)
        else:
            text = text.replace(old, new)
    for old, new in LEGACY_READ_PATCHES:
        text = text.replace(old, new)
    return text


def replace_in_files() -> list[str]:
    changed: list[str] = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = Path(root) / name
            if should_skip(path):
                continue
            if path.suffix not in TEXT_EXTENSIONS and path.name not in {"activate", "pyvenv.cfg"}:
                continue
            if path.name == "_rename_to_panda.py":
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
