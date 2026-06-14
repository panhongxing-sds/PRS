"""Answer extraction and grading for logic benchmarks (string match)."""

from __future__ import annotations

import json
import re


_FINAL_ANSWER_RE = re.compile(
    r"(?:the\s+)?final\s+answer\s+is\s*[:\-]?\s*(.+?)(?:\.|$)",
    re.IGNORECASE | re.DOTALL,
)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_logic_answer(text: str) -> str:
    if not text:
        return ""
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().strip("\"'")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def normalize_logic_answer(text: str) -> str:
    s = extract_logic_answer(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_grid_ref(ref: str) -> dict | None:
    """Parse a zebra-style solution grid into {name: {value, ...}}.

    Reference shape: {"header": ["House", "Name", <attrs...>], "rows": [[...], ...]}.
    Returns a mapping from the lowercased Name value to the set of its (lowercased)
    attribute values across all non-House/Name columns. None if unparseable.
    """
    try:
        obj = json.loads(ref)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    header = obj.get("header") or []
    rows = obj.get("rows") or []
    if not header or not rows:
        return None
    lower = [str(h).strip().lower() for h in header]
    try:
        name_idx = lower.index("name")
    except ValueError:
        return None
    attr_idx = [i for i, h in enumerate(lower) if h not in ("house", "name")]
    grid: dict[str, set[str]] = {}
    for row in rows:
        if name_idx >= len(row):
            return None
        nm = str(row[name_idx]).strip().lower()
        vals = {str(row[i]).strip().lower() for i in attr_idx if i < len(row)}
        if not nm:
            return None
        grid[nm] = vals
    return grid or None


def grade_grid_answer(pred_text: str, ref: str) -> str:
    """Three-way verdict (``correct``/``wrong``/``drop``) for zebra logic grids.

    The model emits free-form prose, so we recover a name->values assignment by
    scanning lines that mention exactly one solution name and collecting the gold
    attribute values that co-occur on that line. A sample is graded only when every
    gold name is located with a unique line attribution; otherwise it is dropped so
    aggregation can exclude rather than misjudge it.
    """
    grid = _parse_grid_ref(ref)
    if not grid:
        return "drop"
    text = (pred_text or "").lower()
    if not text:
        return "drop"
    names = list(grid.keys())
    all_values = set().union(*grid.values()) if grid else set()

    def vals_on(line: str) -> set[str]:
        return {v for v in all_values if re.search(rf"\b{re.escape(v)}\b", line)}

    # For each name use the LAST single-name line that carries >=1 gold value: the
    # final "arrangement" block, ignoring intermediate reasoning that may mention
    # other values. A name with no such line cannot be confidently graded -> drop.
    pred_assign: dict[str, set[str]] = {}
    for line in text.splitlines():
        hit = [n for n in names if re.search(rf"\b{re.escape(n)}\b", line)]
        if len(hit) != 1:
            continue
        v = vals_on(line)
        if v:
            pred_assign[hit[0]] = v

    if set(pred_assign.keys()) != set(names):
        return "drop"
    for n in names:
        # The model's final line for this name must list exactly its gold values
        # and no value belonging to another house.
        if pred_assign[n] != grid[n]:
            return "wrong"
    return "correct"


def logic_equal(pred: str, ref: str) -> bool:
    if not pred or not ref:
        return False
    p = normalize_logic_answer(pred)
    r = normalize_logic_answer(ref)
    if p == r:
        return True
    # Zebra puzzle JSON solutions
    if ref.startswith("{") or ref.startswith("["):
        try:
            ref_obj = json.loads(ref)
            try:
                pred_obj = json.loads(pred)
                return pred_obj == ref_obj
            except json.JSONDecodeError:
                pass
        except json.JSONDecodeError:
            pass
    # Numeric leg-counting
    p_num = re.search(r"-?\d+(?:\.\d+)?", p)
    r_num = re.search(r"-?\d+(?:\.\d+)?", r)
    if p_num and r_num:
        try:
            return int(float(p_num.group())) == int(float(r_num.group()))
        except ValueError:
            pass
    return False
