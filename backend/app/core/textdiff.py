"""Unified diffs without difflib's autojunk heuristic, so frequently repeated lines (such as `files:` in models.yaml)
stay usable as anchors and reviewers see only the lines that changed."""
from __future__ import annotations

from difflib import SequenceMatcher


def unified_diff(a: list[str], b: list[str], *, fromfile: str, tofile: str, n: int = 3) -> str:
    """Same format as `difflib.unified_diff`; `a` and `b` are lines with their line endings."""
    out: list[str] = []
    for group in SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(n):
        if not out:
            out += [f"--- {fromfile}\n", f"+++ {tofile}\n"]
        first, last = group[0], group[-1]
        out.append(f"@@ -{_range(first[1], last[2])} +{_range(first[3], last[4])} @@\n")
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                out += [" " + line for line in a[i1:i2]]
                continue
            if tag in ("replace", "delete"):
                out += ["-" + line for line in a[i1:i2]]
            if tag in ("replace", "insert"):
                out += ["+" + line for line in b[j1:j2]]
    return "".join(out)


def _range(start: int, stop: int) -> str:
    """difflib's unified range format: 1-based start, omitted length when 1."""
    length = stop - start
    beginning = start + 1
    if length == 1:
        return f"{beginning}"
    if not length:
        beginning -= 1
    return f"{beginning},{length}"
