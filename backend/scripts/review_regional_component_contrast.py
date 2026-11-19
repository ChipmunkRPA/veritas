"""Recount anonymous regional-localization results against the pinned parent panels."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PANELS = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
PARENTS = {"native": "image-edge-context-transfer-2026-11-14.json",
           "other_png": "image-edge-context-new-sources-2026-11-14.json"}
CONTRAST = "chipmunk.regional-component-contrast/1"
GROWTH = "chipmunk.regional-boundary-growth/1"
ARMS = {CONTRAST: {"native": ("current", "prune", "lower_addition", "both"),
                   "other_png": ("current", "prune", "flip", "flip_prune")},
        GROWTH: {"native": ("current", "grow"), "other_png": ("current", "grow", "flip", "grow_repeat")}}


def include_edit(area, overlap, changed):
    return changed > 0 and 2 * overlap >= changed and 10 * overlap >= 3 * (area + changed - overlap)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review(receipt):
    require(receipt["schema"] in ARMS and set(receipt["panels"]) == PARENTS.keys(), "Unexpected receipt")
    growth = receipt["schema"] == GROWTH
    result = {}
    for panel, filename in PARENTS.items():
        entry = receipt["panels"][panel]
        body = (PANELS / filename).read_bytes()
        require(hashlib.sha256(body).hexdigest() == entry["parent_sha256"], "The parent panel differs")
        parent = json.loads(body)
        all_cases = parent["freeze"]["cases"] if panel == "native" else parent["cases"]
        prior = [(case, row) for case, row in zip(all_cases, parent["rows"], strict=True)
                 if panel != "native" or case["cohort"] == "native_png"]
        arms = ARMS[receipt["schema"]][panel]
        require(len(prior) == (236 if panel == "native" else 156), "Missing prior cases")
        counts = {key: Counter() for key in arms}
        alerted = {key: set() for key in arms}
        for (case, old), measured in zip(prior, entry["rows"], strict=True):
            require(measured[0] == old["body_sha256"] == case["body_sha256"] and len(measured) == len(arms) + 1,
                    "The delivered image no longer matches its parent")
            targets = case["truth_pixels"]
            baseline = old["current"]
            if panel == "other_png":
                kept = [part for part, flip in zip(old["support"], old["flip_shared"], strict=True) if 2 * flip >= part[0]]
                baseline = [sum(part[key] for part in kept) for key in (2, 3)]
                require(measured[3] == baseline, "The original flip measurements differ")
            require(measured[1] == old["current"], "The original measurements differ")
            width, height = case["size"]
            exact_baseline = include_edit(*baseline, targets)
            for arm, data in zip(arms, measured[1:], strict=True):
                area, overlap = data
                require(type(area) is int and type(overlap) is int and 0 <= overlap <= min(area, targets)
                        and area <= width * height, "Invalid overlap or coverage")
                if growth and arm in {"grow", "grow_repeat"}:
                    base_area, base_overlap = baseline
                    require(base_area <= area <= base_area + base_area // 4 and base_overlap <= overlap,
                            "The expanded boundary differs from its fixed limits")
                useful = include_edit(area, overlap, targets)
                counts[arm].update(images=1, edits=bool(targets), useful=useful,
                                   misplaced=bool(targets and area) and not useful,
                                   ordinary_alerts=not targets and bool(area),
                                   gained=useful and not exact_baseline, lost=exact_baseline and not useful)
                if not targets and area:
                    alerted[arm].add(case["group"])
                if growth:
                    counts[arm]["ordinary_area"] += area if not targets else 0
        result[panel] = {arm: dict(counts[arm], ordinary_alerted_groups=len(alerted[arm])) for arm in arms}
    return result


if __name__ == "__main__":
    raw = json.loads(Path(sys.argv[1]).read_bytes())
    result = review(raw)
    require(result == raw["summary"], "Reported totals differ from the independent recount")
    print(json.dumps(result, indent=2, sort_keys=True))
