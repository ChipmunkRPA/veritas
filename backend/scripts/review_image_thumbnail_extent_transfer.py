# SPDX-License-Identifier: Apache-2.0
"""Independently check the frozen thumbnail extent ledger without media or dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {"added_groups": 3, "added_generated_groups": 1, "added_conventional_groups": 2, "maximum_lost": 1,
            "maximum_new_ordinary_groups": 0, "maximum_changed_ordinary_groups": 1, "maximum_incompatible": 0}
ORDINARY = {"unchanged", "jpeg75", "global_tone", "resize", "camera", "camera_jpeg75", "camera_global_tone"}
INCOMPATIBLE = {"no_preview", "foreign_preview"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def useful(area, truth):
    selected, common = area
    require(type(selected) is int and type(common) is int and 0 <= common <= min(selected, truth), "Invalid pixel overlap")
    return truth > 0 and common * 2 >= truth and common * 10 >= (selected + truth - common) * 3


def review(result):
    frozen = result["frozen"]
    exact = (json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(hashlib.sha256(exact).hexdigest() == result["freeze_sha256"] and frozen["criteria"] == EXPECTED
            and frozen["scope"] == {"confidence_weight": 0, "extent": [.75, .32, 4]}, "Frozen gate or record changed")
    sources = frozen["sources"]
    families = Counter(row["family"] for row in sources.values())
    require(families == {"sdxl": 3, "kandinsky": 3, "flux": 2, "conventional": 10, "camera": 10}
            and all(row["license"].startswith(("https://creativecommons.org/licenses/by/", "https://creativecommons.org/licenses/by-sa/"))
                    and row["author"] and row["page"].startswith("https://") for row in sources.values()), "Wrong source roster or terms")
    rows = result["cases"]
    require(len(rows) == len(frozen["cases"]) == 222 and len({tuple(row["key"]) for row in rows}) == len(rows), "Incomplete or duplicated pairs")
    tally = defaultdict(list)
    additions, generated, conventional, newly_flagged, changed_ordinary = set(), set(), set(), set(), set()
    losses = incompatible = wrong_orientation = 0
    for lock, row in zip(frozen["cases"], rows, strict=True):
        require({k: row[k] for k in lock} == lock and len(row["input_sha256"]) == 64, "Case does not match its freeze")
        group, _, role = row["key"]
        old, new, truth = row["baseline"], row["candidate"], row["target"]
        require((truth > 0) == (role not in ORDINARY | INCOMPATIBLE), "Control or exact pixel truth role changed")
        old_box, new_box = [useful(item["boxes_overlap"], truth) for item in (old, new)]
        old_mask, new_mask = [useful(item["mask"], truth) for item in (old, new)]
        old_flag, new_flag = [bool(item["count"]) for item in (old, new)]
        changed = old["mask_sha256"] != new["mask_sha256"]
        require(all(bool(item["boxes"]) == bool(item["count"]) and len(item["boxes"]) <= 3 for item in (old, new)), "Invalid crop roster")
        tally[role].append((old_box, new_box, old_mask, new_mask, old_flag, new_flag, changed))
        if new["status"] == "comparable":
            incompatible += role in INCOMPATIBLE
            wrong_orientation += role not in INCOMPATIBLE and new["orientation"] != row["expected_preview_orientation"]
        if role in ORDINARY:
            unit = sources[group].get("same_contributor_group", group)
            if new_flag and not old_flag:
                newly_flagged.add(unit)
            if changed:
                changed_ordinary.add(unit)
        if truth:
            losses += old_box and not new_box
            if new_box and not old_box:
                additions.add(group)
                (generated if role == "generated" else conventional).add(group)
    keys = ("baseline_boxes", "candidate_boxes", "baseline_mask", "candidate_mask", "baseline_flags", "candidate_flags", "extent_changes")
    counts = {role: {"total": len(values), **dict(zip(keys, map(sum, zip(*values, strict=True)), strict=True))}
              for role, values in tally.items()}
    gate = (len(additions) >= 3 and len(generated) >= 1 and len(conventional) >= 2 and losses <= 1
            and not newly_flagged and len(changed_ordinary) <= 1 and not incompatible and not wrong_orientation)
    exact_summary = {"by_role": counts, "groups": {"added": sorted(additions), "generated": sorted(generated),
                     "conventional": sorted(conventional), "new_ordinary": sorted(newly_flagged), "changed_ordinary": sorted(changed_ordinary)},
                     "lost_useful_views": losses, "incompatible_comparisons": incompatible,
                     "wrong_preview_orientation": wrong_orientation, "gate_passed": gate}
    require(exact_summary == result["summary"], "Paired arithmetic differs from the reported result")
    return exact_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    summary = review(json.loads(args.ledger.read_bytes()))
    print(json.dumps(summary, sort_keys=True))
