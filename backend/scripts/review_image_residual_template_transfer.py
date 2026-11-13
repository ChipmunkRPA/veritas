# SPDX-License-Identifier: Apache-2.0
"""Audit the frozen residual-template ledger with Python's standard library."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EDITS = ("sharp", "feathered")
CONTROLS = ("original", "local_blur", "local_tone", "local_copy", "global_resize", "second_jpeg")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_box(pixels, box, width, height, truth):
    if box:
        left, top, right, bottom = box
        require(0 <= left < right <= width and 0 <= top < bottom <= height, "Box exceeds native coordinates")
        area = (right - left) * (bottom - top)
    else:
        area = 0
    require(pixels["predicted_positive_pixels"] == area, "Selected area differs from the recorded box")
    overlap = pixels["intersection"]
    require(0 <= overlap <= min(area, truth), "Impossible overlap")
    union = area + truth - overlap
    require(abs(pixels["iou"] - (overlap / union if union else 1)) < 1e-12, "IoU arithmetic differs")
    if truth:
        require(abs(pixels["recall"] - overlap / truth) < 1e-12, "Recall arithmetic differs")
    else:
        require(pixels["recall"] is None, "A control has recall")
    return area > 0, bool(truth and 10 * overlap >= 3 * union and 2 * overlap >= truth)


def audit(folder, receipt_path):
    lock_path = folder / "freeze.json"
    dev_path = folder / "development.json"
    held_path = folder / "held.json"
    lock, dev, held, receipt = map(load, (lock_path, dev_path, held_path, receipt_path))
    require(dev["freeze_sha256"] == receipt["freeze_sha256"] == digest(lock_path), "Frozen inputs differ")
    require(held["development_sha256"] == receipt["development_sha256"] == digest(dev_path), "Calibration digest differs")
    require(receipt["held_sha256"] == digest(held_path), "Held digest differs")
    rows = dev["rows"] + held["rows"]
    frozen = {(row["split"], row["group_sha256"], row["role"]): row for row in lock["rows"]}
    require(len(frozen) == len(rows) == receipt["exact_replays"], "Row count differs")
    identities = defaultdict(set)
    maxima = defaultdict(float)
    for row in rows:
        key = tuple(row[name] for name in ("split", "group_sha256", "role"))
        original = frozen.pop(key)
        require(all(row[name] == value for name, value in original.items()), "Frozen delivery or mask metadata differs")
        identities[row["split"]].add(row["group_sha256"])
        if row["split"] == "development":
            maxima[row["group_sha256"]] = max(row["score"], maxima[row["group_sha256"]])
            require(row["role"] in CONTROLS and row["truth_pixels"] == 0, "Calibration used an edit")
    require(not frozen and not identities["development"] & identities["held"], "Source-role crossover")
    require(len(maxima) == receipt["development_groups"] == 8, "Unexpected calibration cohort")
    require(len(identities["held"]) == receipt["held_groups"], "Unexpected held cohort")
    ordered = sorted(maxima.values(), reverse=True)
    require(ordered == dev["ordinary_source_maxima"], "Source maxima differ")
    point = max(.15, ordered[1])
    require(dev["point"] == receipt["scoring_point"] == point, "Cutoff differs")

    outcomes = defaultdict(Counter)
    flags_by_source = defaultdict(set)
    ordinary = defaultdict(set)
    gained_groups = set()
    additional_useful = 0
    for row in held["rows"]:
        role, group, truth = row["role"], row["group_sha256"], row["truth_pixels"]
        h, w = row["dimensions"]
        require(bool(truth) == (role in EDITS), "Wrong truth for role")
        aligned = verify_box(row["aligned_pixels"], row["aligned_box"], w, h, truth)
        ungated = verify_box(row["proposal_pixels"], row["box"], w, h, truth)
        gated = ungated if row["score"] > point else (False, False)
        combined = aligned if aligned[0] else gated
        if combined[1] and not aligned[1]:
            additional_useful += 1
            gained_groups.add(group)
        arms = {"aligned_luma": aligned, "template": gated, "luma_or_template": combined, "ungated_template": ungated}
        for arm, (flag, hit) in arms.items():
            outcomes[arm, role].update({"files": 1, "flagged": int(flag), "useful": int(hit)})
            if flag:
                flags_by_source[arm, role].add(group)
                if role in CONTROLS:
                    ordinary[arm].add(group)

    result = receipt["held_metrics"]
    for (arm, role), observed in outcomes.items():
        require(observed | {"flagged_groups": len(flags_by_source[arm, role])} == result[arm]["roles"][role],
                f"Per-role held counts differ for {arm}/{role}")
    for arm in arms:
        controls_flagged = sum(outcomes[arm, role]["flagged"] for role in CONTROLS)
        require(result[arm]["ordinary_flagged"] == controls_flagged, f"Ordinary flags differ for {arm}")
        require(result[arm]["ordinary_groups"] == len(ordinary[arm]), f"Ordinary groups differ for {arm}")
    extra_flags = result["luma_or_template"]["ordinary_flagged"] - result["aligned_luma"]["ordinary_flagged"]
    extra_groups = len(ordinary["luma_or_template"]) - len(ordinary["aligned_luma"])
    require(result["admission"] == {
        "additional_useful_files": additional_useful,
        "additional_useful_groups": len(gained_groups),
        "additional_ordinary_flags": extra_flags,
        "additional_ordinary_groups": extra_groups,
        "passed": additional_useful >= 3 and len(gained_groups) >= 2 and extra_flags <= 2 and extra_groups <= 1,
    }, "Advancement decision differs")
    print(f"Verified {len(rows)} frozen rows, {len(identities['held'])} held groups and all decision counts.")


if __name__ == "__main__":
    audit(Path(sys.argv[1]), Path(sys.argv[2]))
