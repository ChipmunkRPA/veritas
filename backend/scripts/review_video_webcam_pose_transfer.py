# SPDX-License-Identifier: Apache-2.0
"""Independently check the within-track comparison journal and exact native intervals."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(folder):
    freeze_path = folder / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    result = json.loads((folder / "aggregate.json").read_text())
    camera = list(map(json.loads, (folder / "camera.jsonl").open()))
    native = list(map(json.loads, (folder / "native.jsonl").open()))
    expected_camera = {(case["slot"], case["role"]) for case in freeze["camera"]}
    expected_native = {(group, role) for group in range(1, len(freeze["swaps"]) + 1)
                       for role in ("original", "swap_a", "swap_b")}
    assert len(camera) == len(expected_camera) and {(r["slot"], r["role"]) for r in camera} == expected_camera
    assert len(native) == len(expected_native) and {(r["group"], r["role"]) for r in native} == expected_native
    selected, useful, contextualized = [], [], []
    time_checks = 0
    for row in camera + native:
        if row["incumbent"] is None:
            assert row["context"] is None and row["status"] in {"incumbent_measured_no_flag", "incumbent_unassessed"}
            continue
        selected.append(row)
        start, end = map(Fraction, row["incumbent"])
        assert 0 <= start < end
        time_checks += 1
        if row["truth"]:
            truth_start, truth_end = map(Fraction, row["truth"])
            assert truth_start < truth_end
            intersection = max(0, min(end, truth_end) - max(start, truth_start))
            if 2 * intersection >= max(end, truth_end) - min(start, truth_start):
                useful.append(row)
            time_checks += 1
        if row["status"] == "measured":
            support = row["triplet_counts"]
            triplets = row["middle_triplets"]
            minimum = (2 * triplets + 2) // 3
            assert row["minimum_context_triplets"] == minimum
            assert all(0 <= value <= triplets for value in support.values()) and sum(support.values()) <= triplets
            context = next((key for key in ("landmark_inversion", "bilateral_support_change") if support[key] >= minimum), None)
            assert context == row["context"]
            if row in camera and context:
                contextualized.append(row)
        else:
            assert row["status"] == "flank_geometry_unassessed" and row["context"] is None
    losses = sum(bool(row["context"]) for row in useful)
    coverage = sum(row["status"] == "measured" for row in selected)
    negative_sources = len({r["slot"] for r in contextualized})
    positive_sources = len({r["group"] for r in useful})
    advances = (len(contextualized) >= 4 and negative_sources >= 2 and losses == 0 and len(useful) >= 4
                and positive_sources >= 3 and 10 * coverage >= 9 * len(selected))
    checked = {"camera_statuses": dict(Counter(r["status"] for r in camera)),
               "native_statuses": dict(Counter(r["status"] for r in native)),
               "camera_context_flags": len(contextualized), "camera_context_groups": negative_sources,
               "native_incumbent_correct_intervals": len(useful), "native_incumbent_correct_groups": positive_sources,
               "correct_swap_contextualized": losses, "geometry_measured": coverage, "geometry_selected": len(selected),
               "pilot_advances": advances}
    assert all(result[key] == value for key, value in checked.items())
    return {"camera_journal_sha256": digest(folder / "camera.jsonl"),
            "native_journal_sha256": digest(folder / "native.jsonl"), "freeze_sha256": digest(freeze_path),
            "reviewed_decisions": len(camera) + len(native), "exact_time_checks": time_checks, **checked}


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    result = compare(folder)
    with (folder / "independent-audit.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
