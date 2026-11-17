# SPDX-License-Identifier: Apache-2.0
"""Recompute the interval-training calibration and held event counts without inference."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence.audio_focused_edit_binding import CENTERED_STEP, CENTERED_WIDTHS
from scripts import audio_focused_context_transfer as files
from scripts import audio_long_context_review as reference

SCHEMA = "chipmunk.audio-interval-ranking/1"
ARMS = ("pooled", "contrast", "combined")


def predict(row, points, policy):
    existing = [tuple(span) for span in row["current"]]
    options = {"current": existing}
    for method in ARMS:
        located = reference.recalled(row["readings"][method], points[method], policy, row["scope"][0])
        options[method] = located
        options["plus_" + method] = existing or located
    return options


def summarize(rows, points):
    counts, sources = defaultdict(Counter), defaultdict(set)
    for row in rows:
        target = reference.common_truth(row)
        existing = files.matches(row["current"], target)
        available = {arm: any(row["readings"][arm]) for arm in ARMS}
        available.update({"plus_" + arm: value or row["current_assessed"] for arm, value in list(available.items())})
        available["current"] = row["current_assessed"]
        for policy in reference.POLICIES:
            for method, locations in predict(row, points, policy).items():
                key = "/".join((row["condition"], policy, method))
                matched = files.matches(locations, target)
                added = matched - existing
                kind = row["kind"]
                counts[key].update({kind: 1, kind + "_flagged": bool(locations), kind + "_unassessed": not available[method],
                                    "targets": len(target), "located": len(matched), "lost": len(existing - matched),
                                    "added": len(added), "misplaced": kind == "edited" and bool(locations) and not matched})
                if added:
                    sources[key].add(row["group"])
    return {key: dict(value, added_groups=len(sources[key])) for key, value in sorted(counts.items())}


def verify(result):
    fit, rows = result["fit"], result["rows"]
    assert result["schema"] == fit["schema"] == SCHEMA and result["design"] == fit["design"]
    assert fit["held_observations_at_fit"] == 0
    assert set(fit["heads"]) == set(fit["points"]) == set(ARMS)
    assert len({row["id"] for row in rows}) == len(rows)
    assert {str(row["group"]) for row in rows}.isdisjoint(fit["source_readers"])
    for row in (*rows, *fit["calibration"]):
        assert set(row["readings"]) == set(ARMS)
        for readings in row["readings"].values():
            for width, window in zip(CENTERED_WIDTHS, readings, strict=True):
                if window:
                    a, b = window["range"]
                    assert 0 <= a < b <= reference.SCOPE_SAMPLES and b - a == width and a % CENTERED_STEP == 0
                    assert math.isfinite(window["score"])
    for arm in ARMS:
        assert len(fit["heads"][arm]["weights"]) == 384 and fit["heads"][arm]["rank_pairs"] > 0
        assert all(math.isfinite(weight) for weight in fit["heads"][arm]["weights"])
        inspected = [row | {"readings": row["readings"][arm]} for row in fit["calibration"]]
        # The reference scalar reviewer takes separately calibrated contexts.
        reference.verify_points({"points": dict.fromkeys(reference.CONTEXTS, fit["points"][arm]),
                                 "calibration": dict.fromkeys(reference.CONTEXTS, inspected)})
    data = summarize(rows, fit["points"])
    if data != result["summary"]:
        raise ValueError("Held event counts do not reproduce")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = files.read(args.receipt)
    print(json.dumps(verify(result), sort_keys=True))
