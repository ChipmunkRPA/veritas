"""Post-hoc scalar-ranking bound, never a replacement/calibration decision."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from scripts.review_mouth_context_v10 import overlap


def bound(body):
    rows = body["panels"]["development"]["cases"]
    summary = body["summary"]
    windows = summary["scored_windows"]
    reference = body["absolute_reference"]
    if [r["id"] for r in rows] != [r["id"] for r in windows] or [r["id"] for r in rows] != [
        r["id"] for r in reference["cases"]
    ]:
        raise ValueError("Complete fixed comparison order required")
    controls = sorted(
        [
            (w["windows"][0]["response"], r["id"])
            for r, w in zip(rows, windows, strict=True)
            if not r["label"]
        ],
        reverse=True,
    )
    if len(controls) < 2:
        raise ValueError("Two ordinary cases required for this bound")
    point = math.nextafter(controls[1][0], math.inf)
    outcomes = []
    for r, w, old, old_outcome in zip(rows, windows, reference["cases"], reference["outcomes"], strict=True):
        absolute = old["probe_windows"] and old["probe_windows"][0]["response"] >= reference["review_point"]
        if absolute:
            flagged, localized = old_outcome["flagged"], old_outcome["localized"]
        else:
            candidate = w["windows"][0]
            flagged = candidate["response"] >= point
            localized = bool(r["label"] and flagged and overlap(candidate, r.get("edit_time_range_s")) >= 0.2)
        outcomes.append(
            {
                "id": r["id"],
                "label": r["label"],
                "family": r["family"],
                "flagged": bool(flagged),
                "localized": bool(localized),
            }
        )
    return {
        "role": "Post-hoc bound on consumed development, not a calibrated/adopted point or fresh validation. V11 remains rejected.",
        "comparison_head_sha256": summary["head_sha256"],
        "minimum_raw_point_for_at_most_one_ordinary_flag": point,
        "second_largest_ordinary_max": controls[1][0],
        "ordinary_top_maxima": controls[:8],
        "maximum_combined_locations": sum(r["localized"] for r in outcomes),
        "ordinary_flags_at_bound": sum(r["flagged"] for r in outcomes if not r["label"]),
        "family_locations": {
            f: sum(r["localized"] for r in outcomes if r["family"] == f) for f in ("liveportrait", "musetalk")
        },
        "original_controls_flagged_at_frozen_point": sum(
            r["flagged"] for r in summary["cases"]["combined"] if r["condition"] == "original"
        ),
        "scope": "Fixed absolute-first priority and top-window selection. At higher scalar points locations can only be lost. No original files, physical spatial-IoU replay or new point selection.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = bound(json.loads(gzip.decompress(args.capsule.read_bytes())))
    result["capsule_sha256"] = hashlib.sha256(args.capsule.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
