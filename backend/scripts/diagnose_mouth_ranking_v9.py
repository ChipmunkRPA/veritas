"""Post-hoc scalar-ranking bound on consumed V9 data; never a fitted release point."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def temporal(a, b):
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union else 0.0


def diagnose(responses):
    data = json.loads(gzip.decompress(responses.read_bytes()))
    rows = data["cases"]
    if len(rows) != 88 or any(r["status"] != "ok" or not r["probe_windows"] for r in rows):
        raise ValueError("Complete assessed 88-case V9 panel required")
    for row in rows:
        values = [w["response"] for w in row["probe_windows"]]
        if any(not math.isfinite(v) for v in values) or values != sorted(values, reverse=True):
            raise ValueError("Retained windows are not finite/ranked")
    ordinary = sorted([r["probe_windows"][0]["response"] for r in rows if not r["label"]], reverse=True)
    if len(ordinary) != 56:
        raise ValueError("All ordinary controls required for bound")
    lower_bound = math.nextafter(ordinary[1], math.inf)
    flagged = [r for r in rows if r["probe_windows"][0]["response"] >= lower_bound]
    families = {}
    for family in ("liveportrait", "musetalk"):
        ai = [r for r in rows if r["family"] == family and r["label"]]
        eligible = [r for r in flagged if r["family"] == family and r["label"]]
        families[family] = {
            "ai_cases": len(ai),
            "maximum_ai_window_response": max(r["probe_windows"][0]["response"] for r in ai),
            "maximum_flaggable_ai_cases_at_one_ordinary_flag": len(eligible),
            "maximum_correct_locations_at_one_ordinary_flag": sum(
                temporal(
                    [r["probe_windows"][0]["start_s"], r["probe_windows"][0]["end_s"]], r["edit_time_range_s"]
                )
                >= 0.2
                for r in eligible
            ),
        }
    return {
        "schema": "chipmunk.mouth-paired-posthoc-ranking-bound/9",
        "public_responses_sha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
        "original_score_receipt_sha256": data["original_receipt_sha256"],
        "frozen_review_point": data["header"]["review_point"],
        "ordinary_cases": len(ordinary),
        "allowed_ordinary_flags": 1,
        "second_highest_ordinary_response": ordinary[1],
        "minimum_float64_scalar_point_for_at_most_one_ordinary_flag": lower_bound,
        "ordinary_flags_at_bound": sum(not r["label"] for r in flagged),
        "families": families,
        "reasoning": "The detector uses response >= point. Every scalar point allowing at most one ordinary flag must be strictly above the second-highest ordinary response; nextafter toward infinity is the smallest allowed float64 point. AI flag sets at higher points can only shrink.",
        "scope": "Post-hoc diagnostic on eight consumed actors and correlated variants. Neither an adopted point nor fresh accuracy/false-alarm evidence. Frozen point, weights and all outcomes remain unchanged.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("responses", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    result = diagnose(a.responses)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
