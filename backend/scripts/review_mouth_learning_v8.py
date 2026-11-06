"""Standard-library review of published retained-window arithmetic and denominators.

This does not replay pretrained/learned inference, source masks, omitted windows
or spatial IoU. Native/media audits are separate and are not outside validation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("Retained window arithmetic differs from bound raw frame margins")


def overlap(a, b):
    if not a or not b:
        return 0.0
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union else 0.0


def review(responses, summary, head):
    data = json.loads(gzip.decompress(responses.read_bytes()))
    result = json.loads(summary.read_text())
    fitted = json.loads(head.read_text())
    if (
        data["original_receipt_sha256"] != result["score_receipt_sha256"]
        or data["header"]["head_sha256"] != hashlib.sha256(head.read_bytes()).hexdigest()
        or data["header"]["review_point"] != fitted["review_point"]
    ):
        raise ValueError("Public response/summary/head identity differs")
    cases = data["cases"]
    if (
        len(cases) != 88
        or len({r["id"] for r in cases}) != 88
        or len({r["group"] for r in cases}) != 8
        or sum(r["label"] for r in cases) != 32
    ):
        raise ValueError("Expected 88 cases/32 AI edits/eight consumed source actors")
    checked_windows = 0
    counts = {}
    for arm, point in (("base", data["current_review_point"]), ("probe", fitted["review_point"])):
        calculated = []
        expected_cases = result["cases"]["baseline" if arm == "base" else "candidate"]
        if [r["id"] for r in expected_cases] != [r["id"] for r in cases]:
            raise ValueError("Public evaluated case identities/order differ")
        for row, expected in zip(cases, expected_cases, strict=True):
            windows = row[arm + "_windows"]
            if arm == "probe":
                inputs = {r["frame_index"]: r for r in row["model_inputs"]}
                for window in windows:
                    centers = window["centers"]
                    if len(centers) < 7 or len(set(centers)) != len(centers) or centers != sorted(centers):
                        raise ValueError("Retained source-context order differs")
                    means = []
                    for i in centers:
                        means.append(sum(inputs[k]["margin"] for k in (i - 1, i, i + 1)) / 3)
                    close(window["response"], sum(means[2:-2]) / len(means[2:-2]))
                    close(window["before_margin"], sum(means[:2]) / 2)
                    close(window["after_margin"], sum(means[-2:]) / 2)
                    close(window["start_s"], inputs[centers[2]]["time_s"])
                    close(window["end_s"], inputs[centers[-3]]["time_s"])
                    checked_windows += 1
                if windows != sorted(windows, key=lambda w: (-w["response"], w["start_s"], w["track_id"])):
                    raise ValueError("Retained candidate windows are not ranked as declared")
            chosen = next((w for w in windows if w["response"] >= point), None)
            interval = [chosen["start_s"], chosen["end_s"]] if chosen else None
            temporal = overlap(interval, row["edit_time_range_s"])
            calculated.append(
                {
                    "label": row["label"],
                    "assessed": bool(windows),
                    "flagged": bool(chosen),
                    "localized": bool(row["label"] and temporal >= 0.2),
                }
            )
            close(temporal, expected["temporal_iou"])
            if any(calculated[-1][k] != expected[k] for k in ("label", "assessed", "flagged", "localized")):
                raise ValueError("Public source interval/flag/outcome differs")
        counts[arm] = {
            "planned": len(calculated),
            "assessed": sum(r["assessed"] for r in calculated),
            "ai_edits": sum(r["label"] for r in calculated),
            "non_ai_controls": sum(not r["label"] for r in calculated),
            "localized_ai": sum(r["localized"] for r in calculated),
            "non_ai_flags": sum(not r["label"] and r["flagged"] for r in calculated),
            "ai_without_flag": sum(r["label"] and not r["flagged"] for r in calculated),
        }
        expected_summary = result["baseline" if arm == "base" else "candidate"]
        if any(value != expected_summary[key] for key, value in counts[arm].items()):
            raise ValueError("Published summary dropped or reclassified an outcome")
    return {
        "schema": "chipmunk.mouth-learning-public-arithmetic-review/8",
        "passed": True,
        "retained_candidate_windows_checked": checked_windows,
        "outcomes_checked": 176,
        "counts": counts,
        "public_responses_sha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
        "scope": "Retained learned-window means, source times, flags, temporal IoU and complete denominators. Does not verify model inference, omitted windows, source masks or spatial IoU. Not outside validation.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("responses", type=Path)
    p.add_argument("summary", type=Path)
    p.add_argument("head", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    result = review(a.responses, a.summary, a.head)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
