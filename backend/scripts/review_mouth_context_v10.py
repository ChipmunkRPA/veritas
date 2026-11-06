"""Standard-library reconstruction of every eligible V10 context window and point.

Source pixels, learned forwards, spatial IoU and generalization remain outside
this arithmetic review. Local reproduction is not outside validation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def close(a, b):
    if not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Context arithmetic differs from bound raw inputs")


def key(window):
    return window["track_id"], window["start_s"], window["end_s"], tuple(window["centers"])


def reconstruct(head):
    inputs = {r["frame_index"]: r for r in head["model_inputs"]}
    expected = {}
    for track, indices in head["source_tracks"].items():
        eligible = [
            i
            for i in head["source_centers"]
            if all(k in indices and k in inputs for k in (i - 1, i, i + 1))
            and inputs[i + 1]["time_s"] - inputs[i - 1]["time_s"] <= 0.13
        ]
        if len(eligible) < 7:
            continue
        anchor = inputs[eligible[0]]["margin"]
        means = {i: sum(inputs[k]["margin"] - anchor for k in (i - 1, i, i + 1)) / 3 for i in eligible}
        times = [inputs[i]["time_s"] for i in eligible]
        for start in range(2, len(eligible) - 4):
            for end in range(start + 3, len(eligible) - 1):
                span = times[end - 1] - times[start]
                if span > 2.6:
                    break
                if (
                    span < 0.52
                    or max(
                        b - a
                        for a, b in zip(times[start - 2 : end + 1], times[start - 1 : end + 2], strict=True)
                    )
                    > 0.55
                ):
                    continue
                middle = sum(means[i] for i in eligible[start:end]) / (end - start)
                before = sum(means[i] for i in eligible[start - 2 : start]) / 2
                after = sum(means[i] for i in eligible[end : end + 2]) / 2
                item = {
                    "track_id": track,
                    "start_s": times[start],
                    "end_s": times[end - 1],
                    "centers": eligible[start - 2 : end + 2],
                    "reference_margin": anchor,
                    "middle_margin": middle + anchor,
                    "before_margin": before + anchor,
                    "after_margin": after + anchor,
                    "response": min(middle - before, middle - after) - 0.25 * abs(before - after),
                }
                expected[key(item)] = item
    retained = head["context_windows_all"]
    if (
        len(retained) != len(expected)
        or len({key(w) for w in retained}) != len(retained)
        or {key(w) for w in retained} != set(expected)
    ):
        raise ValueError("An eligible source-context window was omitted or repeated")
    for window in retained:
        calculated = expected[key(window)]
        for field in ("response", "middle_margin", "before_margin", "after_margin", "reference_margin"):
            close(window[field], calculated[field])
    if (
        retained != sorted(retained, key=lambda w: (-w["response"], w["start_s"], w["track_id"]))
        or head["probe_windows"] != retained[:12]
    ):
        raise ValueError("Windows were not ranked completely before retaining the top12")
    return len(retained)


def overlap(window, interval):
    if not window or not interval:
        return 0.0
    a, b = window["start_s"], window["end_s"]
    intersection = max(0.0, min(b, interval[1]) - max(a, interval[0]))
    union = max(b, interval[1]) - min(a, interval[0])
    return intersection / union if union else 0.0


def review(root):
    panels = root / "evaluation-panels"
    calibration = unpack(panels / "mouth-v10-calibration-responses.json.gz")
    development = unpack(panels / "mouth-v10-development-responses.json.gz")
    old = unpack(panels / "mouth-v8-review-responses.json.gz")
    old9 = unpack(panels / "mouth-v9-review-responses.json.gz")
    lock = json.loads((panels / "mouth-v10-lock.json").read_text())
    calibration_manifest = unpack(panels / "mouth-v10-calibration-inputs.json.gz")
    result = json.loads((root / "mouth-context-v10-evaluation.json").read_text())
    points = json.loads((panels / "mouth-v10-points.json").read_text())
    old_head = json.loads((panels / "mouth-v8-head.json").read_text())
    actual_heads = {
        arm: hashlib.sha256((panels / ("mouth-" + arm + "-head.json")).read_bytes()).hexdigest()
        for arm in ("v8", "v9")
    }
    if (
        calibration["header"]["heads"] != actual_heads
        or development["header"]["heads"] != actual_heads
        or {arm: entry["sha256"] for arm, entry in lock["heads"].items()} != actual_heads
    ):
        raise ValueError("Published head identities differ from the frozen response records")
    if (
        points["calibration_scores_sha256"] != calibration["original_receipt_sha256"]
        or result["score_receipt_sha256"] != development["original_receipt_sha256"]
        or result["points_sha256"] != points["original_points_sha256"]
        or calibration["header"]["lock_sha256"] != development["header"]["lock_sha256"]
    ):
        raise ValueError("Public scores, points and summary have different bindings")
    ordinary, cases = calibration["cases"], development["cases"]
    if (
        lock["original_lock_sha256"] != development["header"]["lock_sha256"]
        or lock["prior_scores"]["v8"]["sha256"] != old["original_receipt_sha256"]
        or lock["prior_scores"]["v9"]["sha256"] != old9["original_receipt_sha256"]
        or lock["calibration_manifest_sha256"] != calibration_manifest["original_manifest_sha256"]
    ):
        raise ValueError("Frozen source/prior response identities differ")
    if (
        len(ordinary) != 24
        or len(cases) != 88
        or len({r["id"] for r in ordinary}) != 24
        or len({r["id"] for r in cases}) != 88
    ):
        raise ValueError("Every planned calibration/development case is required")
    expected_ordinary = calibration_manifest["items"]
    if [r["id"] for r in ordinary] != [r["id"] for r in expected_ordinary]:
        raise ValueError("Ordinary source manifest identities/order differ")
    for row, source in zip(ordinary, expected_ordinary, strict=True):
        if any(
            row[k] != source[k] for k in ("group", "role", "label", "condition", "sha256", "source_sha256")
        ):
            raise ValueError("Ordinary source role/label/bytes differ")
    for row, source in zip(cases, old["cases"], strict=True):
        if any(
            row[k] != source[k]
            for k in (
                "id",
                "group",
                "label",
                "condition",
                "sha256",
                "source_sha256",
                "frames",
                "time_base",
                "pts",
                "edit_time_range_s",
            )
        ):
            raise ValueError("Development source identity/clock/label differs from prior")
    for arm, prior_body in (("v8", old), ("v9", old9)):
        lookup = {r["id"]: r for r in prior_body["cases"]}
        for row in cases:
            new_inputs = row["heads"][arm]["model_inputs"]
            prior_inputs = lookup[row["id"]]["model_inputs"]
            removed = {"features_float32_sha256", "margin"}
            if [{k: v for k, v in r.items() if k not in removed} for r in new_inputs] != [
                {k: v for k, v in r.items() if k not in removed} for r in prior_inputs
            ] or any(
                abs(a["margin"] - b["margin"]) > 1e-7 for a, b in zip(new_inputs, prior_inputs, strict=True)
            ):
                raise ValueError("Frozen native input or learned raw response differs")
    if any(r["role"] != "point" or r["label"] != 0 for r in ordinary) or {r["group"] for r in ordinary} & {
        r["group"] for r in cases
    }:
        raise ValueError("Ordinary point roles overlap video sources or contain positives")
    windows_checked = 0
    for row in ordinary + cases:
        if row.get("failure_reason"):
            raise ValueError("Cannot claim complete arithmetic replay for failed input")
        for arm in ("v8", "v9"):
            if row["heads"][arm]["status"] != "ok":
                raise ValueError("An expected source-context outcome was unassessed")
            windows_checked += reconstruct(row["heads"][arm])
    selected_points = {
        arm: math.nextafter(
            max(0.0, max(row["heads"][arm]["probe_windows"][0]["response"] for row in ordinary)), math.inf
        )
        for arm in ("v8", "v9")
    }
    if selected_points != points["points"] or selected_points != result["points"]:
        raise ValueError("Point was not selected from complete ordinary video-window responses")
    prior = {r["id"]: r for r in old["cases"]}
    counts = {}
    for arm in ("v9", "v8"):
        expected = result["arms"][arm]["combined_cases"]
        if [r["id"] for r in expected] != [r["id"] for r in cases]:
            raise ValueError("Combined outcome identity/order differs")
        flagged = located = ordinary_flags = 0
        for row, saved, signal in zip(cases, expected, result["arms"][arm]["selected_signals"], strict=True):
            absolute = prior[row["id"]]["probe_windows"]
            relative = row["heads"][arm]["probe_windows"]
            if absolute and absolute[0]["response"] >= old_head["review_point"]:
                chosen, selection = absolute[0], "absolute"
            elif relative and relative[0]["response"] >= selected_points[arm]:
                chosen, selection = relative[0], "relative"
            else:
                chosen, selection = None, None
            temporal = overlap(chosen, row["edit_time_range_s"])
            location = bool(row["label"] and temporal >= 0.2)
            if (
                saved["flagged"] != bool(chosen)
                or saved["localized"] != location
                or signal != {"id": row["id"], "signal": selection}
            ):
                raise ValueError("Absolute preservation/relative selection outcome differs")
            close(saved["temporal_iou"], temporal)
            flagged += bool(chosen)
            located += location
            ordinary_flags += not row["label"] and bool(chosen)
        summary = result["arms"][arm]["combined"]
        if summary["localized_ai"] != located or summary["non_ai_flags"] != ordinary_flags:
            raise ValueError("Combined summary dropped/reclassified an outcome")
        counts[arm] = {
            "clips": 88,
            "flagged": flagged,
            "located_ai": located,
            "ordinary_flags": ordinary_flags,
        }
    return {
        "schema": "chipmunk.mouth-context-public-arithmetic-review/10",
        "passed": True,
        "calibration_cases": 24,
        "development_cases": 88,
        "all_eligible_context_windows_checked": windows_checked,
        "points_reproduced": selected_points,
        "combined_counts": counts,
        "scope": "Complete context enumeration/arithmetic/ranking, ordinary-window points and absolute/relative selection. Does not replay neural inference, native pixels/masks/spatial IoU, statistical generalization or outside participation.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("docs", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    result = review(a.docs)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
