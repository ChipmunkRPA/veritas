"""Review published window arithmetic and complete temporal outcomes without ML packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

CONTROLS = {"original", "local-light", "local-resample", "constant-audio-delay"}


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("A finite raw measurement is required")
    return float(value)


def equal(actual, expected):
    if not math.isclose(finite(actual), finite(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Published window arithmetic differs from retained model responses")


def check_windows(row, source):
    inputs = row["model_inputs"]
    if [p["frame_index"] for p in inputs] != sorted({p["frame_index"] for p in inputs}):
        raise ValueError("Duplicated or unordered frame inputs")
    table, time_base = {}, Fraction(source["time_base"])
    for p in inputs:
        i = p["frame_index"]
        if (
            not 0 <= i < len(source["pts"])
            or p["pts"] != source["pts"][i]
            or p["time_base"] != source["time_base"]
        ):
            raise ValueError("Model response is bound to the wrong source timestamp")
        equal(p["time_s"], float(p["pts"] * time_base))
        table[i] = (p["time_s"], finite(p["fake_minus_real_logit"]))
    windows = row["artifact_windows"]
    if len(windows) > 12 or windows != sorted(
        windows, key=lambda w: (-w["response"], w["start_s"], w["track_id"])
    ):
        raise ValueError("Reported window order or count changed")
    for w in windows:
        indices = w["centers"]
        if len(indices) < 7 or indices != sorted(set(indices)):
            raise ValueError("Reported context does not have distinct ordered centres")
        blocks = []
        for i in indices:
            if not all(k in table for k in (i - 1, i, i + 1)) or table[i + 1][0] - table[i - 1][0] > 0.13:
                raise ValueError("A measured source triplet is missing or too widely spaced")
            blocks.append(math.fsum(table[k][1] for k in (i - 1, i, i + 1)) / 3)
        times = [table[i][0] for i in indices]
        if any(b - a > 0.55 for a, b in zip(times, times[1:])) or not 0.52 <= times[-3] - times[2] <= 2.6:
            raise ValueError("Reported window violates frozen source-context spacing")
        left = math.fsum(blocks[:2]) / 2
        middle = math.fsum(blocks[2:-2]) / len(blocks[2:-2])
        right = math.fsum(blocks[-2:]) / 2
        for field, expected in (
            ("before_logit", left),
            ("middle_logit", middle),
            ("after_logit", right),
            ("start_s", times[2]),
            ("end_s", times[-3]),
            ("response", min(middle - left, middle - right) - 0.35 * abs(left - right)),
        ):
            equal(w[field], expected)


def iou(a, b):
    if not a or not b:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / union if union > 0 else 0.0


def summarize(catalogue, receipt):
    if (
        catalogue["schema"] != "chipmunk.public-mouth-edit-development/4"
        or receipt["schema"] != "chipmunk.mouth-artifact-public-review/5"
    ):
        raise ValueError("Unknown public review schema")
    rows, sources = receipt["cases"], catalogue["cases"]
    if [r["id"] for r in rows] != [r["id"] for r in sources]:
        raise ValueError("Every planned input must appear exactly once in source order")
    for row, source in zip(rows, sources, strict=True):
        if any(row[k] != source[k] for k in ("id", "group", "condition", "label", "sha256")):
            raise ValueError("Published source identity or label changed")
        if row["status"] not in {"ok", "not_applicable", "failed"}:
            raise ValueError("Unknown terminal outcome")
        if row["status"] != "ok" and not row.get("reason"):
            raise ValueError("Unassessed outcomes require a reason")
        check_windows(row, source)
    controls = [r for r in rows if not r["label"] and r["artifact_windows"] and r["status"] == "ok"]
    point = (
        math.nextafter(max(0.0, max(r["artifact_windows"][0]["response"] for r in controls)), math.inf)
        if {r["condition"] for r in controls} == CONTROLS
        else None
    )
    if point != receipt["candidate_review_point"]:
        raise ValueError("Candidate point differs from the frozen control-only recipe")
    baseline_point = catalogue["comparison_lock"]["baseline_review_point"]
    all_outcomes = {}
    for arm, arm_point in (("base", baseline_point), ("artifact", point)):
        outcomes = []
        for row, source in zip(rows, sources, strict=True):
            windows = row[arm + "_windows"]
            assessed = bool(windows) and (arm == "base" or row["status"] == "ok")
            flagged = assessed and arm_point is not None and finite(windows[0]["response"]) >= arm_point
            overlap = iou(
                [windows[0]["start_s"], windows[0]["end_s"]] if flagged else None, source["edit_time_range_s"]
            )
            outcomes.append(
                {
                    "id": row["id"],
                    "condition": row["condition"],
                    "label": row["label"],
                    "assessed": assessed,
                    "flagged": flagged,
                    "localized": bool(row["label"] and overlap >= 0.2),
                    "temporal_iou": overlap,
                }
            )
        all_outcomes[arm] = outcomes

    def counts(outcomes):
        ai = [r for r in outcomes if r["label"]]
        controls = [r for r in outcomes if not r["label"]]
        return {
            "planned": len(outcomes),
            "ai_edits": len(ai),
            "non_ai_controls": len(controls),
            "assessed": sum(r["assessed"] for r in outcomes),
            "localized_ai": sum(r["localized"] for r in ai),
            "ai_without_flag": sum(not r["flagged"] for r in ai),
            "non_ai_flags": sum(r["flagged"] for r in controls),
            "mean_ai_temporal_iou": math.fsum(r["temporal_iou"] for r in ai) / len(ai) if ai else None,
        }

    return {
        "schema": "chipmunk.mouth-artifact-arithmetic-review/5",
        "candidate_review_point": receipt["candidate_review_point"],
        "baseline": counts(all_outcomes["base"]),
        "candidate": counts(all_outcomes["artifact"]),
        "conditions": {
            c: {
                arm: counts([r for r in outcomes if r["condition"] == c])
                for arm, outcomes in all_outcomes.items()
            }
            for c in sorted({r["condition"] for r in rows})
        },
        "scope": "Published-window arithmetic, source-clock metadata, control-selected point and complete temporal denominators only",
        "does_not_verify": [
            "Classifier execution or pixel/tensor bytes without media and weights",
            "Completeness of eligible-window enumeration or face-track membership",
            "Spatial masks, probability calibration, fresh accuracy or outside reviewer participation",
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalogue", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--catalogue-sha256", required=True)
    p.add_argument("--receipt-sha256", required=True)
    a = p.parse_args()
    for path, expected in ((a.catalogue, a.catalogue_sha256), (a.receipt, a.receipt_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            p.error("Public review input differs from the selected published digest")
    result = summarize(json.loads(a.catalogue.read_text()), json.loads(a.receipt.read_text()))
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
