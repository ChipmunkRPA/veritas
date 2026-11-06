"""Independent feature arithmetic, complete windows, decisions and optional CPU refit.

The default review uses only the standard library. It does not establish source
pixels, learned neural forwards, spatial truth, generalization or outside review.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import struct
from collections import Counter
from pathlib import Path

from scripts.review_mouth_context_v10 import key, overlap, reconstruct


def close(a, b):
    if not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Temporal feature/fitting arithmetic differs")


def feature_vector(window, heads, measurements):
    centers = window["centers"]
    vector = []
    for arm in ("v8", "v9"):
        margins = {b["frame_index"]: b["margin"] for b in heads[arm]["model_inputs"]}
        values = [statistics.mean(margins[i + k] for k in (-1, 0, 1)) for i in centers]
        left, middle, right = values[:2], values[2:-2], values[-2:]
        l, m, r = map(statistics.mean, (left, middle, right))
        flank = left + right
        relative = min(m - l, m - r) - 0.25 * abs(l - r)
        sd = statistics.pstdev(middle)
        vector.extend(
            [
                m,
                statistics.mean(flank),
                relative,
                abs(l - r),
                sd,
                relative / (0.1 + sd),
                sd / (0.1 + statistics.pstdev(flank)),
            ]
        )
    native = {r["frame_index"]: r["values"] for r in measurements}
    for column in range(9):
        values = [statistics.mean(native[i + k][column] for k in (-1, 0, 1)) for i in centers]
        left, middle, right = values[:2], values[2:-2], values[-2:]
        vector.extend([statistics.mean(middle), abs(statistics.mean(middle) - statistics.mean(left + right))])
        if column < 6:
            vector.append(abs(statistics.mean(left) - statistics.mean(right)))
        vector.append(statistics.pstdev(middle))
    return vector + [window["end_s"] - window["start_s"], len(centers) - 4]


def check_measurements(row):
    head = row["heads"]["v8"]
    native = {r["frame_index"]: r for r in head["model_inputs"]}
    geometry = {}
    offset, previous = 0.0, None
    width, height = row["windows"][0]["region"]["width"], row["windows"][0]["region"]["height"]
    for trace in row["geometry_trace"]:
        upper, lower, left, right, e1, e2, nose = [(p[0] * width, p[1] * height) for p in trace["points"]]
        eye = (e2[0] - e1[0], e2[1] - e1[1])
        distance = math.hypot(*eye)
        mouth = math.dist(left, right)
        axis = (eye[0] / distance, eye[1] / distance)
        delta = (nose[0] - (e1[0] + e2[0]) / 2, nose[1] - (e1[1] + e2[1]) / 2)
        raw = math.atan2(eye[1], eye[0])
        if previous is not None:
            difference = raw - previous
            if difference > math.pi:
                offset -= 2 * math.pi
            elif difference < -math.pi:
                offset += 2 * math.pi
        previous = raw
        close(raw, trace["raw_roll"])
        geometry[trace["frame_index"]] = [
            math.dist(upper, lower) / mouth,
            mouth / distance,
            (delta[0] * axis[0] + delta[1] * axis[1]) / distance,
            (-delta[0] * axis[1] + delta[1] * axis[0]) / distance,
            raw + offset,
            distance / width,
        ]
    measured = {m["frame_index"]: m for m in row["measurements"]}
    if len(measured) != len(row["measurements"]) or set(measured) != set(native):
        raise ValueError("Native measurement schedule differs")
    for i, m in measured.items():
        b = native[i]
        if any(m[k] != b[k] for k in ("pts", "time_s", "decoded_rgb_sha256", "mouth_chip_rgb_sha256")):
            raise ValueError("Geometry/quality is bound to another source input")
        for a, b in zip(m["values"][:6], geometry[i], strict=True):
            close(a, b)


def margin(window, head):
    return (
        sum(
            (x - m) / s * c
            for x, m, s, c in zip(window["features"], head["mean"], head["scale"], head["coef"], strict=True)
        )
        + head["intercept"]
    )


def fitting_weights(rows):
    counts = Counter((r["group"], r["label"], r["condition"]) for r in rows)
    groups = {r["group"] for r in rows}
    conditions = {key: {c for g, y, c in counts if (g, y) == key} for key in {(g, y) for g, y, _ in counts}}
    return [
        len(rows)
        / (
            2
            * len(groups)
            * len(conditions[r["group"], r["label"]])
            * counts[r["group"], r["label"], r["condition"]]
        )
        for r in rows
    ]


def review(body, refit=False):
    panels, head, summary = body["panels"], body["head"], body["summary"]
    all_windows = measured = 0
    groups = []
    for stage, count in (("fit", 160), ("calibration", 24), ("development", 88)):
        rows = panels[stage]["cases"]
        if (
            len(rows) != count
            or len({r["id"] for r in rows}) != count
            or panels[stage]["original_receipt_sha256"]
            != head.get(stage + "_receipt_sha256", summary["receipt_sha256"])
        ):
            raise ValueError("Missing or changed complete feature panel")
        groups.append({r["group"] for r in rows})
        for row in rows:
            if row["status"] != "ok" or not row["native_input_replay"]:
                raise ValueError("Unassessed video cannot enter fitting or evaluation")
            for arm in ("v8", "v9"):
                reconstruct(row["heads"][arm])
            expected = {key(w) for w in row["heads"]["v8"]["context_windows_all"]}
            if len(row["windows"]) != len(expected) or {key(w) for w in row["windows"]} != expected:
                raise ValueError("A complete temporal window was omitted or repeated")
            check_measurements(row)
            measured += len(row["measurements"])
            for w in row["windows"]:
                values = feature_vector(w, row["heads"], row["measurements"])
                for a, b in zip(w["features"], values, strict=True):
                    close(a, b)
                if (
                    hashlib.sha256(struct.pack("<49d", *w["features"])).hexdigest()
                    != w["features_float64_sha256"]
                ):
                    raise ValueError("Retained feature bytes changed")
                all_windows += 1
    if (
        any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
        or groups[0] != set(head["fit_groups"])
        or groups[1] != set(head["point_groups"])
    ):
        raise ValueError("Source-role leakage")
    fitting = body["fitting_windows"]
    if len(fitting) != head["fit_windows"] or len(fitting) != sum(
        len(r["windows"]) for r in panels["fit"]["cases"]
    ):
        raise ValueError("Incomplete fitting window population")
    offset = 0
    for row in panels["fit"]["cases"]:
        for w in row["windows"]:
            r = fitting[offset]
            if (
                r["id"] != row["id"]
                or r["group"] != row["group"]
                or r["role"] != "fit"
                or r["condition"] != row["condition"]
                or r["features"] != w["features"]
            ):
                raise ValueError("Fitting identity/feature mismatch")
            if r["label"] and (not row["label"] or overlap(w, row["edit_time_range_s"]) < 0.2):
                raise ValueError("Invalid positive fitting time/clip label")
            offset += 1
    weights = fitting_weights(fitting)
    total = sum(weights)
    means = [
        sum(w * r["features"][j] for w, r in zip(weights, fitting, strict=True)) / total for j in range(49)
    ]
    scales = [
        max(
            1e-3,
            math.sqrt(
                sum(w * (r["features"][j] - means[j]) ** 2 for w, r in zip(weights, fitting, strict=True))
                / total
            ),
        )
        for j in range(49)
    ]
    for a, b in zip(head["mean"] + head["scale"], means + scales, strict=True):
        close(a, b)
    calibration = panels["calibration"]["cases"]
    if (
        any(r["label"] or r["role"] != "point" for r in calibration)
        or len(groups[1]) != 4
        or any(
            {r["condition"] for r in calibration if r["group"] == g}
            != {"original", "local-light", "local-resample", "muse-resample", "local-blur-2", "local-blur-4"}
            for g in groups[1]
        )
    ):
        raise ValueError("Wrong ordinary point-selection panel")
    control_max = max(margin(w, head) for r in calibration for w in r["windows"])
    close(control_max, head["ordinary_max_margin"])
    close(math.nextafter(max(0.0, control_max), math.inf), head["review_point"])
    scored = summary["scored_windows"]
    if [r["id"] for r in scored] != [r["id"] for r in panels["development"]["cases"]]:
        raise ValueError("Development result order mismatch")
    temporal = summary["cases"]["temporal"]
    for row, outputs, outcome in zip(panels["development"]["cases"], scored, temporal, strict=True):
        expected = sorted(
            [{**w, "response": margin(w, head)} for w in row["windows"]],
            key=lambda w: (-w["response"], w["start_s"], w["track_id"]),
        )
        if [key(w) for w in outputs["windows"]] != [key(w) for w in expected]:
            raise ValueError("Temporal windows not completely ranked")
        for actual, calculated in zip(outputs["windows"], expected, strict=True):
            close(actual["response"], calculated["response"])
        flagged = outputs["windows"][0]["response"] >= head["review_point"]
        window = outputs["windows"][0] if flagged else None
        if (
            outcome["id"] != row["id"]
            or outcome["flagged"] != flagged
            or outcome["localized"]
            != bool(row["label"] and overlap(window, row.get("edit_time_range_s")) >= 0.2)
        ):
            raise ValueError("Temporal decision/time outcome mismatch")
    absolute = body["absolute_reference"]
    if [r["id"] for r in absolute["cases"]] != [r["id"] for r in panels["development"]["cases"]]:
        raise ValueError("Absolute-reference order mismatch")
    combined = summary["cases"]["combined"]
    for old, old_outcome, temporal_outcome, signal, actual in zip(
        absolute["cases"], absolute["outcomes"], temporal, summary["selected_signals"], combined, strict=True
    ):
        old_flag = bool(
            old["probe_windows"] and old["probe_windows"][0]["response"] >= absolute["review_point"]
        )
        expected_signal = "absolute" if old_flag else "relative" if temporal_outcome["flagged"] else None
        expected_outcome = old_outcome if old_flag else temporal_outcome
        if signal != {"id": old["id"], "signal": expected_signal} or actual != expected_outcome:
            raise ValueError("Absolute-first evidence preservation differs")
    ordinary = {r["condition"] for r in absolute["outcomes"] if not r["label"]}
    ai = [r for r in combined if r["label"]]
    old_ai = [r for r in absolute["outcomes"] if r["label"]]
    groups_by_family = {
        f: [
            r
            for r, source in zip(combined, panels["development"]["cases"], strict=True)
            if source["family"] == f
        ]
        for f in ("liveportrait", "musetalk")
    }
    expected_gates = {
        "six_net_additional_locations": sum(r["localized"] for r in ai)
        >= sum(r["localized"] for r in old_ai) + 6,
        "first_family_six_locations": sum(r["localized"] for r in groups_by_family["liveportrait"]) >= 6,
        "both_first_family_conditions": all(
            sum(r["localized"] for r in groups_by_family["liveportrait"] if r["condition"] == c) >= 2
            for c in ("ai-redraw", "ai-retime")
        ),
        "retained_second_family": sum(r["localized"] for r in groups_by_family["musetalk"]) >= 13,
        "at_most_one_lost_location": sum(
            a["localized"] and not b["localized"] for a, b in zip(absolute["outcomes"], combined, strict=True)
        )
        <= 1,
        "total_ordinary_flags": sum(r["flagged"] for r in combined if not r["label"])
        <= sum(r["flagged"] for r in absolute["outcomes"] if not r["label"]) + 1,
        "condition_ordinary_flags": all(
            sum(r["flagged"] for r in combined if r["condition"] == c)
            <= sum(r["flagged"] for r in absolute["outcomes"] if r["condition"] == c) + 1
            for c in ordinary
        ),
        "spatial_iou_gain": statistics.mean(r["spatial_iou"] for r in ai)
        >= statistics.mean(r["spatial_iou"] for r in old_ai) + 0.02,
        "retained_temporal_iou": statistics.mean(r["temporal_iou"] for r in ai)
        >= statistics.mean(r["temporal_iou"] for r in old_ai),
        "retained_coverage": all(
            not a["assessed"] or b["assessed"] for a, b in zip(absolute["outcomes"], combined, strict=True)
        ),
        "complete_replays": all(
            r["native_input_replay"]
            and all(
                h["baseline_replay_parity"] and h["prior_native_and_margin_parity"]
                for h in r["heads"].values()
            )
            for r in panels["development"]["cases"]
        ),
    }
    if summary["gates"] != expected_gates or summary["passes"] != all(expected_gates.values()):
        raise ValueError("Frozen advancement gates differ")
    refit_error = None
    if refit:
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        x = np.asarray([r["features"] for r in fitting], np.float64)
        # Reproduce the specified weighted normalizer with the same reduction dtype.
        mean = np.average(x, weights=weights, axis=0)
        scale = np.maximum(np.sqrt(np.average((x - mean) ** 2, weights=weights, axis=0)), 1e-3)
        model = LogisticRegression(C=0.1, solver="lbfgs", tol=1e-8, max_iter=2000, random_state=2718)
        model.fit((x - mean) / scale, [r["label"] for r in fitting], sample_weight=weights)
        refit_error = float(
            max(
                abs(a - b)
                for a, b in zip(
                    list(model.coef_[0]) + [model.intercept_[0]],
                    head["coef"] + [head["intercept"]],
                    strict=True,
                )
            )
        )
        if refit_error > 1e-7:
            raise ValueError("Independent CPU refit differs")
    return {
        "passed": True,
        "cases": 272,
        "complete_windows": all_windows,
        "native_measurements": measured,
        "fit_windows": len(fitting),
        "refit_max_coefficient_error": refit_error,
        "scope": "Complete feature arithmetic and raw-margin decisions; optional CPU refit. Not source pixels, truth-mask intersection, neural forwards, spatial IoU, generalization or outside review.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capsule", type=Path)
    p.add_argument("--refit", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = review(json.loads(gzip.decompress(a.capsule.read_bytes())), a.refit)
    if a.output:
        with a.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result))
