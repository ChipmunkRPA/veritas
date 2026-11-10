"""Independent array/clock/lag arithmetic replay of both locked held-out roles."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import types
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import native_mouth_components_v128 as evaluation
from scripts.video_appearance_corrected_v112 import scoped

f, original = evaluation.f, evaluation.original
ROOT, BASE, PANEL = f.ROOT, f.BASE, f.PANEL
TRIAL = BASE / "native-mouth-arithmetic-review-v129"
PUBLIC = PANEL / "native-mouth-arithmetic-review-freeze-v129.json"
CODE = (
    "backend/scripts/review_native_mouth_v129.py",
    "backend/tests/test_review_native_mouth_v129.py",
    "docs/native-mouth-arithmetic-review-v129-method.md",
)
read, write, sha, require = f.read, f.write, f.sha, f.require


def close(actual, expected):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    require(
        a.shape == b.shape
        and np.isfinite(a).all()
        and np.isfinite(b).all()
        and np.allclose(a, b, rtol=1e-9, atol=1e-9),
        "Independent finite array/scalar arithmetic differs",
    )


def probability(raw, cal):
    z = cal["slope"] * ((np.asarray(raw, np.float64) - cal["centre"]) / cal["scale"]) + cal["intercept"]
    return np.exp(-np.logaddexp(0.0, -z))


def reconstruct(record, learned):
    """Separate equations; does not call v122/v125 feature/project/margin helpers."""
    from app.detectors.video.decode import FrameTable

    if record["status"] == "unavailable":
        return [], {a: np.empty(0) for a in original.fitted.ARMS}
    table = FrameTable.from_json(record["table"])
    image = original.fitted.reviewed.array_file(record, np.float32).astype(np.float64)
    audio = original.fitted.reviewed.array_file(record["speech"], np.float64)
    lookup = {r["frame_index"]: r["feature_row"] for r in record["frames"] if r["status"] == "ok"}
    sound = {i: j for j, i in enumerate(record["speech"]["indices"])}
    indices = []
    visual = []
    speech = []
    for i in range(1, len(table) - 1):
        left, centre, right = [Fraction(table.pts[j]) * table.tb for j in (i - 1, i, i + 1)]
        if (
            i not in sound
            or not all(j in lookup for j in (i - 1, i, i + 1))
            or max(centre - left, right - centre) > Fraction(2, 25)
        ):
            continue
        a, b, c = [image[lookup[j]] for j in (i - 1, i, i + 1)]
        interpolation = a + (c - a) * float((centre - left) / (right - left))
        visual.append(
            np.r_[b, np.abs(b - interpolation), np.abs(c - a) * float(Fraction(1, 25) / (right - left))]
        )
        speech.append(audio[sound[i]])
        indices.append(i)
    if not indices:
        return [], {a: np.empty(0) for a in original.fitted.ARMS}
    reference = learned["reference"]
    projected = []
    for matrix, name in ((np.asarray(visual), "visual_projection"), (np.asarray(speech), "audio_projection")):
        spec = reference[name]
        projected.append(
            np.dot(matrix - np.asarray(spec["center"]), np.asarray(spec["components"]).T)
            / np.asarray(spec["scale"])
        )
    v = projected[0]
    a = np.dot(projected[1], np.asarray(reference["audio_to_visual"]))
    representations = {
        "visual_only": v,
        "concatenation": np.c_[v, a],
        "interaction": np.c_[v, a, v * a, np.abs(v - a)],
    }
    margins = {}
    for arm, matrix in representations.items():
        head = learned["heads"][arm]
        margins[arm] = (
            np.dot((matrix - np.asarray(head["mean"])) / np.asarray(head["scale"]), np.asarray(head["coef"]))
            + head["intercept"]
        )
    return indices, margins


def frame_union(table, terminal_ms, selected, bounds):
    """Duration from union of native picture cells, independent of interval merge."""
    flags = {i for w in selected for i in range(*w["support_frames_half_open"])}
    require(all(0 <= i < len(table) for i in flags), "Flagged native cell outside source")
    edge = [Fraction(t) * table.tb for t in table.pts]
    edge.append(edge[-1] + Fraction(terminal_ms, 1000))
    length = sum((edge[i + 1] - edge[i] for i in flags), Fraction())
    truth = set(range(*bounds)) if bounds is not None else set()
    overlap = sum((edge[i + 1] - edge[i] for i in flags & truth), Fraction())
    truth_length = sum((edge[i + 1] - edge[i] for i in truth), Fraction())
    union = length + truth_length - overlap
    iou = float(overlap / union) if bounds is not None else 0.0
    return {
        "flagged_duration_seconds": float(length),
        "edit_overlap_seconds": float(overlap),
        "outside_edit_flagged_seconds": float(length - overlap),
        "temporal_union_iou": iou,
        "localized_iou_0_2": bounds is not None and iou >= 0.2,
    }


def pearson(x, y):
    require(len(x) == len(y) >= 3, "Complete lag input vectors required")
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    a, b = [v - mx for v in x], [v - my for v in y]
    xx, yy = math.fsum(v * v for v in a), math.fsum(v * v for v in b)
    if min(math.sqrt(xx / len(x)), math.sqrt(yy / len(y))) < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, math.fsum(v * w for v, w in zip(a, b, strict=True)) / math.sqrt(xx * yy)))


def lag_replay(current):
    windows = coefficients = 0
    proof = current.get("details", {}).get("av_input", {})
    if proof.get("status") != "ready":
        return windows, coefficients
    for track in proof["tracks"]:
        mouth = {r["tick"]: r["mouth_value"] for r in track["frame_inputs"]}
        for window in track["windows"]:
            start, end = window["start_tick"], window["end_tick"]
            audio = {r["tick"]: r["rms"] for r in window["audio_curve"]}
            profile = window["lag_profile"]
            require(
                [r["offset_ticks"] for r in profile] == list(range(-15, 16)),
                "Full original lag search required",
            )
            for item in profile:
                lag = item["offset_ticks"]
                expected = pearson(
                    [mouth[k] for k in range(start, end)], [audio[k + lag] for k in range(start, end)]
                )
                close(item["correlation"], expected)
                coefficients += 1
            winner = max(range(31), key=lambda i: profile[i]["correlation"])
            require(
                window["offset_ticks"] == profile[winner]["offset_ticks"],
                "Recorded best lag contradicts actual lag profile",
            )
            close(window["peak_correlation"], profile[winner]["correlation"])
            windows += 1
    return windows, coefficients


def completed(role):
    require(role in original.ROLES, "One fixed held-out role required")
    stage = evaluation.TRIAL / f"bind-{original.ROLES.index(role):02d}"
    r, p = read(stage / "output/result.json"), read(stage / "profile.json")
    public = read(PANEL / f"native-mouth-{role}-result-v128.json")
    folder = evaluation.PRIVATE / role
    require(
        r["status"] == "complete"
        and r["role"] == role
        and p["child_exit_code"] == 0
        and p["guard"] is None
        and p["sample_errors"] == 0
        and public["private_receipt_sha256"]
        == p["worker_receipt_sha256"]
        == sha(stage / "output/result.json")
        and r["private_score_journal_sha256"] == sha(folder / "scored-cases.jsonl")
        and r["private_outcomes_sha256"] == sha(folder / "outcomes.json"),
        "Successful exact frozen held-out artifacts required",
    )
    return r, folder


def identity():
    evaluation.verify()
    for role in original.ROLES:
        completed(role)
    return {
        "schema": "chipmunk.native-mouth-arithmetic-review-freeze/129",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "evaluation_freeze_sha256": sha(evaluation.PUBLIC),
        "role_result_sha256": {
            role: sha(PANEL / f"native-mouth-{role}-result-v128.json") for role in original.ROLES
        },
        "numeric_rtol": 1e-9,
        "numeric_atol": 1e-9,
        "scope": "Separate native descriptors/projections/raw margins/sigmoids, whole-window input clocks, native-cell time union, confusion counts and saved actual current lag-curve arithmetic. No network forwards, refit, independent face/speech re-encoding, region-mask reconstruction or independent unavailability judgment.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged independent arithmetic replay required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only fixed data audit permitted",
    )
    args = list(argv)
    args[3] = "scripts.review_native_mouth_v129"
    return subprocess.Popen(args, **kwargs)


def run(role):
    require(role in original.ROLES, "One exact audit role required")
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        f.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda c, b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", original.ROLES.index(role))


def review(role, output):
    from threadpoolctl import threadpool_limits

    from app.detectors.video.decode import FrameTable

    threadpool_limits(limits=1)
    result, folder = completed(role)
    rows, constructs = original.calibration.records_for(role)
    scores = [json.loads(line) for line in (folder / "scored-cases.jsonl").read_text().splitlines()]
    outcomes = read(folder / "outcomes.json")["candidates"]
    learned = read(original.fitted.PRIVATE / "reference-and-heads.json")
    cals = read(original.calibration.PRIVATE / "calibrations-and-research-points.json")["arms"]
    plans = {r["source"]["asset_index"]: r["source"] for r in read(f.clips.TRIAL / "plan.json")}
    point_count = margin_count = window_count = outcome_count = lag_windows = lag_coefficients = 0
    require(len(rows) == len(scores) == 60, "Every held-out source/condition required")
    with (output / "reviewed-cases.jsonl").open("x") as journal:
        for ordinal, (row, score) in enumerate(zip(rows, scores, strict=True)):
            require(
                all(row[k] == score[k] for k in ("asset_index", "source_group", "study_role", "condition")),
                "Source/role/condition pairing differs",
            )
            indices, raw = reconstruct(row, learned)
            require(
                indices == score["eligible_centres"], "Independent blind eligible centre schedule differs"
            )
            point_count += len(indices)
            construction = constructs[row["asset_index"], row["condition"]]
            table = FrameTable.from_json(row["table"]) if row["status"] == "extracted" else None
            for arm, values in raw.items():
                part = score["arms"][arm]
                close(values, part["point_margins"])
                margin_count += len(values)
                close(probability(values, cals[arm]["point"]), part["study_point_probabilities"])
                by_index = dict(zip(indices, part["point_margins"], strict=True))
                actual = {w["centre_frame_index"]: w for w in part["windows"]}
                expected = []
                for i in indices:
                    if not all(j in by_index for j in (i - 1, i, i + 1)):
                        continue
                    times = [Fraction(table.pts[j]) * table.tb for j in range(i - 2, i + 3)]
                    if max(b - a for a, b in zip(times, times[1:])) > Fraction(2, 25):
                        continue
                    expected.append(i)
                    w = actual[i]
                    stop = i + 3
                    end = (
                        Fraction(table.pts[stop]) * table.tb
                        if stop < len(table)
                        else times[-1]
                        + Fraction(construction["expected"]["constructed_last_picture_duration_ms"], 1000)
                    )
                    require(
                        w["support_frames_half_open"] == [i - 2, stop]
                        and Fraction(w["start_seconds"]) == times[0]
                        and Fraction(w["end_seconds"]) == end
                        and [w["before_frame_index"], w["during_frame_index"], w["after_frame_index"]]
                        == [max(0, i - 3), i, min(len(table) - 1, stop)],
                        "Native window clocks or evidence indices differ",
                    )
                    close(w["response"], math.fsum(by_index[j] for j in (i - 1, i, i + 1)) / 3)
                    window_count += 1
                require(
                    set(actual) == set(expected) and len(actual) == len(part["windows"]),
                    "Incomplete or duplicated native windows",
                )
                if actual:
                    close(part["window_score"], max(w["response"] for w in actual.values()))
                    close(
                        part["study_clip_probability"], probability(part["window_score"], cals[arm]["clip"])
                    )
                else:
                    require(
                        part["window_score"] is None and part["study_clip_probability"] is None,
                        "Missing scope received a probability",
                    )
                for budget, point in cals[arm]["research_points"].items():
                    saved = outcomes[arm][budget][ordinal]
                    selected = [w for w in actual.values() if w["response"] > point["raw_margin"]]
                    require(
                        saved["flagged"] == bool(selected) and saved["assessed"] == bool(actual),
                        "Strict research decision or missing scope differs",
                    )
                    if table is not None:
                        truth = plans[row["asset_index"]]["edit_frames_half_open"] if score["label"] else None
                        calculated = frame_union(
                            table,
                            construction["expected"]["constructed_last_picture_duration_ms"],
                            selected,
                            truth,
                        )
                        for key, value in calculated.items():
                            close(saved[key], value)
                    outcome_count += 1
            n, c = lag_replay(score["current"])
            lag_windows += n
            lag_coefficients += c
            journal.write(
                json.dumps({"eligible_centres": len(indices), "lag_windows": n, "lag_coefficients": c}) + "\n"
            )
            journal.flush()
        for arm, budgets in outcomes.items():
            for budget, cases in budgets.items():
                report = result["arms"][arm]["research_budgets"][budget]
                for key, expected in {
                    "ai_flags": sum(r["flagged"] and r["label"] for r in cases),
                    "genuine_false_flags": sum(r["flagged"] and not r["label"] for r in cases),
                    "assessed_cases": sum(r["assessed"] for r in cases),
                    "ai_localized_iou_0_2": sum(r["label"] and r["localized_iou_0_2"] for r in cases),
                    "missed_assessed_ai": sum(
                        r["label"] and r["assessed"] and not r["flagged"] for r in cases
                    ),
                }.items():
                    require(report[key] == expected, "Published confusion/localization counts differ")
                usable = [r for r in cases if r["study_probability"] is not None]
                if usable:
                    close(
                        report["clip_reliability_unweighted"]["brier"],
                        math.fsum((r["study_probability"] - r["label"]) ** 2 for r in usable) / len(usable),
                    )
        baseline = result["current_common_input"]["overall"]
        for key, expected in {
            "assessed_cases": sum(r["current"]["assessed"] for r in scores),
            "ai_flags": sum(r["label"] and r["current"]["flagged"] for r in scores),
            "genuine_false_flags": sum(not r["label"] and r["current"]["flagged"] for r in scores),
            "missed_assessed_ai": sum(
                r["label"] and r["current"]["assessed"] and not r["current"]["flagged"] for r in scores
            ),
        }.items():
            require(baseline[key] == expected, "Current adapter coverage/error totals differ")
    receipt = {
        "schema": "chipmunk.native-mouth-arithmetic-review/129",
        "status": "complete",
        "role": role,
        "cases": len(scores),
        "independent_eligible_centres": point_count,
        "raw_margins_replayed": margin_count,
        "native_windows_replayed": window_count,
        "research_outcomes_replayed": outcome_count,
        "current_lag_windows_replayed": lag_windows,
        "current_lag_coefficients_replayed": lag_coefficients,
        "private_review_journal_sha256": sha(output / "reviewed-cases.jsonl"),
        "evaluated_result_sha256": sha(PANEL / f"native-mouth-{role}-result-v128.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": 0,
        "new_head_or_calibration_fits": 0,
        "defaults_changed": False,
        "scope": "Separate array/projection/margin/sigmoid, native-cell time-union and saved current lag arithmetic. No independent encoder/face/PCM rerun, mask reconstruction, unavailability judgment or outside human reviewer.",
    }
    write(output / "result.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run", "bind"))
    p.add_argument("arguments", nargs="*")
    args = p.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze takes no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One audit role required")
        run(args.arguments[0])
    else:
        verify()
        require(
            len(args.arguments) == 2 and args.arguments[1] in {"0", "1"}, "Fixed audit output/role required"
        )
        output = Path(args.arguments[0])
        require(output.resolve().is_relative_to(TRIAL.resolve()), "Exclusive private audit required")
        output.mkdir(exist_ok=False)
        review(original.ROLES[int(args.arguments[1])], output)
