"""Recompute the Full voice-grid receipt from anonymous scores, without model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "chipmunk.audio-communication-serving-grid/1"
GRID = ((0, 64000), (32000, 96000))
ARMS = ("original", "communication", "both")
CONTROLS = {"untouched", "ordinary_same_host", "ordinary_other_voice"}
SOURCES = {
    "discovery": "audio-communication-transfer-2026-11-12",
    "confirmation": "audio-communication-replication-2026-11-12",
}
DOCS = ROOT / "docs/evaluation-panels"
RECEIPT = DOCS / "audio-communication-serving-grid-2026-11-12.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def rows_digest(rows):
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def geometry(window, truth):
    if window is None:
        return 0, 1, truth[1] - truth[0]
    overlap = max(0, min(window[1], truth[1]) - max(window[0], truth[0]))
    width = truth[1] - truth[0]
    return overlap, window[1] - window[0] + width - overlap, width


def overlaps(window, truth):
    intersection, union, width = geometry(window, truth)
    return {
        "any_overlap": intersection > 0,
        "half_covered": intersection * 2 >= width,
        "fully_covered": intersection == width,
        "useful_localization": intersection * 10 >= union * 3 and intersection * 2 >= width,
    }


def iou(window, truth):
    intersection, union, _ = geometry(window, truth)
    return intersection / union


def best(values, point):
    candidates = [(value, -i) for i, value in enumerate(values) if value is not None and value > point]
    return None if not candidates else GRID[-max(candidates)[1]]


def arms(row):
    original, communication = (row["views"][arm]["scores"] for arm in ARMS[:2])
    paired = [None if None in (a, b) else min(a, b) for a, b in zip(original, communication, strict=True)]
    return dict(zip(ARMS, (original, communication, paired), strict=True))


def summarize_arm(rows, arm, point):
    counts, false_groups = Counter(), set()
    for row in rows:
        values = arms(row)[arm]
        windows = [window for window, value in zip(GRID, values, strict=True) if value is not None]
        counts["assessed_files"] += bool(windows)
        counts["admitted_windows"] += len(windows)
        selected = best(values, point)
        counts["flagged_files"] += selected is not None
        if truth := row["truth"]:
            counts["ai_flags"] += selected is not None
            admitted = max(windows, key=lambda w: geometry(w, truth)[0], default=None)
            for metric, found in overlaps(admitted, truth).items():
                counts[f"admitted_{metric}"] += found
            for metric, found in overlaps(selected, truth).items():
                counts[f"flagged_{metric}"] += found
        elif selected is not None:
            counts["genuine_false_files"] += 1
            counts["background_only_false_files"] += row["arm"] == "background_only"
            false_groups.add(row["group"])
    return {key: counts[key] for key in (
        "assessed_files", "admitted_windows", "flagged_files", "ai_flags", "genuine_false_files",
        "background_only_false_files", "admitted_any_overlap", "admitted_half_covered", "admitted_fully_covered",
        "admitted_useful_localization", "flagged_any_overlap", "flagged_half_covered", "flagged_fully_covered",
        "flagged_useful_localization",
    )} | {"false_genuine_groups": len(false_groups)}


def validate_rows(rows):
    groups = defaultdict(list)
    check(len(rows) == 52, "Unexpected retained row count")
    for row in rows:
        check(set(row) == {"arm", "group", "family", "coverage", "truth", "views"} and type(row["group"]) is int,
              "Unexpected anonymous row schema")
        check(set(row["views"]) == set(ARMS[:2]), "Unexpected delivery views")
        for view in row["views"].values():
            values = view["scores"]
            check(len(values) == 2 and all(p is None or type(p) in {float, int} and math.isfinite(p) for p in values),
                  "Invalid model-window scores")
            assessed = any(p is not None for p in values)
            check(view == {"scores": values, "status": "ok" if assessed else "not_applicable",
                           "decision": "inconclusive" if assessed else None,
                           "scope_supported": False if assessed else None, "probability_ai": None},
                  "Adapter status and calibration scope disagree")
        groups[row["group"]].append(row)
        if truth := row["truth"]:
            check(row["arm"] == "synthetic" and type(truth) is list and len(truth) == 2
                  and all(type(i) is int for i in truth) and 0 <= truth[0] < truth[1] <= 96000
                  and truth[1] - truth[0] in {10240, 12800, 15360}, "Unexpected known synthetic span")
        else:
            check(row["truth"] is None and row["arm"] in CONTROLS | {"background_only"}, "Unexpected control")
    check(set(groups) == set(range(1, 17)), "Unexpected anonymous group count")
    for group in groups.values():
        categories = {row["arm"] for row in group}
        family = {row["family"] for row in group}
        coverage = {row["coverage"] for row in group}
        check((len(group) == 1 and categories == {"background_only"} and family == {None} and coverage == {"room"})
              or (len(group) == 4 and categories == CONTROLS | {"synthetic"} and len(family) == 1
                  and family <= {"clone", "expressive", "stock"} and len(coverage) == 1
                  and coverage <= {"early", "interior", "late"}), "Inconsistent paired control group")
    check(Counter(row["arm"] for row in rows) == Counter(
        dict.fromkeys(CONTROLS | {"synthetic"}, 12) | {"background_only": 4}), "Missing trial control or edit")


def summarize(rows, point):
    validate_rows(rows)
    native = {arm: {
        "assessment_status": dict(Counter(row["views"][arm]["status"] for row in rows)),
        "calibrated_whole_recording_confidence": sum(row["views"][arm]["probability_ai"] is not None for row in rows),
        "detector_scope_supported": sum(row["views"][arm]["scope_supported"] is True for row in rows),
    } for arm in ARMS[:2]}
    edited = [row for row in rows if row["truth"]]
    planned = [max(iou(window, row["truth"]) for window in GRID) for row in edited]
    max_iou_by_width = {str(width): max(value for row, value in zip(edited, planned, strict=True)
                                        if row["truth"][1] - row["truth"][0] == width)
                        for width in (10240, 12800, 15360)}
    subsets = {"all": rows, "interior": [row for row in edited if row["coverage"] == "interior"],
               "edge": [row for row in edited if row["coverage"] != "interior"]}
    fixed = {key: {arm: summarize_arm(group, arm, point) for arm in ARMS} for key, group in subsets.items()}
    selected = {arm: {index: best(arms(row)[arm], point) for index, row in enumerate(rows)} for arm in ARMS}
    original_half = {i for i, row in enumerate(rows) if row["truth"]
                     and overlaps(selected["original"][i], row["truth"])["half_covered"]}
    retained_half = {i for i in original_half if overlaps(selected["both"][i], rows[i]["truth"])["half_covered"]}
    false_groups = {arm: {row["group"] for i, row in enumerate(rows) if not row["truth"] and selected[arm][i] is not None}
                    for arm in ARMS}
    return {
        "planned_windows_per_arm": len(rows) * len(GRID),
        "known_edits": len(edited),
        "genuine_recordings": len(rows) - len(edited),
        "genuine_control_groups": len({row["group"] for row in rows}),
        "known_edge_edits": len(subsets["edge"]),
        "known_interior_edits": len(subsets["interior"]),
        "edits_geometrically_eligible_for_useful_localization": sum(iou >= 0.30 for iou in planned),
        "best_attainable_grid_iou_by_edit_length_samples": max_iou_by_width,
        "native_production_scope": native,
        "frozen_point": fixed,
        "comparison": {
            "original_false_groups_removed_by_pairing": len(false_groups["original"] - false_groups["both"]),
            "new_false_groups_from_pairing": len(false_groups["both"] - false_groups["original"]),
            "original_half_covered_flags_retained": len(retained_half),
            "original_half_covered_flags_retained_at_exact_location": sum(
                selected["original"][i] == selected["both"][i] for i in retained_half),
        },
    }


def validate_receipt(receipt):
    check(receipt["schema"] == SCHEMA and receipt["grid_samples"] == [list(window) for window in GRID]
          and receipt["sample_rate_hz"] == 16000 and receipt["samples_per_recording"] == 96000,
          "Wrong serving grid or evidence clock")
    check(receipt["customer_confidence_changed"] is False and receipt["serving_admitted"] is False,
          "This experiment cannot be counted as an admitted serving change")
    check(receipt["commitments"]["runner"] == sha(Path(__file__).with_name("audio_communication_serving_grid.py"))
          and receipt["commitments"]["reviewer"] == sha(Path(__file__)), "Runner or reviewer changed")
    point = receipt["fixed_short_window_logit"]
    for name, filename in SOURCES.items():
        path = DOCS / f"{filename}.json"
        prior = read(path)
        check(point == prior["fixed_point"] and receipt["commitments"]["prior_receipts"][name] == sha(path),
              "Prior operating point or public receipt changed")
        panel = receipt["panels"][name]
        check(panel["short_window_fixed"] == prior["tables"]["all"]["fixed"]
              and panel["serving_grid"] == summarize(panel["anonymous_rows"], point),
              f"Serving-grid comparison failed to replay on {name}")
        check(receipt["commitments"][f"{name}_anonymous_rows"] == rows_digest(panel["anonymous_rows"]),
              f"Published window ledger changed on {name}")


def compile_native(run, output):
    from scripts import audio_communication_serving_grid as native

    lock_path = run / "freeze.json"
    lock = read(lock_path)
    check(lock == native.freeze(), "Frozen serving configuration or implementation changed")
    result = {
        "schema": SCHEMA, "grid_samples": lock["grid"], "sample_rate_hz": lock["sampling_rate"],
        "samples_per_recording": lock["samples_per_file"], "fixed_short_window_logit": lock["fixed_short_window_logit"],
        "customer_confidence_changed": False, "serving_admitted": False,
        "commitments": {"freeze": sha(lock_path), "prior_receipts": lock["prior_receipts"],
                        "runner": lock["runner_sha256"], "reviewer": sha(Path(__file__))},
        "panels": {},
    }
    for name in SOURCES:
        ledger_path = run / f"{name}-scores.json"
        ledger = read(ledger_path)
        check(ledger["freeze_sha256"] == sha(lock_path), "Score ledger belongs to another freeze")
        result["commitments"][f"{name}_scores"] = sha(ledger_path)
        groups, public = {}, []
        for (row, original, communication, path), record in zip(native.cases(name), ledger["records"], strict=True):
            check(row["id"] == record["id"], "Score row belongs to another input")
            native.verify_call(communication, path)
            views = {}
            for arm, pcm in zip(ARMS[:2], (original, communication), strict=True):
                entry = record["views"][arm]
                check(entry["pcm_sha256"] == native.digest(pcm), "Serving input samples changed")
                values = native.np.frombuffer(pcm, "<i2").astype(native.np.float32) / 32768
                check(entry["float_pcm_sha256"] == native.digest(values.astype("<f4").tobytes()),
                      "Native floating-point input binding changed")
                windows = entry["windows"]
                for window, (start, end) in zip(windows, GRID, strict=True):
                    check((window["start"], window["end"]) == (start, end)
                          and window["pcm_sha256"] == native.digest(pcm[2*start:2*end]), "Window-to-sample binding changed")
                views[arm] = {key: entry[key] for key in ("status", "decision", "scope_supported", "probability_ai")} | {
                    "scores": [window["score"] for window in windows],
                }
            group = groups.setdefault(row["group"], len(groups) + 1)
            public.append({key: row[key] for key in ("arm", "family", "coverage", "truth")} | {"group": group, "views": views})
        prior = read(native.public_receipt(name))
        result["commitments"][f"{name}_anonymous_rows"] = rows_digest(public)
        result["panels"][name] = {"anonymous_rows": public, "serving_grid": summarize(public, lock["fixed_short_window_logit"]),
                                   "short_window_fixed": prior["tables"]["all"]["fixed"]}
    validate_receipt(result)
    if output.exists():
        check(read(output) == result, "Existing public receipt differs from native replay")
    else:
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print("Verified 104 original/call pairs, 208 complete PCM bindings and 416 planned window bindings. Re-decoded 104 call clocks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--native", type=Path, help="Optional private run directory for sample and clock replay")
    args = parser.parse_args()
    if args.native:
        compile_native(args.native, args.receipt)
    else:
        validate_receipt(read(args.receipt))
        print("Replayed 104 anonymous recording pairs on the frozen Full grid and verified both earlier receipts.")


if __name__ == "__main__":
    main()
