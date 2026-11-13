# SPDX-License-Identifier: Apache-2.0
"""Recompute held speech locations and genuine flagged time without any model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.audio-reduced-background-localization/1"
RATE, WINDOW, STEP = 16_000, 48_000, 24_000
RESCUE = {"inconclusive", "unassessed"}
METHOD = {
    "primary_calibration_sha256": "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf",
    "window_samples": WINDOW, "step_samples": STEP, "include_end_aligned_window": False,
    "decision_point": 7.422796281843017, "speech_fraction_minimum": 0.5,
    "pair": "One original AI-like response and one adjacent processed rescue, or two processed rescues. Report only the intersection.",
    "veto": "A primary original direction on a window cannot be rescued. Leave current long and short reviews intact.",
    "minimum_noisy_added_partial_groups": 4, "minimum_generator_families": 3, "minimum_scenes_with_gains": 2,
    "maximum_noisy_added_genuine_groups": 1, "maximum_added_clean_genuine_groups": 0,
    "maximum_added_isolated": 0, "maximum_increase_noisy_genuine_flagged_time": 0.02,
    "release_role": "Transfer study with prior speaker exposure reported. Serving requires a separate source-held confirmation.",
}
SCENES = {"gain_control", "construction", "leicester"}
ARMS = ("current", "dense_original", "dense_reduced", "combined")
SHA = re.compile("[0-9a-f]{64}")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(path.read_bytes())


def schedules(length):
    require(type(length) is int and 6 * RATE <= length <= 14 * RATE, "Native clock outside the measured scope")
    last = length - WINDOW
    starts = (0, last) if length < 9 * RATE else (0, last // 2, last)
    return ([[a, a + WINDOW] for a in starts], [[a, a + WINDOW] for a in range(0, last + 1, STEP)])


def intersections(windows, positive):
    return [[right[0], left[1]] for left, right in zip(windows, windows[1:])
            if tuple(left) in positive and tuple(right) in positive]


def intervals_union(ranges):
    merged = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def overlap(first, second):
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def useful(ranges, edits):
    return any(2 * (shared := overlap(span, edit)) >= edit[1] - edit[0]
               and 10 * shared >= 3 * (span[1] - span[0] + edit[1] - edit[0] - shared)
               for span in ranges for edit in edits)


def predict(source, checks, rescues, point):
    sparse, dense = schedules(source["samples"])
    scheduled = sorted(set(map(tuple, sparse + dense)))
    require([tuple(row["span"]) for row in checks["checks"]] == scheduled, "Native window schedule changed")
    directions = {tuple(row["span"]): row["direction"] for row in checks["checks"]}
    require(all(value in RESCUE | {"ai", "original"} for value in directions.values()), "Invalid native response")
    eligible = [tuple(span) for span in dense if directions[tuple(span)] in RESCUE]
    require([tuple(row["span"]) for row in rescues] == eligible, "Missing or unexpected processed window")
    positive = {span for span, direction in directions.items() if direction == "ai"}
    for row in rescues:
        require(type(row["admitted"]) is bool and bool(SHA.fullmatch(row["processed_pcm_sha256"])), "Invalid processed reading")
        score = float.fromhex(row["logit"]) if row["admitted"] else None
        require((score is None and row["logit"] is None) or (score is not None and math.isfinite(score)),
                "Unusable processed response")
        if score is not None and score > point:
            positive.add(tuple(row["span"]))
    native = {span for span, direction in directions.items() if direction == "ai"}
    current = [span for span in sparse if tuple(span) in native]
    if len(current) < 2 or any(directions[tuple(span)] == "original" for span in sparse):
        current = []
    if short := checks["short_selected"]:
        a, b = short["start_sample"], short["end_sample"]
        require(source["samples"] == 6 * RATE and a in range(0, 5 * RATE, RATE) and b == a + 2 * RATE,
                "Current short-review location differs")
        current.append([a, b])
    original = intersections(dense, native)
    reduced = intersections(dense, positive)
    return {"current": current, "dense_original": original, "dense_reduced": reduced, "combined": current + reduced}, directions


def truth(source):
    if source["source"] is None:
        require(source["scene"] != "gain_control" and source["samples"] == 8 * RATE, "Invalid isolated background")
        return "isolated", []
    record = source["source"]
    clock, length = record["clock"], source["samples"]
    require(record["samples"] == length and clock[0][0] == 0 and clock[-1][1] == length
            and all(0 <= a < b <= length and label in {"spoof", "bonafide"} for a, b, label in clock)
            and all(left[1] == right[0] for left, right in zip(clock, clock[1:])), "Invalid publisher source clock")
    edits = [[a, b] for a, b, label in clock if label == "spoof"]
    label = "genuine" if not edits else "full" if edits == [[0, length]] else "partial"
    require(record["edits"] == edits and record["label"] == label, "Publisher edit label differs")
    return label, edits


def analyze(lock, roster, native, processed, run=None):
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["enhanced_point"] == METHOD["decision_point"]
            and lock["known_model_responses_at_freeze"] == 0 and len(roster) == len(native) == 116, "Frozen protocol differs")
    rescues = defaultdict(list)
    for row in processed:
        rescues[row["id"]].append(row)
    require(set(rescues) <= set(range(len(roster))), "Processed window has no matching original")
    by_scene = defaultdict(Counter)
    groups = defaultdict(dict)
    gains, clean_alarms, noisy_alarms, families, gained_scenes = set(), set(), set(), set(), set()
    new_group_gains = set()
    isolated_alarms, original_checks = 0, 0
    for i, (source, checks) in enumerate(zip(roster, native, strict=True)):
        require(source["id"] == checks["id"] == i and source["scene"] in SCENES and SHA.fullmatch(source["pcm_sha256"]),
                "Missing or moved source delivery")
        if run is not None:
            pcm = (run / "pcm" / f"{i:03}.pcm").read_bytes()
            require(len(pcm) == 2 * source["samples"] and digest(pcm) == source["pcm_sha256"], "Prepared input differs")
        for check in checks["checks"]:
            a, b = check["span"]
            require(SHA.fullmatch(check["pcm_sha256"]) and (run is None or digest(pcm[2*a:2*b]) == check["pcm_sha256"]),
                    "Original window bytes differ from its location")
        for row in rescues[i]:
            a = row["span"][0]
            if run is not None:
                clip = (run / "processed" / f"{i:03}-{a}.pcm").read_bytes()
                require(len(clip) == 2 * WINDOW and digest(clip) == row["processed_pcm_sha256"], "Saved processed PCM differs")
        ranges, directions = predict(source, checks, rescues[i], lock["enhanced_point"])
        scene = source["scene"]
        label, edits = truth(source)
        cell = by_scene[f"{scene}/{label}"]
        cell["deliveries"] += 1
        cell["assessed_original"] += int(any(direction != "unassessed" for direction in directions.values()))
        cell["assessed_processed"] += int(any(row["admitted"] for row in rescues[i]))
        fake_samples = sum(b - a for a, b in edits)
        cell["genuine_samples"] += source["samples"] - fake_samples
        cell["synthetic_samples"] += fake_samples
        located = {}
        for arm in ARMS:
            located[arm] = bool(ranges[arm]) if label == "full" else useful(ranges[arm], edits)
            union = intervals_union(ranges[arm])
            flag_samples = sum(b - a for a, b in union)
            correct_samples = sum(overlap(span, edit) for span in union for edit in edits)
            cell[f"{arm}_flagged"] += int(bool(union))
            cell[f"{arm}_useful"] += int(located[arm])
            cell[f"{arm}_synthetic_samples_flagged"] += correct_samples
            cell[f"{arm}_genuine_samples_flagged"] += flag_samples - correct_samples
        original_checks += len(checks["checks"])
        if label == "isolated":
            isolated_alarms += int(bool(ranges["combined"]) and not ranges["current"])
            continue
        row = source["source"]
        require(type(row["prior_group_exposure"]) is bool, "Missing speaker exposure record")
        group = row["group"]
        require(scene not in groups[group], "Repeated source within a scene")
        groups[group][scene] = row
        if label == "partial" and scene != "gain_control" and located["combined"] and not located["current"]:
            gains.add(group)
            families.add(row["family"])
            gained_scenes.add(scene)
            if not row["prior_group_exposure"]:
                new_group_gains.add(group)
        if label == "genuine" and ranges["combined"] and not ranges["current"]:
            (clean_alarms if scene == "gain_control" else noisy_alarms).add(group)
    labels, family_counts, new_speakers = Counter(), Counter(), Counter()
    for views in groups.values():
        require(set(views) == SCENES and views["gain_control"] == views["construction"] == views["leicester"],
                "Paired source identities differ")
        item = views["gain_control"]
        labels[item["label"]] += 1
        new_speakers[item["label"]] += int(not item["prior_group_exposure"])
        if item["family"]:
            family_counts[item["label"], item["family"]] += 1
    require(labels == {"genuine": 12, "partial": 18, "full": 6} and len(family_counts) == 12
            and all(n == (3 if label == "partial" else 1) for (label, _), n in family_counts.items()),
            "Distinct source roles or generator coverage changed")
    noisy = [cell for key, cell in by_scene.items() if not key.startswith("gain_control/")]
    duration = sum(cell["genuine_samples"] for cell in noisy)
    extra = sum(cell["combined_genuine_samples_flagged"] - cell["current_genuine_samples_flagged"] for cell in noisy)
    delta = extra / duration
    passes = (len(gains) >= METHOD["minimum_noisy_added_partial_groups"]
              and len(families) >= METHOD["minimum_generator_families"] and len(gained_scenes) >= METHOD["minimum_scenes_with_gains"]
              and len(noisy_alarms) <= METHOD["maximum_noisy_added_genuine_groups"]
              and len(clean_alarms) <= METHOD["maximum_added_clean_genuine_groups"]
              and isolated_alarms <= METHOD["maximum_added_isolated"]
              and delta <= METHOD["maximum_increase_noisy_genuine_flagged_time"])
    return {"by_scene_and_class": {key: dict(value) for key, value in sorted(by_scene.items())},
            "source_groups": dict(labels), "native_windows": original_checks, "processed_windows": len(processed),
            "groups_outside_both_prior_panels": dict(new_speakers),
            "added_noisy_partial_groups_outside_both_prior_panels": len(new_group_gains),
            "added_partial_groups_under_noise": len(gains), "added_generator_families": sorted(families),
            "scenes_with_gains": sorted(gained_scenes), "added_genuine_groups_under_noise": len(noisy_alarms),
            "added_clean_genuine_groups": len(clean_alarms), "added_isolated_alerts": isolated_alarms,
            "additional_genuine_samples_flagged_under_noise": extra, "genuine_samples_under_noise": duration,
            "additional_genuine_time_fraction_under_noise": delta, "transfer_gate_passed": passes, "serving_qualified": False}


def verify(receipt, run=None):
    require(digest(encode(receipt["freeze"])) == receipt["freeze_sha256"]
            and digest(encode(receipt["roster"])) == receipt["freeze"]["roster_sha256"],
            "Public frozen commitments differ")
    result = analyze(receipt["freeze"], receipt["roster"], receipt["primary"], receipt["processed"], run)
    require(result == receipt["result"], "Recorded decisions differ from the saved readings")
    return result


def export(run):
    freeze, roster, native, enhanced, processed = (read(run / name) for name in (
        "freeze.json", "roster.json", "primary.json", "enhancement.json", "processed.json"))
    require(native["freeze_sha256"] == digest((run / "freeze.json").read_bytes())
            and enhanced["primary_sha256"] == digest((run / "primary.json").read_bytes())
            and processed["enhancement_sha256"] == digest((run / "enhancement.json").read_bytes())
            and [{key: row[key] for key in ("id", "span", "processed_pcm_sha256")} for row in processed["rows"]] == enhanced["rows"],
            "Exact inference chain differs")
    receipt = {"freeze": freeze, "freeze_sha256": native["freeze_sha256"], "roster": roster, "primary": native["rows"],
               "processed": processed["rows"], "result": analyze(freeze, roster, native["rows"], processed["rows"], run)}
    verify(receipt)
    return receipt


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("export", "review"))
    cli.add_argument("source", type=Path)
    cli.add_argument("--out", type=Path)
    cli.add_argument("--native-run", type=Path)
    args = cli.parse_args()
    if args.action == "export":
        receipt = export(args.source)
        if args.out:
            with args.out.open("xb") as file:
                file.write(encode(receipt))
    else:
        receipt = read(args.source)
    print(json.dumps(verify(receipt, args.native_run), sort_keys=True))


if __name__ == "__main__":
    main()
