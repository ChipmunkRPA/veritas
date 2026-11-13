# SPDX-License-Identifier: Apache-2.0
"""Recompute the frozen speech-window comparison without models or dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

RATE, SIZE, STRIDE = 16_000, 48_000, 12_000
ARMS = ("sparse", "dense", "staggered", "reduced", "combined")
SCENES = {"selection": ["gain_control", "restaurant", "coastal"], "held": ["gain_control", "vienna", "market"]}
DECISIONS = {"ai", "original", "inconclusive", "unassessed"}
CALIBRATION = "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf"
ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_FILE = ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json"
RUNNER_FILE = Path(__file__).with_name("audio_noise_window_transfer.py")
SHA = re.compile(r"[0-9a-f]{64}")
GIT_REVISION = re.compile(r"[0-9a-f]{40}")
DEPENDENCIES = {
    "backend/app/detectors/audio/calibration.py",
    "backend/app/detectors/audio/calibrations/local-speech-v3.json",
    "backend/app/detectors/audio/content.py",
    "backend/app/detectors/audio/upgraded.py",
    "backend/app/detectors/video/adapters/audio.py",
    "backend/app/engine/profiles.py",
    "backend/scripts/audio_gtcrn_traffic.py",
    "backend/scripts/audio_noise_views_transfer.py",
    "backend/scripts/audio_partial_publisher_transfer.py",
    "backend/scripts/audio_radar_transfer.py",
    "backend/scripts/audio_reduced_background_localization.py",
    "backend/scripts/audio_vocal_event_mixtures.py",
    "backend/scripts/review_audio_reduced_background_localization.py",
    "backend/scripts/run_audio_specialists_v15.py",
}


def require(valid, message):
    if not valid:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def pin(value):
    return hashlib.sha256(encode(value)).hexdigest()


def is_pin(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def integer_pair(value, limit):
    require(isinstance(value, list) and len(value) == 2 and all(type(n) is int for n in value)
            and 0 <= value[0] < value[1] <= limit, "Invalid exact sample interval")
    return tuple(value)


def score(value):
    if value is None:
        return None
    number = float.fromhex(value)
    require(math.isfinite(number) and value == number.hex(), "Invalid stored binary floating-point score")
    return number


def schedules(length):
    last = length - SIZE
    incumbent = [0, last] if length < 9 * RATE else [0, last // 2, last]
    staggered = list(range(0, last + 1, STRIDE))
    dense = staggered[::2]
    return incumbent, dense, staggered


def calibration_bundle(path):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == CALIBRATION, "The source calibration bundle differs")
    return json.loads(data)


def prior_direction(raw, bundle):
    if raw is None:
        return "unassessed"
    slope = bundle["parameters"]["slope"]
    intercept = bundle["parameters"]["intercept"]
    value = min(max(raw, 1e-6), 1-1e-6)
    probability = 1/(1+math.exp(-(slope*(math.log(value)-math.log1p(-value))+intercept)))
    eligible = []
    for label, deviation in ((0, probability), (1, 1-probability)):
        guard = bundle["guards"][str(label)]
        if len(guard) < 9:
            return "inconclusive"
        rank = (1+sum(entry is None or entry >= deviation for entry in guard))/(len(guard)+1)
        if rank > bundle["alpha"]:
            eligible.append(label)
    if probability >= .8 and eligible == [1]:
        return "ai"
    if probability <= .2+1e-12 and eligible == [0]:
        return "original"
    return "inconclusive"


def check_rows(held, roster, point, bundle):
    for record, case in zip(held, roster, strict=True):
        length = case["samples"]
        sparse, dense, shifted = schedules(length)
        expected = [(a, a + SIZE) for a in sorted(set(sparse + shifted))]
        windows = record["windows"]
        require(record["id"] == case["id"] and is_pin(record["reduced_sha256"])
                and [integer_pair(item["span"], length) for item in windows] == expected, "Window layout differs")
        directions, rescued = {}, set()
        for item in windows:
            direction = item["direction"]
            require(direction in DECISIONS, "Invalid original direction")
            start = item["span"][0]
            directions[start] = direction
            raw, reduced = score(item["original"]), score(item["processed"])
            require((direction == "unassessed") == (raw is None) and (raw is None or 0 <= raw <= 1),
                    "Original score does not support its recorded assessment status")
            prior = prior_direction(raw, bundle)
            require(direction == prior or (prior == "original" and direction == "inconclusive"),
                    "Original direction differs from its pinned calibration or allowed stability veto")
            if reduced is not None:
                require(direction in {"unassessed", "inconclusive"} and start in shifted,
                        "Reduced response does not qualify for rescue")
                require(prior != "original", "Original-leaning raw response vetoes reduced rescue")
                if reduced > point:
                    rescued.add(start)
        original = {a for a, direction in directions.items() if direction == "ai"}
        comparator = [a for a in sparse if a in original]
        if len(comparator) < 2 or any(directions[a] == "original" for a in sparse):
            comparator = []
        baseline = [(b, a + SIZE) for a, b in zip(dense, dense[1:]) if a in original and b in original]
        staggered, reduced = [], []
        for a, b, c in zip(shifted, shifted[1:], shifted[2:]):
            span = c, a + SIZE
            if {a, b, c} <= original:
                staggered.append(span)
            if {a, b, c} <= original | rescued:
                reduced.append(span)
        yield dict(zip(ARMS, ([(a, a + SIZE) for a in comparator], baseline, staggered, reduced, baseline + reduced),
                       strict=True)), len(set(directions.values()) - {"unassessed"}) > 0, any(
                           item["processed"] is not None for item in windows)


def validate_roster(roster, freeze, run):
    groups, sources, isolated = defaultdict(set), {}, defaultdict(set)
    hashes = set()
    for index, row in enumerate(roster):
        role, scene, kind, source, length = (row[key] for key in ("role", "scene", "kind", "source", "samples"))
        require(row["id"] == index and type(length) is int and 6 * RATE <= length <= 14 * RATE
                and role in SCENES and scene in SCENES[role] and is_pin(row["pcm_sha256"])
                and row["pcm_sha256"] not in hashes, "Invalid, duplicate or out-of-scope waveform")
        hashes.add(row["pcm_sha256"])
        if run is not None:
            data = (run / "pcm" / f"{index:03}.pcm").read_bytes()
            require(len(data) == 2 * length and hashlib.sha256(data).hexdigest() == row["pcm_sha256"],
                    "Local PCM differs from the frozen waveform")
        if scene == "gain_control":
            require(row["offset"] is None and kind != "isolated", "Invalid gain-only control")
        else:
            require(row["offset"] in [n * RATE for n in freeze["backgrounds"][scene][2]], "Unknown scene offset")
        if kind == "isolated":
            require(source is None and length == 8 * RATE and row["offset"] not in isolated[role, scene],
                    "Invalid isolated control")
            isolated[role, scene].add(row["offset"])
            continue
        require(kind in {"genuine", "partial"} and isinstance(source["group"], str)
                and all(is_pin(source[key]) for key in ("source_sha256", "source_pcm_sha256")), "Invalid source identity")
        spans = [integer_pair(item, length) for item in source["edits"]]
        require(all(a[1] < b[0] for a, b in zip(spans, spans[1:])), "Repeated or overlapping edit labels")
        if kind == "partial":
            require(role == "held" and source["recipe"] in {"cp", "oa"} and source["family"] in
                    {"cosyvoice", "elevenlab", "gptsovits", "ljjets", "xttsv2", "yourtts"}
                    and .10 * length <= sum(b-a for a,b in spans) <= .60 * length
                    and any(9600 <= b-a <= 38400 for a,b in spans), "Invalid publisher edit scope")
        else:
            require(not spans and source["family"] is None and source["recipe"] is None, "Invalid genuine labels")
        group = source["group"]
        require(sources.setdefault(group, (role, kind, length, source)) == (role, kind, length, source)
                and scene not in groups[group], "Speaker crossed the frozen comparison roles")
        groups[group].add(scene)
    counts = Counter((item[0], item[1]) for item in sources.values())
    families = Counter(row[3]["family"] for row in sources.values() if row[1] == "partial")
    require(counts == {("selection", "genuine"): 12, ("held", "genuine"): 12, ("held", "partial"): 18}
            and len(families) == 6 and set(families.values()) == {3}
            and all(groups[group] == set(SCENES[item[0]]) for group, item in sources.items())
            and len(isolated) == 4 and all(len(offsets) == 2 for offsets in isolated.values()), "Incorrect frozen denominators")


def useful(ranges, truth):
    for a, b in ranges:
        for c, d in truth:
            intersect = max(0, min(b, d) - max(a, c))
            if 2 * intersect >= d - c and 10 * intersect >= 3 * (b - a + d - c - intersect):
                return True
    return False


def marked_genuine(ranges, truth):
    boundaries = sorted({value for pair in [*ranges, *truth] for value in pair})
    return sum(b-a for a,b in zip(boundaries, boundaries[1:])
               if any(start <= a < stop for start, stop in ranges)
               and not any(start <= a < stop for start, stop in truth))


def verify_local_run(document, run, runner):
    for name in ("freeze", "roster", "selection", "held", "point"):
        actual = hashlib.sha256((run / f"{name}.json").read_bytes()).hexdigest()
        require(actual == pin(document[name]), f"Retained {name} record differs from the receipt")
    require(hashlib.sha256(runner.read_bytes()).hexdigest() == document["freeze"]["runner_sha256"],
            "Local runner differs from its frozen source")


def verify_dependencies(path, freeze):
    receipt = json.loads(path.read_bytes())
    require(set(receipt) == {"base_revision", "dependency_sha256", "freeze_sha256"}
            and isinstance(receipt["base_revision"], str) and GIT_REVISION.fullmatch(receipt["base_revision"])
            and receipt["freeze_sha256"] == pin(freeze), "Dependency receipt does not match the study freeze")
    pins = receipt["dependency_sha256"]
    require(isinstance(pins, dict) and set(pins) == DEPENDENCIES and all(is_pin(value) for value in pins.values()),
            "Dependency receipt has an unexpected file list or digest")
    for name, expected in pins.items():
        local = ROOT / name
        require(local.resolve().is_relative_to(ROOT) and hashlib.sha256(local.read_bytes()).hexdigest() == expected,
                f"Local dependency differs from the receipt: {name}")


def audit(document, run=None, *, calibration=CALIBRATION_FILE, runner=RUNNER_FILE, dependencies=None):
    freeze, roster, selection, held, point = (document[key] for key in ("freeze", "roster", "selection", "held", "point"))
    method = freeze["method"]
    require(freeze["schema"] == "chipmunk.audio-noise-window-transfer/1" and freeze["model_responses_at_freeze"] == 0
            and method["calibration_sha256"] == CALIBRATION and method["window_samples"] == SIZE
            and method["staggered_step"] == STRIDE and method["dense_step"] == 2 * STRIDE and method["role_scenes"] == SCENES
            and method["background_db"] == -5 and is_pin(freeze["runner_sha256"])
            and all(is_pin(value) for field in ("pins", "part_a", "voice_assets", "previous") for value in freeze[field].values()),
            "The frozen method or source commitments differ")
    require(pin(roster) == freeze["roster_sha256"] and selection["freeze_sha256"] == held["freeze_sha256"] == pin(freeze)
            and selection["point_sha256"] is None and held["point_sha256"] == pin(point)
            and point["selection_sha256"] == pin(selection), "Freeze, selection or held stage changed")
    if run is not None:
        verify_local_run(document, run, runner)
    if dependencies is not None:
        verify_dependencies(dependencies, freeze)
    bundle = calibration_bundle(calibration)
    validate_roster(roster, freeze, run)
    values = [score(item["processed"]) for row in selection["rows"] for item in row["windows"] if item["processed"] is not None]
    cutoff = score(point["point_hex"])
    require(values and max(values) == cutoff and len(values) == point["observations"], "Selection point differs")
    list(check_rows(selection["rows"], [row for row in roster if row["role"] == "selection"], cutoff, bundle))
    held_rows = [row for row in roster if row["role"] == "held"]
    cells = defaultdict(Counter)
    gained, gained_families, ordinary, clean = set(), set(), set(), set()
    for row, (arms, assess, processed) in zip(held_rows, check_rows(held["rows"], held_rows, cutoff, bundle), strict=True):
        scene, kind, source = row["scene"], row["kind"], row["source"]
        truth = source["edits"] if source else []
        key = source["group"] if source else f"{scene}/{row['offset']}"
        cell = cells[f"{scene}/{kind}"]
        cell.update(files=1, assessed_original=assess, assessed_reduced=processed,
                    genuine_samples=row["samples"] - sum(b-a for a,b in truth))
        located = {}
        for arm, ranges in arms.items():
            located[arm] = useful(ranges, truth)
            cell.update({f"{arm}_flags": bool(ranges), f"{arm}_useful": located[arm],
                         f"{arm}_genuine_samples_flagged": marked_genuine(ranges, truth)})
        if kind == "partial" and scene != "gain_control" and located["combined"] and not located["dense"]:
            gained.add(key)
            gained_families.add(source["family"])
        if kind != "partial" and arms["combined"] and not arms["dense"]:
            (clean if scene == "gain_control" or kind == "isolated" else ordinary).add(key)
    noisy = [cell for label, cell in cells.items() if not label.startswith("gain_control/")]
    excess = sum(cell["combined_genuine_samples_flagged"]-cell["dense_genuine_samples_flagged"] for cell in noisy) / sum(
        cell["genuine_samples"] for cell in noisy)
    target = method["target"]
    gate = {"added_noisy_groups": len(gained) >= target["minimum_added_partial_groups"],
            "added_families": len(gained_families) >= target["minimum_gained_families"],
            "new_noisy_genuine_groups": len(ordinary) <= target["maximum_added_genuine_groups"],
            "new_clean_or_isolated_groups": len(clean) <= target["maximum_new_clean_or_isolated_groups"],
            "genuine_time": excess <= target["maximum_added_genuine_time_fraction"]}
    output = {"cells": dict(cells), "new_noisy_partial_groups": len(gained), "new_families": sorted(gained_families),
              "new_noisy_genuine_groups": len(ordinary), "new_clean_or_isolated_groups": len(clean),
              "added_noisy_genuine_time_fraction": excess, "fixed_gate": gate, "passes": all(gate.values()), "serving": False}
    require(output == document["summary"], "Saved error and useful-location totals differ")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--run", type=Path, help="Also verify retained stage receipts, runner bytes and native input PCM")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_FILE, help="Path to the pinned calibration JSON")
    parser.add_argument("--runner", type=Path, default=RUNNER_FILE, help="Path to the original runner when verifying a local run")
    parser.add_argument("--dependencies", type=Path, help="Also verify the separate dependency receipt and checkout bytes")
    args = parser.parse_args()
    summary = audit(json.loads(args.receipt.read_bytes()), args.run, calibration=args.calibration,
                    runner=args.runner, dependencies=args.dependencies)
    print(encode(summary).decode(), end="")


if __name__ == "__main__":
    main()
