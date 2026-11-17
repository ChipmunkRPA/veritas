# SPDX-License-Identifier: Apache-2.0
"""Compare time-local speech responses on exact and estimated voices."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_frame_logit_localization as portable
from scripts.audio_focused_context_transfer import matches, overlap

SCHEMA = "chipmunk.separated-speech-segments/1"
ASSET_SHA256 = "fde76030df782e140658aa5a2726d18ebffa9fa359e11ce2b237f6ec46114d2f"
STEP = 2560
WIDTHS = (1, 2, 3, 4, 5, 6)
FLANK_WIDTH = 2
FAMILIES = ("mixture", "angular", "independent")
SCORES = ("level", "contrast")
CONTROL = {"untouched", "neither"}
POLICIES = {"strict": 0, "one_pair": 1}
QUIET_SCALE = 10 ** (-5 / 20)


def model_driver(model, source):
    spec = importlib.util.spec_from_file_location("speech_segment_trial", source / "sal_smoke.py")
    driver = importlib.util.module_from_spec(spec)
    original = sys.argv
    sys.argv = [str(spec.origin), str(model)]
    sys.path.insert(0, str(source))
    try:
        spec.loader.exec_module(driver)
    finally:
        sys.path.pop(0)
        sys.argv = original
    return driver


def peak_readings(bins, speech):
    best = dict.fromkeys(SCORES)
    for width in WIDTHS:
        for start in range(len(bins) + 1 - width):
            end = start + width
            if 2 * sum(speech[start:end]) < (end - start) * STEP:
                continue
            average = sum(bins[start:end]) / width
            readings = {"level": average}
            if start >= FLANK_WIDTH and end + FLANK_WIDTH <= len(bins):
                left = sum(bins[start-FLANK_WIDTH:start]) / FLANK_WIDTH
                right = sum(bins[end:end+FLANK_WIDTH]) / FLANK_WIDTH
                readings["contrast"] = average - max(left, right)
            for key, value in readings.items():
                if best[key] is None or value > best[key]["score"]:
                    best[key] = {"range": [start * STEP, end * STEP], "score": value}
    return best


def fixed_readings(bins, reference):
    support = [overlap((i * STEP, (i + 1) * STEP), reference) for i in range(len(bins))]
    total = sum(support)
    return sum(value * count for value, count in zip(bins, support, strict=True)) / total if total else None


def calibrated_points(rows):
    cal = defaultdict(dict)
    for row in rows:
        if row["condition"] not in CONTROL:
            continue
        for method, lanes in row["measurements"].items():
            for lane in lanes:
                for field in SCORES:
                    reading = lane["peaks"][field]
                    if reading is not None:
                        values = cal[method, field]
                        pair = row["pair"]
                        values[pair] = max(values.get(pair, -math.inf), reading["score"])
    cuts = {}
    for method in FAMILIES:
        cuts[method] = {}
        for field in SCORES:
            maxima = sorted(cal[method, field].values(), reverse=True)
            if len(maxima) <= max(POLICIES.values()):
                raise ValueError("Insufficient ordinary development pairs for both requested points")
            cuts[method][field] = {policy: maxima[index] for policy, index in POLICIES.items()}
    return cuts


def acoustic_quality(original, estimate):
    from scripts.audio_source_isolation_diagnostic import acoustic_quality as read

    value = read(original, estimate)
    return round(float(value), 5) if value is not None and math.isfinite(value) else None


def measure(args):
    import numpy as np

    from app.evidence.audio_short_review_binding import speech_counts
    from scripts import audio_two_channel_separator as blind
    from scripts.audio_joint_generator_transfer import prior
    from scripts.audio_spatial_mask_voice_transfer import separate as angular
    from scripts.audio_two_voice_isolation import frozen_plan, parent, study

    reference = parent()
    frozen = frozen_plan(args.inputs, reference)
    plan = study.SOURCE.path_sha(args.inputs / "freeze.json")
    # Duplicated 16 kHz PCM bytes and both native source clocks are verified first.
    cases = list(study.cases(frozen, args.phase))
    driver = model_driver(args.model, args.driver)
    activity = prior.runtime(Path("models/weights/audio-upgrade/silero_vad.jit"))
    input_pins = {"plan_sha256": plan, "variant": args.variant,
                  "reader_sha256": study.SOURCE.path_sha(Path(__file__)),
                  "separator_sha256": study.SOURCE.path_sha(Path(blind.__file__)),
                  "model_state_sha256": study.SOURCE.path_sha(args.model),
                  "sal_driver_sha256": study.SOURCE.path_sha(args.driver / "sal_smoke.py"),
                  "published_checkpoint": ASSET_SHA256}
    if args.phase == "held":
        calibration = portable.read(args.development)
        if calibration["input_pins"] != input_pins or calibration["phase"] != "development":
            raise ValueError("Frozen development configuration differs")
        expected_points = calibrated_points(calibration["rows"])
        if expected_points != calibration["points"]:
            raise ValueError("Frozen development points differ from the original controls")
    else:
        expected_points = None
    rows = []
    for index, scene, condition, native in cases:
        original = study.load_stereo(args.inputs, native["pcm_sha256"])
        first, last = (study.load_stereo(args.inputs, item) for item in native["stems"])
        if not np.array_equal(original, first.astype(np.int32) + last.astype(np.int32)):
            raise ValueError("The two retained stems do not sum to the admitted source")
        quiet = np.rint(first.astype(np.float64) * QUIET_SCALE)
        signal = quiet + last
        normalized = signal / 32768.
        exact = [signal.mean(axis=1) / 32768. for signal in (quiet, last)]
        spatial, _ = angular(signal)
        spatial = [wave.astype(np.float64) / 32768. for wave in spatial if wave is not None]
        frame = int(args.variant.split("/")[0])
        separated = blind.separate(normalized, frame_size=frame, whiten=args.variant.endswith("/white"))
        clean = {"mixture": [normalized.mean(axis=1)], "angular": spatial,
                 "independent": [] if separated is None else list(separated.T)}
        measured = {}
        clocks = [voice["span"] for voice in scene["voices"]]
        for method, lanes in clean.items():
            readings = []
            for samples in lanes:
                pcm = np.asarray(samples, dtype=np.float32)
                bins = driver.measure(pcm).tolist()
                spans = [(i * STEP, (i + 1) * STEP) for i in range(len(bins))]
                speech = speech_counts(activity.spans(pcm), spans)
                peaks = peak_readings(bins, speech)
                for best in (row for row in peaks.values() if row is not None):
                    window = slice(*best["range"])
                    best["source_quality_db"] = [acoustic_quality(stem[window], pcm[window]) for stem in exact]
                readings.append({"peaks": peaks, "whole_sdr_db": [acoustic_quality(stem, pcm) for stem in exact],
                                 "fixed_clock": [fixed_readings(bins, clock) for clock in clocks],
                                 "wave_sha256": portable.sha(pcm.astype("<f4").tobytes())})
            measured[method] = readings
        states = study.CONDITIONS[condition]
        truths = [{"side": side, "span": voice["span"], "family": voice["family"]}
                  for side, voice in enumerate(scene["voices"]) if states[side] == "synthetic"]
        rows.append({"scene": index, "pair": scene["pair"], "cohort": scene["cohort"], "condition": condition,
                     "parent_pcm_sha256": native["pcm_sha256"], "stems": native["stems"],
                     "input_sha256": portable.sha(signal.astype("<i2").tobytes()),
                     "truth": truths, "measurements": measured,
                     "mix_db": round(20 * math.log10(np.linalg.norm(exact[0]) / np.linalg.norm(exact[1])), 5)})
        if len(rows) % 10 == 0 or len(rows) == len(cases):
            print(args.phase, len(rows), "/", len(cases), flush=True)
    points = expected_points if expected_points is not None else calibrated_points(rows)
    receipt = {"schema": SCHEMA, "phase": args.phase, "input_pins": input_pins, "points": points,
               "rows": rows, "summary": summarize(rows, points)}
    portable.write(args.output, receipt)
    print(json.dumps(receipt["summary"], sort_keys=True), flush=True)


def attributed(candidate, side):
    direct, other = (candidate["source_quality_db"][i] for i in (side, 1-side))
    return direct is not None and other is not None and direct >= other + 1.


def summarize(rows, points):
    results = defaultdict(Counter)
    for row in rows:
        truth = [entry["span"] for entry in row["truth"]]
        for method, lanes in row["measurements"].items():
            for field in SCORES:
                available = [lane["peaks"][field] for lane in lanes if lane["peaks"][field] is not None]
                for policy, point in points[method][field].items():
                    selected = [item for item in available if item["score"] > point]
                    areas = sorted(set(tuple(item["range"]) for item in selected))
                    exact = matches(areas, truth)
                    key = "/".join((row["cohort"], method, field, policy))
                    count = results[key]
                    count["edits"] += len(truth)
                    count["correct"] += len(exact)
                    count["edited_wrong"] += bool(truth and selected and not exact)
                    count["controls"] += row["condition"] in CONTROL
                    count["controls_flagged"] += bool(selected) and row["condition"] in CONTROL
                    count["unassessed"] += not available
                    count["right_voice"] += sum(any(attributed(item, event["side"]) and matches([item["range"]], [event["span"]])
                                                         for item in selected) for event in row["truth"]) if method != "mixture" else 0
    return {key: dict(record) for key, record in sorted(results.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="command", required=True)
    scoring = phases.add_parser("measure")
    scoring.add_argument("phase", choices=("development", "held"))
    for argument in ("inputs", "model", "driver", "output"):
        scoring.add_argument(argument, type=Path)
    scoring.add_argument("--variant", required=True, choices=("1024/plain", "2048/plain", "1024/white", "2048/white"))
    scoring.add_argument("--development", type=Path)
    verify = phases.add_parser("verify")
    verify.add_argument("receipt", type=Path)
    verify.add_argument("--development", type=Path)
    args = parser.parse_args()
    if args.command == "measure":
        if (args.phase == "held") != (args.development is not None):
            parser.error("held measurement requires the completed development receipt")
        measure(args)
    else:
        data = portable.read(args.receipt)
        if data["summary"] != summarize(data["rows"], data["points"]):
            raise ValueError("Interval outcomes differ from the recorded data")
        if data["phase"] == "development" or args.development:
            first = portable.read(args.development) if args.development else data
            if (data["input_pins"] != first["input_pins"] or data["points"] != calibrated_points(first["rows"])):
                raise ValueError("Calibrated points differ from their development sources")
        print(json.dumps(data["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
