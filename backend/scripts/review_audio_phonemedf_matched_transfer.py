"""Recompute the frozen PhonemeDF and retrospective ODSS decisions without model weights."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import review_audio_odss_matched_room as prior

ARMS = ("mono", "wpe", "frontal")
ORIGINS = ("natural", "melotts", "chatterbox", "background")
LEVELS = (0, -5)


def selective(row, points):
    scores = row["responses"]
    native = scores["mono"]
    if native is None:
        return False
    return native > points["mono"]["zero"] or (
        native > points["mono"]["one"] and all(
            scores.get(arm) is not None and scores[arm] > points[arm]["zero"] for arm in ARMS[1:]
        )
    )


def decision_counts(rows, points, origins, arm):
    def assessed(row):
        return row["responses"].get("mono" if arm == "selective" else arm) is not None

    def flagged(row):
        return selective(row, points) if arm == "selective" else row["responses"][arm] > points[arm]["zero"]

    counts = {}
    for origin in origins:
        all_rows = [row for row in rows if row["origin"] == origin]
        admitted = [row for row in all_rows if assessed(row)]
        calls = [row for row in admitted if flagged(row)]
        additions = [row for row in calls if not (row["responses"]["mono"] is not None
                                                  and row["responses"]["mono"] > points["mono"]["zero"])]
        counts[origin] = {
            "eligible": len(all_rows), "assessed": len(admitted), "flagged": len(calls),
            "flagged_prompts": len({row["case"] for row in calls}),
            "flagged_reader_groups": len({row["group"] for row in calls}),
            "additional_flags_vs_mono": len(additions),
        }
    return counts


def ranks(rows, arm):
    pairs = {(row["case"], row["level_db"], row["origin"]): row["responses"][arm] for row in rows}
    counts = {}
    for generator in ORIGINS[1:3]:
        complete = [(score, pairs[case, level, "natural"]) for (case, level, origin), score in pairs.items()
                    if origin == generator and score is not None and pairs[case, level, "natural"] is not None]
        counts[generator] = {"compared": len(complete), "synthetic_higher": sum(a > b for a, b in complete),
                             "median_logit_gap": statistics.median(a - b for a, b in complete) if complete else None}
    return counts


def validate(rows, lock):
    planned = {(f"source-reader-{index:02d}-prompt-{prompt}", origin, level)
               for index, prompt in itertools.product(range(4), range(3))
               for origin in ORIGINS for level in (LEVELS if origin == "background" else (None, *LEVELS))}
    inputs = set()
    keys = {"id", "case", "group", "role", "origin", "level_db", "measured_snr_db", "noise_position", "window_hashes",
            "input_sha256", "acoustic_gain_db", "fft_max_lsb", "responses"}
    for row in rows:
        prior.require(set(row) == keys, "Unexpected receipt fields")
        case, origin, level = row["case"], row["origin"], row["level_db"]
        prior.require((case, origin, level) in planned and row["role"] == "assessment"
                      and row["group"] == case.rsplit("-prompt-", 1)[0]
                      and row["id"] == f"{case}:{origin}:{'room' if level is None else level}"
                      and row["noise_position"] == ("right" if case.endswith("-1") else "left"),
                      "Unexpected held identity or geometry")
        planned.remove((case, origin, level))
        sha = row["input_sha256"]
        prior.require(prior.hashed(sha) and sha not in inputs, "Missing or duplicate waveform commitment")
        inputs.add(sha)
        controls = origin == "background" or level is None
        measured, acoustic = row["measured_snr_db"], row["acoustic_gain_db"]
        if controls:
            prior.require(measured is None and acoustic is None, "Control row gained target information")
        else:
            prior.require(type(measured) in (int, float) and abs(measured - level) <= .02
                          and set(acoustic) == set(ARMS[1:])
                          and all(gain is None or math.isfinite(gain) for gain in acoustic.values()), "Unbound acoustic gain")
        expected = ("mono",) if level is None else ARMS
        for field in ("responses", "window_hashes"):
            prior.require(set(row[field]) == set(expected), "Incomplete view set")
        for arm, score in row["responses"].items():
            pin = row["window_hashes"][arm]
            prior.require((pin is None or prior.hashed(pin)) and (score is None or pin is not None and math.isfinite(score)),
                          "Unbound response")
        fft = row["fft_max_lsb"]
        prior.require(set(fft) == (set() if level is None else set(ARMS[1:]))
                      and all(value in (0, 1) for value in fft.values()), "FFT replay differs")
    prior.require(not planned and lock["delivery_roster_sha256"] ==
                  prior.digest([{k: v for k, v in row.items() if k != "responses"} for row in rows]),
                  "Frozen delivery roster differs")
    return Counter(arm for row in rows for arm, response in row["responses"].items() if response is not None)


def acoustic_readout(rows):
    result = {}
    for arm in ARMS[1:]:
        levels = {}
        for level in LEVELS:
            gain = [row["acoustic_gain_db"][arm] for row in rows
                    if row["level_db"] == level and row["origin"] != "background" and row["acoustic_gain_db"][arm] is not None]
            levels[str(level)] = {"available": len(gain), "improved": sum(value > 0 for value in gain),
                                  "median_sisdr_gain_db": statistics.median(gain) if gain else None}
        result[arm] = levels
    return result


def review(receipt, odss_bytes):
    lock, assessment = receipt["freeze"], receipt["assessment"]
    prior.require(lock["schema"] == "chipmunk.audio-phonemedf-matched-transfer/1"
                  and lock["window_samples"] == list(prior.SCOPE) and lock["sample_rate"] == 16000
                  and lock["levels_db"] == list(LEVELS) and lock["intake"]["common_counts"] == [32, 81, 96, 102]
                  and all(len(pins) == 12 and all(prior.hashed(pin) for pin in pins)
                          for pins in lock["intake"]["chunks_sha256"].values()), "Fresh scope or release range differs")
    prior.require(hashlib.sha256(odss_bytes).hexdigest() == lock["odss_receipt_sha256"], "Prior panel commitment differs")
    old = json.loads(odss_bytes)
    prior.review(old)
    points = prior.points_from(old["development"]["rows"])
    prior.require(points == lock["points"], "Pre-assessment operating points differ")
    rows = assessment["rows"]
    prior.require(validate(rows, lock) == assessment["forwards"], "Model-forward denominator differs")
    noisy = [row for row in rows if row["level_db"] is not None]
    retrospective = [row for row in old["assessment"]["rows"] if row["level_db"] is not None]
    return {
        "deliveries": len(rows), "forwards": assessment["forwards"], "points": points,
        "noisy": {arm: decision_counts(noisy, points, ORIGINS, arm) for arm in (*ARMS, "selective")},
        "by_level": {str(level): {arm: decision_counts([row for row in noisy if row["level_db"] == level], points, ORIGINS, arm)
                                  for arm in (*ARMS, "selective")} for level in LEVELS},
        "matched_rank": {arm: ranks(noisy, arm) for arm in ARMS},
        "room_only_mono": decision_counts([row for row in rows if row["level_db"] is None], points, ORIGINS, "mono"),
        "acoustic_gain": acoustic_readout(noisy),
        "odss_retrospective": {arm: decision_counts(retrospective, points, ("natural", *prior.GENERATORS, "background"), arm)
                               for arm in ("mono", "selective")},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("odss", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes()), args.odss.read_bytes()), sort_keys=True, indent=2, allow_nan=False))
