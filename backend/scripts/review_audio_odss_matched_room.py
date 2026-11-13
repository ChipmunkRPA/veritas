"""Replay matched noisy-speech decisions from the anonymous ODSS receipt."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.audio-odss-matched-room/1"
ARMS = ("mono", "wpe", "frontal")
GENERATORS = ("vits", "fastpitch-hifigan")
LEVELS = (0, -5)
SCOPE = (24000, 72000)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    serialized = json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n"
    return hashlib.sha256(serialized.encode()).hexdigest()


def hashed(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_rows(rows, role, groups):
    plans = {(f"{group}-prompt-{index}", origin, level) for group, index in itertools.product(groups, range(2))
             for origin in ("natural", *GENERATORS, "background")
             for level in (LEVELS if origin == "background" else (None, *LEVELS))}
    actual = set()
    keys = {"id", "case", "group", "role", "origin", "level_db", "measured_snr_db", "input_sha256", "window_hashes",
            "acoustic_gain_db", "fft_max_lsb", "noise_position", "utterance_extent", "responses"}
    for row in rows:
        require(set(row) == keys, "Unrecognized receipt fields")
        case, origin, level = row["case"], row["origin"], row["level_db"]
        route = (case, origin, level)
        expected = ("mono",) if level is None else ARMS
        require(route in plans and route not in actual and row["role"] == role and row["group"] == case.rsplit("-prompt-", 1)[0]
                and row["id"] == f"{case}:{origin}:{'room' if level is None else level}"
                and row["noise_position"] == ("left" if case.endswith("-0") else "right") and hashed(row["input_sha256"]),
                "Role, paired source, SNR, position or input identity differs")
        actual.add(route)
        extent = row["utterance_extent"]
        if origin == "background":
            require(extent is None and row["acoustic_gain_db"] is None and row["measured_snr_db"] is None, "Background gained target data")
        else:
            require(isinstance(extent, list) and len(extent) == 2 and SCOPE[0] <= extent[0] < extent[1] <= SCOPE[1]
                    and 26400 <= extent[1] - extent[0] <= 48000 and abs(sum(extent) - sum(SCOPE)) <= 1, "Fixed utterance placement differs")
            if level is not None:
                require(type(row["measured_snr_db"]) in (int, float) and abs(row["measured_snr_db"] - level) <= .02
                        and set(row["acoustic_gain_db"]) == set(ARMS[1:]) and all(gain is None or math.isfinite(gain)
                            for gain in row["acoustic_gain_db"].values()), "Bad source SNR")
            else:
                require(row["measured_snr_db"] is None and row["acoustic_gain_db"] is None, "Room-only source gained background data")
        for field in ("responses", "window_hashes"):
            require(set(row[field]) == set(expected), "Receipt dropped a comparison arm")
        for arm, value in row["responses"].items():
            pin = row["window_hashes"][arm]
            require((pin is None or hashed(pin)) and (value is None or pin is not None and math.isfinite(value)), "Unbound model response")
        require(set(row["fft_max_lsb"]) == (set() if level is None else set(ARMS[1:]))
                and all(delta in (0, 1) for delta in row["fft_max_lsb"].values()), "Incomplete FFT replay")
    require(actual == plans, "Missing frozen delivery")
    return Counter(arm for row in rows for arm, response in row["responses"].items() if response is not None)


def points_from(rows):
    values = {}
    for arm in ARMS:
        accepted = [(row["group"], row["responses"][arm]) for row in rows if row["origin"] == "natural"
                    and row["level_db"] is not None and row["responses"][arm] is not None]
        speakers = {group for group, _ in accepted}
        require(len(speakers) == 4, "A development speaker lacks an admitted human reading")
        ordered = sorted((max(score for group, score in accepted if group == speaker) for speaker in speakers), reverse=True)
        values[arm] = {"zero": ordered[0], "one": ordered[1]}
    return values


def readout(rows, points):
    assayed = [row for row in rows if any(row["responses"].get(arm) is not None for arm in points)]
    flagged = [row for row in assayed if any(row["responses"].get(arm) is not None and row["responses"][arm] > point
                                             for arm, point in points.items())]
    return {origin: {"total": sum(row["origin"] == origin for row in rows),
                     "assessed": sum(row["origin"] == origin for row in assayed),
                     "flagged": sum(row["origin"] == origin for row in flagged),
                     "matched_prompts_or_background_excerpts_flagged": len({row["case"] for row in flagged if row["origin"] == origin}),
                     "groups_flagged": len({row["group"] for row in flagged if row["origin"] == origin})}
            for origin in ("natural", *GENERATORS, "background")}


def comparisons(rows, arm):
    values = {(row["case"], row["level_db"], row["origin"]): row["responses"].get(arm) for row in rows}
    comparisons = {}
    for generator in GENERATORS:
        complete = [(value, values[case, level, "natural"]) for (case, level, origin), value in values.items()
                    if origin == generator and value is not None and values[case, level, "natural"] is not None]
        comparisons[generator] = {"compared": len(complete), "synthetic_higher": sum(a > b for a, b in complete),
                                  "median_logit_separation": statistics.median(a - b for a, b in complete) if complete else None}
    return comparisons


def decisions(rows, points):
    category = {(False, False): "neither_flag", (True, False): "mono_only",
                (False, True): "candidate_only", (True, True): "both_flag"}
    results = {}
    for arm in ARMS[1:]:
        mono_point, candidate_point = (points[name]["zero"] for name in ("mono", arm))
        counts = {}
        for origin in ("natural", *GENERATORS, "background"):
            pairs = [(row["responses"]["mono"], row["responses"][arm]) for row in rows if row["origin"] == origin]
            counts[origin] = {"both_flag": 0, "mono_only": 0, "candidate_only": 0, "neither_flag": 0,
                              "mono_unassessed": 0, "candidate_unassessed": 0}
            for native, candidate in pairs:
                original_flag = native is not None and native > mono_point
                candidate_flag = candidate is not None and candidate > candidate_point
                counts[origin][category[original_flag, candidate_flag]] += 1
                counts[origin]["mono_unassessed"] += native is None
                counts[origin]["candidate_unassessed"] += candidate is None
        results[arm] = counts
    return results


def review(data):
    lock, points, development, assessment = (data[key] for key in ("freeze", "operating_points", "development", "assessment"))
    require(lock["schema"] == SCHEMA and lock["windows"] == list(SCOPE) and lock["clock_hz"] == 16000
            and lock["stereo_samples"] == 96000 and lock["levels_db"] == list(LEVELS), "Frozen audio scope differs")
    combined = development["rows"] + assessment["rows"]
    require(digest([{k: v for k, v in row.items() if k != "responses"} for row in combined]) == lock["delivery_roster_sha256"]
            and len({row["input_sha256"] for row in combined}) == len(combined), "Delivery bindings differ or contain aliases")
    for role, roster, indices in (("development", development, range(4)), ("assessment", assessment, range(4, 8))):
        groups = {f"reader-{index:02d}" for index in indices}
        require(validate_rows(roster["rows"], role, groups) == roster["forwards"], "Native model call denominator differs")
    require(points == {"freeze_sha256": digest(lock), "development_sha256": digest(development), "points": points_from(development["rows"])},
            "Fitted point or development role commitment differs")
    held = assessment["rows"]
    noisy = [row for row in held if row["level_db"] is not None]
    fitted = points["points"]
    return {"delivery_commitments": len(combined), "development_forwards": development["forwards"], "assessment_forwards": assessment["forwards"],
            "points": fitted,
            "room_only_mono": readout([row for row in held if row["level_db"] is None], {"mono": fitted["mono"]["zero"]}),
            "noisy": {arm: {"pooled": {name: readout(noisy, {arm: point}) for name, point in budgets.items()},
                            "by_level_primary": {str(level): readout([row for row in noisy if row["level_db"] == level], {arm: budgets["zero"]})
                                                 for level in LEVELS},
                            "matched_rank": comparisons(noisy, arm)} for arm, budgets in fitted.items()},
            "noisy_union_diagnostic": {arm: readout(noisy, {name: fitted[name]["zero"] for name in ("mono", arm)}) for arm in ARMS[1:]},
            "paired_decisions": decisions(noisy, fitted)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), sort_keys=True, indent=2, allow_nan=False))
