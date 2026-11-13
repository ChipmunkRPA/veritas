"""Verify flank cleanup comparison and replay earlier mixture-failure diagnoses."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import review_audio_odss_matched_room as odss
import review_audio_phonemedf_matched_transfer as phoneme

SCHEMA = "chipmunk.audio-flank-background/1"
LEVELS = (0, -5)
ARMS = ("mono", "wpe", "frontal", "flank")
ORIGINS = ("natural", "vits", "background")


def validate(rows, role, prior_rows=()):
    is_dev = role == "development"
    groups = [f"reader-{index:02d}" if is_dev else f"vctk-reader-{index:02d}" for index in range(4)]
    planned = {(f"{group}-prompt-{prompt}", origin, level)
               for group, prompt in itertools.product(groups, range(2))
               for origin in (("natural", "background") if is_dev else ORIGINS)
               for level in (LEVELS if is_dev or origin == "background" else (None, *LEVELS))}
    prior = {row["id"]: row["input_sha256"] for row in prior_rows}
    input_hashes, counts = set(), Counter()
    for row in rows:
        origin, level = row["origin"], row["level_db"]
        case = row["case"]
        key = (case, origin, level)
        odss.require(key in planned and row["role"] == role and row["group"] == case.rsplit("-prompt-", 1)[0]
                     and row["position"] == ("left" if case.endswith("-0") else "right")
                     and row["id"] == f"{case}:{origin}:{'room' if level is None else level}", "Delivery role differs")
        planned.remove(key)
        sha = row["input_sha256"]
        odss.require(odss.hashed(sha) and sha not in input_hashes and (not is_dev or prior[row["id"]] == sha),
                     "Unexpected or aliased input waveform")
        input_hashes.add(sha)
        expected = ("flank",) if is_dev else ("mono",) if level is None else ARMS
        odss.require(set(row["responses"]) == set(row["window_hashes"]) == set(expected)
                     and set(row["fft_max_lsb"]) == (set(expected) - {"mono"})
                     and all(value in (0, 1) for value in row["fft_max_lsb"].values()), "Comparison views differ")
        for arm, response in row["responses"].items():
            pin = row["window_hashes"][arm]
            odss.require((pin is None or odss.hashed(pin)) and (response is None or pin and math.isfinite(response)),
                         "Unbound response")
            counts[arm] += response is not None
        if origin != "background" and level is not None:
            odss.require(abs(row["snr_db"] - level) <= .02, "Rendered level differs")
        else:
            odss.require(row["snr_db"] is None, "Control row acquired noise measurements")
        gain = row["gain_db"]
        expected_gains = {"flank"} if is_dev else None if origin == "background" or level is None else set(ARMS[1:])
        odss.require((gain is None if expected_gains is None else set(gain) == expected_gains)
                     and (gain is None or all(value is None or math.isfinite(value) for value in gain.values())),
                     "Acoustic readout differs")
    odss.require(not planned, "Missing planned delivery")
    return +counts


def decisions(rows, points, flank_point):
    names = ("mono", "selective", "selective_plus_flank")
    results = {}
    for name in names:
        results[name] = {}
        for origin in ORIGINS:
            eligible = [row for row in rows if row["origin"] == origin]
            admitted = [row for row in eligible if row["responses"]["mono"] is not None]
            base = [row for row in admitted if row["responses"]["mono"] > points["mono"]["zero"]]
            added = []
            if name != "mono":
                for row in admitted:
                    if row in base or not phoneme.selective(row, points):
                        continue
                    flank = row["responses"].get("flank")
                    if name == "selective" or flank is not None and flank > flank_point:
                        added.append(row)
            flagged = base + added
            results[name][origin] = {"total": len(eligible), "admitted": len(admitted), "flags": len(flagged),
                                     "flagged_groups": len({row["group"] for row in flagged}),
                                     "additional_vs_mono": len(added),
                                     "additional_groups": len({row["group"] for row in added})}
    return results


def earlier_diagnosis(rows, points):
    by_key = {(row["case"], row["origin"], row["level_db"]): row for row in rows}
    results = {}
    for origin_type in ("genuine", "synthetic"):
        selected = []
        for row in rows:
            origin, level, case = row["origin"], row["level_db"], row["case"]
            if origin == "background" or level is None or (origin == "natural") != (origin_type == "genuine"):
                continue
            native = row["responses"]["mono"]
            if native is None or native > points["mono"]["zero"] or not phoneme.selective(row, points):
                continue
            dry = by_key[case, origin, None]["responses"]["mono"]
            mask = by_key[case, "background", level]["responses"]
            selected.append((native, dry, any(mask[arm] is not None and mask[arm] > points[arm]["zero"]
                                              for arm in ARMS[:3])))
        shifts = [native - dry for native, dry, _ in selected if dry is not None]
        results[origin_type] = {"additions": len(selected), "room_only_above_point": sum(
                                dry is not None and dry > points["mono"]["zero"] for _, dry, _ in selected),
                                "matched_background_only_any_above_point": sum(hit for _, _, hit in selected),
                                "room_to_noisy_mono_median_logit_change": statistics.median(shifts) if shifts else None,
                                "room_to_noisy_mono_pairs": len(shifts)}
    return results


def acoustic(rows):
    results = {}
    for arm in ARMS[1:]:
        results[arm] = {}
        for level in LEVELS:
            values = [row["gain_db"][arm] for row in rows if row["level_db"] == level and row["origin"] != "background"
                      and row["gain_db"][arm] is not None]
            results[arm][str(level)] = {"assessed": len(values), "improved": sum(value > 0 for value in values),
                                        "median_sisdr_gain_db": statistics.median(values) if values else None}
    return results


def review(receipt, old_bytes, phoneme_bytes):
    lock = receipt["freeze"]
    odss.require(lock["schema"] == SCHEMA and lock["window_samples"] == list(odss.SCOPE)
                 and lock["levels_db"] == list(LEVELS) and set(lock["room_responses"]) == {"front", "left", "right"}
                 and hashlib.sha256(old_bytes).hexdigest() == lock["odss_receipt_sha256"]
                 and hashlib.sha256(phoneme_bytes).hexdigest() == lock["phonemedf_receipt_sha256"], "Frozen ancestry differs")
    old, fresh = json.loads(old_bytes), json.loads(phoneme_bytes)
    odss.review(old)
    phoneme.review(fresh, old_bytes)
    prior_points = odss.points_from(old["development"]["rows"])
    dev, held = (receipt[role]["rows"] for role in ("development", "assessment"))
    odss.require(lock["primary_points"] == prior_points and all((
        validate(dev, "development", old["development"]["rows"]) == receipt["development"]["forwards"],
        validate(held, "assessment") == receipt["assessment"]["forwards"],
        odss.digest([{k: v for k, v in row.items() if k != "responses"} for row in dev + held]) == lock["delivery_roster_sha256"],
    )), "Frozen points, delivery schedule or count differs")
    values = [row["responses"]["flank"] for row in dev if row["responses"]["flank"] is not None]
    flank_point = max(values)
    odss.require(receipt["operating_point"] == {"freeze_sha256": odss.digest(lock), "flank_point": flank_point,
                 "development_sha256": odss.digest(receipt["development"])}, "Frozen development-only point differs")
    noisy = [row for row in held if row["level_db"] is not None]
    return {
        "development_forwards": receipt["development"]["forwards"],
        "assessment_forwards": receipt["assessment"]["forwards"],
        "flank_point": flank_point,
        "noisy": decisions(noisy, prior_points, flank_point),
        "by_level": {str(level): decisions([row for row in noisy if row["level_db"] == level], prior_points, flank_point)
                     for level in LEVELS},
        "room_only": decisions([row for row in held if row["level_db"] is None], prior_points, flank_point)["mono"],
        "acoustic_gain": acoustic(noisy),
        "earlier_diagnoses": {"odss": earlier_diagnosis(old["assessment"]["rows"], prior_points),
                              "phonemedf": earlier_diagnosis(fresh["assessment"]["rows"], prior_points)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("odss", type=Path)
    parser.add_argument("phonemedf", type=Path)
    args = parser.parse_args()
    value = review(json.loads(args.receipt.read_bytes()), args.odss.read_bytes(), args.phonemedf.read_bytes())
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
