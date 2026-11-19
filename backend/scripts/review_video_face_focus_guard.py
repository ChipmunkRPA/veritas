# SPDX-License-Identifier: Apache-2.0
"""Recount the held face-transition and surrounding sharpness comparison offline."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.off-face-focus-guard-holdout-scalars/1"
CADENCES = {"30": 1000, "30000/1001": 1001}
CARRIERS = ("direct", "tandem")
SIMPLE = {"one_sided_prefix", "one_sided_suffix"}
COMPOUND = "swap_plus_simultaneous_focus"
BROAD = {"soften_across_scene", "sharpen_across_scene"}
ROLES = SIMPLE | BROAD | {COMPOUND, "genuine_original", "background_only_defocus", "central_defocus",
                          "gain_ramp", "same_recording_jump", "cross_recording_cut"}
FOCUS_RULE = {
    "shift_hex": (.18).hex(), "agreement_hex": (.55).hex(), "min_tiles": 12,
    "minimum_quadrants": 3, "minimum_tiles_each": 2,
    "minimum_coarse_rms_hex": (.7).hex(), "minimum_coarse_correlation_hex": (.91).hex(),
    "frame_offsets": [-7, -6, -5, 5, 6, 7],
    "tile_readout": "Every admitted off-face tile contributes only quadrant and exact signed log "
                    "fine/coarse contrast shift. Correlation and RMS gates were applied before export; "
                    "excluded tiles, pixel positions and image/face features are omitted.",
}
TRIAL = {"paired_source_groups": 8, "authored_roles_per_group": 11, "frame_count": 120,
         "staged_focus_controls": True, "real_optical_focus_validation": False, "authorship_confidence_weight": 0}
INTAKE = {"authored_source_picture_pins": 10560, "decoded_rgb_clock_pins": 21120, "encoded_file_pins": 176,
          "matched_untouched_picture_pins": 6152, "publisher_swap_picture_pins": 1056,
          "redelivered_clock_rgb_replay_pins": 21120, "unchanged_picture_pins": 6760}
ARM_FIELDS = {"capture_rate", "clock_is_reconformed", "delivery_pts_step", "delivery_time_base", "gates",
              "intake_counts", "rows", "witness_counts"}
COMMITMENT_FIELDS = {"source_trial_plan_sha256", "render_code_seal_sha256", "score_code_seal_sha256", "recount_code_seal_sha256",
                     "cross_cadence_intake_sha256", "development_point_selection_sha256"}
ARM_COMMITMENT_FIELDS = {"media_intake_sha256", "independent_intake_sha256", "completed_score_sha256",
                         "independent_scalar_custody_sha256"}
ROW_FIELDS = {"group", "role", "carrier", "expected_frame", "authored_at", "raw_boundary", "raw_pts", "guarded_boundary",
              "face_coverage", "face_coverage_reason", "focus_status", "focus_direction", "eligible_tiles"}


def require(condition, problem):
    if not condition:
        raise ValueError(problem)


def frame(value):
    return type(value) is int and 0 <= value < 120


def commitments_valid(values, fields):
    return isinstance(values, dict) and set(values) == fields and all(
        isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) for value in values.values()
    )


def sharpness(row):
    tiles = row["eligible_tiles"]
    require(isinstance(tiles, list) and len(tiles) <= 60, "Invalid number of surrounding-picture tiles")
    quadrants = Counter()
    support = {-1: Counter(), 1: Counter()}
    for pair in tiles:
        require(isinstance(pair, list) and len(pair) == 2 and type(pair[0]) is int and 0 <= pair[0] < 4
                and isinstance(pair[1], str) and len(pair[1]) <= 32, "Invalid sharpness tile")
        quadrant, printed = pair
        shift = float.fromhex(printed)
        require(abs(shift) <= 100 and shift.hex() == printed, "Noncanonical tile reading")
        quadrants[quadrant] += 1
        if abs(shift) >= .18:
            support[-1 if shift < 0 else 1][quadrant] += 1
    require(all(count <= 15 for count in quadrants.values()), "A quadrant includes more tiles than its fixed grid")

    boundary = row["raw_boundary"]
    if boundary is None or not 7 <= boundary < 113:
        require(not tiles, "Missing source flanks cannot provide tile measurements")
        expected = None if boundary is None else "insufficient_source_flanks"
    else:
        expected = "assessed" if len(tiles) >= 12 else "insufficient_off_face_tiles"
    require(expected == row["focus_status"], "Recorded sharpness coverage differs from the eligible tiles")
    if expected == "assessed":
        for sign, direction in ((-1, "softening"), (1, "sharpening")):
            readings = support[sign]
            if sum(readings.values()) / len(tiles) >= .55 and sum(n >= 2 for n in readings.values()) >= 3:
                return direction
    return None


def combine(counts, carrier, roles):
    result = Counter()
    for role in roles:
        result.update(counts[carrier, role])
    return dict(result)


def replay_cadence(profile, tick, labels):
    rows = profile["rows"]
    require(set(profile) == ARM_FIELDS and profile["intake_counts"] == INTAKE
            and profile["capture_rate"] == "30" and type(profile["delivery_pts_step"]) is int
            and profile["delivery_pts_step"] == tick and profile["delivery_time_base"] == "1/30000"
            and profile["clock_is_reconformed"] is (tick == 1001) and isinstance(rows, list) and len(rows) == 176,
            "Invalid roster or delivery clock")
    seen = set()
    anchors = flanks = 0
    counts, groups = defaultdict(Counter), defaultdict(set)
    for row in rows:
        require(set(row) == ROW_FIELDS, "Unexpected scalar fields")
        group, role, carrier = (row[key] for key in ("group", "role", "carrier"))
        key = (group, role, carrier)
        require(type(group) is int and group in range(1, 9) and role in ROLES and carrier in CARRIERS
                and key not in seen, "Invalid or repeated group, role or carrier")
        seen.add(key)
        truth, authored = (row[key] for key in ("expected_frame", "authored_at"))
        edited = role in SIMPLE or role == COMPOUND
        planned = (37 + 2 * (group - 1) if role == "one_sided_prefix" else 62 + 4 * (group - 1))
        require((frame(truth) if edited else truth is None)
                and (authored is None if role == "genuine_original" else frame(authored) and authored == planned)
                and (not edited or truth == authored), "Truth differs from the authored source clock")
        require(labels.setdefault((group, role), (truth, authored)) == (truth, authored),
                "Comparison arms differ in their labeled timing")
        boundary, pts, guarded = (row[key] for key in ("raw_boundary", "raw_pts", "guarded_boundary"))
        status, reason = (row[key] for key in ("face_coverage", "face_coverage_reason"))
        require(status in ("localized", "no_match", "unavailable") and (boundary is not None) == (status == "localized")
                and (reason in ("shot_change", "insufficient_native_boundary") if status == "unavailable" else reason is None),
                "Recorded face coverage disagrees with the available result")
        require((pts is None if boundary is None else frame(boundary) and type(pts) is int and pts == tick * boundary),
                "Face-transition frame differs from its delivery timestamp")
        direction = sharpness(row)
        require(row["focus_direction"] == direction
                and (guarded is None if boundary is None or direction is not None else frame(guarded) and guarded == boundary),
                "Guarded boundary disagrees with the recorded sharpness rule")
        if boundary is not None:
            anchors += 3
            flanks += 6 if 7 <= boundary < 113 else 0
        entry = counts[carrier, role]
        entry.update(count=1)
        entry[f"coverage_{status}"] += 1
        if edited:
            useful = boundary is not None and abs(boundary - truth) <= 1
            entry.update(raw_useful=useful, guarded_useful=useful and guarded is not None,
                         raw_exact=boundary == truth, guarded_exact=guarded == truth,
                         raw_wrong=boundary is not None and not useful, guarded_wrong=guarded is not None and not useful)
            if useful:
                groups[carrier, "standalone" if role in SIMPLE else "compound"].add(group)
        else:
            entry.update(raw_ordinary=boundary is not None, guarded_ordinary=guarded is not None)
            if role in BROAD and boundary is not None:
                groups[carrier, "broad"].add(group)

    require(profile["witness_counts"] == {"native_aligned_face_sources": anchors, "sharpness_flanks": flanks},
            "Witness counts disagree with the available source clocks")

    report, gates = {}, {}
    for carrier in CARRIERS:
        simple = combine(counts, carrier, SIMPLE)
        compound = combine(counts, carrier, {COMPOUND})
        ordinary = combine(counts, carrier, ROLES - SIMPLE - {COMPOUND})
        broad = combine(counts, carrier, BROAD)
        gates[f"{carrier}_standalone_retention"] = (simple["raw_useful"] == simple["guarded_useful"]
                                                       and simple["raw_useful"] >= 6
                                                       and len(groups[carrier, "standalone"]) >= 4)
        gates[f"{carrier}_compound_retention"] = compound["raw_useful"] == compound["guarded_useful"]
        gates[f"{carrier}_ordinary_benefit"] = broad["raw_ordinary"] - broad["guarded_ordinary"] >= 2
        report[carrier] = {"simple": simple, "compound": compound, "all_ordinary": ordinary, "full_frame_focus": broad,
                           "by_role": {role: dict(counts[carrier, role]) for role in sorted(ROLES)}}

    original = sum(r["full_frame_focus"]["raw_ordinary"] for r in report.values())
    remaining = sum(r["full_frame_focus"]["guarded_ordinary"] for r in report.values())
    compound = sum(r["compound"]["raw_useful"] for r in report.values())
    gates["pooled_ordinary_min_support"] = original >= 4 and len(groups["direct", "broad"] | groups["tandem", "broad"]) >= 3
    gates["pooled_ordinary_min_effect"] = bool(original and (original - remaining) * 2 >= original)
    gates["pooled_compound_min_support"] = compound >= 3 and len(groups["direct", "compound"] | groups["tandem", "compound"]) >= 3
    require(gates == profile["gates"], "Calculated requirements disagree with the published outcome")
    return {"carrier": report, "gates": gates, "staged_guard_passes": all(gates.values())}


def replay(receipt):
    require(set(receipt) == {"cadences", "commitments", "focus_rule", "schema", "trial"}
            and receipt["schema"] == SCHEMA and receipt["trial"] == TRIAL and receipt["focus_rule"] == FOCUS_RULE
            and set(receipt["cadences"]) == set(CADENCES), "Unexpected scalar schema, cadence, or sharpness rule")
    arms = receipt["commitments"]["cadences"]
    require(commitments_valid({k: v for k, v in receipt["commitments"].items() if k != "cadences"}, COMMITMENT_FIELDS)
            and set(arms) == set(CADENCES) and all(commitments_valid(item, ARM_COMMITMENT_FIELDS) for item in arms.values()),
            "Invalid release commitments")
    labels = {}
    return {rate: replay_cadence(receipt["cadences"][rate], tick, labels) for rate, tick in CADENCES.items()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("receipt", type=Path)
    args = cli.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as reader:
        report = replay(json.load(reader))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
