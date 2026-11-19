# SPDX-License-Identifier: Apache-2.0
"""Recount anonymous SAL native-noise decision tables without models or recordings."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

NAME = "audio-sal-native-strong-noise-2026-11-18"
SCHEMA = "chipmunk.audio-sal-native-strong-noise-summary/1"
PIN = "05d68a7e77dbaa3e8abca423f5224c6d0fa979b1eba47121945e98e60659a983"
CONDITIONS = ("clean", "washing", "river", "restaurant")
KINDS = {"synthetic", "ordinary", "background"}
CELLS = {"neither", "baseline_only", "candidate_only", "both"}
ROLES = {"untouched", "same_reader_splice", "different_reader_splice", "roundtrip_piper", "roundtrip_kitten"}
ARMS = {"level_strict", "level_wider", "contrast_strict", "contrast_wider"}
FAMILIES = {"family_a", "family_b"}
POINTS = {"level": {"strict": 10.786407470703125, "wider": 9.697216033935547},
          "contrast": {"strict": 6.430497348308563, "wider": 6.051407992839813}}
METHOD = {"sample_rate": 16000, "recording_samples": 192000, "prediction_bin_samples": 2560,
          "range_bins": [1, 2, 3, 4, 5, 6], "flank_bins": 2, "minimum_speech_fraction": [1, 2],
          "maximum_ranges": 5, "minimum_edit_coverage": [1, 2], "minimum_temporal_iou": [3, 10],
          "cpu_threads": 2, "newly_fitted_parameters": 0, "input_views": 1}
COHORT = {"deliveries": 496, "unique_deliveries": 496, "hosts": 16, "readers": 8,
          "donor_connected_components": 4, "generator_families": 2, "background_sessions": 3,
          "foreground_variants_per_host_condition": 7, "ordinary_variants": 5,
          "fresh_to_segment_model": 496, "previously_examined_source_deliveries": 496}
VERIFICATION = {"original_hosts_replayed": 16, "changed_clean_foregrounds": 96,
                "additive_noisy_deliveries_replayed": 336, "outside_edit_samples_compared": 79134720,
                "native_prediction_bins": 37200, "range_admission_checks": 215760,
                "actual_selection_parity_comparisons": 1984, "authored_selection_parity_comparisons": 68,
                "authored_boundary_checks": 8, "inference_elapsed_seconds": 465.706}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), label + " has an unexpected field set")


def count(value, maximum=496):
    require(type(value) is int and 0 <= value <= maximum, "Expected a bounded anonymous integer count")
    return value


def exact(value, expected, label):
    # Strict JSON encoding also rejects bool-for-int and changed number types.
    require(json.dumps(value, sort_keys=True, allow_nan=False)
            == json.dumps(expected, sort_keys=True, allow_nan=False), label + " differs from its frozen value")


def table(value, denominator):
    keys(value, CELLS, "Decision table")
    for entry in value.values():
        count(entry, denominator)
    require(sum(value.values()) == denominator, "A decision table has the wrong denominator")
    return {"baseline": value["baseline_only"] + value["both"],
            "candidate": value["candidate_only"] + value["both"],
            "combined": denominator - value["neither"], "added": value["candidate_only"]}


def ranks(value, comparisons):
    keys(value, {"wins", "losses", "ties", "comparisons"}, "Rank counts")
    for entry in value.values():
        count(entry, comparisons)
    require(value["comparisons"] == comparisons, "Rank comparison denominator differs")
    require(value["wins"] + value["losses"] + value["ties"] == comparisons,
            "Rank comparison outcomes do not add up")


def validate(data):
    keys(data, {"schema", "study", "custody_commitment_sha256", "decision", "frozen_points", "method",
                "cohort", "verification", "conditions"}, "Receipt")
    exact(data["schema"], SCHEMA, "Schema")
    exact(data["study"], NAME, "Study")
    exact(data["decision"], "diagnostic_only_no_product_acceptance_gate", "Decision scope")
    require(isinstance(data["custody_commitment_sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", data["custody_commitment_sha256"]), "Invalid whole-study commitment")
    exact(data["frozen_points"], POINTS, "Operating points")
    exact(data["method"], METHOD, "Method")
    exact(data["cohort"], COHORT, "Cohort")
    exact(data["verification"], VERIFICATION, "Verification scope")
    require(COHORT["deliveries"] == 16 * 4 * 7 + 16 * 3, "Source role arithmetic differs")
    keys(data["conditions"], CONDITIONS, "Conditions")
    report = {}
    for condition in CONDITIONS:
        row = data["conditions"][condition]
        keys(row, {"denominators", "unassessed", "baseline", "arms", "clocked_rank_comparisons"}, "Condition")
        expected = {"synthetic": 32, "ordinary": 80, "background": 0 if condition == "clean" else 16}
        exact(row["denominators"], expected, "Condition denominators")
        keys(row["unassessed"], KINDS, "Unassessed counts")
        for kind in KINDS:
            count(row["unassessed"][kind], expected[kind])
        base = row["baseline"]
        keys(base, {"synthetic_locations", "synthetic_flags", "ordinary_flags", "background_flags"}, "Baseline")
        count(base["synthetic_locations"], 32)
        count(base["synthetic_flags"], 32)
        require(base["synthetic_locations"] <= base["synthetic_flags"], "Baseline location lacks a flagged file")
        count(base["ordinary_flags"], 80)
        count(base["background_flags"], expected["background"])
        keys(row["arms"], ARMS, "Arms")
        result = {}
        for name, arm in row["arms"].items():
            keys(arm, {"synthetic_location_cells", "ordinary_flag_cells", "background_flag_cells",
                       "candidate_false_only_synthetic", "zero_overlap_candidate_intervals",
                       "ordinary_flags_by_role", "candidate_locations_by_family", "candidate_located_host_count",
                       "candidate_located_component_count", "candidate_ordinary_host_count",
                       "candidate_ordinary_component_count"}, "Arm")
            synthetic = table(arm["synthetic_location_cells"], 32)
            ordinary = table(arm["ordinary_flag_cells"], 80)
            background = table(arm["background_flag_cells"], expected["background"])
            require(synthetic["baseline"] == base["synthetic_locations"], "Baseline locations differ across arms")
            require(ordinary["baseline"] == base["ordinary_flags"], "Baseline ordinary alerts differ across arms")
            require(background["baseline"] == base["background_flags"], "Baseline background alerts differ across arms")
            false_only = count(arm["candidate_false_only_synthetic"], 32 - synthetic["candidate"])
            candidate_flags = synthetic["candidate"] + false_only
            count(arm["zero_overlap_candidate_intervals"], 5 * candidate_flags)
            keys(arm["ordinary_flags_by_role"], ROLES, "Ordinary role counts")
            for value in arm["ordinary_flags_by_role"].values():
                count(value, 16)
            require(sum(arm["ordinary_flags_by_role"].values()) == ordinary["candidate"], "Ordinary roles do not recount")
            keys(arm["candidate_locations_by_family"], FAMILIES, "Anonymous family counts")
            for value in arm["candidate_locations_by_family"].values():
                count(value, 16)
            require(sum(arm["candidate_locations_by_family"].values()) == synthetic["candidate"], "Family locations do not recount")
            for prefix, flags in (("located", synthetic["candidate"]), ("ordinary", ordinary["candidate"])):
                hosts = count(arm["candidate_" + prefix + "_host_count"], min(16, flags))
                components = count(arm["candidate_" + prefix + "_component_count"], min(4, hosts))
                require((flags == 0) == (hosts == 0) == (components == 0), "Source-count presence differs from decisions")
                require(flags <= hosts * (2 if prefix == "located" else 5), "Source count is too small for its decisions")
                require(hosts <= components * 4, "Host counts exceed component extent")
            result[name] = {"candidate_locations": synthetic["candidate"], "combined_locations": synthetic["combined"],
                            "missed_locations": 32 - synthetic["combined"], "added_locations": synthetic["added"],
                            "candidate_false_only_synthetic": false_only,
                            "combined_ordinary_flags": ordinary["combined"], "added_ordinary_flags": ordinary["added"],
                            "combined_background_flags": background["combined"], "added_background_flags": background["added"]}
        paired = row["clocked_rank_comparisons"]
        keys(paired, {"all", "by_role", "by_family"}, "Clocked diagnostic")
        keys(paired["all"], {"wins", "losses", "ties", "comparisons", "above_all_five"}, "All matched ranks")
        ranks({key: value for key, value in paired["all"].items() if key != "above_all_five"}, 160)
        above = count(paired["all"]["above_all_five"], 32)
        require(5 * above <= paired["all"]["wins"], "All-control wins exceed pairwise wins")
        keys(paired["by_role"], ROLES, "Matched role ranks")
        keys(paired["by_family"], FAMILIES, "Anonymous family ranks")
        for role in paired["by_role"].values():
            ranks(role, 32)
        for family in paired["by_family"].values():
            ranks(family, 80)
        for key in ("wins", "losses", "ties", "comparisons"):
            require(sum(value[key] for value in paired["by_role"].values()) == paired["all"][key], "Role ranks do not recount")
            require(sum(value[key] for value in paired["by_family"].values()) == paired["all"][key], "Family ranks do not recount")
        report[condition] = {"baseline_locations": base["synthetic_locations"], "arms": result,
                             "unassessed": row["unassessed"], "matched_clock_wins": paired["all"]["wins"],
                             "matched_clock_comparisons": 160, "above_all_five_controls": above}
    return report


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def review(path):
    body = path.read_bytes()
    require(len(body) <= 100000, "Receipt exceeds the anonymous aggregate size limit")
    require(hashlib.sha256(body).hexdigest() == PIN, "Receipt differs from the published pinned aggregate")
    return validate(json.loads(body, object_pairs_hook=unique_fields))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[2] / "docs" / "evaluation-panels" / (NAME + ".json"))
    args = parser.parse_args()
    try:
        result = review(args.receipt)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps({"passed": True, "deliveries": COHORT["deliveries"], "conditions": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
