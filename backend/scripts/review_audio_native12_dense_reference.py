"""Recount the privacy-limited twelve-second earlier-copy research aggregate.

This checks grouped counts and the frozen decision, not source audio or model
inference. Reader-component counts are native-audited input facts because the
public file deliberately contains no individual reader or case identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

EXPECTED_SHA256 = "01ebb0b79b21219ab1577f3f7cffe5a6eaa18ccb96d6443b7453e43e02ee1ad0"
CONDITIONS = ("clean", "traffic", "competing_speech")
FOREGROUND = frozenset({"untouched", "same_reader_splice", "different_reader_splice", "roundtrip_piper",
                        "roundtrip_kitten", "edited_piper", "edited_kitten"})
WRONG_REFERENCE = frozenset({"wrong_reader", "shifted_earlier", "half_matched_earlier"})
PROCESSING = frozenset({"whole_gain", "whole_lowpass", "whole_high_shelf"})
CONTROLS = PROCESSING | {"untouched", "roundtrip_piper", "roundtrip_kitten"}
ROLES = FOREGROUND | WRONG_REFERENCE | PROCESSING
STATES = frozenset({"local_change", "no_corroborated_local_change", "reference_match_not_supported",
                    "consistent_frequency_balance_change", "local_support_not_available"})


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, minimum=0, maximum=624):
    return type(value) is int and minimum <= value <= maximum


def review(receipt):
    require(set(receipt) == {"schema", "scope", "method", "counts", "decision", "native_review",
                            "commitments", "privacy", "limitations"}, "Aggregate fields differ")
    require(receipt["schema"] == "chipmunk.audio-native12-dense-reference-aggregate/1", "Aggregate schema differs")
    expected_scope = {
        "post_hoc_development": True, "independent_new_source_holdout": False,
        "requires_supplied_clean_earlier_copy": True, "single_upload_detection": False,
        "new_model_calls": 0, "affects_overall_confidence": False,
        "foreground_hosts": 16, "declared_host_reader_components": 8, "generator_families": 2,
        "synthetic_targets_repeated_across_conditions": 32, "background_captures": 4,
        "conditions": list(CONDITIONS), "pairs": 624, "pairs_per_condition": 208,
        "sample_rate": 16000, "samples_per_copy": 192000,
    }
    require(receipt["scope"] == expected_scope, "Source scope or denominators differ")
    require(receipt["method"] == {
        "six_second_window_starts": [0, 19200, 38400, 57600, 76800, 96000], "dense_point": .31758884,
        "maximum_start_difference": 5760, "preserves_existing_range": True,
        "preserves_reference_admission": True, "preserves_frequency_guard": True, "zero_model_weight": True,
    }, "Frozen method differs")
    require(receipt["privacy"] == dict.fromkeys(("raw_audio", "individual_decisions", "per_case_scores",
            "source_identifiers", "wording_or_transcripts", "local_paths", "individual_edit_positions"), False),
            "Publication privacy scope differs")
    require(set(receipt["counts"]) == set(CONDITIONS), "A condition is missing or duplicated")
    summary, new_families, new_locations, new_misplaced, wrong_flags = {}, set(), 0, 0, 0
    misplaced_kinds = {"ordinary": 0, "synthetic": 0}
    for condition in CONDITIONS:
        roles = receipt["counts"][condition]
        require(set(roles) == ROLES, "A source role is missing or unexpected")
        for role, cell in roles.items():
            require(set(cell) == {"total", "baseline", "candidate"} and cell["total"] == 16,
                    "Source role denominator differs")
            for arm in ("baseline", "candidate"):
                numbers = cell[arm]
                require(set(numbers) == {"flags", "target_locations", "unassessed_reference", "states"},
                        "Grouped decision fields differ")
                require(all(integer(numbers[key], maximum=16) for key in
                            ("flags", "target_locations", "unassessed_reference")), "Invalid decision counts")
                require(numbers["target_locations"] <= numbers["flags"], "Locations exceed flags")
                states = numbers["states"]
                require(set(states) <= STATES and all(integer(v, 1, 16) for v in states.values())
                        and sum(states.values()) == 16, "Coverage states do not cover the source role")
                require(numbers["flags"] == states.get("local_change", 0)
                        and numbers["unassessed_reference"] == states.get("reference_match_not_supported", 0),
                        "Grouped states disagree with the decisions")
                require(numbers["flags"] + numbers["unassessed_reference"] <= 16,
                        "An unassessed comparison has a flag")
            before, after = cell["baseline"], cell["candidate"]
            require(before["flags"] <= after["flags"] and before["target_locations"] <= after["target_locations"],
                    "An existing range or location was displaced")
            require(before["unassessed_reference"] == after["unassessed_reference"],
                    "Reference admission changed")
            if role in WRONG_REFERENCE:
                wrong_flags += after["flags"]
            if role in FOREGROUND:
                difference = ((after["flags"] - after["target_locations"])
                              - (before["flags"] - before["target_locations"]))
                require(difference >= 0, "An inherited off-target range disappeared")
                new_misplaced += difference
                misplaced_kinds["synthetic" if role.startswith("edited_") else "ordinary"] += difference
            if condition != "clean" and role.startswith("edited_"):
                addition = after["target_locations"] - before["target_locations"]
                new_locations += addition
                if addition:
                    new_families.add(role.removeprefix("edited_"))
        table = {}
        for arm in ("baseline", "candidate"):
            table[arm] = {
                "synthetic_locations": sum(roles[role][arm]["target_locations"] for role in
                                           ("edited_piper", "edited_kitten")),
                "genuine_splice_locations": sum(roles[role][arm]["target_locations"] for role in
                                               ("same_reader_splice", "different_reader_splice")),
                "ordinary_processing_flags": sum(roles[role][arm]["flags"] for role in CONTROLS),
                "wrong_reference_flags": sum(roles[role][arm]["flags"] for role in WRONG_REFERENCE),
            }
        summary[condition] = table
    decision = receipt["decision"]
    require(set(decision) == {"new_generator_families", "new_noisy_synthetic_locations",
            "new_ordinary_or_processing_reader_components", "new_reader_components", "new_wrong_locations",
            "passes", "wrong_reference_flags"}, "Frozen decision fields differ")
    require(decision["new_noisy_synthetic_locations"] == new_locations
            and decision["new_generator_families"] == sorted(new_families)
            and decision["new_wrong_locations"] == new_misplaced
            and decision["wrong_reference_flags"] == wrong_flags, "Frozen decision totals do not recount")
    components = decision["new_ordinary_or_processing_reader_components"]
    require(set(components) == set(CONDITIONS) and all(integer(v, maximum=8) for v in components.values())
            and integer(decision["new_reader_components"], maximum=min(8, new_locations)),
            "Native-audited component counts are outside the cohort")
    passed = (new_locations >= 4 and decision["new_reader_components"] >= 2
              and new_families == {"piper", "kitten"} and max(components.values()) <= 1
              and new_misplaced == 0 and wrong_flags == 0)
    require(type(decision["passes"]) is bool and decision["passes"] == passed, "Frozen gate differs")
    require(receipt["native_review"] == {
        "passed": True, "source_deliveries_checked": 368, "source_pairs_recomputed": 624,
        "dense_evidence_windows_recomputed": 1464, "new_misplaced_kind_counts": misplaced_kinds,
    }, "Recorded independent native review differs")
    require(set(receipt["commitments"]) == {"private_plan", "private_measurements", "private_aggregate",
            "independent_native_review", "independent_native_reviewer"}
            and all(isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v)
                    for v in receipt["commitments"].values()), "Native custody commitments differ")
    require(type(receipt["limitations"]) is list and len(receipt["limitations"]) == 5
            and all(isinstance(v, str) for v in receipt["limitations"]), "Limitations differ")
    return {"pairs": 624, "summary": summary, "new_noisy_synthetic_locations": new_locations,
            "new_off_target_foreground_highlights": new_misplaced, "frozen_gate_passed": passed,
            "native_source_replay_required_for_individual_verification": True}


def unique(fields):
    result = {}
    for name, value in fields:
        require(name not in result, "Duplicate aggregate field")
        result[name] = value
    return result


def load(path):
    body = path.read_bytes()
    require(len(body) <= 128 * 1024 and hashlib.sha256(body).hexdigest() == EXPECTED_SHA256,
            "The pinned public aggregate changed")
    receipt = json.loads(body, object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    return review(receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(load(args.receipt), sort_keys=True))
