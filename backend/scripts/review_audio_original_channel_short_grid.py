#!/usr/bin/env python3
"""Verify anonymous original-channel experiment counts; native/model replay is private."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

PIN = "d1859a49b107a043cfa44e0c86b8433d02862b26ee1041ddd05d05e4669a962b"
ARMS = ("two_second", "two_second_plus_mono750", "two_second_plus_native750")
ROOMS = ("aachen", "huddersfield")
TOP = {
    "schema", "closed", "scope", "sampling", "counts", "comparison", "frozen_gate",
    "proceed_to_independent_native_confirmation", "production_activation_supported",
    "speaker_attribution_supported", "verification", "elapsed_seconds",
    "maximum_rss_bytes", "whole_protocol_commitment", "whole_numeric_closure_commitment",
    "limitations", "publisher_sources", "privacy",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def integer(value: object, upper: int) -> int:
    require(type(value) is int and 0 <= value <= upper, "Invalid scalar count")
    return value


def validate(data: dict) -> dict:
    require(isinstance(data, dict) and set(data) == TOP, "Unexpected aggregate fields")
    require(data["schema"] == "chipmunk.original-channel-short-grid-aggregate/1"
            and data["closed"] is True, "A closed named study is required")
    require(data["scope"] == {
        "native_deliveries": 60, "native_seconds": 6, "native_rate_hz": 16000,
        "native_channels": 2, "declared_readers": 24, "reader_pairs": 12,
        "measured_rooms": 2, "synthetic_files": 36, "annotated_interventions": 48,
        "ordinary_controls": 24, "untouched_two_voice_controls": 12,
        "same_reader_splice_two_voice_controls": 12, "isolated_natural_room_controls": 0,
        "different_reader_within_voice_splice_controls": 0,
        "latest_mls_declared_reader_overlap": 0, "latest_mls_native_source_overlap": 0,
    }, "Source scope changed")
    require(data["sampling"] == {
        "canonical_mono_views": 60, "native_channel_views": 120,
        "brief_windows_per_view": 11, "brief_window_seconds": .75,
        "brief_hop_seconds": .5, "mono_two_second_windows_per_file": 5,
        "outside_brief_schedule_seconds": .25,
    }, "Native input or timing scope changed")
    require(set(data["counts"]) == set(ARMS), "Missing comparison")
    for arm in ARMS:
        require(set(data["counts"][arm]) == set(ROOMS), "Missing room")
        for row in data["counts"][arm].values():
            require(set(row) == {
                "ordinary_total", "ordinary_flags", "edited_total", "edited_flags",
                "edit_labels_total", "edit_labels_located", "useful_files",
                "missed_files", "wrong_only_files",
            }, "Unexpected decision fields")
            require(row["ordinary_total"] == 12 and row["edited_total"] == 18
                    and row["edit_labels_total"] == 24, "A denominator changed")
            integer(row["ordinary_flags"], 12)
            flags, hits, misses = (integer(row[k], 18) for k in
                                  ("edited_flags", "useful_files", "missed_files"))
            require(hits + misses == 18 and hits <= flags, "Misses and flags differ")
            require(integer(row["wrong_only_files"], 18) == flags - hits,
                    "Off-target-only count differs")
            require(hits <= integer(row["edit_labels_located"], 24),
                    "Useful file count exceeds located interventions")
    long, mono, native = (data["counts"][arm] for arm in ARMS)
    for room in ROOMS:
        for earlier, later in ((long[room], mono[room]), (mono[room], native[room])):
            require(later["ordinary_flags"] >= earlier["ordinary_flags"]
                    and later["edited_flags"] >= earlier["edited_flags"]
                    and later["useful_files"] >= earlier["useful_files"],
                    "The supplement lost a retained result")
    summary = data["comparison"]
    require(set(summary) == {
        "new_useful_files", "new_source_pairs", "new_rooms",
        "new_single_edit_families", "new_quieter_only_files", "new_ordinary_files",
        "new_ordinary_pairs", "added_ordinary_seconds", "matched_ordinary_collision_files",
    }, "Unexpected source-level summary")
    gains = sum(native[r]["useful_files"] - mono[r]["useful_files"] for r in ROOMS)
    alarms = sum(native[r]["ordinary_flags"] - mono[r]["ordinary_flags"] for r in ROOMS)
    require(summary["new_useful_files"] == gains and summary["new_ordinary_files"] == alarms,
            "New gain or false-alert total differs")
    groups = integer(summary["new_source_pairs"], 12)
    families = integer(summary["new_single_edit_families"], 2)
    rooms = integer(summary["new_rooms"], 2)
    quiet = integer(summary["new_quieter_only_files"], 12)
    ordinary_groups = integer(summary["new_ordinary_pairs"], 12)
    collisions = integer(summary["matched_ordinary_collision_files"], 36)
    require(groups <= gains and families <= gains and rooms <= gains and quiet <= gains
            and ordinary_groups <= alarms and collisions <= gains,
            "Group or matched-control counts exceed delivery gains")
    seconds = summary["added_ordinary_seconds"]
    require(type(seconds) in (float, int) and 0 <= seconds <= 144,
            "Invalid ordinary highlighted duration")
    gate = {
        "three_added_useful_files": gains >= 3, "three_source_pairs": groups >= 3,
        "two_quieter_only_additions": quiet >= 2, "gains_in_both_rooms": rooms == 2,
        "both_families_on_single_edits": families == 2,
        "at_most_two_new_ordinary_files": alarms <= 2,
        "at_most_two_new_ordinary_pairs": ordinary_groups <= 2,
        "wrong_only_limit_each_room": all(
            native[r]["wrong_only_files"] - mono[r]["wrong_only_files"] <= 2 for r in ROOMS),
        "no_paired_control_collisions": collisions == 0,
        "ordinary_time_budget": seconds <= 3,
    }
    require(data["frozen_gate"] == gate
            and data["proceed_to_independent_native_confirmation"] is all(gate.values()),
            "Frozen continuation gate differs")
    require(data["production_activation_supported"] is False
            and data["speaker_attribution_supported"] is False,
            "Research results cannot establish product activation or speaker attribution")
    proof = data["verification"]
    require(proof == {
        "native_source_files_replayed": 60, "independent_canonical_mono_replays": 60,
        "native_stereo_samples_compared": 11_520_000,
        "additive_intervention_sample_identities": 2_304_000,
        "view_records_checked": 180, "window_source_admission_bindings": 2280,
        "distinct_authenticity_windows": 1477, "classifier_calls_with_repeat": 1478,
        "exact_repeat_passed": True, "authored_controls": 7,
        "independent_outcomes_passed": True,
    }, "Independent verification counts changed")
    return {
        "deliveries": 60, "ordinary_controls": 24, "synthetic_files": 36,
        "known_edit_clocks": 48, "additional_useful_files": gains,
        "additional_ordinary_flags": alarms, "continuation_gate_passed": all(gate.values()),
    }


def self_test(data: dict) -> int:
    mutations = [
        lambda x: x["counts"][ARMS[2]]["aachen"].__setitem__("missed_files", 16),
        lambda x: x["counts"][ARMS[0]]["huddersfield"].__setitem__("wrong_only_files", 3),
        lambda x: x["comparison"].__setitem__("new_useful_files", 1),
        lambda x: x["scope"].__setitem__("ordinary_controls", 36),
        lambda x: x["sampling"].__setitem__("native_channel_views", 180),
        lambda x: x.__setitem__("speaker_attribution_supported", True),
        lambda x: x.__setitem__("proceed_to_independent_native_confirmation", True),
        lambda x: x.__setitem__("private_source_record", "authored forbidden field"),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(data)
        mutate(changed)
        try:
            validate(changed)
        except (ValueError, KeyError):
            continue
        raise ValueError("Authored corruption passed the aggregate checks")
    return len(mutations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PIN, "Named aggregate commitment differs")
    data = json.loads(raw)
    summary = validate(data)
    if args.self_test:
        summary["authored_corruptions_rejected"] = self_test(data)
    print(json.dumps(summary, sort_keys=True))
    print("This checks aggregate arithmetic. Native audio and model replay require the retained private sources.")


if __name__ == "__main__":
    main()
