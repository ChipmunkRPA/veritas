"""Portable aggregate arithmetic and integrity review, with no media or model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SHA256 = "7b6d58e5ba67b0331ee4327687817be65c9e3bd9dce66641e1ef3e068b40c375"


def validate(data):
    assert data["schema"] == "chipmunk.room-layer-short-transfer-aggregate/1"
    assert data["closed"] is True
    scope = data["scope"]
    assert (scope["fit_inputs"], scope["point_inputs"], scope["transfer_inputs"]) == (132, 48, 60)
    assert scope["transfer_ordinary_files"] + scope["transfer_edited_files"] == scope["transfer_inputs"]
    assert (scope["fit_reader_pairs"], scope["point_reader_pairs"], scope["transfer_reader_pairs"]) == (
        3,
        4,
        12,
    )
    assert (scope["fit_readers"], scope["point_readers"], scope["transfer_readers"]) == (6, 8, 24)
    assert scope["frozen_layer_count"] + 1 == scope["fit_parameters"] == 26
    assert scope["new_downloads"] == scope["new_synthesis"] == 0
    counts = data["counts"]
    expected = {
        "incumbent": {"aachen": (1, 17, 0, 0), "huddersfield": (2, 16, 1, 2)},
        "point_only": {"aachen": (1, 17, 1, 3), "huddersfield": (4, 14, 1, 0)},
        "learned": {"aachen": (1, 17, 1, 3), "huddersfield": (7, 11, 4, 4)},
    }
    for arm, rooms in counts.items():
        assert set(rooms) == {"aachen", "huddersfield"}
        for room, cell in rooms.items():
            assert all(type(value) is int and value >= 0 for value in cell.values())
            assert cell["edited_total"] == cell["useful_files"] + cell["missed_files"] == 18
            assert cell["ordinary_total"] == 12 and cell["label_total"] == 24
            assert cell["wrong_only_files"] <= cell["missed_files"]
            assert cell["ordinary_flags"] <= cell["ordinary_total"]
            assert cell["useful_files"] <= cell["labels_located"] <= 2 * cell["useful_files"]
            actual = tuple(
                cell[key] for key in ("useful_files", "missed_files", "ordinary_flags", "wrong_only_files")
            )
            assert actual == expected[arm][room]
    paired = data["point_only_paired"]
    assert paired["added_useful_deliveries"] == 2 and paired["lost_useful_deliveries"] == 0
    assert (
        paired["gain_source_pairs"]
        == paired["gain_physical_labelled_events"]
        == paired["gain_synthesis_families"]
        == paired["gain_rooms"]
        == 1
    )
    assert paired["gain_quieter_only_deliveries"] == 0
    assert paired["new_ordinary_files"] == paired["new_ordinary_source_pairs"] == 1
    assert paired["new_ordinary_kind"] == "matched_genuine_splice"
    assert paired["added_ordinary_seconds"] == 1.25 and paired["matched_ordinary_collision_files"] == 0
    assert paired["wrong_only_additions"] == {"aachen": 3, "huddersfield": 0}
    assert paired["wrong_only_converted_to_useful"] == {"aachen": 0, "huddersfield": 2}
    assert paired["existing_wrong_intervals_retained"] is True
    assert paired["formal_new_adoption_gate_created_after_results"] is False
    summary = data["learned_summary"]
    def useful(arm):
        return sum(c["useful_files"] for c in counts[arm].values())

    def ordinary(arm):
        return sum(c["ordinary_flags"] for c in counts[arm].values())
    assert useful("point_only") - useful("incumbent") == paired["added_useful_deliveries"]
    assert ordinary("point_only") - ordinary("incumbent") == paired["new_ordinary_files"]
    assert summary["new_useful_files"] == useful("learned") - useful("incumbent") == 5
    assert summary["new_ordinary_flags"] == ordinary("learned") - ordinary("incumbent") == 4
    assert summary["gain_source_pairs"] == 3 and summary["gain_rooms"] == 1
    assert summary["gain_single_families"] == 2 and summary["quieter_only_gains"] == 0
    assert summary["added_ordinary_seconds"] == 5.0 and summary["matched_ordinary_collision_files"] == 0
    gate = {
        "four_useful_gains": summary["new_useful_files"] >= 4,
        "three_source_pairs": summary["gain_source_pairs"] >= 3,
        "two_quieter_gains": summary["quieter_only_gains"] >= 2,
        "gains_both_rooms": summary["gain_rooms"] == 2,
        "both_single_edit_families": summary["gain_single_families"] == 2,
        "no_room_useful_loss": all(
            counts["learned"][room]["useful_files"] >= counts["incumbent"][room]["useful_files"]
            for room in counts["learned"]
        ),
        "ordinary_budget": summary["new_ordinary_flags"] <= 2,
        "wrong_only_each_room": all(
            counts["learned"][room]["wrong_only_files"] - counts["incumbent"][room]["wrong_only_files"] <= 2
            for room in counts["learned"]
        ),
        "matched_control_collision_budget": summary["matched_ordinary_collision_files"] <= 1,
        "ordinary_seconds_budget": summary["added_ordinary_seconds"] <= 3,
        "layer_fit_beats_point_only": useful("learned") >= useful("point_only") + 2,
        "layer_fit_false_vs_point_only": ordinary("learned") <= ordinary("point_only") + 1,
    }
    assert gate == data["original_frozen_learned_gate"]
    assert all(gate.values()) == data["original_gate_passed"] is False
    assert data["production_activation_supported"] is False and data["speaker_attribution_supported"] is False
    proof = data["verification"]
    assert proof["independent_replay_passed"] is True
    assert proof["native_inputs_checked"] == proof["independent_canonical_pcm_replays"] == 240
    assert proof["native_stereo_sample_identities"] == 240 * 96000 * 2
    assert proof["source_window_admission_bindings"] == 240 * 11
    assert proof["exact_original_short_window_parity"] == 60 * 11
    assert proof["development_actual_calls"] == proof["development_distinct_classifier_inputs"] + 1 == 1073
    assert proof["transfer_actual_calls"] == proof["transfer_distinct_classifier_inputs"] + 1 == 414
    assert proof["authored_compatibility_calls"] == 3 and proof["authored_controls"] == 9
    assert proof["exact_initial_repeats"] == 2 and proof["max_original_head_reconstruction_error"] < 3e-5
    return {
        "useful_deliveries": {arm: useful(arm) for arm in counts},
        "ordinary_flags": {arm: ordinary(arm) for arm in counts},
        "point_only_distinct_gain_events": paired["gain_physical_labelled_events"],
        "frozen_learned_gate_passed": all(gate.values()),
    }


def self_test(data):
    corruptions = []
    for name, change in (
        ("useful", lambda x: x["counts"]["point_only"]["huddersfield"].update(useful_files=5)),
        ("ordinary", lambda x: x["counts"]["point_only"]["aachen"].update(ordinary_flags=0)),
        ("independent_event", lambda x: x["point_only_paired"].update(gain_physical_labelled_events=2)),
        (
            "retained_wrong_times",
            lambda x: x["point_only_paired"].update(existing_wrong_intervals_retained=False),
        ),
        ("ordinary_time", lambda x: x["point_only_paired"].update(added_ordinary_seconds=0)),
        (
            "new_gate",
            lambda x: x["point_only_paired"].update(formal_new_adoption_gate_created_after_results=True),
        ),
        ("learned_gate", lambda x: x.update(original_gate_passed=True)),
        ("source_bindings", lambda x: x["verification"].update(source_window_admission_bindings=2639)),
    ):
        altered = copy.deepcopy(data)
        change(altered)
        try:
            validate(altered)
        except AssertionError:
            corruptions.append(name)
        else:
            raise AssertionError("Accepted authored corruption " + name)
    return len(corruptions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    body = args.receipt.read_bytes()
    if hashlib.sha256(body).hexdigest() != SHA256:
        raise SystemExit("The aggregate differs from the pinned publication.")
    data = json.loads(body)
    result = validate(data)
    if args.self_test:
        result["authored_corruptions_rejected"] = self_test(data)
    print(json.dumps(result, sort_keys=True))
    print(
        "This replays aggregate arithmetic. Exact native/model replay requires the retained private custody."
    )


if __name__ == "__main__":
    main()
