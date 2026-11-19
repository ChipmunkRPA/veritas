"""Standard-library recount of anonymous grouped decisions, not private media replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PIN = "d2f5f994bff2a4525634c0d57924beb28dad79444ca272ce63109f59c4fd2c55"
ROLES = ("development", "held")
CONDITIONS = ("clean", "minus5", "minus10")
KINDS = (
    "synthetic_converted",
    "untouched",
    "same_reader_splice",
    "different_reader_splice",
    "local_resampling",
    "background",
)
ARMS = ("original", "true_reference", "wrong_reference", "additive")
GROUP_COUNTERS = (
    "files",
    "flagged",
    "useful",
    "missed",
    "wrong_only",
    "off_target_intervals",
    "off_target_union_samples",
    "wholly_outside_intervals",
    "unassessed",
    "edit_unassessed",
    "flag_union_samples",
)


def keys(obj, names):
    assert isinstance(obj, dict) and set(obj) == set(names)


def count(x):
    assert type(x) is int and x >= 0
    return x


def review(data):
    keys(
        data,
        (
            "schema",
            "scope",
            "acoustic_groups",
            "groups",
            "paired_transitions",
            "new_genuine_files_per_reader_histograms",
            "new_noisy_useful_deliveries_per_event_histograms",
            "new_useful_control_collisions",
            "acoustic_guard",
            "research_gate",
            "verification",
            "counts",
            "limitations",
        ),
    )
    assert data["schema"] == "chipmunk.openvoice-v2-enrollment-serving750-anonymous/1"
    expected = dict(
        readers=6,
        development_readers=2,
        held_readers=4,
        held_meetings=1,
        meeting_groups=2,
        base_utterances=2,
        conversion_pipelines=1,
        physical_synthetic_events=6,
        held_physical_synthetic_events=4,
        source_deliveries=102,
        synthetic_deliveries=18,
        genuine_deliveries=72,
        background_deliveries=12,
        views=222,
        extractor_outputs=120,
        native_rate=16000,
        source_samples=192000,
        forensic_window_samples=12000,
        forensic_step_samples=8000,
        scheduled_windows_per_view=23,
        uncovered_tail_samples=4000,
        forensic_point=6.642364579351768,
    )
    assert data["scope"] == expected
    group = {}
    for row in data["groups"]:
        keys(row, ("role", "condition", "kind", "arm", *GROUP_COUNTERS))
        key = tuple(row[k] for k in ("role", "condition", "kind", "arm"))
        assert (
            key not in group
            and key[0] in ROLES
            and key[1] in CONDITIONS
            and key[2] in KINDS
            and key[3] in ARMS
        )
        for k in GROUP_COUNTERS:
            count(row[k])
        n = 2 if row["role"] == "development" else 4
        assert row["files"] == n and all(
            row[k] <= n
            for k in (
                "flagged",
                "useful",
                "missed",
                "wrong_only",
                "unassessed",
                "edit_unassessed",
            )
        )
        assert row["useful"] <= row["flagged"] <= n - row["unassessed"]
        assert row["flag_union_samples"] <= n * 188000
        assert row["off_target_union_samples"] <= row["flag_union_samples"]
        assert row["wholly_outside_intervals"] <= row["off_target_intervals"] <= n * 23
        if row["kind"] == "synthetic_converted":
            assert (
                row["missed"] == n - row["useful"]
                and row["wrong_only"] == row["flagged"] - row["useful"]
            )
        else:
            assert all(
                row[k] == 0
                for k in (
                    "useful",
                    "missed",
                    "wrong_only",
                    "off_target_intervals",
                    "off_target_union_samples",
                    "wholly_outside_intervals",
                )
            )
            assert row["arm"] != "wrong_reference"
        assert not (row["condition"] == "clean" and row["kind"] == "background")
        group[key] = row
    assert len(group) == 108
    paired = {}
    new_files = {role: 0 for role in ROLES}
    new_bg = {role: 0 for role in ROLES}
    noisy_gains = {role: 0 for role in ROLES}
    gains_by_condition = {}
    lost = 0
    for row in data["paired_transitions"]:
        keys(row, ("role", "condition", "kind", "files", "transitions"))
        key = tuple(row[k] for k in ("role", "condition", "kind"))
        assert key not in paired and key + ("original",) in group
        n = row["files"]
        transitions = row["transitions"]
        fields = (
            ("flag", "assessed", "edit_assessed", "useful")
            if row["kind"] == "synthetic_converted"
            else ("flag", "assessed", "edit_assessed")
        )
        keys(transitions, fields)
        for label, values in transitions.items():
            assert (
                isinstance(values, list)
                and len(values) == 4
                and sum(count(x) for x in values) == n
            )
            left = values[1] + values[3]
            right = values[2] + values[3]
            union = n - values[0]
            for arm, target in (
                ("original", left),
                ("true_reference", right),
                ("additive", union),
            ):
                name = {
                    "flag": "flagged",
                    "assessed": "unassessed",
                    "edit_assessed": "edit_unassessed",
                    "useful": "useful",
                }[label]
                assert group[key + (arm,)][name] == (
                    n - target if label in ("assessed", "edit_assessed") else target
                )
        original, processed, additive = (
            group[key + (a,)] for a in ("original", "true_reference", "additive")
        )
        assert (
            max(original["flag_union_samples"], processed["flag_union_samples"])
            <= additive["flag_union_samples"]
            <= original["flag_union_samples"] + processed["flag_union_samples"]
        )
        role = row["role"]
        if row["kind"] == "synthetic_converted":
            gains_by_condition[role, row["condition"]] = transitions["useful"][2]
            if row["condition"] != "clean":
                noisy_gains[role] += transitions["useful"][2]
            lost += transitions["useful"][1]
        elif row["kind"] == "background":
            new_bg[role] += transitions["flag"][2]
        else:
            new_files[role] += transitions["flag"][2]
        paired[key] = row
    assert len(paired) == 34
    lineage = {}
    for row in data["new_genuine_files_per_reader_histograms"]:
        keys(row, ("role", "histogram"))
        role = row["role"]
        h = row["histogram"]
        assert role in ROLES and role not in lineage
        keys(h, map(str, range(13)))
        assert sum(count(x) for x in h.values()) == (2 if role == "development" else 4)
        assert sum(int(k) * v for k, v in h.items()) == new_files[role]
        lineage[role] = sum(v for k, v in h.items() if k != "0")
    assert set(lineage) == set(ROLES)
    gain_events = {}
    for row in data["new_noisy_useful_deliveries_per_event_histograms"]:
        keys(row, ("role", "histogram"))
        role = row["role"]
        h = row["histogram"]
        assert role in ROLES and role not in gain_events
        keys(h, ("0", "1", "2"))
        assert sum(count(x) for x in h.values()) == (2 if role == "development" else 4)
        assert h["1"] + 2 * h["2"] == noisy_gains[role]
        gain_events[role] = h["1"] + h["2"]
    assert set(gain_events) == set(ROLES)
    collisions = {}
    for row in data["new_useful_control_collisions"]:
        keys(row, ("role", "condition", "new_useful", "with_matched_control_collision"))
        key = (row["role"], row["condition"])
        assert key not in collisions and key in gains_by_condition
        assert row["new_useful"] == gains_by_condition[key]
        assert count(row["with_matched_control_collision"]) <= row["new_useful"]
        collisions[key] = row["with_matched_control_collision"]
    assert len(collisions) == 6
    acoustic = {}
    for row in data["acoustic_groups"]:
        keys(
            row,
            (
                "role",
                "condition",
                "kind",
                "reference",
                "cells",
                "retained_at_least_minus3",
                "snr_gain_at_least_point5",
                "snr_gain_cells",
                "absent_target_projection_only",
            ),
        )
        key = tuple(row[k] for k in ("role", "condition", "kind", "reference"))
        assert (
            key not in acoustic
            and key[:3] in paired
            and key[3] in ("true_reference", "wrong_reference")
        )
        assert key[3] == "true_reference" or key[2] == "synthetic_converted"
        assert row["cells"] == (2 if row["role"] == "development" else 4)
        assert count(row["retained_at_least_minus3"]) <= row["cells"]
        assert count(row["snr_gain_at_least_point5"]) <= count(row["snr_gain_cells"])
        assert row["snr_gain_cells"] == (
            0 if row["condition"] == "clean" else row["cells"]
        )
        assert row["absent_target_projection_only"] == (row["kind"] == "background")
        acoustic[key] = row
    assert len(acoustic) == 40
    held_ai = [
        acoustic["held", c, "synthetic_converted", "true_reference"] for c in CONDITIONS
    ]
    held_genuine = [
        acoustic["held", c, "untouched", "true_reference"] for c in CONDITIONS
    ]
    guard = dict(
        all_held_synthetic_retention_at_least_minus3=all(
            r["retained_at_least_minus3"] == r["cells"] for r in held_ai
        ),
        all_held_untouched_retention_at_least_minus3=all(
            r["retained_at_least_minus3"] == r["cells"] for r in held_genuine
        ),
        all_held_noisy_untouched_snr_gain_at_least_point5=all(
            r["snr_gain_at_least_point5"] == r["cells"] for r in held_genuine[1:]
        ),
    )
    assert data["acoustic_guard"] == guard
    targets = [
        r
        for r in group.values()
        if r["role"] == "held"
        and r["kind"] == "synthetic_converted"
        and r["arm"] == "additive"
    ]
    ordinary = [
        r
        for r in group.values()
        if r["role"] == "held"
        and r["kind"] not in ("synthetic_converted", "background")
        and r["arm"] == "additive"
    ]
    gate = dict(
        two_new_held_noisy_physical_events=gain_events["held"] >= 2,
        at_most_one_new_genuine_file_overall=sum(new_files.values()) <= 1,
        at_most_one_new_genuine_lineage_overall=sum(lineage.values()) <= 1,
        at_most_one_new_held_genuine_file=new_files["held"] <= 1,
        at_most_one_new_held_genuine_lineage=lineage["held"] <= 1,
        zero_added_held_background_flags=new_bg["held"] == 0,
        no_new_useful_same_clock_matched_control_collision=all(
            collisions["held", c] == 0 for c in CONDITIONS[1:]
        ),
        held_synthetic_edit_coverage_at_least80percent=5
        * sum(r["files"] - r["edit_unassessed"] for r in targets)
        >= 4 * sum(r["files"] for r in targets),
        held_genuine_edit_coverage_at_least80percent=5
        * sum(r["files"] - r["edit_unassessed"] for r in ordinary)
        >= 4 * sum(r["files"] for r in ordinary),
    )
    assert data["research_gate"] == gate
    counters = data["counts"]
    keys(
        counters,
        (
            "extractor_calls",
            "forensic_calls",
            "distinct_classifier_inputs",
            "processed_only_lost_original_useful_deliveries",
            "held_new_useful_noisy_deliveries",
            "held_new_noisy_physical_events",
            "new_genuine_files_overall",
            "new_genuine_lineages_overall",
            "held_new_genuine_files",
            "held_new_genuine_lineages",
            "held_new_background_files",
        ),
    )
    for value in counters.values():
        count(value)
    assert (
        counters["extractor_calls"] == 121
        and counters["forensic_calls"] == counters["distinct_classifier_inputs"] + 1
    )
    assert counters["processed_only_lost_original_useful_deliveries"] == lost
    assert counters["held_new_useful_noisy_deliveries"] == noisy_gains["held"]
    assert counters["held_new_noisy_physical_events"] == gain_events["held"]
    assert counters["new_genuine_files_overall"] == sum(new_files.values())
    assert counters["new_genuine_lineages_overall"] == sum(lineage.values())
    assert (
        counters["held_new_genuine_files"] == new_files["held"]
        and counters["held_new_genuine_lineages"] == lineage["held"]
    )
    assert counters["held_new_background_files"] == new_bg["held"]
    v = data["verification"]
    expected_verify = dict(
        independent_native_review=True,
        independent_panel_review=True,
        independent_acoustic_review=True,
        independent_detector_review=True,
        native_input_sample_values=102 * 192000,
        outside_edit_identities=72 * (192000 - 15360),
        extraction_output_values=120 * 192000,
        forensic_view_sample_values=222 * 192000,
        scheduled_window_bindings=222 * 23,
        admitted_window_bindings=v["admitted_window_bindings"],
        authored_acoustic_controls=7,
        authored_selection_controls=8,
    )
    assert (
        v == expected_verify
        and counters["distinct_classifier_inputs"]
        <= count(v["admitted_window_bindings"])
        <= 222 * 23
    )
    expected_limits = dict(
        prior_project_source_exposure=True,
        possible_extractor_training_corpus_overlap=True,
        biometric_identity_verified=False,
        base_generator_exact_checkpoint_unverified=True,
        latest_live_background_refresh_completed=False,
        retained_native_background_replay_verified=True,
        raw_audio_published=False,
        per_case_outcomes_published=False,
        acoustic_features_published=False,
        private_closure_required_for_source_replay=True,
        confidence_changed=False,
        production_activation_supported=False,
    )
    assert data["limitations"] == expected_limits
    return dict(
        passed=True,
        paired_source_groups=len(paired),
        decision_groups=len(group),
        acoustic_groups=len(acoustic),
        new_held_noisy_physical_events=gain_events["held"],
        new_held_genuine_files=new_files["held"],
        research_gate_passed=all(gate.values()),
        acoustic_guard_passed=all(guard.values()),
        production_activation_supported=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "receipt",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("receipt.json"),
    )
    args = parser.parse_args()
    body = args.receipt.read_bytes()
    assert hashlib.sha256(body).hexdigest() == PIN, (
        "Named aggregate receipt hash changed"
    )
    print(json.dumps(review(json.loads(body)), sort_keys=True))


if __name__ == "__main__":
    main()
