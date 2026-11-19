"""Standard-library count recount. Native/model/threshold replay remains private."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PIN = "73a71723824a95ab7e58dbdeeb530af9d7e43bc5c3098f8eaf44cd6248e0c1e4"
GROUP_FIELDS = (
    "role",
    "condition",
    "target_windows",
    "target_admitted",
    "target_current_admitted",
    "target_new_admitted",
    "target_removed_current",
    "other_windows",
    "other_admitted",
    "background_windows",
    "background_admitted",
    "original_other_admitted",
    "original_background_admitted",
)


def keys(obj, expected):
    assert isinstance(obj, dict) and set(obj) == set(expected)


def count(n):
    assert type(n) is int and n >= 0
    return n


def review(data):
    keys(data, ("schema", "scope", "primary", "posthoc", "verification", "limitations"))
    assert (
        data["schema"] == "chipmunk.original-enrollment-admission-feasibility-counts/1"
    )
    scope = dict(
        planned_target_readers=3,
        planned_development_readers=1,
        planned_held_readers=2,
        qualified_target_readers=2,
        qualified_development_readers=1,
        qualified_held_readers=1,
        rejected_held_readers=1,
        ordinary_nuisance_readers=2,
        original_utterances_including_nuisances=12,
        source_deliveries=18,
        target_present_deliveries=6,
        other_genuine_deliveries=6,
        background_deliveries=6,
        isolated_recorded_noise_deliveries=4,
        native_rate=16000,
        source_samples=192000,
        reference_samples=64000,
        window_samples=12000,
        step_samples=8000,
        scheduled_windows_per_view=23,
        window_observations=414,
        embedding_dimensions=192,
        development_absent_windows=138,
        synthetic_deliveries=0,
    )
    assert data["scope"] == scope
    primary = data["primary"]
    keys(
        primary,
        (
            "point_locked_before_held_encoding",
            "source_goal_met",
            "groups",
            "held_metric_guard",
        ),
    )
    assert (
        primary["point_locked_before_held_encoding"] is True
        and primary["source_goal_met"] is False
    )
    groups = {}
    for row in primary["groups"]:
        keys(row, GROUP_FIELDS)
        key = row["role"], row["condition"]
        assert (
            key not in groups
            and key[0] in ("development", "held")
            and key[1] in ("clean", "traffic", "competing")
        )
        for name in GROUP_FIELDS[2:]:
            count(row[name])
        assert 1 <= row["target_windows"] <= 23
        assert row["other_windows"] == row["background_windows"] == 23
        assert (
            row["target_admitted"] <= row["target_windows"]
            and row["target_current_admitted"] <= row["target_windows"]
        )
        assert row["target_removed_current"] <= row["target_current_admitted"]
        assert (
            row["target_new_admitted"]
            <= row["target_windows"] - row["target_current_admitted"]
        )
        assert (
            row["target_admitted"]
            == row["target_current_admitted"]
            - row["target_removed_current"]
            + row["target_new_admitted"]
        )
        for kind in ("other", "background"):
            assert (
                row[kind + "_admitted"] <= 23
                and row["original_" + kind + "_admitted"] <= 23
            )
        groups[key] = row
    assert len(groups) == 6
    assert (
        len({r["target_windows"] for r in groups.values() if r["role"] == "held"}) == 1
    )
    assert (
        len(
            {r["target_windows"] for r in groups.values() if r["role"] == "development"}
        )
        == 1
    )
    held = [r for r in groups.values() if r["role"] == "held"]
    guard = {
        r["condition"]: dict(
            retains80percent=r["target_admitted"] * 5 >= r["target_windows"] * 4,
            absent_admits_at_most10percent=(
                r["other_admitted"] + r["background_admitted"]
            )
            * 10
            <= r["other_windows"] + r["background_windows"],
        )
        for r in held
    }
    assert guard == primary["held_metric_guard"]
    posthoc = data["posthoc"]
    keys(
        posthoc,
        (
            "diagnostic_only",
            "held_selected",
            "new_point_adopted",
            "new_inference_calls",
            "independently_recounted",
            "groups",
        ),
    )
    assert (
        posthoc["diagnostic_only"]
        is posthoc["held_selected"]
        is posthoc["independently_recounted"]
        is True
    )
    assert posthoc["new_point_adopted"] is False and posthoc["new_inference_calls"] == 0
    bounds = {}
    for row in posthoc["groups"]:
        keys(
            row,
            (
                "condition",
                "target_windows",
                "absent_windows",
                "frozen_target_admitted",
                "positive_margin_target_any_likeness",
                "positive_margin_absent_any_likeness",
                "optimistic_held_selected_best_target_at10percent_absent",
                "optimistic_held_selected_absent_flags",
            ),
        )
        condition = row["condition"]
        assert condition not in bounds and ("held", condition) in groups
        for key, value in row.items():
            if key != "condition":
                count(value)
        primary_row = groups["held", condition]
        assert (
            row["target_windows"] == primary_row["target_windows"]
            and row["absent_windows"] == 46
        )
        assert row["frozen_target_admitted"] == primary_row["target_admitted"]
        assert (
            row["frozen_target_admitted"]
            <= row["optimistic_held_selected_best_target_at10percent_absent"]
            <= row["positive_margin_target_any_likeness"]
            <= row["target_windows"]
        )
        assert row["positive_margin_absent_any_likeness"] <= 46
        assert row["optimistic_held_selected_absent_flags"] * 10 <= 46
        bounds[condition] = row
    assert len(bounds) == 3
    v = data["verification"]
    expected = dict(
        native_original_sample_values=1280000,
        native_delivery_sample_values=18 * 192000,
        exact_additive_sample_values=18 * 192000,
        original_window_sample_values=414 * 12000,
        cosine_pairs_replayed=414 * 4,
        original_window_clocks_replayed=414,
        role_original_source_overlap=0,
        participant_encoder_calls=378,
        distinct_encoder_inputs=377,
        reference_encoder_calls=8,
        exact_native_initial_repeat=True,
        independent_encoder_calls=377,
        independent_speech_activity_calls=18,
        separator_calls=0,
        synthesis_calls=0,
        forensic_calls=0,
        source_clock_and_point_parity=True,
    )
    assert (
        v == expected
        and v["participant_encoder_calls"] == v["distinct_encoder_inputs"] + 1
    )
    limits = dict(
        single_held_reader=True,
        overlapping_windows=True,
        windows_are_independent_people=False,
        source_conditions_are_correlated=True,
        reused_background_captures=True,
        reused_nuisance_readers=True,
        reader_identity_historical_project_exposure=True,
        encoder_training_independence_established=False,
        physical_room_independence_established=False,
        synthetic_retention_measured=False,
        detector_improvement_measured=False,
        original_audio_modified_for_representation=False,
        per_source_scores_or_features_published=False,
        private_custody_required_for_native_replay=True,
        production_activation_supported=False,
    )
    assert data["limitations"] == limits
    return dict(
        passed=True,
        groups=6,
        held_readers=1,
        independent_population_accuracy_supported=False,
        noisy_metric_guard_passed=all(
            all(guard[c].values()) for c in ("traffic", "competing")
        ),
        full_source_target_met=False,
        production_activation_supported=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "receipt",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("receipt.json"),
    )
    args = p.parse_args()
    raw = args.receipt.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PIN, "Named scalar receipt changed"
    print(json.dumps(review(json.loads(raw)), sort_keys=True))


if __name__ == "__main__":
    main()
