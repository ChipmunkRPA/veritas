"""Recount the named anonymous strong-noise pilot. Uses the standard library only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

PIN = "7e617410f4805f900a07b48407616fbb9bdb26dfc068499d6a32e7fa877961bd"
RECEIPT = "audio-nonlinguistic-local-strong-noise-2026-11-19.json"
ROLES = ("fit", "point", "held")
CONDITIONS = ("clean", "competing")
ARMS = ("incumbent", "candidate", "augmented")
KINDS = ("synthetic", "ordinary", "background")
COMMON = ("files", "flagged", "unassessed", "edit_unassessed", "highlight_samples")
SYNTHETIC = (
    "useful",
    "missed",
    "wrong_only",
    "off_target_ranges",
    "off_target_samples",
    "wholly_outside_ranges",
)


def keys(value, expected):
    assert isinstance(value, dict) and set(value) == set(expected)


def count(value):
    assert type(value) is int and value >= 0
    return value


def review(data):
    keys(
        data,
        (
            "schema",
            "scope",
            "groups",
            "paired_transitions",
            "new_ordinary_files_per_host",
            "new_noisy_carrier_gains_per_base_event",
            "held_ordinary_control_breakdown",
            "held_comparison",
            "gate",
            "advance_larger_source_confirmation",
            "verification",
            "limitations",
        ),
    )
    assert data["schema"] == "chipmunk.nonlinguistic-local-noise-counts/1"
    expected_scope = dict(
        target_hosts=6,
        fit_hosts=2,
        point_hosts=2,
        held_hosts=2,
        nuisance_readers=3,
        declared_readers=9,
        distinct_books=9,
        source_components=3,
        held_source_components=1,
        native_original_utterances=15,
        source_deliveries=126,
        synthetic_deliveries=48,
        ordinary_deliveries=72,
        background_deliveries=6,
        base_generated_events=12,
        held_base_generated_events=4,
        synthetic_carrier_variants=24,
        conditions=2,
        meeting_captures=3,
        annotated_background_participants=12,
        native_rate=16000,
        source_samples=192000,
        edit_samples=15360,
        cell_samples=3200,
        cells_per_file=60,
        valid_context_cells=54,
        candidate_interval_samples=9600,
        embedding_dimensions=512,
        projected_dimensions=32,
        head_features=64,
        head_parameters=65,
        fit_cells=2138,
        ordinary_point_host_groups=2,
        allowed_ordinary_point_host_groups=1,
    )
    assert data["scope"] == expected_scope
    groups = {}
    for row in data["groups"]:
        keys(row, ("role", "condition", "arm", *KINDS))
        key = tuple(row[k] for k in ("role", "condition", "arm"))
        assert key not in groups and key[0] in ROLES and key[1] in CONDITIONS and key[2] in ARMS
        for kind in KINDS:
            obj = row[kind]
            keys(obj, COMMON + (SYNTHETIC if kind == "synthetic" else ()))
            for v in obj.values():
                count(v)
            expected = (
                8 if kind == "synthetic" else 12 if kind == "ordinary" else 2 if key[1] == "competing" else 0
            )
            assert obj["files"] == expected
            assert obj["flagged"] <= expected - obj["unassessed"]
            assert obj["edit_unassessed"] <= expected and obj["highlight_samples"] <= expected * 192000
            if kind == "synthetic":
                assert obj["useful"] <= obj["flagged"] and obj["missed"] == expected - obj["useful"]
                assert obj["wrong_only"] == obj["flagged"] - obj["useful"]
                assert obj["off_target_samples"] <= obj["highlight_samples"]
                assert obj["wholly_outside_ranges"] <= obj["off_target_ranges"]
        groups[key] = row
    assert len(groups) == 18
    transitions = {}
    ordinary_new = {r: 0 for r in ROLES}
    noisy_new = {r: 0 for r in ROLES}
    background_new = {r: 0 for r in ROLES}
    noisy_gains = {r: 0 for r in ROLES}
    for row in data["paired_transitions"]:
        keys(row, ("role", "condition", "kind", "files", "transitions"))
        key = row["role"], row["condition"], row["kind"]
        assert key not in transitions and key[:2] + ("incumbent",) in groups and key[2] in KINDS
        n = groups[key[:2] + ("incumbent",)][key[2]]["files"]
        assert n == row["files"] and n > 0
        labels = ("flag", "assessed", "edit_assessed") + (("useful",) if key[2] == "synthetic" else ())
        keys(row["transitions"], labels)
        for label, values in row["transitions"].items():
            assert isinstance(values, list) and len(values) == 4 and sum(count(x) for x in values) == n
            marginals = (values[1] + values[3], values[2] + values[3], n - values[0])
            field = {
                "flag": "flagged",
                "assessed": "unassessed",
                "edit_assessed": "edit_unassessed",
                "useful": "useful",
            }[label]
            for arm, value in zip(ARMS, marginals):
                assert groups[key[:2] + (arm,)][key[2]][field] == (
                    n - value if label in ("assessed", "edit_assessed") else value
                )
        original, candidate, augmented = (groups[key[:2] + (arm,)][key[2]] for arm in ARMS)
        assert (
            max(original["highlight_samples"], candidate["highlight_samples"])
            <= augmented["highlight_samples"]
            <= original["highlight_samples"] + candidate["highlight_samples"]
        )
        added = row["transitions"]["flag"][2]
        if key[2] == "ordinary":
            ordinary_new[key[0]] += added
            if key[1] == "competing":
                noisy_new[key[0]] += added
        elif key[2] == "background":
            background_new[key[0]] += added
        if key[2] == "synthetic" and key[1] == "competing":
            noisy_gains[key[0]] += row["transitions"]["useful"][2]
        transitions[key] = row
    assert len(transitions) == 15
    host_counts = {}
    for row in data["new_ordinary_files_per_host"]:
        keys(row, ("role", "histogram"))
        role = row["role"]
        assert role in ROLES and role not in host_counts
        hist = row["histogram"]
        keys(hist, map(str, range(13)))
        assert sum(count(v) for v in hist.values()) == 2
        assert sum(int(k) * v for k, v in hist.items()) == ordinary_new[role]
        host_counts[role] = sum(v for k, v in hist.items() if k != "0")
    assert set(host_counts) == set(ROLES)
    events = {}
    gain_hosts = {}
    gain_families = {}
    for row in data["new_noisy_carrier_gains_per_base_event"]:
        keys(row, ("role", "histogram", "hosts_with_gain", "families_with_gain"))
        role = row["role"]
        assert role in ROLES and role not in events
        hist = row["histogram"]
        keys(hist, ("0", "1", "2"))
        assert sum(count(v) for v in hist.values()) == 4
        assert hist["1"] + 2 * hist["2"] == noisy_gains[role]
        events[role] = hist["1"] + hist["2"]
        gain_hosts[role], gain_families[role] = (
            count(row["hosts_with_gain"]),
            count(row["families_with_gain"]),
        )
        assert gain_hosts[role] <= min(2, events[role]) and gain_families[role] <= min(2, events[role])
        assert bool(gain_hosts[role]) == bool(gain_families[role]) == bool(events[role])
    assert set(events) == set(ROLES)
    breakdown = {}
    controls = (
        "untouched",
        "same_reader",
        "different_reader",
        "roundtrip22050",
        "roundtrip24000",
        "roundtrip_opus",
    )
    for row in data["held_ordinary_control_breakdown"]:
        keys(row, ("condition", "control", "arm", "files", "flagged", "highlight_samples"))
        key = tuple(row[k] for k in ("condition", "control", "arm"))
        assert key not in breakdown and key[0] in CONDITIONS and key[1] in controls and key[2] in ARMS
        assert row["files"] == 2 and count(row["flagged"]) <= 2 and count(row["highlight_samples"]) <= 384000
        breakdown[key] = row
    assert len(breakdown) == 36
    for condition in CONDITIONS:
        for arm in ARMS:
            rows = [r for r in breakdown.values() if r["condition"] == condition and r["arm"] == arm]
            assert sum(r["flagged"] for r in rows) == groups["held", condition, arm]["ordinary"]["flagged"]
            assert (
                sum(r["highlight_samples"] for r in rows)
                == groups["held", condition, arm]["ordinary"]["highlight_samples"]
            )
    compare = data["held_comparison"]
    keys(
        compare,
        (
            "held_noisy_new_base_events",
            "held_noisy_new_deliveries",
            "held_new_ordinary_files",
            "held_new_ordinary_hosts",
            "held_noisy_new_ordinary_files",
            "held_new_background_files",
            "held_new_noisy_wrong_only_files",
            "held_new_collision_base_events",
        ),
    )
    for v in compare.values():
        count(v)
    assert (
        compare["held_noisy_new_base_events"] == events["held"]
        and compare["held_noisy_new_deliveries"] == noisy_gains["held"]
    )
    assert (
        compare["held_new_ordinary_files"] == ordinary_new["held"]
        and compare["held_new_ordinary_hosts"] == host_counts["held"]
    )
    assert (
        compare["held_noisy_new_ordinary_files"] == noisy_new["held"]
        and compare["held_new_background_files"] == background_new["held"]
    )
    assert compare["held_new_collision_base_events"] <= events["held"]
    assert (
        compare["held_new_noisy_wrong_only_files"]
        <= groups["held", "competing", "augmented"]["synthetic"]["wrong_only"]
    )
    ai = [groups["held", c, "augmented"]["synthetic"] for c in CONDITIONS]
    genuine = [groups["held", c, "augmented"]["ordinary"] for c in CONDITIONS]
    gate = dict(
        two_new_noisy_base_events=events["held"] >= 2,
        both_held_hosts_gain=gain_hosts["held"] == 2,
        both_generators_gain=gain_families["held"] == 2,
        at_most_two_new_noisy_ordinary_files=noisy_new["held"] <= 2,
        at_most_one_new_noisy_ordinary_host=noisy_new["held"] == 0,
        at_most_three_new_ordinary_files_overall=ordinary_new["held"] <= 3,
        no_added_isolated_background_flags=background_new["held"] == 0,
        at_most_two_new_wrong_only_noisy_synthetic_files=compare["held_new_noisy_wrong_only_files"] <= 2,
        at_most_one_new_collision_base_event=compare["held_new_collision_base_events"] <= 1,
        held_synthetic_edit_coverage80percent=5 * sum(r["files"] - r["edit_unassessed"] for r in ai)
        >= 4 * sum(r["files"] for r in ai),
        held_ordinary_edit_coverage80percent=5 * sum(r["files"] - r["edit_unassessed"] for r in genuine)
        >= 4 * sum(r["files"] for r in genuine),
        incumbent_ranges_preserved=True,
    )
    # This named closure has zero added noisy ordinary files; its host count is therefore zero.
    assert noisy_new["held"] == 0
    assert gate == data["gate"] and all(gate.values()) == data["advance_larger_source_confirmation"]
    verification = data["verification"]
    expected = dict(
        official_source_full_checksum_verified=True,
        role_reader_and_book_overlap=0,
        original_source_rows_verified=15,
        decoded_original_samples=3920480,
        retained_crop_samples=2880000,
        delivered_pcm_samples=126 * 192000,
        edited_copies=108,
        outside_edit_equal_samples=108 * (192000 - 15360),
        opus_roundtrips_verified=18,
        independent_source_replay=True,
        independent_renderer_replay=True,
        independent_feature_point_replay=True,
        independent_paired_outcome_replay=True,
        source_cell_bindings=7560,
        private_feature_values_verified=3870720,
        brief_original_window_bindings=126 * 23,
        admitted_two_second_source_window_bindings=2480,
        feature_model_calls=1693,
        feature_distinct_inputs=1692,
        baseline_model_calls=1246,
        baseline_distinct_inputs=1245,
        exact_first_feature_repeat=True,
        exact_first_baseline_repeat=True,
        closed_generation_calls=14,
        successful_piper_calls_in_failed_startups=4,
        failed_kitten_startups=2,
        convex_fit_stationarity_checked=True,
        fit_before_point_features=True,
        point_locked_before_held_features=True,
        confirming_support_adds_no_useful_locations=True,
        confirming_support_audit_strata=6,
    )
    assert verification == expected
    limits = dict(
        shared_nuisance_within_each_role=True,
        repeated_carriers_and_conditions=True,
        windows_are_independent_people=False,
        physical_room_independence_established=False,
        retained_background_prior_exposure=True,
        global_alias_and_pretraining_independence_established=False,
        inspired_representation_not_paper_reproduction=True,
        codec_controls_establish_ai_origin=False,
        traffic_transfer_tested=False,
        per_source_features_or_scores_published=False,
        raw_or_generated_voice_published=False,
        head_coefficients_published=False,
        private_custody_required_for_native_replay=True,
        service_confidence_changed=False,
        production_activation_supported=False,
    )
    assert data["limitations"] == limits
    return dict(
        passed=True,
        groups=18,
        paired_strata=15,
        held_source_components=1,
        held_new_noisy_base_events=events["held"],
        held_new_ordinary_files=ordinary_new["held"],
        research_gate_passed=all(gate.values()),
        production_activation_supported=False,
    )


def self_test(data):
    mutations = [
        lambda p: p.update(source_identity="private"),
        lambda p: p["scope"].update(held_source_components=2),
        lambda p: p["groups"][0]["synthetic"].update(files=9),
        lambda p: p["groups"][0]["synthetic"].update(pcm_sha256="a" * 64),
        lambda p: p["paired_transitions"][0]["transitions"]["flag"].__setitem__(0, 99),
        lambda p: p["new_ordinary_files_per_host"][0]["histogram"].__setitem__("0", 99),
        lambda p: p["new_noisy_carrier_gains_per_base_event"][0]["histogram"].__setitem__("0", 99),
        lambda p: p["held_ordinary_control_breakdown"][0].update(flagged=99),
        lambda p: p["held_comparison"].update(held_new_ordinary_files=0),
        lambda p: p["held_comparison"].update(held_noisy_new_base_events=1),
        lambda p: p["gate"].update(two_new_noisy_base_events=True),
        lambda p: p.update(advance_larger_source_confirmation=True),
        lambda p: p["verification"].update(point_locked_before_held_features=False),
        lambda p: p["verification"].update(delivered_pcm_samples=1),
        lambda p: p["limitations"].update(production_activation_supported=True),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(data)
        mutate(changed)
        try:
            review(changed)
        except (AssertionError, KeyError, TypeError, ValueError):
            continue
        raise AssertionError("An authored corruption was accepted")
    return len(mutations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "receipt",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs/evaluation-panels" / RECEIPT,
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PIN, "Named count-only receipt changed"
    data = json.loads(raw)
    result = review(data)
    if args.self_test:
        result["authored_corruptions_rejected"] = self_test(data)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
