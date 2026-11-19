"""Portable recount of grouped unchanged-point speech detector outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SHA256 = "52643c58d1212254654d847efd451ee111396a8a7edd496cb6173613f9133c84"
ARMS = ("original", "true_reference", "wrong_reference", "additive")
ROLES = ("development", "held")
CONDITIONS = ("clean", "minus5", "minus10")


def gates(data, groups):
    p = data["held_pair"]
    held = [groups["held", c, "additive"] for c in CONDITIONS]
    return {
        "new_held_noisy_physical_event": p["new_noisy_physical_events"] >= 1,
        "at_most_one_new_genuine_file": p["new_genuine_files"] <= 1,
        "at_most_one_new_genuine_lineage": p["new_genuine_lineages"] <= 1,
        "no_new_background_flags": p["new_background_files"] == 0,
        "no_matched_same_clock_collision": p["new_useful_noisy_deliveries"] == 0,
        "held_synthetic_edit_coverage_at_least80percent": sum(
            g["synthetic"] - g["synthetic_edit_unassessed"] for g in held
        )
        * 5
        >= sum(g["synthetic"] for g in held) * 4,
        "held_genuine_target_coverage_at_least80percent": sum(
            g["genuine"] - g["genuine_target_unassessed"] for g in held
        )
        * 5
        >= sum(g["genuine"] for g in held) * 4,
    }


def validate(data):
    assert data["schema"] == "chipmunk.learned-enrollment-serving750-aggregate/1" and data["closed"] is True
    s, m, proof = (data[k] for k in ("scope", "method", "verification"))
    assert s["generator_families"] == 1 and s["physical_synthetic_events"] == s["target_readers"] == 4
    assert s["development_lineages"] == s["held_lineages"] == 2 and s["distinct_meeting_groups"] == 4
    assert (
        s["source_input_deliveries"]
        == s["synthetic_deliveries"] + s["genuine_deliveries"] + s["background_deliveries"]
        == 32
    )
    assert s["synthetic_deliveries"] == s["genuine_deliveries"] == s["physical_synthetic_events"] * 3 == 12
    assert (
        s["background_deliveries"] == 8
        and s["measured_views"] == s["source_input_deliveries"] * s["views_per_source"] == 96
    )
    assert s["views_per_source"] == 3 and s["conditions"] == list(CONDITIONS)
    assert s["native_rate_hz"] == 16000 and s["native_samples_per_input"] == 192000
    assert s["native_channels"] == 1 and s["native_pcm_bits"] == 16 and s["edit_duration_samples"] == 15360
    assert s["project_source_reuse"] is True and s["possible_train_clean_100_overlap"] is True
    assert (
        s["original_clone_prompt_audio_overlap_unresolved"] is True
        and s["ordinary_splice_controls_present"] is False
    )
    assert m["brief_point"] == 6.642364579351768 and m["strict_greater_than"] is True
    assert m["adjacent_scheduled_confirmation"] is True and m["source_assignment_is_diagnostic"] is True
    assert (
        m["points_fitted"] is False and m["overall_confidence_changed"] is False and m["clipped_samples"] == 0
    )
    assert m["window_samples"] == 12000 and m["step_samples"] == 8000
    assert m["scheduled_windows_per_view"] == (192000 - 12000) // 8000 + 1 == 23
    assert m["uncovered_tail_samples"] == 192000 - (22 * 8000 + 12000) == 4000
    groups = {(g["role"], g["condition"], g["arm"]): g for g in data["groups"]}
    assert len(groups) == len(data["groups"]) == 24
    assert set(groups) == {(r, c, a) for r in ROLES for c in CONDITIONS for a in ARMS}
    for (_, condition, _), g in groups.items():
        counts = {k: v for k, v in g.items() if k not in {"role", "condition", "arm"}}
        assert all(type(v) is int and v >= 0 for v in counts.values())
        assert g["synthetic"] == g["genuine"] == 2 and g["background"] == (0 if condition == "clean" else 2)
        assert g["useful"] + g["missed"] == g["synthetic"]
        assert g["useful"] + g["wrong_only"] == g["synthetic_flagged"] <= g["synthetic"]
        assert g["genuine_flagged"] <= g["genuine"] and g["background_flagged"] <= g["background"]
        assert g["synthetic_unassessed"] <= g["synthetic_edit_unassessed"] <= g["synthetic"]
        assert g["genuine_unassessed"] <= g["genuine_target_unassessed"] <= g["genuine"]
        assert g["background_unassessed"] <= g["background"]
        assert g["wholly_outside_intervals"] <= g["off_target_intervals"]
    paired = {(p["role"], p["condition"]): p for p in data["paired"]}
    assert len(paired) == len(data["paired"]) == 6
    for (role, condition), p in paired.items():
        old, new = (groups[role, condition, a] for a in ("original", "additive"))
        assert new["useful"] - old["useful"] == p["new_useful_deliveries"] - p["lost_useful_deliveries"]
        assert (
            p["lost_useful_deliveries"] == 0
            and 0 <= p["new_useful_physical_events"] <= p["new_useful_deliveries"]
        )
        assert new["genuine_flagged"] - old["genuine_flagged"] == p["new_genuine_flag_files"]
        assert 0 <= p["new_genuine_flag_lineages"] <= p["new_genuine_flag_files"]
        for source, output in (
            ("genuine_flag_union_samples", "added_genuine_flag_union_samples"),
            ("off_target_intervals", "added_off_target_intervals"),
            ("off_target_union_samples", "added_off_target_union_samples"),
            ("background_flagged", "new_background_flag_files"),
        ):
            assert new[source] - old[source] == p[output] >= 0
    p = data["overall_pair"]
    assert p["new_useful_synthetic_deliveries"] == sum(x["new_useful_deliveries"] for x in paired.values())
    assert (
        p["lost_useful_synthetic_deliveries"]
        == sum(x["lost_useful_deliveries"] for x in paired.values())
        == 0
    )
    assert p["new_useful_physical_events"] == p["new_useful_synthetic_deliveries"] == 0
    assert p["new_genuine_flag_files"] == sum(x["new_genuine_flag_files"] for x in paired.values()) == 2
    assert (
        max(x["new_genuine_flag_lineages"] for x in paired.values())
        <= p["new_genuine_flag_lineages"]
        <= sum(x["new_genuine_flag_lineages"] for x in paired.values())
    )
    assert p["new_genuine_flag_lineages"] == 2
    assert p["new_background_flag_files"] == sum(x["new_background_flag_files"] for x in paired.values()) == 0
    assert (
        p["extraction_only_lost_clean_held_detections"]
        == groups["held", "clean", "original"]["useful"] - groups["held", "clean", "true_reference"]["useful"]
        == 1
    )
    h = data["held_pair"]
    assert (
        h["new_useful_noisy_deliveries"]
        == sum(paired["held", c]["new_useful_deliveries"] for c in ("minus5", "minus10"))
        == 0
    )
    assert h["new_noisy_physical_events"] == h["new_useful_noisy_deliveries"]
    assert h["new_genuine_files"] == sum(paired["held", c]["new_genuine_flag_files"] for c in CONDITIONS) == 0
    assert (
        h["new_genuine_lineages"] == 0
        and h["new_background_files"]
        == sum(paired["held", c]["new_background_flag_files"] for c in CONDITIONS)
        == 0
    )
    assert (
        h["added_genuine_flag_union_samples"]
        == sum(paired["held", c]["added_genuine_flag_union_samples"] for c in CONDITIONS)
        == 48000
    )
    frozen = gates(data, groups)
    assert data["frozen_gate"] == frozen and data["advance_source_disjoint_followup"] is all(frozen.values())
    assert proof["independent_native_preprocessing_and_outcome_replay_passed"] is True
    assert proof["input_views_verified"] == s["measured_views"]
    assert proof["native_sample_values_reconstructed"] == s["measured_views"] * s["native_samples_per_input"]
    assert proof["scheduled_window_bindings"] == s["measured_views"] * m["scheduled_windows_per_view"] == 2208
    assert 0 <= proof["admitted_prepared_window_bindings"] <= proof["scheduled_window_bindings"]
    assert proof["admitted_prepared_window_bindings"] == 1670
    assert proof["authenticity_calls"] == proof["distinct_classifier_inputs"] + 1 == 1255
    assert proof["exact_initial_repeat"] is True and proof["authored_selector_controls"] == 8
    assert proof["new_generator_calls"] == 0 and proof["cpu_threads"] == 2 and proof["interop_threads"] == 1
    assert proof["publisher_clone_native_user_channel_replays"] == 4
    assert data["previous_acoustic_findings"]["general_synthetic_preservation_supported"] is False
    assert data["production_activation_supported"] is False and all(
        v is False for v in data["claims"].values()
    )
    return {
        "original_useful": sum(groups[r, c, "original"]["useful"] for r in ROLES for c in CONDITIONS),
        "additive_useful": sum(groups[r, c, "additive"]["useful"] for r in ROLES for c in CONDITIONS),
        "synthetic_deliveries": s["synthetic_deliveries"],
        "original_genuine_flags": sum(
            groups[r, c, "original"]["genuine_flagged"] for r in ROLES for c in CONDITIONS
        ),
        "additive_genuine_flags": sum(
            groups[r, c, "additive"]["genuine_flagged"] for r in ROLES for c in CONDITIONS
        ),
        "extraction_only_lost_clean_held_detections": p["extraction_only_lost_clean_held_detections"],
        "held_added_genuine_highlight_seconds": h["added_genuine_flag_union_samples"] / 16000,
        "advance_source_disjoint_followup": all(frozen.values()),
        "production_activation_supported": False,
    }


def self_test(data):
    changes = (
        lambda d: d["scope"].update(generator_families=2),
        lambda d: d["scope"].update(physical_synthetic_events=12),
        lambda d: d["scope"].update(held_lineages=4),
        lambda d: d["overall_pair"].update(new_genuine_flag_lineages=1),
        lambda d: d["scope"].update(possible_train_clean_100_overlap=False),
        lambda d: d["method"].update(brief_point=4.38),
        lambda d: d["groups"][0].update(useful=1),
        lambda d: d["overall_pair"].update(extraction_only_lost_clean_held_detections=0),
        lambda d: d["held_pair"].update(added_genuine_flag_union_samples=0),
        lambda d: d["verification"].update(scheduled_window_bindings=1),
        lambda d: d["verification"].update(authenticity_calls=1254),
        lambda d: d.update(advance_source_disjoint_followup=True),
        lambda d: d.update(production_activation_supported=True),
    )
    for mutate in changes:
        changed = copy.deepcopy(data)
        mutate(changed)
        try:
            validate(changed)
        except (AssertionError, KeyError, TypeError):
            pass
        else:
            raise AssertionError("An authored corruption was accepted")
    return len(changes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    body = args.receipt.read_bytes()
    if hashlib.sha256(body).hexdigest() != SHA256:
        raise SystemExit("The aggregate differs from the pinned publication.")
    data = json.loads(body)
    result = validate(data)
    if args.self_test:
        result["authored_corruptions_rejected"] = self_test(data)
    print(json.dumps(result, sort_keys=True))
    print("This recounts grouped arithmetic. Exact source/model replay requires retained private custody.")


if __name__ == "__main__":
    main()
