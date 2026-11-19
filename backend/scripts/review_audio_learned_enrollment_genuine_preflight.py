"""Recount anonymous acoustic engineering groups without third-party packages."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SHA256 = "cb0e178ee7a07984afe330f5c0b39e91888822a8fb049e80693809bb392518ef"
SCHEMA = "chipmunk.learned-enrollment-genuine-preflight-aggregate/1"
SNR_BANDS = ("below_0p5", "0p5_to_5", "5_to_10", "10_to_15", "15_or_more")
RETENTION_BANDS = ("below_minus3", "minus3_to_0", "0_to_3", "3_or_more")


def gates(rows):
    held = [r for r in rows if r["role"] == "held"]
    total = sum(r["genuine_deliveries"] for r in held)
    return {
        "held_true_beats_wrong_at_least_two_thirds": (
            sum(r["true_reference_better_snr_than_wrong"] for r in held) * 3 >= 2 * total
        ),
        "held_snr_gain_each_cell_at_least_0p5_db": all(
            r["true_reference_snr_gain_at_least_0p5_db"] == r["genuine_deliveries"] for r in held
        ),
        "held_retention_each_cell_at_least_minus3_db": all(
            r["true_reference_retention_at_least_minus3_db"] == r["genuine_deliveries"] for r in held
        ),
    }


def validate(data):
    assert data["schema"] == SCHEMA and data["closed"] is True
    scope, proof, method = (data[k] for k in ("scope", "verification", "method"))
    assert scope["target_readers"] == 4 and scope["distinct_meeting_groups"] == 4
    assert scope["development_reader_meeting_lineages"] == scope["held_reader_meeting_lineages"] == 2
    assert scope["declared_annotated_participants"] == 4 * scope["distinct_meeting_groups"] == 16
    assert (
        scope["role_reader_overlap"]
        == scope["role_meeting_overlap"]
        == scope["role_participant_overlap"]
        == 0
    )
    assert (
        scope["total_input_deliveries"]
        == scope["genuine_deliveries"] + scope["background_only_deliveries"]
        == 20
    )
    assert scope["genuine_deliveries"] == 12 and scope["background_only_deliveries"] == 8
    assert scope["reference_choices_per_delivery"] == 2 and scope["noise_conditions_db"] == [-5, -10]
    assert scope["native_rate_hz"] == 16000 and scope["native_samples_per_delivery"] == 192000
    assert scope["native_channels"] == 1 and scope["native_pcm_bits"] == 16
    assert scope["original_native_host_seconds"] == 12 and scope["separate_genuine_enrollment_seconds"] == 4
    assert scope["acoustic_control_interval_samples"] == 15360
    assert scope["synthetic_deliveries"] == 0 and scope["noisy_genuine_deliveries_per_role"] == 4
    assert scope["project_source_reuse"] is True
    assert scope["genuine_upstream_training_corpus_overlap_possible"] is True
    assert scope["raw_waveforms_published"] is False and scope["per_case_results_published"] is False
    expected = {(role, condition) for role in ("development", "held") for condition in ("minus5", "minus10")}
    rows = data["noisy_genuine_groups"]
    assert len(rows) == 4 and {(r["role"], r["condition"]) for r in rows} == expected
    for row in rows:
        n = row["genuine_deliveries"]
        assert type(n) is int and n == row["reader_meeting_lineages"] == 2
        for field in (
            "true_reference_snr_gain_at_least_0p5_db",
            "true_reference_retention_at_least_minus3_db",
            "true_reference_better_snr_than_wrong",
        ):
            assert type(row[field]) is int and 0 <= row[field] <= n
        snr, retention = row["snr_gain_db_bands"], row["retention_db_bands"]
        assert set(snr) == set(SNR_BANDS) and set(retention) == set(RETENTION_BANDS)
        assert all(type(x) is int and 0 <= x <= n for x in (*snr.values(), *retention.values()))
        assert sum(snr.values()) == sum(retention.values()) == n
        assert n - snr["below_0p5"] == row["true_reference_snr_gain_at_least_0p5_db"]
        assert n - retention["below_minus3"] == row["true_reference_retention_at_least_minus3_db"]
    assert sum(r["genuine_deliveries"] for r in rows) == 8
    clean = data["clean_genuine_groups"]
    assert len(clean) == 2 and {r["role"] for r in clean} == {"development", "held"}
    assert all(r["genuine_deliveries"] == r["retention_db_0_to_3"] == 2 for r in clean)
    assert (
        sum(r["genuine_deliveries"] for r in clean) + sum(r["genuine_deliveries"] for r in rows)
        == scope["genuine_deliveries"]
    )
    thresholds = method["thresholds"]
    assert thresholds == {
        "true_beats_wrong_required_numerator": 2,
        "true_beats_wrong_required_denominator": 3,
        "minimum_each_held_snr_gain_db": 0.5,
        "minimum_each_held_foreground_retention_db": -3,
    }
    assert method["strict_tensor_state_entries"] == 948 and method["unsafe_pickle_globals"] == 0
    assert method["checkpoint_bytes"] == 251856016
    assert method["output_peak_normalization"] is False and method["enrollment_vad_or_trimming"] is False
    assert method["source_alignment_or_truth_based_rescaling"] is False
    assert (
        method["clean_snr_improvement_interpreted"] is False
        and method["background_projection_is_speaker_attribution"] is False
    )
    assert method["original_clock_preserved"] is True and method["published_config_preserved"] is True
    assert method["cpu_threads"] == 2 and method["interop_threads"] == 1
    gate = gates(rows)
    assert data["frozen_gate"] == gate and data["genuine_engineering_gate_passed"] is all(gate.values())
    assert proof["independent_native_and_outcome_replays_passed"] is True
    assert proof["ffmpeg_input_replays"] == scope["total_input_deliveries"]
    assert (
        proof["native_input_samples_checked"]
        == scope["total_input_deliveries"] * scope["native_samples_per_delivery"]
    )
    assert proof["known_component_additive_samples"] == 8 * scope["native_samples_per_delivery"]
    assert (
        proof["output_records"] == scope["total_input_deliveries"] * scope["reference_choices_per_delivery"]
    )
    assert (
        proof["native_output_samples_checked"]
        == proof["output_records"] * scope["native_samples_per_delivery"]
    )
    assert proof["scalar_fields_independently_recomputed"] == proof["output_records"] * 2 * 6
    assert (
        proof["actual_extraction_calls"] == proof["output_records"] + 1
        and proof["exact_initial_repeat"] is True
    )
    assert proof["authored_baseline_extraction_calls"] == 2
    assert (
        proof["total_extraction_calls_including_authored"]
        == proof["actual_extraction_calls"] + proof["authored_baseline_extraction_calls"]
    )
    assert proof["original_source_flacs_verified"] == 8 and proof["original_parquet_pins_verified"] == 3
    assert proof["same_reader_different_chapter_enrollments"] == proof["wrong_reader_enrollments"] == 4
    assert proof["authored_grader_controls"] == 7
    assert (
        proof["authenticity_calls"]
        == proof["synthetic_extraction_calls"]
        == proof["original_clone_model_calls"]
        == 0
    )
    assert all(value is False for value in data["claims"].values())
    assert data["licensing_exclusions"] == {
        "optional_usef_enabled": False,
        "optional_usef_source_executed": False,
        "noncommercial_optional_source_removed_before_import": True,
        "disabled_branch_has_fail_closed_stub": True,
        "checkpoint_has_usef_tensor_entries": False,
    }
    held = [r for r in rows if r["role"] == "held"]
    return {
        "held_lineages": scope["held_reader_meeting_lineages"],
        "held_noisy_deliveries": sum(r["genuine_deliveries"] for r in held),
        "held_snr_gain_passes": sum(r["true_reference_snr_gain_at_least_0p5_db"] for r in held),
        "held_foreground_retention_passes": sum(
            r["true_reference_retention_at_least_minus3_db"] for r in held
        ),
        "held_true_beats_wrong": sum(r["true_reference_better_snr_than_wrong"] for r in held),
        "genuine_engineering_gate_passed": all(gate.values()),
        "synthetic_retention_measured": data["claims"]["synthetic_retention_measured"],
        "production_activation_supported": data["claims"]["production_activation_supported"],
    }


def self_test(data):
    changes = (
        lambda d: d["scope"].update(held_reader_meeting_lineages=4),
        lambda d: d["scope"].update(genuine_upstream_training_corpus_overlap_possible=False),
        lambda d: d["scope"].update(total_input_deliveries=24),
        lambda d: d["noisy_genuine_groups"][0].update(genuine_deliveries=3),
        lambda d: d["noisy_genuine_groups"][0]["snr_gain_db_bands"].update(below_0p5=1),
        lambda d: d["noisy_genuine_groups"][2].update(true_reference_better_snr_than_wrong=0),
        lambda d: d["method"]["thresholds"].update(minimum_each_held_snr_gain_db=0),
        lambda d: d["verification"].update(native_output_samples_checked=1),
        lambda d: d["verification"].update(synthetic_extraction_calls=1),
        lambda d: d["claims"].update(production_activation_supported=True),
        lambda d: d["claims"].update(unseen_reader_validation=True),
        lambda d: d.update(genuine_engineering_gate_passed=False),
    )
    rejected = 0
    for mutate in changes:
        changed = copy.deepcopy(data)
        mutate(changed)
        try:
            validate(changed)
        except (AssertionError, KeyError, TypeError):
            rejected += 1
        else:
            raise AssertionError("An authored corruption was accepted")
    assert rejected == len(changes)
    return rejected


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
    print("This recounts anonymous aggregate arithmetic. Exact source/model replay requires private custody.")


if __name__ == "__main__":
    main()
