"""Review grouped one-family acoustic retention arithmetic using the standard library."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SHA256 = "a4a5a4ccc708b596cdecef6991e2eefe5601945d2b51f7debe873b555a0463fe"


def review(data):
    assert (
        data["schema"] == "chipmunk.learned-enrollment-one-family-retention-aggregate/1"
        and data["closed"] is True
    )
    s = data["scope"]
    assert s["generator_families"] == 1 and s["declared_clone_targets"] == 4
    assert s["development_lineages"] == s["held_lineages"] == 2
    assert (
        s["native_inputs"] == 12
        and s["native_rate_hz"] == 16000
        and s["native_samples_per_input"] == 192000
    )
    assert (
        s["native_channels"] == 1
        and s["native_bits"] == 16
        and s["edit_duration_samples"] == 15360
    )
    assert (
        s["project_source_reuse"] is True
        and s["possible_train_clean_100_overlap"] is True
    )
    assert s["original_clone_prompt_audio_overlap_unresolved"] is True
    assert (
        s["declared_role_reader_collisions"]
        == s["normalized_script_key_role_collisions"]
        == 0
    )
    groups = data["groups"]
    expected = {
        (r, c, f)
        for r in ("development", "held")
        for c in ("clean", "minus5", "minus10")
        for f in ("true_reference", "wrong_reference")
    }
    assert (
        len(groups) == 12
        and {(g["role"], g["condition"], g["reference"]) for g in groups} == expected
    )
    for g in groups:
        assert (
            g["deliveries"] == g["lineages"] == 2
            and type(g["retained_at_least_minus3_db"]) is int
        )
        b = g["retention_db_bands"]
        assert set(b) == {
            "below_minus30",
            "minus30_to_minus3",
            "minus3_to_0",
            "0_or_more",
        }
        assert (
            all(type(v) is int and 0 <= v <= 2 for v in b.values())
            and sum(b.values()) == 2
        )
        assert b["minus3_to_0"] + b["0_or_more"] == g["retained_at_least_minus3_db"]
    held = [
        g
        for g in groups
        if g["role"] == "held"
        and g["condition"] != "clean"
        and g["reference"] == "true_reference"
    ]
    clean = [
        g
        for g in groups
        if g["role"] == "held"
        and g["condition"] == "clean"
        and g["reference"] == "true_reference"
    ]
    development = [
        g
        for g in groups
        if g["role"] == "development"
        and g["condition"] != "clean"
        and g["reference"] == "true_reference"
    ]
    assert len(held) == 2 and sum(g["deliveries"] for g in held) == 4
    guard = all(g["retained_at_least_minus3_db"] == g["deliveries"] for g in held)
    assert data["frozen_noisy_guard_passed"] is guard
    d = data["diagnostic"]
    assert d["held_noisy_deliveries"] == 4 and d["held_noisy_retained"] == sum(
        g["retained_at_least_minus3_db"] for g in held
    )
    assert d["held_clean_deliveries"] == 2 and d["held_clean_below_minus3"] == sum(
        g["deliveries"] - g["retained_at_least_minus3_db"] for g in clean
    )
    assert d["held_clean_below_minus30"] == sum(
        g["retention_db_bands"]["below_minus30"] for g in clean
    )
    assert d["development_noisy_deliveries"] == 4 and d[
        "development_noisy_below_minus3"
    ] == sum(g["deliveries"] - g["retained_at_least_minus3_db"] for g in development)
    p = data["verification"]
    assert p["independent_native_and_outcome_checks_passed"] is True
    assert p["native_input_samples_checked"] == 12 * 192000
    assert p["outside_edit_sample_identities"] == 12 * (192000 - 15360)
    assert p["output_records"] == 24 and p["raw_output_samples_checked"] == 24 * 192000
    assert (
        p["scalar_fields_recomputed"] == 24 * 5
        and p["extraction_calls"] == 25
        and p["exact_initial_repeat"] is True
    )
    assert (
        p["authenticity_calls"] == p["generator_calls"] == 0
        and p["authored_grader_controls"] == 8
    )
    assert (
        p["publisher_whole_tar_archives_rehashed"] == 3
        and p["publisher_native_user_channel_replays"] == 4
    )
    assert data["method"]["minimum_each_held_noisy_clone_retention_db"] == -3
    assert (
        data["method"]["noise_only_gate_predeclared"] is True
        and data["method"]["clean_guard_predeclared"] is False
    )
    assert data["claims"]["acoustic_clone_preservation_measured"] is True
    assert all(
        v is False
        for k, v in data["claims"].items()
        if k != "acoustic_clone_preservation_measured"
    )
    return dict(
        noisy_guard_passed=guard,
        held_lineages=2,
        held_noisy_retained=d["held_noisy_retained"],
        held_clean_below_minus30=d["held_clean_below_minus30"],
        development_noisy_below_minus3=d["development_noisy_below_minus3"],
        general_preservation_supported=False,
        detector_accuracy_measured=False,
    )


def self_test(data):
    changes = (
        lambda d: d["scope"].update(generator_families=2),
        lambda d: d["scope"].update(held_lineages=4),
        lambda d: d["scope"].update(possible_train_clean_100_overlap=False),
        lambda d: d["groups"][0].update(deliveries=3),
        lambda d: d["diagnostic"].update(held_clean_below_minus30=0),
        lambda d: d["verification"].update(raw_output_samples_checked=1),
        lambda d: d.update(frozen_noisy_guard_passed=False),
        lambda d: d["claims"].update(production_activation_supported=True),
    )
    for mutate in changes:
        changed = copy.deepcopy(data)
        mutate(changed)
        try:
            review(changed)
        except (AssertionError, KeyError, TypeError):
            pass
        else:
            raise AssertionError("Altered aggregate was accepted")
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
    result = review(data)
    if args.self_test:
        result["authored_corruptions_rejected"] = self_test(data)
    print(json.dumps(result, sort_keys=True))
    print(
        "This recounts aggregate arithmetic. Source/model replay requires retained private custody."
    )


if __name__ == "__main__":
    main()
