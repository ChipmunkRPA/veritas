#!/usr/bin/env python3
"""Recount the anonymous exact-room-voice diagnostic; native replay remains private."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

PIN = "d7a70135017cc07e3b85a84c16fb5d8c319c1cbda4050605f46af2d3e151862c"
TOP = {
    "schema", "closed", "scope", "counts", "admission_detail", "paired_context_summary",
    "summary", "frozen_gate", "supports_further_mixture_recovery_research",
    "production_activation_supported", "speaker_attribution_supported", "verification",
    "elapsed_seconds", "whole_protocol_commitment", "whole_numeric_closure_commitment",
    "limitations", "source_procedure", "publisher_sources",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def validate(data: dict) -> dict:
    require(isinstance(data, dict) and set(data) == TOP, "Unexpected public fields")
    require(data["schema"] == "chipmunk.room-voice-recovery-ceiling-aggregate/1"
            and data["closed"] is True, "A closed named aggregate is required")
    require(data["scope"] == {
        "selected_reader_pairs": 6, "phoneme_pairs": 2, "stock_pairs": 4,
        "source_voice_instances": 12, "same_carrier_inputs": 66, "mixtures": 30,
        "exact_room_voice_states": 36, "ordinary_isolated_voice_observations": 24,
        "unique_ordinary_mixture_files": 12, "ordinary_mixture_source_observations": 24,
        "measured_held_rooms": 1, "native_seconds": 6, "sample_rate_hz": 16000,
        "source_stereo_channels": 2, "source_states_per_voice": 3,
        "source_generation_calls": 0, "oracle_component_selection": True,
    }, "Source or oracle scope changed")
    cohorts = {"phoneme_transfer": 4, "stock_transfer": 8}
    require(set(data["counts"]) == set(cohorts)
            and set(data["admission_detail"]) == set(cohorts), "Missing source cohort")
    gain_total = 0
    for cohort, voices in cohorts.items():
        row = data["counts"][cohort]
        require(set(row) == {
            "voices", "geometrically_reachable", "mixture_brief_useful",
            "oracle_brief_useful", "mixture_two_second_useful", "oracle_two_second_useful",
            "oracle_brief_clear_recoveries", "ordinary_stem_observations",
            "ordinary_stem_flags", "ordinary_mixture_observations",
            "ordinary_mixture_flags", "new_ordinary_stem_flags",
        }, "Unexpected source decision fields")
        require(row["voices"] == voices, "Voice-instance denominator differs")
        for key, value in row.items():
            require(type(value) is int and value >= 0, "Invalid scalar decision count")
            if key not in {"ordinary_stem_observations", "ordinary_stem_flags",
                           "ordinary_mixture_observations", "ordinary_mixture_flags",
                           "new_ordinary_stem_flags"}:
                require(value <= voices, "Source counts exceed voice instances")
        require(row["ordinary_stem_observations"] == row["ordinary_mixture_observations"]
                == 2 * voices, "Ordinary source-observation denominator differs")
        require(row["ordinary_stem_flags"] <= 2 * voices
                and row["ordinary_mixture_flags"] <= 2 * voices
                and row["new_ordinary_stem_flags"] <= row["ordinary_stem_flags"],
                "Ordinary flags exceed their denominator")
        require(row["oracle_brief_clear_recoveries"] <= row["oracle_brief_useful"],
                "Recoveries exceed useful source clocks")
        admitted = data["admission_detail"][cohort]
        require(set(admitted) == {
            "mixture_files_with_adjacent_speech", "mixture_admitted_geometry_at_edit",
            "oracle_files_with_adjacent_speech", "oracle_admitted_geometry_at_edit",
        }, "Unexpected admission fields")
        for value in admitted.values():
            require(type(value) is int and 0 <= value <= voices, "Invalid speech coverage")
        require(admitted["mixture_admitted_geometry_at_edit"]
                <= admitted["mixture_files_with_adjacent_speech"]
                and admitted["oracle_admitted_geometry_at_edit"]
                <= admitted["oracle_files_with_adjacent_speech"],
                "Useful admitted geometry exceeds assessed files")
        gain_total += row["oracle_brief_clear_recoveries"]
    summary = data["summary"]
    require(set(summary) == {
        "clear_recoveries", "gain_source_pairs", "gain_cohorts", "gain_families",
        "new_ordinary_stem_flags",
    }, "Unexpected recovery summary")
    require(summary["clear_recoveries"] == gain_total, "Recovery count differs")
    for key in ("gain_source_pairs", "gain_cohorts", "gain_families"):
        require(type(summary[key]) is int and 0 <= summary[key] <= gain_total,
                "Gain source count exceeds recovered observations")
    require(summary["new_ordinary_stem_flags"]
            == sum(r["new_ordinary_stem_flags"] for r in data["counts"].values()),
            "Added ordinary flags differ")
    paired = data["paired_context_summary"]
    require(set(paired) == {"planned", "fully_admitted", "synthetic_above_genuine_in_both"}
            and paired["planned"] == 12
            and 0 <= paired["synthetic_above_genuine_in_both"] <= paired["fully_admitted"] <= 12,
            "Paired-context denominator or ordering differs")
    gate = {
        "three_clear_recoveries": gain_total >= 3,
        "three_source_pairs": summary["gain_source_pairs"] >= 3,
        "both_cohorts": summary["gain_cohorts"] == 2,
        "two_families": summary["gain_families"] >= 2,
        "ordinary_stem_additions_within_budget": summary["new_ordinary_stem_flags"] <= 2,
    }
    require(data["frozen_gate"] == gate
            and data["supports_further_mixture_recovery_research"] is all(gate.values()),
            "Frozen diagnostic gate differs")
    require(data["production_activation_supported"] is False
            and data["speaker_attribution_supported"] is False,
            "An oracle diagnostic cannot establish activation or attribution")
    require(data["verification"] == {
        "native_inputs_checked": 66, "source_window_admission_bindings": 1056,
        "distinct_classifier_inputs": 520, "actual_classifier_calls": 521,
        "exact_initial_repeat": True, "exact_additive_stereo_sample_identities": 5_760_000,
        "authored_controls": 7, "independent_outcomes_passed": True,
    }, "Independent verification totals differ")
    return {"source_voice_instances": 12, "inputs": 66, "clear_recoveries": gain_total,
            "ordinary_stem_observations": 24, "diagnostic_gate_passed": all(gate.values())}


def self_test(data: dict) -> int:
    mutations = [
        lambda x: x["scope"].__setitem__("oracle_component_selection", False),
        lambda x: x["counts"]["stock_transfer"].__setitem__("voices", 9),
        lambda x: x["counts"]["stock_transfer"].__setitem__("ordinary_stem_observations", 8),
        lambda x: x["summary"].__setitem__("clear_recoveries", 1),
        lambda x: x["admission_detail"]["phoneme_transfer"].__setitem__("oracle_admitted_geometry_at_edit", 5),
        lambda x: x["paired_context_summary"].__setitem__("fully_admitted", 7),
        lambda x: x.__setitem__("production_activation_supported", True),
        lambda x: x.__setitem__("source_identity", "authored forbidden field"),
    ]
    for mutate in mutations:
        broken = copy.deepcopy(data)
        mutate(broken)
        try:
            validate(broken)
        except (KeyError, ValueError):
            continue
        raise ValueError("An authored corruption escaped aggregate validation")
    return len(mutations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    body = args.receipt.read_bytes()
    require(hashlib.sha256(body).hexdigest() == PIN, "The named aggregate commitment differs")
    data = json.loads(body)
    result = validate(data)
    if args.self_test:
        result["authored_corruptions_rejected"] = self_test(data)
    print(json.dumps(result, sort_keys=True))
    print("This replays aggregate arithmetic. Exact native/model replay needs the retained private source custody.")


if __name__ == "__main__":
    main()
