#!/usr/bin/env python3
"""Recount the named anonymous aggregate; original-audio/model replay stays private."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

PIN = "4dad6821a472d7506d4c749a7a9922f79f2d39278bc9d384882ce3efd6d45091"
TOP = {
    "schema", "closed", "experimental_scope", "preparation", "source_counts",
    "exposure", "point_selection", "counts", "paired_ordering_at_supplied_clock",
    "new_noisy_source_groups", "new_noisy_generator_families",
    "matched_ordinary_collisions", "gate", "proceed_to_separate_free_search",
    "production_change_supported", "verification", "model_forwards",
    "elapsed_seconds", "whole_private_protocol_commitment",
    "whole_private_numeric_closure_commitment", "publisher_sources", "privacy",
    "new_ordinary_groups_by_condition",
}
CONDITIONS = {
    "development": ("clean", "washing", "river", "restaurant"),
    "transfer": ("clean", "traffic", "competing_speech"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def count(value: object, lower: int = 0, upper: int = 10_000) -> int:
    require(type(value) is int and lower <= value <= upper, "Invalid aggregate count")
    return value


def validate(data: dict) -> dict:
    require(isinstance(data, dict) and set(data) == TOP, "Unexpected public aggregate fields")
    require(data["schema"] == "chipmunk.spectra-original-one-second-aggregate/1"
            and data["closed"] is True, "The study is not a closed named aggregate")
    require(data["preparation"] == {
        "source_rate_hz": 16000, "source_extent_samples": 192000,
        "native_excerpt_samples": 16000, "graph_input_samples": 64600,
        "pre_emphasis": .97, "whole_excerpt_repeats": 5,
        "source_audio_modified_for_denoising": False, "new_speech_gate": False,
    }, "Original one-second input scope changed")
    require(data["source_counts"] == {
        "development_deliveries": 496, "development_foreground_host_recordings": 16,
        "development_connected_reader_donor_components": 4,
        "development_physical_background_sessions": 3,
        "transfer_deliveries": 368, "transfer_host_recordings": 16,
        "transfer_host_donor_groups": 8, "transfer_physical_background_captures": 4,
        "generator_families": 2, "delivered_one_second_roles_disjoint": True,
    }, "Source-group or delivery scope changed")
    require(set(data["counts"]) == set(CONDITIONS)
            and set(data["paired_ordering_at_supplied_clock"]) == set(CONDITIONS),
            "Missing study role")
    paired_total = 0
    for role, conditions in CONDITIONS.items():
        cells = data["counts"][role]
        ordering = data["paired_ordering_at_supplied_clock"][role]
        require(set(cells) == set(conditions) and set(ordering) == set(conditions),
                "Missing or added noise condition")
        total = 0
        for condition in conditions:
            row = cells[condition]
            kinds = ("ordinary", "synthetic") if condition == "clean" else (
                "ordinary", "synthetic", "background")
            expected = set()
            for kind in kinds:
                expected.update((kind + "_total", kind + "_candidate_flagged"))
                total += count(row[kind + "_total"])
                require(row[kind + "_total"] == {
                    "ordinary": 80, "synthetic": 32, "background": 16,
                }[kind], "A denominator changed")
                flagged = count(row[kind + "_candidate_flagged"])
                require(flagged <= row[kind + "_total"], "Candidate flags exceed deliveries")
                if role == "transfer":
                    expected.update((kind + "_baseline_flagged", kind + "_combined_flagged"))
                    base = count(row[kind + "_baseline_flagged"])
                    combined = count(row[kind + "_combined_flagged"])
                    require(base <= combined <= row[kind + "_total"], "Original flags were lost")
                    require(max(base, flagged) <= combined <= base + flagged,
                            "Candidate and baseline intersection is impossible")
                    if kind != "synthetic":
                        key = "new_" + kind + "_flags"
                        expected.add(key)
                        require(count(row[key]) == combined - base,
                                "Added control flags do not match the preserved baseline")
            if role == "transfer":
                expected.update(("baseline_useful", "combined_useful_at_supplied_clock",
                                 "missed", "baseline_wrong_only", "combined_wrong_only",
                                 "added_useful_at_supplied_clock"))
                baseline = count(row["baseline_useful"])
                combined = count(row["combined_useful_at_supplied_clock"])
                require(baseline <= combined <= 32, "A useful result was lost")
                require(count(row["missed"]) == 32 - combined, "Misses do not match useful results")
                require(count(row["added_useful_at_supplied_clock"]) == combined - baseline,
                        "Added useful counts differ")
                require(count(row["baseline_wrong_only"]) == row["synthetic_baseline_flagged"] - baseline,
                        "Baseline off-target count differs")
                require(count(row["combined_wrong_only"]) == row["synthetic_combined_flagged"] - combined,
                        "Combined off-target count differs")
                candidate = row["synthetic_candidate_flagged"]
                require(max(baseline, candidate) <= combined <= baseline + candidate,
                        "Known-clock candidate and baseline overlap is impossible")
            require(set(row) == expected, "Unexpected scalar condition fields")
            order = ordering[condition]
            require(set(order) == {"pairs", "wins", "ties", "losses", "synthetic_files",
                                   "above_all_five_controls"}, "Unexpected paired fields")
            require(count(order["synthetic_files"]) == 32 and count(order["pairs"]) == 160,
                    "Paired denominator differs")
            wins, ties, losses = (count(order[key]) for key in ("wins", "ties", "losses"))
            require(wins + ties + losses == 160, "Paired outcomes do not sum")
            require(count(order["above_all_five_controls"]) <= 32
                    and 5 * order["above_all_five_controls"] <= wins,
                    "All-five comparison count is impossible")
            paired_total += 160
        require(total == {"development": 496, "transfer": 368}[role],
                "Role deliveries do not sum")
    cells = data["counts"]["transfer"]
    groups = data["new_ordinary_groups_by_condition"]
    require(set(groups) == set(CONDITIONS["transfer"]), "Missing ordinary-group condition")
    for condition, value in groups.items():
        require(count(value, upper=8) <= cells[condition]["new_ordinary_flags"],
                "Newly flagged groups exceed newly flagged files")
    new_groups = count(data["new_noisy_source_groups"], upper=8)
    families = count(data["new_noisy_generator_families"], upper=2)
    additions = sum(cells[c]["added_useful_at_supplied_clock"]
                    for c in ("traffic", "competing_speech"))
    require(new_groups <= additions and families <= additions,
            "New sources exceed new known-clock gains")
    collisions = count(data["matched_ordinary_collisions"])
    require(collisions <= additions, "Paired collisions exceed new noisy gains")
    gate = {
        "three_added_noisy_at_supplied_clock": additions >= 3,
        "three_source_groups": new_groups >= 3,
        "both_generators": families == 2,
        "ordinary_files_within_budget": all(c["new_ordinary_flags"] <= 4 for c in cells.values()),
        "ordinary_groups_within_budget": all(v <= 2 for v in groups.values()),
        "zero_added_isolated": all(c.get("new_background_flags", 0) == 0 for c in cells.values()),
        "wrong_only_within_budget": all(
            cells[c]["combined_wrong_only"] - cells[c]["baseline_wrong_only"] <= 2
            for c in ("traffic", "competing_speech")),
        "zero_matched_ordinary_collisions": collisions == 0,
    }
    require(data["gate"] == gate and data["proceed_to_separate_free_search"] is all(gate.values()),
            "Continuation decision differs from the frozen gate")
    require(data["production_change_supported"] is False,
            "A supplied-clock experiment cannot authorize product activation")
    require(data["model_forwards"] == {
        "authored_compatibility": 2, "development": 497, "transfer": 369,
    }, "Actual graph forward counts differ")
    verify = data["verification"]
    require(verify["passed"] is True and verify["native_deliveries"] == 864
            and verify["graph_input_bindings"] == 864
            and verify["graph_samples_checked"] == 55_814_400
            and verify["decision_rows_recounted"] == 864
            and verify["paired_orderings_recounted"] == paired_total
            and verify["point_reconstructed_exact"] is True
            and verify["gate_reconstructed_exact"] is True,
            "Independent review counts differ")
    return {"delivery_requests": 864, "paired_comparisons": paired_total,
            "continuation_gate_passed": all(gate.values()),
            "production_change_supported": False}


def self_test(data: dict) -> int:
    mutations = [
        lambda x: x["counts"]["transfer"]["traffic"].__setitem__("missed", 31),
        lambda x: x["counts"]["development"]["clean"].__setitem__("ordinary_total", 81),
        lambda x: x["paired_ordering_at_supplied_clock"]["transfer"]["traffic"].__setitem__("wins", 78),
        lambda x: x["new_ordinary_groups_by_condition"].__setitem__("traffic", 3),
        lambda x: x.__setitem__("proceed_to_separate_free_search", True),
        lambda x: x.__setitem__("customer_path", "/private/source.wav"),
        lambda x: x.__setitem__("closed", False),
        lambda x: x["preparation"].__setitem__("native_excerpt_samples", 48000),
    ]
    for mutate in mutations:
        broken = copy.deepcopy(data)
        mutate(broken)
        try:
            validate(broken)
        except (ValueError, KeyError):
            continue
        raise ValueError("An authored corruption escaped aggregate verification")
    return len(mutations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    body = args.receipt.read_bytes()
    require(hashlib.sha256(body).hexdigest() == PIN, "The named aggregate commitment differs")
    data = json.loads(body)
    summary = validate(data)
    if args.self_test:
        summary["authored_corruptions_rejected"] = self_test(data)
    print(json.dumps(summary, sort_keys=True))
    print("This command checks anonymous aggregate arithmetic. It does not rerun native audio or model inference.")


if __name__ == "__main__":
    main()
