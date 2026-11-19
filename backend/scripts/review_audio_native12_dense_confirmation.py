"""Recount the public grouped independent twelve-second reference confirmation.

This validates arithmetic and the prospective error budget. Reader-component
counts and native replay facts are audited aggregate inputs. Individual case
reproduction requires the retained private source pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

EXPECTED_SHA256 = "e35c4cce9733acc7378d394e7b3e4e2dff9ab012afdf342cc9e05c54a633e6a0"
CONDITIONS = ("clean", "washing", "river", "restaurant")
SYNTHETIC = ("edited_piper", "edited_kitten")
GENUINE = ("same_reader_splice", "different_reader_splice")
CONTROLS = ("untouched", "roundtrip_piper", "roundtrip_kitten", "whole_gain", "whole_lowpass", "whole_high_shelf")
WRONG = ("wrong_reader", "shifted_earlier", "half_matched_earlier", "clock_faster", "clock_slower")
GAINS = tuple(f"{level}/{family}" for level in ("earlier_gain_half", "earlier_gain_high") for family in ("piper", "kitten"))
STATES = {"local_change", "no_corroborated_local_change", "reference_match_not_supported",
          "consistent_frequency_balance_change", "local_support_not_available"}


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, maximum=16):
    return type(value) is int and 0 <= value <= maximum


def review(receipt):
    require(set(receipt) == {"schema", "scope", "method", "error_budget", "counts", "decision", "native_review",
            "sources", "commitments", "arithmetic_clarification", "privacy", "limitations"}, "Aggregate fields differ")
    require(receipt["schema"] == "chipmunk.audio-native12-dense-confirmation-aggregate/1", "Schema differs")
    scope = receipt["scope"]
    require(scope == {
        "prospective_confirmation": True, "host_readers_disjoint_from_recorded_prior_roles": True,
        "unrestricted_corpus_blind_validation": False, "requires_supplied_clean_earlier_copy": True,
        "standalone_voice_check_remeasured": False, "new_detector_calls": 0, "generator_calls": 32,
        "authorship_confidence_weight": 0, "declared_host_readers": 8, "hosts": 16, "hosts_per_reader": 2,
        "connected_reader_components": 4, "generator_families": 2,
        "unique_synthetic_targets_repeated_across_conditions": 32, "background_captures": 3,
        "conditions": list(CONDITIONS), "pairs": 1264, "sample_rate": 16000, "samples_per_copy": 192000,
        "native_source_admission_attempts": 19, "qualified_source_records": 16,
    }, "Source or privacy scope differs")
    require(receipt["method"] == {"point": .31758884, "six_second_starts": [0, 19200, 38400, 57600, 76800, 96000],
            "neighbor_separation": 19200, "maximum_start_difference": 5760, "current_result_first": True,
            "reference_and_frequency_guards_unchanged": True, "model_calls": 0, "ai_confidence_weight": 0},
            "Frozen measurement differs")
    require(receipt["privacy"] == dict.fromkeys(("raw_audio", "individual_decisions", "per_case_scores",
            "reader_identifiers", "transcripts", "local_paths", "individual_edit_positions"), False),
            "Publication privacy scope differs")
    counts = receipt["counts"]
    require(set(counts) == set(CONDITIONS), "Condition roster differs")
    summaries, additions, extra_controls, extra_misplaced = {}, {}, {}, {}
    total, wrong_flags, background_flags = 0, 0, 0
    family_additions = dict.fromkeys(("piper", "kitten"), 0)
    for condition in CONDITIONS:
        roles = counts[condition]
        expected = {*SYNTHETIC, *GENUINE, *CONTROLS, *WRONG, *GAINS}
        if condition != "clean":
            expected.add("background_only")
        require(set(roles) == expected, "Condition role roster differs")
        for role, row in roles.items():
            require(set(row) == {"total", "baseline", "candidate"} and row["total"] == 16,
                    "Role denominator differs")
            total += row["total"]
            for arm in ("baseline", "candidate"):
                value = row[arm]
                require(set(value) == {"flags", "target_locations", "unassessed_reference",
                        "unassessed_local_support", "states"}, "Grouped outcome fields differ")
                require(all(integer(value[key]) for key in ("flags", "target_locations", "unassessed_reference",
                        "unassessed_local_support")), "Invalid grouped counts")
                require(value["target_locations"] <= value["flags"], "Locations exceed flags")
                states = value["states"]
                require(set(states) <= STATES and all(integer(v) and v > 0 for v in states.values())
                        and sum(states.values()) == 16, "Coverage denominator differs")
                require(value["flags"] == states.get("local_change", 0)
                        and value["unassessed_reference"] == states.get("reference_match_not_supported", 0)
                        and value["unassessed_local_support"] == states.get("local_support_not_available", 0),
                        "Coverage states and counts disagree")
            before, after = row["baseline"], row["candidate"]
            require(before["flags"] <= after["flags"] and before["target_locations"] <= after["target_locations"],
                    "An existing result disappeared")
            require(after["flags"] - before["flags"] >= after["target_locations"] - before["target_locations"],
                    "An existing off-target range changed position")
        summaries[condition] = {
            arm: {"synthetic_locations": sum(roles[r][arm]["target_locations"] for r in SYNTHETIC),
                  "genuine_splice_locations": sum(roles[r][arm]["target_locations"] for r in GENUINE),
                  "ordinary_control_flags": sum(roles[r][arm]["flags"] for r in CONTROLS)}
            for arm in ("baseline", "candidate")
        }
        additions[condition] = sum(roles[r]["candidate"]["target_locations"]
                                  - roles[r]["baseline"]["target_locations"] for r in SYNTHETIC)
        extra_controls[condition] = sum(roles[r]["candidate"]["flags"] - roles[r]["baseline"]["flags"] for r in CONTROLS)
        extra_misplaced[condition] = sum((roles[r]["candidate"]["flags"] - roles[r]["candidate"]["target_locations"])
                                        - (roles[r]["baseline"]["flags"] - roles[r]["baseline"]["target_locations"])
                                        for r in SYNTHETIC)
        wrong_flags += sum(roles[r]["candidate"]["flags"] for r in WRONG)
        background_flags += roles.get("background_only", {}).get("candidate", {}).get("flags", 0)
        if condition != "clean":
            for family in family_additions:
                cell = roles["edited_" + family]
                family_additions[family] += cell["candidate"]["target_locations"] - cell["baseline"]["target_locations"]
    require(total == 1264, "Overall denominator differs")
    decision, gate = receipt["decision"], receipt["error_budget"]
    require(set(decision) == {"new_noisy_synthetic_locations", "new_connected_components", "new_families",
            "new_control_rows_and_components", "new_misplaced_synthetic_rows", "wrong_reference_or_clock_flags",
            "background_only_flags", "gate_checks", "passed"}, "Decision fields differ")
    require(gate == {
        "minimum_added_noisy_synthetic_locations": 4, "minimum_connected_components": 2,
        "required_families": ["piper", "kitten"], "maximum_new_ordinary_control_rows_per_condition": 4,
        "maximum_new_ordinary_reader_components_per_condition": 1,
        "maximum_new_misplaced_synthetic_rows_per_condition": 1,
        "maximum_wrong_reference_or_clock_flags": 0, "maximum_background_only_flags": 0,
        "preserve_all_baseline_ranges": True,
        "scope": "Only principal equal-level reference cases count toward the gain gate. Earlier-gain variants are a separately reported robustness condition. No post-result source/control replacement or point fitting.",
    }, "Prospective error budget differs")
    new_locations = sum(v for c, v in additions.items() if c != "clean")
    families = sorted(f for f, count in family_additions.items() if count)
    require(decision["new_noisy_synthetic_locations"] == new_locations and decision["new_families"] == families
            and integer(decision["new_connected_components"], 4), "Gain or component totals differ")
    controls, misplaced = decision["new_control_rows_and_components"], decision["new_misplaced_synthetic_rows"]
    require(set(controls) == set(misplaced) == set(CONDITIONS), "Component condition roster differs")
    for c in CONDITIONS:
        require(set(controls[c]) == {"rows", "components", "hosts"}
                and controls[c]["rows"] == extra_controls[c]
                and integer(controls[c]["components"], min(4, extra_controls[c]))
                and integer(controls[c]["hosts"], min(16, extra_controls[c])), "Control increment differs")
        require(set(misplaced[c]) == {"rows", "components"} and misplaced[c]["rows"] == extra_misplaced[c]
                and integer(misplaced[c]["components"], min(4, extra_misplaced[c])), "Misplaced increment differs")
    require(decision["wrong_reference_or_clock_flags"] == wrong_flags
            and decision["background_only_flags"] == background_flags, "Negative control totals differ")
    checks = {"location_gain": new_locations >= 4, "component_gain": decision["new_connected_components"] >= 2,
              "family_gain": set(families) == {"piper", "kitten"},
              "ordinary_rows": all(v <= 4 for v in extra_controls.values()),
              "ordinary_components": all(v["components"] <= 1 for v in controls.values()),
              "misplaced_synthetic": all(v <= 1 for v in extra_misplaced.values()),
              "wrong_reference_and_clock": wrong_flags == 0, "background_only": background_flags == 0}
    require(decision["gate_checks"] == checks and type(decision["passed"]) is bool
            and decision["passed"] == all(checks.values()), "Prospective gate does not recount")
    require(receipt["native_review"] == {"passed": True, "source_pairs_reconstructed": 1264,
            "source_clocks_checked": 2528, "native_synthesis_phrases_and_fragments_replayed": 32,
            "dense_evidence_windows_recomputed": 2838, "original_source_alternate_decodes": 16},
            "Recorded native proof differs")
    require(receipt["arithmetic_clarification"] == {"required_family_collection": "unordered membership",
            "identified_before_closed_results": True, "measured_decisions_or_inputs_changed": False,
            "original_result_preserved": True}, "Recorded arithmetic clarification differs")
    require(set(receipt["commitments"]) == {"protocol", "native_intake", "independent_source_review", "rendered_inputs",
            "rendered_truth", "independent_render_review", "score_lock", "measurements", "original_result",
            "canonical_result", "independent_outcome_review", "preclosure_family_membership_clarification"}
            and all(isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v)
            for v in receipt["commitments"].values()), "Native custody commitments differ")
    require(len(receipt["sources"]) == 4, "Publisher source roster differs")
    require({s["role"] for s in receipt["sources"]} == {"foreground", "appliance_background", "water_background",
            "restaurant_background"}, "Publisher source roles differ")
    for source in receipt["sources"]:
        require(set(source) <= {"role", "publisher", "license", "terms_url", "object_url", "object_sha256",
                "object_bytes", "published_object_md5"} and source["publisher"] in {"OpenSLR LibriSpeech", "DEMAND"},
                "Unexpected publisher fields")
        require(source["license"] in {"CC-BY-4.0", "CC-BY-SA-3.0"}, "Source rights differ")
        for key, value in source.items():
            if key.endswith("_url"):
                link = urlsplit(value)
                require(link.scheme == "https" and link.hostname in {"www.openslr.org", "huggingface.co", "zenodo.org"}
                        and link.username is None and link.password is None and not link.query,
                        "Unexpected publisher citation")
    require(type(receipt["limitations"]) is list and len(receipt["limitations"]) == 6, "Limitations differ")
    return {"pairs": total, "summary": summaries, "new_noisy_synthetic_locations": new_locations,
            "new_ordinary_control_rows": extra_controls, "new_misplaced_synthetic_rows": extra_misplaced,
            "prospective_gate_passed": decision["passed"], "individual_source_replay_requires_private_pairs": True}


def unique(fields):
    result = {}
    for name, value in fields:
        require(name not in result, "Duplicate JSON field")
        result[name] = value
    return result


def load(path):
    body = path.read_bytes()
    require(len(body) < 128 * 1024 and hashlib.sha256(body).hexdigest() == EXPECTED_SHA256,
            "The pinned public aggregate changed")
    value = json.loads(body, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    return review(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(load(args.receipt), sort_keys=True))
