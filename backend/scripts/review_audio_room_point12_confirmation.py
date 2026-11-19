"""Public aggregate arithmetic review for the frozen twelve-second point trial."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SHA256 = "bcb65ce1e9c5613d2e5e5ef89c4bd54f98526ddbbe229c09e1ce665fc741f968"


def gates(paired, noise):
    clean = paired["clean"]
    clean_gate = {
        "four_new_clean_events": clean["new_useful_events"] >= 4,
        "three_clean_reader_groups": clean["new_useful_reader_groups"] >= 3,
        "both_clean_families": clean["new_useful_families"] == 2,
        "clean_ordinary_files": clean["new_ordinary_files"] <= 2,
        "clean_ordinary_groups": clean["new_ordinary_reader_groups"] <= 2,
        "clean_wrong_only": clean["new_wrong_only_synthetic_files"] <= 2,
        "clean_wrong_seconds": clean["added_off_target_seconds"] <= 4,
        "clean_ordinary_seconds": clean["added_ordinary_seconds"] <= 3,
        "clean_control_collisions": clean["matched_ordinary_collision_events"] <= 1,
    }
    broad = {
        "clean_gate": all(clean_gate.values()),
        "two_noisy_events": noise["new_distinct_physical_events"] >= 2,
        "two_noisy_reader_groups": noise["new_reader_groups"] >= 2,
        "both_noise_conditions": all(
            paired[c]["new_useful_events"] > 0 for c in ("traffic", "competing_speech")
        ),
    }
    for condition in ("traffic", "competing_speech"):
        p = paired[condition]
        for key, yes in {
            "ordinary_files": p["new_ordinary_files"] <= 2,
            "ordinary_groups": p["new_ordinary_reader_groups"] <= 2,
            "wrong_only": p["new_wrong_only_synthetic_files"] <= 2,
            "wrong_seconds": p["added_off_target_seconds"] <= 4,
            "ordinary_seconds": p["added_ordinary_seconds"] <= 3,
            "control_collisions": p["matched_ordinary_collision_events"] <= 1,
            "isolated_backgrounds": p["new_isolated_background_flags"] == 0,
        }.items():
            broad[condition + "_" + key] = yes
    return clean_gate, broad


def validate(data):
    assert data["schema"] == "chipmunk.room-point12-confirmation-aggregate/1" and data["closed"] is True
    s = data["scope"]
    assert (
        s["deliveries"],
        s["synthetic_deliveries"],
        s["ordinary_deliveries"],
        s["background_deliveries"],
    ) == (184, 48, 120, 16)
    assert (
        s["deliveries"] == s["synthetic_deliveries"] + s["ordinary_deliveries"] + s["background_deliveries"]
    )
    assert s["physical_synthetic_events"] == 16 and s["reader_donor_groups"] == 8
    assert s["host_readers"] == s["genuine_donor_readers"] == 8 and s["background_captures"] == 4
    assert s["point_reader_overlap"] == s["prior_room_trial_reader_overlap"] == 0
    assert (
        s["native_samples"] == 192000 and s["native_rate_hz"] == 16000 and s["native_delivery_seconds"] == 12
    )
    assert s["new_synthesis_calls"] == s["new_downloads"] == 0
    counts = data["counts"]
    for condition in ("clean", "traffic", "competing_speech"):
        for arm in ("incumbent", "point_only"):
            c = counts[condition][arm]
            assert all(type(v) is int and v >= 0 for v in c.values())
            assert c["synthetic_total"] == c["useful"] + c["missed"] == 16
            assert c["ordinary_total"] == 40
            assert c.get("background_total", 0) == (0 if condition == "clean" else 8)
            for kind in ("synthetic", "ordinary", "background"):
                assert c.get(kind + "_flags", 0) <= c.get(kind + "_total", 0)
                assert c.get(kind + "_unassessed", 0) <= c.get(kind + "_total", 0)
            assert c["wrong_only"] <= c["missed"] and c["any_wrong_files"] <= c["synthetic_flags"]
            assert c["useful"] + c["wrong_only"] == c["synthetic_flags"]
        p = data["paired"][condition]
        old, new = (counts[condition][a] for a in ("incumbent", "point_only"))
        assert new["useful"] - old["useful"] == p["new_useful_events"]
        assert new["ordinary_flags"] - old["ordinary_flags"] == p["new_ordinary_files"]
        assert (
            new.get("background_flags", 0) - old.get("background_flags", 0)
            == p["new_isolated_background_flags"]
        )
        assert 0 <= p["new_useful_reader_groups"] <= min(8, p["new_useful_events"])
        assert 0 <= p["new_useful_families"] <= min(2, p["new_useful_events"])
        assert p["new_ordinary_reader_groups"] <= min(8, p["new_ordinary_files"])
        assert 0 <= p["matched_ordinary_collision_events"] <= p["new_useful_events"]
        assert new["wrong_intervals"] - old["wrong_intervals"] == p["added_off_target_intervals"]
        assert (new["wrong_union_samples"] - old["wrong_union_samples"]) / 16000 == p[
            "added_off_target_seconds"
        ]
    clean_gate, broad_gate = gates(data["paired"], data["noise_gain_summary"])
    assert clean_gate == data["clean_gate"] and broad_gate == data["broad_gate"]
    assert all(clean_gate.values()) == data["advance_clean_native_integration_trial"]
    assert all(broad_gate.values()) == data["advance_broad_noise_followup"]
    assert data["product_activation_supported"] is False and data["speaker_attribution_supported"] is False
    proof = data["verification"]
    assert proof["independent_native_and_outcome_replays_passed"] is True
    assert proof["native_inputs"] == proof["independent_wav_decodes"] == 184
    assert proof["native_sample_values_replayed"] == 184 * 192000
    assert proof["brief_source_window_bindings"] == 184 * 23
    assert proof["six_second_excerpt_bindings"] == 184 * 4
    assert proof["scheduled_source_window_bindings"] == 184 * 43
    assert proof["authenticity_calls"] == proof["distinct_authenticity_inputs"] + 1
    assert proof["exact_initial_repeat"] is True and proof["authored_controls"] == 8
    return {
        "useful": {c: [counts[c][a]["useful"] for a in ("incumbent", "point_only")] for c in counts},
        "ordinary": {
            c: [counts[c][a]["ordinary_flags"] for a in ("incumbent", "point_only")] for c in counts
        },
        "clean_advancement": all(clean_gate.values()),
        "broad_advancement": all(broad_gate.values()),
    }


def self_test(data):
    changes = (
        lambda d: d["scope"].update(physical_synthetic_events=48),
        lambda d: d["scope"].update(point_reader_overlap=1),
        lambda d: d["counts"]["clean"]["point_only"].update(useful=17),
        lambda d: d["paired"]["clean"].update(new_ordinary_files=100),
        lambda d: d["verification"].update(scheduled_source_window_bindings=1),
        lambda d: d.update(product_activation_supported=True),
        lambda d: d.update(
            advance_clean_native_integration_trial=not d["advance_clean_native_integration_trial"]
        ),
        lambda d: d.update(advance_broad_noise_followup=not d["advance_broad_noise_followup"]),
    )
    rejected = 0
    for mutate in changes:
        candidate = copy.deepcopy(data)
        mutate(candidate)
        try:
            validate(candidate)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError("An authored alteration was accepted")
    assert rejected == 8
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
    print("This reviews aggregate arithmetic. Exact source/model replay needs retained private custody.")


if __name__ == "__main__":
    main()
