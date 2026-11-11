import copy

import numpy as np
import pytest
from scipy.special import expit

from scripts import audio_paired_transfer_v230 as transfer


def head():
    width = transfer.native.DIMENSIONS
    coefficients = np.zeros(width + 2)
    coefficients[0] = 0.5
    coefficients[1] = 0.25
    coefficients[-1] = 0.2
    return {
        "transform": {"center": [0] * width, "scale": [1] * width},
        "coefficients": coefficients.tolist(),
    }


def feature_packet():
    features = [0.25] * transfer.native.DIMENSIONS
    return {
        "wavlm": features,
        "wavlm_sha256": transfer.native.float_sha(features),
        "feature_frames": 149,
        "valid_feature_frames": 149,
        "input_pcm_sha256": "a" * 64,
        "pooled_state_sha256": ["b" * 64] * 3,
    }


def original_case(status="ok"):
    return {
        "case": "slot",
        "triplet": "triplet-00",
        "arm": "genuine",
        "condition": transfer.CLEAN,
        "status": "unavailable" if status == "preparation_unavailable" else "prepared",
        "samples": 48000,
    }, {
        "route_status": status,
        "decision": "inconclusive" if status == "ok" else "unavailable",
        "raw": 0.5 if status == "ok" else None,
        "english_reference_estimate_ai": 0.9 if status == "ok" else None,
        "windows": [[0, 48000]] if status == "ok" else [],
    }


def points():
    return {
        "current": {"original_below": 0.1, "ai_above": 0.9},
        transfer.METHOD: {"original_below": 0.2, "ai_above": 0.5},
    }


def test_fixed_head_projection_and_strict_english_cutpoints():
    case, baseline = original_case()
    model, packet = head(), feature_packet()
    row = transfer.outcome(case, baseline, packet, model, points())
    expected = expit(0.2 + 0.25 * 0.25 * np.sqrt(16 / transfer.native.DIMENSIONS))
    assert row["candidate_score"] == expected
    assert row["current_raw"] == 0.5
    assert row["recorded_direction"] == "inconclusive"
    assert row["current_english_point_direction"] == "inconclusive"
    assert row["candidate_english_point_direction"] == "ai"
    assert row["encoder_forwards"] == 1
    transfer.audit_observation(row, case, baseline, model, points())


@pytest.mark.parametrize("status", ["not_applicable", "preparation_unavailable"])
def test_preparation_and_gate_unavailable_stay_distinct(status):
    case, baseline = original_case(status)
    row = transfer.outcome(case, baseline, None, None, points())
    assert row["route_status"] == status
    assert row["current_raw"] is row["candidate_score"] is None
    assert row["recorded_direction"] == row["candidate_english_point_direction"] == "unavailable"
    assert row["encoder_forwards"] == 0
    with pytest.raises(ValueError, match="never acquire"):
        transfer.outcome(case, baseline, feature_packet(), None, points())


def test_guard_rejects_wrong_window_and_mutated_features():
    case, baseline = original_case()
    with pytest.raises(ValueError, match="whole-window"):
        transfer.admitted(case, {**baseline, "windows": [[160, 48000]]})
    model, packet = head(), feature_packet()
    row = transfer.outcome(case, baseline, packet, model, points())
    broken = copy.deepcopy(row)
    broken["native_features"]["wavlm"][0] += 1
    with pytest.raises(ValueError, match="feature packet"):
        transfer.audit_observation(broken, case, baseline, model, points())
    with pytest.raises(ValueError, match="calculation differs"):
        transfer.audit_observation({**row, "candidate_score": 1}, case, baseline, model, points())


def test_rank_within_triplets_and_pairwise_drift_use_only_common_assessed():
    def record(triplet, arm, condition, raw, candidate, status="ok"):
        direction = "inconclusive" if status == "ok" else "unavailable"
        return {
            "triplet": triplet,
            "arm": arm,
            "condition": condition,
            "route_status": status,
            "current_raw": raw,
            "candidate_score": candidate,
            "recorded_direction": direction,
            "current_english_point_direction": direction,
            "candidate_english_point_direction": direction,
        }

    clean, noise = transfer.CLEAN, transfer.NOISE[0]
    rows = [
        record("a", "genuine", clean, 0.4, 0.3),
        record("b", "genuine", clean, 0.9, 0.6),
        record("a", "alania", clean, 0.4, 0.9),
        record("b", "alania", clean, None, None, "not_applicable"),
        record("a", "drifting", clean, 0.2, 0.8),
        record("a", "alania", noise, 0.7, 0.8),
        record("b", "alania", noise, 0.8, 0.85),
    ]
    summary = transfer.summarize(rows)
    rank = summary["raw_ranking"][clean]
    assert rank["alania"]["current_raw"]["all"] == {
        "comparisons": 2,
        "wins": 0,
        "ties": 1,
        "losses": 1,
        "concordance": 0.25,
    }
    assert rank["alania"]["current_raw"]["within_triplet"]["concordance"] == 0.5
    assert rank["drifting"]["candidate_score"]["all"]["wins"] == 2
    assert rank["all_synthetic"]["current_raw"]["all"]["comparisons"] == 4
    assert summary["coverage_and_directions"]["alania"][clean]["speech_gate_rejected"] == 1
    movement = summary["paired_clean_to_noise_score_drift"]["alania"][noise]
    assert movement["current_raw"]["planned_pairs"] == 2
    assert movement["current_raw"]["both_assessed"] == 1
    assert movement["current_raw"]["up"] == 1
    assert movement["candidate_score"]["down"] == 1
    assert movement["candidate_score"]["mean_absolute_score_change"] == pytest.approx(0.1)


def test_import_remains_model_free():
    transfer.native.PILOT.require_model_free()
