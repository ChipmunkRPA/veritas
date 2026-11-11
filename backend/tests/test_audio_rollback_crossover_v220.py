"""Exercise source and rank pairing without loading either classifier."""

import subprocess
import sys

import numpy as np
import pytest

from scripts import audio_rollback_crossover_v220 as study


def test_import_keeps_a_fresh_model_free_process():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from scripts import audio_rollback_crossover_v220 as v; v.PILOT.require_model_free()",
        ],
        check=True,
    )


def test_preserves_gate_window_and_planned_unavailable_slots(monkeypatch):
    pcm = np.zeros(40001, dtype=np.float32)
    calls = []
    monkeypatch.setattr(study, "waveform", lambda _: pcm)
    case = {"case": "fixture", "pair": "pair-00", "condition": "matched_clean", "origin": "genuine"}
    baseline = {**case, "route_status": "ok", "decision": "original", "raw": 0.2, "windows": [[0, len(pcm)]]}

    def score(values):
        assert values is pcm
        calls.append(len(values))
        return 0.75

    row = study.row_from(case, baseline, score) | {"rollback_forwards": 1}
    study.check_row(row, case, baseline)
    assert row["recorded_direction"] == "original"
    assert row["rollback_raw"] == 0.75
    with pytest.raises(ValueError, match="complete window"):
        study.row_from(case, baseline | {"windows": [[1, len(pcm)]]}, score)
    with pytest.raises(ValueError, match="paired"):
        study.check_row(row | {"rollback_forwards": 0}, case, baseline)

    rejected = baseline | {
        "route_status": "not_applicable",
        "decision": "unavailable",
        "raw": None,
        "windows": [],
    }
    unassessed = study.row_from(case, rejected, score) | {"rollback_forwards": 0}
    study.check_row(unassessed, case, rejected)
    assert unassessed["rollback_raw"] is None
    assert calls == [40001]


def observation(pair, origin, recorded, rollback):
    return {
        "pair": pair,
        "origin": origin,
        "route_status": "ok" if recorded is not None else "not_applicable",
        "recorded_direction": "inconclusive" if recorded is not None else "unavailable",
        "recorded_raw": recorded,
        "rollback_raw": rollback,
    }


def test_common_ranks_hold_ties_and_same_excerpt_pairs_separately():
    rows = [
        observation("pair-00", "genuine", 0.2, 0.1),
        observation("pair-01", "genuine", 0.6, 0.7),
        observation("pair-02", "genuine", None, None),
        observation("pair-00", "synthetic", 0.2, 0.8),
        observation("pair-01", "synthetic", 0.3, 0.7),
        observation("pair-02", "synthetic", 0.5, 0.5),
    ]
    result = study.summarize_condition(rows)
    assert result["planned"] == 6
    assert result["common_assessed"] == {"genuine": 2, "synthetic": 3}
    recorded = result["comparison"]["recorded_raw"]
    assert recorded["all_cross_class_rank"]["comparisons"] == 6
    assert recorded["same_pair_rank"]["comparisons"] == 2
    assert recorded["same_pair_rank"]["concordance"] == 0.25
    rollback = result["comparison"]["rollback_raw"]
    assert rollback["same_pair_rank"]["concordance"] == 0.75
    assert rollback["raw_fake_side_at_legacy_point"] == {"genuine": 1, "synthetic": 3}
    assert result["recorded_directions"]["genuine"]["unavailable"] == 1


def test_movements_compare_each_model_to_its_own_matched_clean_score():
    clean = [
        observation("pair-00", "genuine", 0.2, 0.3),
        observation("pair-00", "synthetic", 0.8, 0.7),
        observation("pair-01", "genuine", 0.1, 0.1),
    ]
    noisy = [
        observation("pair-00", "synthetic", 0.3, 0.7),
        observation("pair-01", "genuine", None, None),
        observation("pair-00", "genuine", 0.4, 0.1),
    ]
    result = study.paired_movements(clean, noisy)
    assert result["genuine"] == {
        "planned_pairs": 2,
        "both_assessed": 1,
        "score_movements": {"recorded_raw": {"up": 1}, "rollback_raw": {"down": 1}},
    }
    assert result["synthetic"]["score_movements"] == {"recorded_raw": {"down": 1}, "rollback_raw": {"tie": 1}}
