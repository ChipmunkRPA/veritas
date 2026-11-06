from __future__ import annotations

import pytest

from scripts.review_tiled_splice import fit_gate, summary


def originals():
    return [({"status": "ok", "tiles": [1], "max_active_fraction": (i+1)/100},
             {"predicted_positive_pixels": 0}) for i in range(24)]


def test_gate_uses_only_additional_original_alert_budget_and_preserves_current_flags():
    rows = originals()
    rows[-1][1]["predicted_positive_pixels"] = 5  # Already flagged by current check.
    result = fit_gate(rows)
    assert result["current_authentic_flags"] == 1 and result["additional_authentic_budget"] == 1
    assert .22 < result["minimum_active_fraction"] < .23
    assert result["enabled"]


def test_gate_can_disable_uninformative_all_positive_tiles_without_clipping_threshold():
    rows = originals()
    for row, _ in rows:
        row["max_active_fraction"] = 1.
    result = fit_gate(rows)
    assert not result["enabled"]


def test_incomplete_reference_cannot_fit_a_tile_gate():
    rows = originals()
    rows[0] = (rows[0][0], None)
    with pytest.raises(ValueError, match="complete"):
        fit_gate(rows)


def test_summary_keeps_unassessed_edits_and_failed_tiles_in_the_record():
    positive = {"intersection": 2, "predicted_positive_pixels": 2, "iou": .5, "pixel_f1": .6}
    negative = {"intersection": 0, "predicted_positive_pixels": 0, "iou": 0., "pixel_f1": 0.}
    rows = [{"label": 0, "baseline": negative, "candidate": negative, "tile_status": "ok", "scheduled_tiles": 4},
            {"label": 1, "baseline": negative, "candidate": positive, "tile_status": "ok", "scheduled_tiles": 4},
            {"label": 1, "baseline": None, "candidate": None, "tile_status": "failed", "scheduled_tiles": 0}]
    result = summary(rows)
    assert result["candidate"]["edits"] == 2
    assert result["candidate"]["mean_edit_iou"] == .25
    assert result["candidate"]["unassessed"] == result["tile_failure_count"] == 1
    assert result["meets_count_tradeoff"]
    assert "qualifies_for_scoped_local_adoption" not in result  # A development summary alone cannot authorize adoption.
