from __future__ import annotations

import pytest

from scripts.summarize_iml_vit import paired, summarize


def example(ident, label, *, status="ok", overlap=True, response=.2):
    pixels = 10 if overlap else 0
    return {"id": ident, "source_id": ident, "input_sha256": ident, "label": label,
            "status": status, "runtime_s": 1., "active_fraction": response,
            "metrics": {"intersection": pixels if label else 0, "predicted_positive_pixels": pixels,
                        "iou": .5 if label and overlap else 0., "pixel_f1": .6 if label and overlap else 0.},
            "ungated": {"intersection": pixels if label else 0, "predicted_positive_pixels": pixels,
                        "iou": .5 if label and overlap else 0., "pixel_f1": .6 if label and overlap else 0.}}


def test_missing_and_out_of_scope_edits_count_as_misses_not_zeroed_denominators():
    rows = [example("a", 0), example("b", 1), example("c", 1, status="failed"),
            example("d", 1, status="not_applicable")]
    actual = summarize(rows)
    assert actual["edits"] == 3 and actual["assessed"] == 2
    assert actual["failures"] == actual["not_applicable"] == 1
    assert actual["missed_edits"] == 2 and actual["edits_with_overlap"] == 1
    assert actual["mean_edit_iou"] == pytest.approx(.5/3)


def test_frozen_candidate_gate_applies_to_false_highlights_and_edit_overlap():
    rows = [example("a", 0, response=.02), example("b", 1, response=.02)]
    actual = summarize(rows, threshold=.03)
    assert actual["originals_with_regions"] == actual["edits_with_overlap"] == 0
    assert actual["missed_edits"] == 1 and actual["mean_edit_iou"] == 0


def test_same_input_bytes_are_required_for_paired_comparison():
    first = [example("a", 0), example("b", 1)]
    second = [example("a", 0), example("b", 1, response=0.)]
    result = paired(first, second, .1)
    assert result["candidate_minus_current"] == -.5
    assert result["source_bootstrap_95_percent"] == [-.5, -.5]
    second[-1]["input_sha256"] = "changed"
    with pytest.raises(ValueError, match="different input"):
        paired(first, second, .1)
