"""Check actual mask combinations rather than oracle per-document winners."""

import numpy as np
import pytest

from scripts.review_multimodal_tradeoffs_v44 import combine_masks, complete_diagnostic_receipt
from scripts.writing_span_geometry_v7 import metrics


def test_either_method_useful_does_not_establish_a_useful_union():
    text = "a" * 100
    labels = [1] * 10 + [0] * 90
    precise = np.asarray([True] * 10 + [False] * 90)
    broad = np.ones(100, dtype=bool)
    assert metrics(text, labels, precise)["character_iou"] == 1
    assert metrics(text, labels, combine_masks([precise, broad], "union"))["character_iou"] == 0.1


def test_union_and_intersection_preserve_exact_source_positions():
    a = np.asarray([True, False, False, True])
    b = np.asarray([False, True, False, True])
    assert combine_masks([a, b], "union").tolist() == [True, True, False, True]
    assert combine_masks([a, b], "intersection").tolist() == [False, False, False, True]
    assert a.tolist() == [True, False, False, True]


@pytest.mark.parametrize("other", [None, np.zeros(3, bool), np.zeros(4, int), np.zeros((2, 2), bool)])
def test_unknown_or_misaligned_mask_does_not_become_clear(other):
    with pytest.raises(ValueError, match="source-aligned"):
        combine_masks([np.zeros(4, bool), other], "union")


def test_combination_rule_must_be_explicit():
    with pytest.raises(ValueError):
        combine_masks([np.zeros(4, bool)], "union")
    with pytest.raises(ValueError):
        combine_masks([np.zeros(4, bool), np.zeros(4, bool)], "best_from_truth")


def test_actual_phase_completion_schema_is_distinct_from_per_row_status():
    complete_diagnostic_receipt({"status": "complete", "role": "diagnostic", "documents": 288, "freeze_sha256": "bound"}, "bound")


@pytest.mark.parametrize("field,value", [("status", "failed"), ("status", "ok"), ("role", "fit"), ("documents", 287), ("freeze_sha256", "changed")])
def test_incomplete_or_unbound_phase_cannot_authorize_combination(field, value):
    receipt = {"status": "complete", "role": "diagnostic", "documents": 288, "freeze_sha256": "bound"}
    receipt[field] = value
    with pytest.raises(ValueError, match="bound observed"):
        complete_diagnostic_receipt(receipt, "bound")
