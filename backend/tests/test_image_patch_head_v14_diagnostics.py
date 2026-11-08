"""Diagnostic completion cannot waive a frozen numerical or quality failure."""
import numpy as np

from scripts import diagnose_image_patch_head_v14 as diagnostics


def test_float32_accumulation_failure_remains_a_failure_without_rewriting_inputs():
    features = np.full((1024, 2), 16.1, np.float32)
    average = features.mean(0, dtype=np.float32)
    before = average.copy()
    result, accurate = diagnostics.assess_mean(features, average)
    assert result['maximum_difference'] > 1e-5
    assert not result['within_original_1e_5_bound']
    assert np.array_equal(average, before)
    assert np.array_equal(accurate, np.full(2, 16.1, np.float32))


def test_diagnostic_receipt_never_claims_original_audit_or_serving_admission():
    for count in (0, 1, 1344):
        record = diagnostics.admission(count)
        assert record['original_strict_audit_passed'] is record['serving_admitted'] is record['used_in_confidence'] is False
        assert record['original_mean_bound_failures'] == count
