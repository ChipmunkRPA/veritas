"""Source-relative truth rejects unexplained changes and keeps controls non-AI."""

import numpy as np
import pytest

from scripts.image_pair_truth_v21 import qualify


def example():
    original = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    edited = original.copy()
    edited[4:8, 3:9] ^= 31
    alpha = np.zeros((12, 12), np.uint8)
    alpha[3:9, 2:10] = 255
    return original, edited, alpha


def test_exact_changed_pixels_are_not_the_entire_declared_selection():
    changed, proof = qualify(*example())
    assert changed.sum() == proof["changed_pixels"] == 24
    assert proof["support_pixels"] == 48 and proof["outside_support_changes"] == 0
    assert set(proof["ordinary_controls"]) == {"tone", "blur"}
    assert all(r["ai_generation_used"] is False for r in proof["ordinary_controls"].values())


def test_one_change_outside_alpha_prevents_positive_pair_admission():
    source, edit, alpha = example()
    edit[0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="outside"):
        qualify(source, edit, alpha)


@pytest.mark.parametrize("fault", ["empty_alpha", "empty_edit", "shape", "float"])
def test_missing_or_ambiguous_reference_is_unassessed(fault):
    source, edit, alpha = example()
    if fault == "empty_alpha":
        alpha[:] = 0
    elif fault == "empty_edit":
        edit = source.copy()
    elif fault == "shape":
        edit = edit[:-1]
    else:
        source = source.astype(float)
    with pytest.raises(ValueError):
        qualify(source, edit, alpha)


def test_any_positive_alpha_can_support_a_changed_boundary_pixel():
    source, edit, alpha = example()
    edit = source.copy()
    alpha[:] = 0
    alpha[2, 3] = 1
    edit[2, 3, 0] ^= 1
    changed, proof = qualify(source, edit, alpha)
    assert changed[2, 3] and proof["changed_pixels"] == proof["support_pixels"] == 1


def test_control_hashes_repeat_from_exact_source_and_selection():
    first, proof = qualify(*example())
    second, repeated = qualify(*example())
    assert np.array_equal(first, second) and proof == repeated
