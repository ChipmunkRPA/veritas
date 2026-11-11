import numpy as np
import pytest

from scripts import image_copy_corroboration_v295 as cue
from scripts import image_copy_positive_product_v300 as product
from scripts import image_copy_positive_transfer_v300 as transfer
from scripts import image_copy_transfer_v298 as ordinary


def setup_case(placement, phase):
    cue.cv2.setNumThreads(1)
    target, donor = ordinary.TARGETS[placement], ordinary.DONORS[phase]
    row = {
        "source_index": 0,
        "target_box_xywh": list(target),
        "donor_box_xywh": list(donor),
        "mixed_copy_box_xywh": list(transfer.REMOTE_BOX),
    }
    source = cue.fixture_texture(300, (256, 256))
    altered = source.copy()
    mask = transfer.rectangle_mask(target)
    altered[mask] = cue.fixture_texture(482, (64, 64)).reshape(-1, 3)
    return source, altered, row


@pytest.mark.parametrize("placement,phase", [("interior", "aligned"), ("edge", "shifted")])
def test_copy_only_and_mixed_views_share_separate_exact_footprints(placement, phase):
    source, altered, row = setup_case(placement, phase)
    variants, masks = transfer.material(source, altered, row)
    assert set(variants) == set(transfer.VARIANTS)
    ai_only, ai_truth, _ = variants["ai"]
    remote_copy, _, copy_truth = variants["remote_reflected_copy"]
    mixed, mixed_ai, mixed_copy = variants["mixed"]

    assert not (masks["target"] & masks["remote"]).any()
    assert np.array_equal(ai_truth, mixed_ai)
    assert np.array_equal(copy_truth, mixed_copy)
    assert np.array_equal(mixed[masks["target"]], ai_only[masks["target"]])
    assert np.array_equal(mixed[masks["remote"]], remote_copy[masks["remote"]])
    assert np.array_equal(np.any(source != mixed, axis=2), ai_truth | copy_truth)
    donor = transfer.rectangle_mask(row["donor_box_xywh"])
    assert np.array_equal(mixed[donor], source[donor])

    results = {
        (item["variant"], item["region"], item["format"]): item
        for item in transfer.corroboration_rows(row, {"status": "qualified_geometry"}, variants, masks)
    }
    assert len(results) == 10
    for fmt in transfer.FORMATS:
        assert results["mixed", "remote", fmt]["exact_ordinary_donor"]
        assert results["remote_reflected_copy", "remote", fmt]["exact_ordinary_donor"]
        assert results["mixed", "target", fmt]["supported_on_generated_pixels"] == 0
        assert results["ai", "target", fmt]["supported_on_generated_pixels"] == 0


def test_absent_generator_output_keeps_planned_cue_rows():
    source, altered, row = setup_case("interior", "aligned")
    variants, masks = transfer.material(source, None, row)
    result = list(transfer.corroboration_rows(row, {"status": "no_in_mask_change"}, variants, masks))
    assert len(result) == 10
    assert sum(item["status"] == "unassessed_generator_geometry" for item in result) == 6
    assert "ai" not in variants and "mixed" not in variants
    altered[0, 0] = 255
    with pytest.raises(ValueError, match="frozen region"):
        transfer.material(source, altered, row)


def test_product_copy_cue_keeps_both_existing_components():
    source, altered, row = setup_case("interior", "aligned")
    variants, masks = transfer.material(source, altered, row)
    selected = masks["target"] | masks["remote"]
    original = selected.copy()
    mixed, ai_truth, ordinary_truth = variants["mixed"]
    measured = product.measure_selected(mixed, selected, ai_truth, ordinary_truth, row, "mixed")

    assert np.array_equal(selected, original)
    assert measured["source_mask_sha256"] == transfer.flux.digest(original)
    assert len(measured["cue"]) == 2
    assert sum(region["exact_ordinary_donor"] for region in measured["cue"]) == 1
    assert sum(region["copy_supported_generated_intersection"] for region in measured["cue"]) == 0
    assert measured["generated"]["intersection"] == int(ai_truth.sum())
    assert measured["ordinary"]["intersection"] == int(ordinary_truth.sum())


def test_product_encoding_replays_exact_v298_bindings():
    import hashlib

    source, _, _ = setup_case("edge", "shifted")
    for fmt in transfer.FORMATS:
        old_pixels, old_hash = ordinary.encode_view(source, fmt)
        payload, image = product.encoded_view(source, fmt)
        assert np.array_equal(np.asarray(image), old_pixels)
        assert hashlib.sha256(payload).hexdigest() == old_hash
