import pytest
from PIL import Image

from scripts import image_flux_detector_comparison_v280 as compare


def test_png_jpeg_keep_native_truth_and_expose_codec_spill():
    source, intended = compare.photo.flux.fixture("interior")
    edited = source.copy()
    edited[102:139, 92:133] ^= 29
    images = {kind: Image.fromarray(edited if kind == "ai" else source) for kind in compare.KINDS}
    footprints = {
        kind: compare.original.changed_pixels(images["original"], image) for kind, image in images.items()
    }
    binding = compare.source_entry({"source_id": "first", "component": "group"}, images, footprints)
    assert binding["changed_pixels"]["ai"] == 37 * 41
    assert binding["changed_pixels"]["ai"] < (intended > 0).sum()
    assert (
        binding["inputs"]["0"]["ai"]["source_size"]
        == binding["inputs"]["75"]["ai"]["source_size"]
        == [256, 256]
    )
    assert binding["decoded_outside_native_footprint_changes"]["native_png"]["ai"] == 0
    assert binding["decoded_outside_native_footprint_changes"]["jpeg75"]["ai"] > 0


def test_complete_order_and_fixed_useful_and_false_region_denominators():
    groups = ["photo-1", "photo-2"]
    rows = []
    for number, kind in compare.tasks(0):
        methods = {}
        for name in compare.original.METHODS:
            value = 1 if kind == "ai" and name == "sensitive" and number == 0 else 0
            positive = 9 if value else 0
            if kind == "tone" and name == "sensitive" and number == 1:
                positive = 3
            methods[name] = {
                "ai": {
                    "iou": value * 0.3,
                    "recall": value * 0.5,
                    "intersection": value,
                    "predicted_positive_pixels": positive,
                    "flagged_area_fraction": positive / 65536,
                }
            }
        rows.append(
            {"n": number, "kind": kind, "component": groups[number], "quality": 75, "methods": methods}
        )
    report = compare.summarize(rows, groups, 75)
    assert report["methods"]["sensitive"]["useful_positive_groups"] == 1
    assert report["methods"]["sensitive"]["controls"]["tone"]["flagged_groups"] == 1
    assert report["methods"]["current"]["useful_positive_groups"] == 0
    with pytest.raises(ValueError, match="Every paired view"):
        compare.summarize(rows[:-1], groups, 75)
    with pytest.raises(ValueError, match="Every paired view"):
        compare.summarize(rows[::-1], groups, 75)
