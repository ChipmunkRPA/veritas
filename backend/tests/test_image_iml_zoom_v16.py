"""Authored source-geometry and genuine-point contracts, not detection accuracy."""

import copy
import hashlib
import json

import numpy as np
import pytest

from scripts import image_iml_zoom_v16 as method
from scripts import review_image_iml_zoom_v16 as reference
from scripts import run_image_iml_zoom_v16 as driver
from scripts.score_splice_localizers import score_map


@pytest.mark.parametrize(
    "shape,target", [((7, 11), (19, 25)), ((41, 37), (256, 256)), ((17, 31), (6, 9)), ((1, 4), (3, 13))]
)
def test_half_pixel_sampling_has_separate_reference_on_odd_and_nonsquare_maps(shape, target):
    values = np.random.default_rng(16).uniform(size=shape).astype(np.float32)
    assert np.max(np.abs(method.resize(values, *target) - reference.resize(values, *target))) < 1e-12


@pytest.mark.parametrize("shape", [(67, 103), (384, 513), (683, 1024)])
def test_all_source_edges_and_tile_interiors_have_actual_coverage(shape):
    height, width = shape
    grids = np.full((5, 256, 256), 51, np.uint8)
    output = method.reconstruct(grids, width, height)
    independent = reference.reconstruct(grids, width, height)
    assert np.allclose(output["zoom"], 0.2, rtol=0, atol=1e-15)
    assert np.max(np.abs(output["zoom"] - independent["zoom"])) < 1e-12
    assert all(
        0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height for x0, y0, x1, y1 in method.boxes(width, height)
    )


def test_nonuniform_tiled_projection_matches_reference_and_retains_all_five_inputs():
    grids = np.random.default_rng(170).integers(0, 256, (5, 256, 256), dtype=np.uint8)
    actual = method.reconstruct(grids, 131, 97)
    expected = reference.reconstruct(grids, 131, 97)
    assert all(np.max(np.abs(actual[k] - expected[k])) < 1e-12 for k in actual)
    with pytest.raises(ValueError):
        method.reconstruct(grids[:-1], 131, 97)


def test_zoom_pixel_extent_is_long_side1024_with_source_rectangle_preserved():
    rgb = np.random.default_rng(19).integers(0, 256, (53, 91, 3), dtype=np.uint8)
    for box in method.boxes(91, 53):
        out = method.zoom(rgb, box)
        assert max(out.shape[:2]) == 1024
        assert abs(out.shape[1] / out.shape[0] - (box[2] - box[0]) / (box[3] - box[1])) < 0.005
    with pytest.raises(ValueError):
        method.zoom(rgb, [-1, 0, 20, 20])


def test_ordinary_operation_changes_only_the_fixed_central_support():
    rgb = np.random.default_rng(20).integers(30, 220, (129, 131, 3), dtype=np.uint8)
    out = method.ordinary(rgb)
    changed = np.any(out != rgb, axis=2)
    assert changed.any() and np.array_equal(out[:32], rgb[:32]) and np.array_equal(out[:, :32], rgb[:, :32])
    assert not changed[97:].any() and not changed[:, 98:].any()


def test_recorded_quantization_and_activity_preserve_half_even_and_exact_sides():
    v = np.full((256, 256), 0.5, np.float32)
    encoded = method.encode(v)
    assert np.all(encoded == 128)
    assert method.fraction(np.full((4, 5), 127 / 255)) == 0
    assert method.fraction(np.full((4, 5), 128 / 255)) == 1


def controls():
    return [
        {
            "role": "point",
            "kind": kind,
            "format": "native",
            "label": 0,
            "status": "ok",
            "source_id": str(i),
            "responses": {
                arm: {"active_fraction": 0.01 + i / 1000, "ungated": {"predicted_positive_pixels": 2}}
                for arm in ("whole", "zoom")
            },
        }
        for i in range(24)
        for kind in ("original", "ordinary")
    ]


def test_genuine_only_point_groups_correlated_operations_and_retains_ties():
    rows = controls()
    result = method.point(rows)
    assert all(p["groups_flagged"] == 1 and not p["uses_ai_or_diagnostic"] for p in result.values())
    for row in rows:
        for v in row["responses"].values():
            v["active_fraction"] = 0.5
    assert all(
        p["groups_flagged"] == 0 and p["minimum_active_fraction"] > 0.5 for p in method.point(rows).values()
    )


@pytest.mark.parametrize(
    "fault", ["AI", "diagnostic", "missing_source", "missing_operation", "mixed_format", "failed"]
)
def test_bad_or_leaking_source_roles_cannot_fit_points(fault):
    rows = controls()
    if fault == "AI":
        rows[0]["label"] = 1
    elif fault == "diagnostic":
        rows[0]["role"] = "diagnostic"
    elif fault == "missing_source":
        rows = rows[2:]
    elif fault == "missing_operation":
        rows = rows[1:]
    elif fault == "mixed_format":
        rows[0]["format"] = "jpeg75"
    else:
        rows[0]["status"] = "failed"
    with pytest.raises(ValueError):
        method.point(rows)


def test_points_and_both_complete_source_journals_are_bound_before_diagnostics(tmp_path):
    names = ("points.json", "point-native.jsonl", "point-jpeg75.jsonl", "freeze.json")
    for name in names:
        (tmp_path / name).write_text("{}\n")
    hashes = {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in names}
    (tmp_path / "point-seal.json").write_text(json.dumps({"sha256": hashes}))
    driver.check_points(tmp_path)
    (tmp_path / "points.json").write_text('{"edited":0}\n')
    with pytest.raises(ValueError):
        driver.check_points(tmp_path)


def test_independent_components_and_metrics_match_current_display_on_authored_pixels():
    rgb = np.random.default_rng(29).integers(25, 230, (97, 131, 3), dtype=np.uint8)
    truth = np.zeros((97, 131), bool)
    truth[15:55, 13:68] = True
    raw = np.zeros((97, 131), np.float64)
    raw[12:58, 12:72] = 0.8
    raw[65:90, 88:119] = 0.7
    data = method.image_bytes(rgb)
    assert (
        reference.display_metric(raw, rgb, data, truth)
        == score_map(raw, True, 0.5, data, rgb, truth)["metrics"]
    )


def test_summary_keeps_every_source_and_rejects_false_alarm_tradeoff():
    rows = []
    for fmt in ("native", "jpeg75"):
        for i in range(48):
            for kind in ("original", "ordinary", "ai"):
                good = {
                    "iou": 0.4 if kind == "ai" else 1.0,
                    "pixel_f1": 0.5,
                    "intersection": int(kind == "ai"),
                    "predicted_positive_pixels": int(kind == "ai"),
                }
                rows.append(
                    {
                        "id": f"{fmt}-{i}-{kind}",
                        "format": fmt,
                        "kind": kind,
                        "source_id": str(i),
                        "family": ("firefly", "sd2", "flux-fill")[i // 16],
                        "metrics": {a: copy.deepcopy(good) for a in ("current", "whole", "zoom")},
                    }
                )
    rows[0]["metrics"]["zoom"]["predicted_positive_pixels"] = 1
    result = reference.aggregate(rows)
    assert not result["formats"]["native"]["qualifies_fresh_confirmation"]
    assert result["formats"]["native"]["arms"]["zoom"]["files"] == 144
    with pytest.raises(ValueError):
        reference.aggregate(rows[:-1])
