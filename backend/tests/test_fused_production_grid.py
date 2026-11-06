from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from scripts import audit_fused_production_grid as audit
from scripts import fused_research_eval as evaluator


def test_source_projection_uses_actual_overlay_pixel_centers():
    grid = np.array([[1, 0, 1], [0, 1, 0]], bool)
    projected = audit.to_source(grid, 7, 5)
    expected = grid[np.ix_([0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2])]
    assert np.array_equal(projected, expected)
    assert not np.array_equal(projected, cv2.resize(grid.astype(np.uint8), (7, 5), interpolation=cv2.INTER_NEAREST))


def test_exact_components_preserve_holes_rank_scores_and_apply_both_caps():
    values = np.zeros((100, 200), np.float32)
    for index in range(12):
        x, y = 10 + (index % 4) * 40, 10 + (index // 4) * 30
        values[y:y + 8, x:x + 8] = (250 - index * 5) / 255
    values[12:16, 12:16] = 0  # A hole in the highest scoring component.
    values[95:98, 190:193] = 1  # Above threshold, below minimum area.
    masks, facts = audit.component_masks(values, np.ones_like(values, bool), 501, 223)
    assert facts["threshold_component_count"] == 13 and facts["qualifying_component_count"] == 12
    assert len(facts["runtime_regions"]) == 10 and facts["runtime_components_exactly_matched"]
    assert masks["area_retained_components"].sum() == 12 * 64 - 16
    assert masks["localiser_region_cap"].sum() == 10 * 64 - 16
    assert masks["per_map_display_cap"].sum() == 2 * 64 - 16
    assert masks["per_map_display_cap"][10, 10] and masks["per_map_display_cap"][10, 50]
    assert not masks["per_map_display_cap"][10, 90]
    assert not masks["per_map_display_cap"][12:16, 12:16].any()
    assert facts["runtime_regions"][0]["region"]["bbox"] == {"x": 25.0, "y": 22.3, "w": 20.1, "h": 17.8}
    source_mask = audit.to_source(masks["per_map_display_cap"], 501, 223)
    assert not source_mask[31, 35]  # The projected hole remains a hole, not a filled bbox.


def test_equal_score_order_uses_area_then_y_then_x():
    values = np.zeros((100, 100), np.float32)
    values[10:15, 10:15] = .8
    values[10:15, 30:35] = .8
    values[30:35, 10:15] = .8
    values[50:60, 50:60] = .8
    _, facts = audit.component_masks(values, np.ones_like(values, bool), 100, 100)
    assert [(r["bbox"]["x"], r["bbox"]["y"]) for r in facts["ranked_components"]] == [
        (50.0, 50.0), (10.0, 10.0), (30.0, 10.0), (10.0, 30.0)]


def test_quantization_conditions_and_gate_never_clear_excluded_truth():
    pixels = np.full((64, 96, 3), 255, np.uint8)  # Fully clipped/flat, therefore not measurable.
    truth = np.zeros((64, 96), bool)
    truth[10:20, 10:20] = True
    native = np.full((512, 512), .5, np.float32)
    projected = np.full(truth.shape, .5, np.float32)
    row, scalar, reliable, masks = audit.replay(native, pixels, truth, jpeg_quality=None,
                                               source_probabilities=projected, score=.5)
    assert (scalar == 128).all() and row["image_gate_positive"]
    assert row["fully_unmeasurable"] and not reliable.any()
    assert row["excluded_truth_pixels_projected_grid"] == 100
    assert not masks["reliable_threshold"].any()
    for gate in ("ungated", "image_gated"):
        assert row["metrics"][gate]["per_map_display_cap"]["false_negative_pixels"] == 100
        assert row["metrics"][gate]["per_map_display_cap"]["iou"] == 0


@pytest.mark.parametrize("threshold", [.5, .7])
def test_component_reconstruction_detects_runtime_disagreement(monkeypatch, threshold):
    def disagree(*args, **kwargs):
        assert kwargs == {"threshold": threshold}
        return [], []

    monkeypatch.setattr(audit.loc, "components", disagree)
    with pytest.raises(ValueError, match="differs"):
        audit.component_masks(np.ones((10, 10), np.float32), np.ones((10, 10), bool), 10, 10,
                              threshold=threshold)


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    items = []
    for label, orientation in ((1, None), (0, 6)):
        rng = np.random.default_rng(48 + label)
        image = Image.fromarray(rng.integers(30, 225, (80, 120, 3), dtype=np.uint8))
        path = tmp_path / f"{label}.image"
        exif = Image.Exif()
        if orientation:
            exif[274] = orientation
        image.save(path, format="PNG", exif=exif)
        mask = np.zeros((80, 120), np.uint8)
        mask[10:30, 20:50] = 255
        mask_path = tmp_path / f"{label}.mask"
        Image.fromarray(mask).save(mask_path, format="PNG")
        items.append({"id": str(label), "family": "fixture", "label": label,
                      "image": path.name, "image_sha256": audit.sha(path.read_bytes()),
                      "mask": mask_path.name if label else None,
                      "mask_sha256": audit.sha(mask_path.read_bytes()) if label else None})
    manifest = tmp_path / "manifest.json"
    audit.write_new(manifest, {"items": items})
    audit.declare(tmp_path)
    responses = iter((.6, .4))

    class FakeModel:
        def __init__(self, *args):
            self.ops = SimpleNamespace(tensors={}, used=set())

        def predict(self, image):
            assert image.size == (120, 80)  # EXIF source remains in its stored coordinates.
            probabilities = np.full((512, 512), .1, np.float32)
            probabilities[60:200, 80:220] = .8
            return probabilities, next(responses)

    monkeypatch.setattr(evaluator, "FusedResearch", FakeModel)
    evaluator.evaluate(SimpleNamespace(manifest=manifest, output="evaluation", per_group=1, max_items=None,
                                       device="cpu", dtype="float32"))
    return tmp_path / "evaluation"


def test_saved_native_pipeline_audit_hashes_exif_and_frozen_outputs(saved_run):
    directory = saved_run
    old = {p.name: p.read_bytes() for p in directory.iterdir()}
    result = audit.audit(directory)
    assert result["all_stored_coordinate_summary"]["all"]["authentic_n"] == 1
    assert result["production_comparable_summary"]["all"]["authentic_n"] == 0
    assert result["production_comparable_summary"]["all"]["edited_n"] == 1
    assert result["rows"][1]["exif_orientation"] == 6
    assert result["rows"][1]["source_dimensions"] == [120, 80]
    assert result["rows"][1]["metrics"]["image_gated"]["native_source_float"]["predicted_positive_pixels"] == 0
    assert all((directory / name).read_bytes() == data for name, data in old.items())
    for row in result["rows"]:
        for record in row["artifacts"].values():
            assert audit.sha((directory / "production-grid-masks" / record["path"]).read_bytes()) == record["sha256"]
    with pytest.raises(FileExistsError):
        audit.audit(directory)


@pytest.mark.parametrize("damage", ["native", "source", "policy", "manifest", "metric", "duplicate"])
def test_frozen_inputs_cannot_be_changed_for_audit(saved_run, damage):
    directory = saved_run
    path = directory / "results.json"
    result = json.loads(path.read_text())
    if damage in {"native", "source"}:
        record = result["rows"][0]["probabilities"][damage]
        (directory / record["path"]).write_bytes(b"changed")
    elif damage in {"policy", "manifest"}:
        name = "production-grid-policy.json" if damage == "policy" else "manifest.json"
        original = directory.parent / name
        original.write_bytes(original.read_bytes() + b" ")
    elif damage == "metric":
        result["rows"][0]["intersection"] += 1
        path.write_text(json.dumps(result))
    else:
        result["rows"].append(result["rows"][0])
        result["selected_items"].append(result["selected_items"][0])
        path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        audit.audit(directory)
    assert not (directory / "production-grid-audit.json").exists()
