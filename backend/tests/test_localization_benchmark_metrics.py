from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from app.detectors.image import localiser
from scripts import localization_benchmark as benchmark


def test_area_filter_preserves_component_holes_and_removes_specks():
    mask = np.zeros((100, 100), bool)
    mask[10:20, 10:20] = True
    mask[12:18, 12:18] = False
    mask[50, 50:59] = True  # nine cells is below the 10-cell minimum
    truth = mask.copy()
    measured = benchmark.measure(mask, truth, benchmark.policy())
    retained = measured["metrics"]["area_retained_components"]
    assert measured["qualifying_component_count"] == 1
    assert measured["minimum_component_cells"] == 10
    assert measured["metrics"]["raw_threshold"]["predicted_positive_pixels"] == 73
    assert retained["intersection"] == 64
    assert retained["false_positive_pixels"] == 0
    assert retained["false_negative_pixels"] == 9


def test_binary_predictions_cannot_recover_score_ranked_cap():
    mask = np.zeros((100, 100), bool)
    for row in (10, 30, 50):
        mask[row:row + 4, 10:15] = True
    measured = benchmark.measure(mask, mask, benchmark.policy())
    assert measured["qualifying_component_count"] == 3
    assert measured["metrics"]["localiser_region_cap"]["iou"] == 1
    assert measured["metrics"]["per_map_display_cap"]["status"] == "unavailable"
    assert "iou" not in measured["metrics"]["per_map_display_cap"]
    tighter = {**benchmark.policy(), "localiser_max_regions": 2}
    assert benchmark.measure(mask, mask, tighter)["metrics"]["localiser_region_cap"]["status"] == "unavailable"


def test_scalar_grids_reproduce_localiser_rank_and_cap():
    values = np.zeros((100, 100), np.float32)
    values[10:14, 10:15] = .8
    values[30:34, 10:15] = .9
    values[50:54, 10:15] = .95
    reliable = np.ones_like(values, bool)
    actual, _ = localiser.components(values, reliable, 100, 100)
    assert [r["region"]["bbox"]["y"] for r in actual] == [50, 30, 10]
    masks, _ = benchmark.component_masks(values >= .5, width=100, height=100, rule=benchmark.policy(), values=values)
    capped = masks["per_map_display_cap"]
    assert capped[50:54, 10:15].all() and capped[30:34, 10:15].all()
    assert not capped[10:14, 10:15].any()


def test_metric_empty_cases_and_summary_do_not_inflate_edited_iou():
    empty = np.zeros((10, 10), bool)
    edited = empty.copy()
    edited[0, 0] = True
    assert benchmark.mask_metrics(empty, empty)["iou"] == 1
    assert benchmark.mask_metrics(empty, edited)["iou"] == 0
    assert benchmark.mask_metrics(empty, edited)["precision"] is None
    assert benchmark.mask_metrics(empty, empty)["recall"] is None
    rows = [{"id": "real", "label": 0, **benchmark.measure(empty, empty, benchmark.policy())},
            {"id": "edit", "label": 1, **benchmark.measure(empty, edited, benchmark.policy())}]
    assert benchmark.summary(rows)["metrics"]["raw_threshold"]["edited_mean_iou"] == 0
    with pytest.raises(ValueError, match="Boolean"):
        benchmark.mask_metrics(empty.astype(np.uint8), edited)


@pytest.fixture
def saved_run(tmp_path):
    items, rows = [], []
    for label in (0, 1):
        ident = f"sample-{label}"
        image = tmp_path / (ident + ".image")
        Image.new("RGB", (64, 64), (80, 100, 150)).save(image, format="PNG")
        truth = np.zeros((64, 64), bool)
        truth[10:14, 12:17] = bool(label)
        mask = tmp_path / (ident + ".mask")
        Image.fromarray(truth.astype(np.uint8) * 255).save(mask, format="PNG")
        predicted = np.zeros_like(truth)
        predicted[10:14, 12:17] = True
        predicted[45, 45] = True
        Image.fromarray(predicted.astype(np.uint8) * 255).save(tmp_path / (ident + ".prediction.png"))
        items.append({"id": ident, "family": "fixture", "label": label, "image": image.name,
                      "image_sha256": benchmark.sha(image.read_bytes()), "mask": mask.name if label else None,
                      "mask_sha256": benchmark.sha(mask.read_bytes()) if label else None})
        rows.append({"id": ident, "family": "fixture", "label": label, "regions": 1,
                     **benchmark.mask_metrics(predicted, truth)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": items}))
    frozen = tmp_path / "mesorch-smoke-results.json"
    frozen.write_text(json.dumps({"model": "test-model", "weights_revision": "a" * 64, "device": "fixture",
                                 "manifest_sha256": benchmark.sha(manifest.read_bytes()), "threshold": .5, "rows": rows}))
    return tmp_path, frozen


def test_remeasurement_replays_masks_without_changing_frozen_results(saved_run):
    root, frozen = saved_run
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = benchmark.remeasure(root)
    assert result["operation"] == "remeasure_saved_predictions_without_inference"
    assert result["frozen_raw_metrics_reproduced"]
    assert result["source_results_sha256"] == benchmark.sha(before[frozen.name])
    assert result["overall"]["metrics"]["raw_threshold"]["edited_mean_iou"] == pytest.approx(20 / 21)
    assert result["overall"]["metrics"]["area_retained_components"]["edited_mean_iou"] == 1
    assert result["overall"]["metrics"]["area_retained_components"]["authentic_with_positive_pixels"] == 1
    assert all((root / name).read_bytes() == data for name, data in before.items())
    assert result["rows"][0]["predictions"]["threshold"]["sha256"]
    with pytest.raises(FileExistsError):
        benchmark.remeasure(root)
    # New records retain checksummed prediction references and are replayable too.
    again = benchmark.remeasure(root, source_results=root / "mesorch-smoke-metrics-v2.json", output=root / "replay.json")
    assert again["overall"] == result["overall"]


@pytest.mark.parametrize("damage", ["image", "mask", "manifest", "raw_metrics", "region_count", "prediction", "label"])
def test_remeasurement_rejects_inconsistent_saved_evidence(saved_run, damage):
    root, frozen = saved_run
    old = json.loads(frozen.read_text())
    if damage == "image":
        (root / "sample-1.image").write_bytes(b"changed")
    elif damage == "mask":
        (root / "sample-1.mask").write_bytes(b"changed")
    elif damage == "manifest":
        (root / "manifest.json").write_text((root / "manifest.json").read_text() + " ")
    elif damage == "raw_metrics":
        old["rows"][1]["intersection"] += 1
    elif damage == "region_count":
        old["rows"][0]["regions"] += 1
    elif damage == "prediction":
        Image.new("L", (64, 64), 0).save(root / "sample-1.prediction.png")
    else:
        old["rows"][1]["label"] = 0
    frozen.write_text(json.dumps(old))
    with pytest.raises(ValueError):
        benchmark.remeasure(root)
    assert not (root / "mesorch-smoke-metrics-v2.json").exists()


def test_remeasurement_scalar_files_must_reproduce_saved_binary(saved_run):
    root, frozen = saved_run
    original = json.loads(frozen.read_text())
    for item in original["rows"]:
        ident = item["id"]
        path = root / (ident + ".prediction.png")
        scalar, reliable = root / (ident + ".scalar.png"), root / (ident + ".reliable.png")
        with Image.open(path) as image:
            image.save(scalar)
        Image.new("L", (64, 64), 255).save(reliable)
        item["predictions"] = {"threshold": benchmark._record(root, path), "scalar": benchmark._record(root, scalar),
                               "reliable": benchmark._record(root, reliable)}
    frozen.write_text(json.dumps(original))
    result = benchmark.remeasure(root)
    assert all(r["component_ranking_available"] for r in result["rows"])
    changed = deepcopy(original)
    path = root / changed["rows"][1]["predictions"]["scalar"]["path"]
    Image.new("L", (64, 64), 0).save(path)
    changed["rows"][1]["predictions"]["scalar"] = benchmark._record(root, path)
    frozen.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="do not reproduce"):
        benchmark.remeasure(root, output=root / "invalid.json")
