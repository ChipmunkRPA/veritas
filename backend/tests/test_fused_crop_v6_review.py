"""Independent reconstruction failures, beyond the crop arithmetic unit checks."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from scripts import review_fused_crop_v6 as reviewer
from scripts.evaluate_iml_vit import input_bytes
from scripts.fused_crop_v6 import boxes
from scripts.localization_benchmark import mask_metrics, sha


@pytest.fixture
def receipt(tmp_path, monkeypatch):
    rgb = np.random.default_rng(12).integers(30, 225, (320, 400, 3), dtype=np.uint8)
    image = tmp_path / "source.png"
    Image.fromarray(rgb).save(image)
    item = {"id": "fixture", "source_id": "one", "family": "fixture", "label": 0,
            "image": image.name, "image_sha256": sha(image.read_bytes()), "mask": None, "mask_sha256": None}
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    maps = tmp_path / "maps"
    maps.mkdir()
    monkeypatch.setattr(reviewer, "selected", lambda *_: ({"items": [item]}, "manifest", [item]))
    policy = {"schema": "chipmunk.fused-crop-development/6", "section": "development",
              "manifest_sha256": "manifest", "selection_sha256": sha(selection.read_bytes()),
              "code_sha256": reviewer.code_pins(), "item_ids": [item["id"]],
              "weights": {"fused": reviewer.fused.CHECKPOINT_SHA256,
                          "semantic": reviewer.fused.BACKBONE_SHA256,
                          "native_baseline": reviewer.splice.CHECKPOINT_SHA256}}
    (maps / "policy.json").write_text(json.dumps(policy))
    rows = []
    for condition in ("native", "jpeg75"):
        data, pixels = input_bytes(rgb, image.read_bytes(), condition)
        image_pixels = Image.fromarray(pixels)
        crops = [{"box": box, "image_score": 0.6,
                  "rgb_sha256": hashlib.sha256(np.asarray(image_pixels.crop(box)).tobytes()).hexdigest()}
                 for box in boxes(400, 320)]
        name = condition + ".npz"
        side = reviewer.splice.INPUT_SIZE if condition == "native" else 512
        np.savez_compressed(maps / name, baseline=np.zeros((side, side), dtype=np.float32),
                            **{f"crop{i}": np.zeros((512, 512), dtype=np.float32) for i in range(4)})
        rows.append({**item, "condition": condition, "input_sha256": sha(data), "status": "ok",
                     "arrays": name, "arrays_sha256": sha((maps / name).read_bytes()), "crops": crops,
                     "baseline_image_score": 0.1,
                     "baseline": mask_metrics(np.zeros((320, 400), bool), np.zeros((320, 400), bool))})
    path = maps / "predictions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path, selection, maps, rows


def test_reconstructs_exact_crop_inputs_arrays_and_baseline(receipt):
    root, selection, maps, _ = receipt
    _, rows = reviewer.prepare(root, selection, maps)
    assert len(rows) == 2 and all(len(r["crops"]) == 4 for r in rows)
    assert all(r["rgb"].shape == (320, 400, 3) for r in rows)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "label", "input", "crop_rgb", "crop_box", "metric", "array", "code"])
def test_changed_evidence_is_refused_before_gate_selection(receipt, damage):
    root, selection, maps, rows = receipt
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows[1] = rows[0]
    elif damage == "label":
        rows[0]["label"] = 1
    elif damage == "input":
        rows[0]["input_sha256"] = "0" * 64
    elif damage == "crop_rgb":
        rows[0]["crops"][0]["rgb_sha256"] = "0" * 64
    elif damage == "crop_box":
        rows[0]["crops"][0]["box"] = [1, 0, 240, 192]
    elif damage == "metric":
        rows[0]["baseline"]["pixels"] += 1
    elif damage == "array":
        (maps / rows[0]["arrays"]).write_bytes(b"changed")
    else:
        p = maps / "policy.json"
        d = json.loads(p.read_text())
        d["code_sha256"]["backend/scripts/fused_crop_v6.py"] = "0" * 64
        p.write_text(json.dumps(d))
    (maps / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        reviewer.prepare(root, selection, maps)


def test_failed_extraction_stays_recorded_and_cannot_select_a_gate(receipt):
    root, selection, maps, rows = receipt
    rows[0].update(status="failed", reason="fixture inference failure")
    (maps / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, prepared = reviewer.prepare(root, selection, maps)
    assert len(prepared) == 2 and prepared[0]["status"] == "failed"
    target = maps / "review.json"
    with pytest.raises(ValueError, match="incomplete extraction"):
        reviewer.review(root, selection, maps, target)
    assert not target.exists()
