"""The amended audit matches persisted bytes, including half-rounding rank ties."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.engine.exhibits import ExhibitWriter
from app.storage.objects import ObjectStore
from scripts import audit_fused_production_grid as original
from scripts import audit_fused_writer_quantization as amended


def test_half_rounding_can_change_capped_region_selection_and_matches_writer(tmp_path, monkeypatch):
    pixels = np.full((512, 512, 3), 128, np.uint8)
    truth = np.zeros((512, 512), bool)
    raw = np.full((512, 512), .1, np.float32)
    raw[10:30, 10:30] = .9  # float32 * 255 rounds to 229.5, but its exact value is below that tie.
    raw[40:60, 10:30] = 230 / 255
    raw[70:90, 10:30] = 230 / 255
    monkeypatch.setattr(amended.loc, "conditions_mask",
                        lambda values, quality: (np.zeros(values.shape[:2], bool), {"unreliable": 0.0}))
    old, old_scalar, reliable, old_masks = original.replay(
        raw, pixels, truth, jpeg_quality=None, source_probabilities=raw, score=.8)
    new, new_scalar, _, new_masks = amended.writer_replay(
        raw, pixels, truth, jpeg_quality=None, source_probabilities=raw, score=.8)
    assert old_scalar[15, 15] == 230 and new_scalar[15, 15] == 229
    assert np.array_equal(old_masks["recorded_threshold"], new_masks["recorded_threshold"])
    assert old["qualifying_component_count"] == new["qualifying_component_count"] == 3
    assert old_masks["per_map_display_cap"][15, 15] and not new_masks["per_map_display_cap"][15, 15]
    assert not old_masks["per_map_display_cap"][75, 15] and new_masks["per_map_display_cap"][75, 15]

    store = ObjectStore(tmp_path / "objects")
    writer = ExhibitWriter("precision-fixture", store)
    writer.heatmap(raw, semantics="p_manipulated", title="Raw response", mask=reliable, value_range=(0, 1))
    with Image.open(io.BytesIO(store.read_bytes(writer.exhibits[0].sha256))) as image:
        np.testing.assert_array_equal(new_scalar, np.asarray(image))


def test_amended_decode_precision_matches_persisted_region_statistics(monkeypatch):
    native = np.full((512, 512), .9, np.float32)
    native[:20] = .2
    pixels = np.zeros((80, 100, 3), np.uint8)
    truth = np.zeros((80, 100), bool)
    truth[10:30, 20:40] = True
    monkeypatch.setattr(amended.loc, "conditions_mask",
                        lambda values, quality: (np.zeros(values.shape[:2], bool), {"unreliable": 0.0}))
    row, encoded, reliable, _ = amended.writer_replay(
        native, pixels, truth, jpeg_quality=None, source_probabilities=np.full(truth.shape, .9, np.float32), score=.2)
    _, stats = amended.loc.components(encoded.astype(np.float64) / 255, reliable, 100, 80)
    assert row["ranked_components"][0]["stats"] == stats[0]
    assert row["metrics"]["image_gated"]["per_map_display_cap"]["predicted_positive_pixels"] == 0
    assert row["metrics"]["image_gated"]["per_map_display_cap"]["false_negative_pixels"] == int(truth.sum())
