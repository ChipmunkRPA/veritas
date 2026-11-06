from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from app.detectors.image import splice
from scripts import tiled_splice_candidate as tiled
from scripts import tiled_splice_localnorm_candidate as localnorm


def test_full_feature_planes_match_existing_preprocessing_bit_for_bit():
    pixels = np.random.default_rng(41).integers(0, 256, (337, 451, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    actual = tiled.features(tiled.feature_planes(image)).unsqueeze(0)
    assert torch.equal(actual, splice.recompression_features(image))


def test_four_tiles_preserve_original_geometry_and_cover_the_source():
    width, height = 641, 481
    regions = tiled.boxes(width, height)
    assert len(regions) == 4
    cover = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in regions:
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        cover[y0:y1, x0:x1] = True
    assert cover.all()
    assert tiled.boxes(319, 600) == []


def test_stitched_maps_use_original_coordinates_without_an_internal_seam():
    width, height = 500, 400
    tiles = [tiled.Tile(box, np.full((384, 384), .9, np.float32)) for box in tiled.boxes(width, height)]
    actual = tiled.stitch(tiles, width, height, .5)
    assert actual.shape == (height, width)
    np.testing.assert_allclose(actual, .9)
    alone = tiled.stitch(tiles[:1], width, height, .5)
    assert alone[0, 0] == pytest.approx(.9)  # Keep an outer source border.
    assert alone[-1, -1] == 0
    x0, y0, x1, y1 = tiles[0].box
    assert alone[y1-1, x1-1] == 0  # Suppress only the tile's internal edge.
    assert alone[20, 20] == pytest.approx(.9)


def test_tile_gate_never_invents_pixels_and_invalid_coordinates_are_rejected():
    raw = np.zeros((384, 384), np.float32)
    raw[:20, :20] = .9
    tile = tiled.Tile((0, 0, 400, 400), raw)
    assert 0 < tile.active_fraction < .01
    assert not tiled.stitch([tile], 400, 400, .01).any()
    with pytest.raises(ValueError, match="outside"):
        tiled.stitch([tiled.Tile((0, 0, 401, 400), raw)], 400, 400, .01)
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        tiled.stitch([tile], 400, 400, .01)


def test_scoring_passes_only_global_residual_crops_to_the_existing_checkpoint():
    class Model:
        def __call__(self, values):
            assert values.shape == (4, 3, 384, 384)
            return torch.zeros((4, 1, 384, 384))
    tiles = tiled.predict(Model(), Image.new("RGB", (500, 400), "gray"))
    assert len(tiles) == 4
    assert all(np.all(tile.response == .5) and tile.active_fraction == 0 for tile in tiles)


def test_local_normalization_is_separate_and_preserves_the_original_full_image_transform():
    import io

    from PIL import ImageChops

    pixels = np.random.default_rng(43).integers(0, 256, (400, 500, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    residuals = []
    for quality in splice.QUALITIES:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        with Image.open(io.BytesIO(buf.getvalue())) as encoded:
            residuals.append(ImageChops.difference(image, encoded.convert("RGB")))
    actual = localnorm.local_features(residuals, (0, 0, 500, 400))
    assert torch.equal(actual.unsqueeze(0), splice.recompression_features(image))
