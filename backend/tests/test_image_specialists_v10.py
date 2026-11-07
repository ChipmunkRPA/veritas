"""Verify source-coordinate plans before any image-model evaluation."""
import pytest

from scripts.image_specialists_v10 import boxes


@pytest.mark.parametrize(('width', 'height'), [(64, 64), (101, 65), (1024, 683)])
def test_four_source_regions_cover_odd_dimensions_once(width, height):
    regions = boxes(width, height)
    assert len(regions) == 4
    assert sum(w * h for _, _, w, h in regions) == width * height
    for index, (x, y, w, h) in enumerate(regions):
        assert 0 <= x < x + w <= width and 0 <= y < y + h <= height
        for u, v, a, b in regions[index + 1:]:
            assert max(x, u) >= min(x + w, u + a) or max(y, v) >= min(y + h, v + b)


@pytest.mark.parametrize(('width', 'height'), [(63, 64), (64, 0), (64., 64)])
def test_unsupported_geometry_is_refused(width, height):
    with pytest.raises(ValueError, match='geometry'):
        boxes(width, height)
