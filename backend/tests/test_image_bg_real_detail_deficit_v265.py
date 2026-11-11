import numpy as np

from scripts import image_bg_real_detail_deficit_v265 as cue


def test_flat_image_has_no_relative_detail_deficit():
    pixels = np.full((83, 89, 3), 128, np.uint8)
    assert np.array_equal(cue.detail_map(pixels), np.zeros((5, 5)))


def test_loss_of_fine_detail_in_one_tile_increases_local_response():
    size = 9 * cue.TILE
    y, x = np.indices((size, size))
    coarse = 128 + 25 * np.sin(x * (2 * np.pi / 8))
    full = coarse + 8 * ((-1) ** (x + y))
    changed = full.copy()
    target = np.s_[4 * cue.TILE : 5 * cue.TILE, 4 * cue.TILE : 5 * cue.TILE]
    changed[target] = coarse[target]
    before = np.repeat(full.astype(np.uint8)[:, :, None], 3, axis=2)
    after = np.repeat(changed.astype(np.uint8)[:, :, None], 3, axis=2)
    assert cue.detail_map(after)[4, 4] > cue.detail_map(before)[4, 4]


def test_concordance_weights_exact_changed_area_and_excludes_adjacent_tiles():
    scores = np.zeros((5, 5))
    scores[1, 1:3] = (1, -1)
    scores[0, :4] = 100
    footprint = np.zeros((5 * cue.TILE, 5 * cue.TILE), dtype=bool)
    footprint[cue.TILE, cue.TILE] = True
    footprint[cue.TILE, 2 * cue.TILE : 2 * cue.TILE + 3] = True
    assert cue.compare(scores, footprint) == {
        "concordance": 0.25,
        "target_response": -0.5,
        "footprint_retained": 1,
    }
    assert cue.compare(np.zeros_like(scores), footprint)["concordance"] == 0.5


def test_incomplete_border_support_is_counted_or_unassessed():
    scores = np.zeros((5, 5))
    footprint = np.zeros((5 * cue.TILE + 1, 5 * cue.TILE), dtype=bool)
    footprint[-1, 0] = True
    assert cue.compare(scores, footprint) is None
    footprint[0, 0] = True
    assert cue.compare(scores, footprint)["footprint_retained"] == 0.5
