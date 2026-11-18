import numpy as np

from app.evidence.video_reference_short_reencode import GRIDS, summarize_grids


def test_short_picture_change_can_fail_only_the_finer_grid():
    coarse, fine = [np.zeros((53, height, width), dtype=np.float32) for width, height in GRIDS]
    coarse[:, :5, :5] = 3
    fine[22:24, :5, :5] = 14

    guard = summarize_grids((coarse, fine))

    assert guard["grids"][0]["within_bounds"]
    assert guard["grids"][1]["measurements"]["local_five_frame_median"] == 0
    assert guard["grids"][1]["instant_cell_max"] == 14
    assert not guard["grids"][1]["within_bounds"]
    assert not guard["within_bounds"]

    fine[22:24, :5, :5] = 13
    assert summarize_grids((coarse, fine))["within_bounds"]
