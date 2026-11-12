import cv2
import numpy as np

from scripts import review_video_mouth_reconstruction_transfer as review
from scripts import video_mouth_reconstruction_transfer as transfer


def test_reconstruction_uses_correct_warp_direction_for_translated_pixels():
    noise = np.random.default_rng(17).integers(0, 256, (transfer.SIZE, transfer.SIZE), dtype=np.uint8)
    original = cv2.GaussianBlur(noise, (5, 5), 0.9)
    moved = cv2.warpAffine(original, np.float32([[1, 0, 2], [0, 1, -1]]), original.shape[::-1],
                           borderMode=cv2.BORDER_REFLECT_101)
    x, y = np.meshgrid(np.arange(transfer.SIZE, dtype=np.float32), np.arange(transfer.SIZE, dtype=np.float32))
    reconstructed = transfer.reconstruct(original, moved, np.dstack((x, y)))
    core, _ = transfer.support_masks((48, 55))
    uncorrected = np.mean(np.abs(original[core].astype(float) - moved[core]))
    error = np.mean(np.abs(original[core].astype(float) - reconstructed[core]))
    assert error < uncorrected / 4


def test_local_range_uses_both_flanks_and_independent_half_open_decision():
    curve = [1.0] * len(transfer.POSITIONS)
    curve[14:24] = [4.0] * 10
    candidate = transfer.best_window(curve)
    assert candidate["score"] == 3
    assert candidate["medians"] == [1, 4, 1]
    lo, hi = candidate["frames"]
    before, during, after = (transfer.POSITIONS[i] for i in candidate["slots"])
    assert before < lo <= during < hi <= after
    assert transfer.best_window([None] * len(curve)) is None
    row = {"truth_clock_s": ["1/2", "3/2"], "arms": {"direct": {"best": {
        "medians": [0.0, 1.0, 0.0], "score": 1.0, "source_clock_s": ["1/2", "1"]}}}}
    assert transfer.result(row, "direct", .5) == review.verified_outcome(row, "direct", .5) == "correct_range"
    assert transfer.result(row, "direct", 1) == "measured_no_range"
