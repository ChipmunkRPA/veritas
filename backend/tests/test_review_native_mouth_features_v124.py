import copy

import numpy as np
import pytest

from scripts import mouth_articulation_core_v122 as core
from scripts import mouth_probe_v6 as native
from scripts import review_native_mouth_features_v124 as review
from scripts.av_conditioning_clock_v109 import frame_plan


def authored_pixels():
    rgb = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    crop = rgb[48:72, 32:68]
    vector = np.ones(768, np.float32) / np.sqrt(np.float32(384))
    metadata = {
        "face": {"box": [20, 10, 70, 80], "landmarks": [[35, 25], [65, 25], [50, 40], [40, 60], [60, 60]]},
        "region_bbox": [32, 48, 68, 72],
        "crop_rgb_sha256": review.fingerprint(crop),
        "model_input_float32_sha256": review.fingerprint(native.preprocessed([crop])[0]),
        "feature_float32_sha256": review.fingerprint(vector),
    }
    return rgb, metadata, vector


def test_independent_pixel_and_normalization_replay_matches_authored_contract():
    review.pixel_replay(*authored_pixels())


@pytest.mark.parametrize("change", ["pixels", "bounds", "landmarks", "input", "feature_row", "scale"])
def test_wrong_picture_region_input_or_feature_assignment_is_rejected(change):
    rgb, record, vector = authored_pixels()
    if change == "pixels":
        rgb[50, 50, 0] ^= 1
    elif change == "bounds":
        record["region_bbox"][0] += 1
    elif change == "landmarks":
        record["face"]["landmarks"][3][0] += 3
    elif change == "input":
        record["model_input_float32_sha256"] = "0" * 64
    elif change == "feature_row":
        vector = np.roll(vector * np.linspace(0.5, 1.5, 768, dtype=np.float32), 1)
    else:
        vector *= 2
        record["feature_float32_sha256"] = review.fingerprint(vector)
    with pytest.raises(ValueError):
        review.pixel_replay(rgb, record, vector)


def test_direct_cached_speech_replay_matches_qualified_summary_without_forward():
    rng = np.random.default_rng(724)
    hidden = rng.normal(size=(50, 5, 384)).astype(np.float32)
    plan = frame_plan(10, "2/5", "0", "0", 16000, "0")
    assert np.array_equal(review.speech_replay(hidden, plan), core.speech_summary(hidden, plan))
    wrong = copy.deepcopy(plan)
    wrong["retained_tick_slots"][0] = False
    with pytest.raises(ValueError, match="support"):
        review.speech_replay(hidden, wrong)


def test_array_contract_rejects_changed_bytes_dtype_or_dimensions(tmp_path):
    file = tmp_path / "authored.npy"
    np.save(file, np.ones((2, 768), np.float32), allow_pickle=False)
    record = {"file": str(file), "sha256": review.sha(file), "shape": [2, 768]}
    assert review.array_file(record, np.float32).shape == (2, 768)
    with pytest.raises(ValueError, match="dtype"):
        review.array_file(record, np.float64)
    record["shape"] = [1, 768]
    with pytest.raises(ValueError, match="dimensions"):
        review.array_file(record, np.float32)
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        review.array_file(record, np.float32)


def test_invalid_batch_or_neural_command_is_not_routed():
    with pytest.raises(ValueError):
        review.completed_batch(True)
    with pytest.raises(ValueError):
        review.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract"])
