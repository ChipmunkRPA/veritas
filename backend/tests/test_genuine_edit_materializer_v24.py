"""Copy-only ordinary edits, exact native RGB encoding, and explicit no-AI truth."""
from pathlib import Path

import numpy as np
import pytest

from scripts.genuine_edit_materializer_v24 import checked_clip, compose, decode, encode


def test_copy_interval_does_not_label_a_human_edit_as_ai():
    target = np.zeros((12, 480, 640, 3), np.uint8)
    donor = np.full_like(target, 101)
    result = compose(target, donor, [3, 8])
    assert not result[:3].any() and not result[8:].any()
    assert (result[3:8] == 101).all()
    assert not target.any()
    with pytest.raises(ValueError):
        compose(target, donor, [3, 12])


def test_native_lossless_rgb_roundtrip_preserves_real_clock_and_ordinary_truth(tmp_path: Path):
    rng = np.random.default_rng(44)
    # Authored noisy fields exercise all RGB channels through the real encoder.
    pixels = rng.integers(0, 256, size=(12, 480, 640, 3), dtype=np.uint8)
    path = tmp_path / 'rgb.mp4'
    encode(path, pixels)
    repeated = tmp_path / 'rgb-repeat.mp4'
    encode(repeated, pixels)
    assert repeated.read_bytes() == path.read_bytes()
    case, plan = checked_clip(path, 'authored', 'authored-component')
    assert np.array_equal(decode(path, case['table']), pixels)
    assert plan['source_geometry']['native_frame_rate'] == '24'
    assert plan['source_geometry']['frame_count'] == 12
    assert plan['label'] == 0 and all(u['label'] == 0 for u in plan['units'])
    with pytest.raises(ValueError, match='New lossless'):
        encode(path, pixels)
