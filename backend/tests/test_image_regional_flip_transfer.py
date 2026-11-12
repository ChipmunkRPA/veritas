import hashlib

import numpy as np
import pytest

from scripts.image_regional_flip_transfer import combine
from scripts.review_image_regional_flip_transfer import restore_masks, useful


def test_keeps_an_eight_connected_component_only_with_half_overlap(tmp_path):
    native = np.zeros((7, 9), bool)
    native[1, 1] = native[2, 2] = native[3, 3] = native[4, 4] = True
    native[1:4, 7] = True
    mirror = np.zeros_like(native)
    mirror[1, 1] = mirror[2, 2] = mirror[1, 7] = True
    mirror[0, 8] = True
    recovered = combine(native, mirror)
    assert recovered["supported_components"].sum() == 4
    assert recovered["shared_pixels"].sum() == 3
    assert recovered["supported_components"][4, 4]
    assert not recovered["supported_components"][1, 7]
    assert not recovered["shared_pixels"][0, 8]

    path = tmp_path / "fixture-masks.bin"
    packed = b"".join(np.packbits(mask, axis=None).tobytes() for mask in (native, mirror, ~native))
    path.write_bytes(packed)
    replay = restore_masks(path, {"shape": list(native.shape), "mask_bytes_sha256": hashlib.sha256(packed).hexdigest()})
    for arm, mask in recovered.items():
        np.testing.assert_array_equal(replay[arm], mask)
    np.testing.assert_array_equal(replay["recompression"], ~native)


def test_mismatched_mask_geometry_and_overlap_limits():
    with pytest.raises(ValueError, match="same source pixel grid"):
        combine(np.zeros((2, 3), bool), np.zeros((3, 2), bool))
    assert useful({"target": 10, "reported": 5, "overlap": 5})
    assert not useful({"target": 10, "reported": 20, "overlap": 5})
    assert not useful({"target": 10, "reported": 4, "overlap": 4})
    assert not useful({"target": 0, "reported": 0, "overlap": 0})
