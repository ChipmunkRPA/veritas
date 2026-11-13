# SPDX-License-Identifier: Apache-2.0
"""Research-only exact delivery identity for an earlier RGB PNG and same-framing JPEG."""

from __future__ import annotations

import hashlib
import io
from collections import Counter

import numpy as np
from PIL import Image, JpegImagePlugin

from app.detectors.image import reference_compare

TILE = 16
MIN_SIDE, MAX_SIDE = 256, 768
MIN_ANCHORS = 24
MIN_ZONE_ANCHORS, MIN_ZONES = 2, 6
MIN_EXACT_SHARE = .65
MIN_DETAIL_RANGE = 24
# Standard quality 75 tables in Pillow/libjpeg natural coefficient order.
LUMA_Q75 = (8, 6, 5, 8, 12, 20, 26, 31, 6, 6, 7, 10, 13, 29, 30, 28, 7, 7, 8, 12, 20, 29, 35, 28,
            7, 9, 11, 15, 26, 44, 40, 31, 9, 11, 19, 28, 34, 55, 52, 39, 12, 18, 28, 32, 41, 52, 57, 46,
            25, 32, 39, 44, 52, 61, 60, 51, 36, 46, 48, 49, 56, 50, 52, 50)
CHROMA_Q75 = (9, 9, 12, 24, 50, 50, 50, 50, 9, 11, 13, 33, 50, 50, 50, 50, 12, 13, 28, 50, 50, 50, 50, 50,
              24, 33, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
              50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50)


def qualified_carriers(earlier, submitted):
    """Admit only the tested upright, unprofiled, same-dimension profile."""
    return (
        earlier.format == "PNG" and submitted.format == "JPEG"
        and earlier.mode == submitted.mode == "RGB"
        and getattr(earlier, "n_frames", 1) == getattr(submitted, "n_frames", 1) == 1
        and earlier.size == submitted.size and MIN_SIDE <= min(earlier.size) <= max(earlier.size) <= MAX_SIDE
        and earlier.getexif().get(274, 1) == submitted.getexif().get(274, 1) == 1
        and not {"icc_profile", "transparency", "gamma", "srgb", "chromaticity", "xmp", "XML:com.adobe.xmp"} & earlier.info.keys()
        and not {"icc_profile", "progressive", "progression", "xmp"} & submitted.info.keys()
        and JpegImagePlugin.get_sampling(submitted) == 2
        and submitted.quantization == {0: list(LUMA_Q75), 1: list(CHROMA_Q75)}
        and [item[3] for item in submitted.layer] == [0, 1, 1]
    )


def encoded_reference(earlier_rgb, submitted):
    """Recreate the tested delivery at the same native origin, then decode it."""
    with io.BytesIO() as stream:
        Image.fromarray(earlier_rgb).save(stream, format="JPEG", qtables=submitted.quantization, subsampling=2)
        stream.seek(0)
        with Image.open(stream, formats=["JPEG"]) as encoded:
            return np.array(encoded.convert("RGB"))


def blocks(rgb):
    rows, columns = (length // TILE for length in rgb.shape[:2])
    return rgb[:rows*TILE, :columns*TILE].reshape(rows, TILE, columns, TILE, 3).transpose(0, 2, 1, 3, 4)


def unique_grid_tiles(expected, submitted, exact):
    """Require high-detail anchor tiles unique within each delivered full-block grid."""
    rows, columns = exact.shape
    expected_hashes = [hashlib.sha256(tile.tobytes()).hexdigest()
                       for tile in expected.reshape(-1, TILE, TILE, 3)]
    submitted_hashes = Counter(hashlib.sha256(tile.tobytes()).hexdigest()
                               for tile in submitted.reshape(-1, TILE, TILE, 3))
    earlier_hashes = Counter(expected_hashes)
    anchors = []
    for row, col in np.argwhere(exact):
        fingerprint = expected_hashes[row * columns + col]
        tile = expected[row, col]
        # Reject flat tiles even when the literal RGB triplet is uncommon.
        if earlier_hashes[fingerprint] != 1 or submitted_hashes[fingerprint] != 1:
            continue
        contrast = np.ptp(tile.astype(np.int16), axis=(0, 1))
        if int(contrast.max()) < MIN_DETAIL_RANGE:
            continue
        anchors.append({"x": int(col)*TILE, "y": int(row)*TILE, "sha256": fingerprint,
                        "zone": [int(row)*3 // rows, int(col)*3 // columns]})
    return anchors


def alignment_proof(expected, submitted):
    """Measure exact decoded RGB support without searching over offsets or orientations."""
    a, b = blocks(expected), blocks(submitted)
    exact = (a == b).all(axis=(-1, -2, -3))
    equal = int(exact.sum())
    if equal < MIN_EXACT_SHARE * exact.size:
        return None
    anchors = unique_grid_tiles(a, b, exact)
    zones = Counter(tuple(row["zone"]) for row in anchors)
    supported = [pair for pair, count in sorted(zones.items()) if count >= MIN_ZONE_ANCHORS]
    if (len(anchors) < MIN_ANCHORS or len(supported) < MIN_ZONES
            or any(len({point[axis] for point in supported}) != 3 for axis in (0, 1))):
        return None
    return {
        "method": "exact_decoded_jpeg_delivery_identity",
        "reference_minus_submitted": {"x": 0, "y": 0},
        "reference_overlap": {"x": 0, "y": 0, "w": expected.shape[1], "h": expected.shape[0]},
        "submitted_overlap": {"x": 0, "y": 0, "w": expected.shape[1], "h": expected.shape[0]},
        "full_tiles": int(exact.size),
        "exact_tiles": equal,
        "anchors": anchors,
        "supported_zones": [list(pair) for pair in supported],
        "expected_decoded_rgb_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
    }


def compare(earlier, submitted):
    """Return a difference mask only after a strict profile and exact identity proof."""
    if not qualified_carriers(earlier, submitted):
        return {"status": "outside_profile", "regions": []}, None
    native = np.array(earlier)
    questioned = np.array(submitted)
    expected = encoded_reference(native, submitted)
    proof = alignment_proof(expected, questioned)
    if proof is None:
        return {"status": "delivery_identity_unresolved", "regions": []}, None
    result, mask = reference_compare.compare_across_deliveries(native, questioned, submitted)
    if result["status"] != "comparable":
        return result, None
    for region in result["regions"]:
        region["reference_region"] = {key: region[key] for key in ("x", "y", "w", "h")}
    return result | {"alignment": proof}, mask
