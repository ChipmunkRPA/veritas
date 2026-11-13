# SPDX-License-Identifier: Apache-2.0
"""Check native 16-bit parsing and phase localization on known pixel grids."""
from __future__ import annotations

import hashlib
import struct
import zlib

import cv2
import numpy as np
import pytest

from app.core.canonical import canonical_hash
from app.detectors.image import native_cfa
from app.evidence import image_interpolation_binding as binding
from app.evidence import image_interpolation_source as source


def png(rgb):
    encoded, body = cv2.imencode(".png", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    assert encoded
    return bytes(body)


@pytest.fixture(scope="module")
def renders():
    height, width = native_cfa.SCOPE
    y, x = np.indices((height, width), dtype=np.int32)
    rng = np.random.default_rng(128)
    # Distinct RGB channels, with red sampled at even/even and blue at odd/odd.
    red = (9000 + (71 * x + 113 * y) % 18000 + rng.integers(0, 300, (height, width))).astype(np.uint16)
    blue = (9500 + (97 * x + 61 * y) % 19000 + rng.integers(0, 300, (height, width))).astype(np.uint16)
    red[::2, 1:-1:2] = (red[::2, :-2:2] + red[::2, 2::2]) // 2
    red[1:-1:2, :] = (red[:-2:2, :] + red[2::2, :]) // 2
    blue[1::2, 2::2] = (blue[1::2, 1:-1:2] + blue[1::2, 3::2]) // 2
    blue[2::2, :] = (blue[1:-1:2, :] + blue[3::2, :]) // 2
    green = np.full_like(red, 18000)
    original = np.stack((red, green, blue), axis=-1)
    edited = original.copy()
    edited[416:544, 480:608] = original[160:288, 129:257]
    return original, edited


def test_native_phase_localizes_copy_in_original_coordinates(renders):
    original, edited = renders
    original_body, edited_body = png(original), png(edited)
    decoded = native_cfa.decoded_rgb(memoryview(edited_body))
    assert decoded.dtype == np.uint16
    assert np.array_equal(decoded, edited)
    normal = native_cfa.inspect_png(original_body)
    assert normal.status == "assessed"
    assert normal.phase == 0
    assert normal.region is None and normal.decoded_rgb_sha256 is None

    result = native_cfa.inspect_png(memoryview(edited_body))
    assert result.status == "assessed"
    assert result.phase == 0
    assert result.region == (480, 416, 608, 544)
    assert result.score > native_cfa.POINT
    assert result.supported_cells > 100
    assert result.reference_closure == 1.
    assert result.decoded_rgb_sha256 == hashlib.sha256(edited.tobytes()).hexdigest()


def chunk(name, payload):
    value = name + payload
    return struct.pack(">I", len(payload)) + value + struct.pack(">I", zlib.crc32(value))


def test_unsupported_transport_and_missing_stencil(renders):
    original, _ = renders
    body = png(original)
    text = chunk(b"tEXt", b"Comment\0Caption")
    rotation = chunk(b"eXIf", b"II*\0\x08\0\0\0")
    assert native_cfa.inspect_png(body[:33] + text + body[33:]).status == "unsupported_png"
    assert native_cfa.inspect_png(body[:33] + rotation + body[33:]).status == "unsupported_png"
    assert native_cfa.inspect_png(body[:-1]).status == "unsupported_png"
    assert native_cfa.inspect_png(body + b"extra").status == "unsupported_png"
    assert native_cfa.inspect_png(png((original >> 8).astype(np.uint8))).status == "unsupported_png"
    assert native_cfa.inspect_png(png(np.full_like(original, 24000))).status == "no_repeatable_stencil"


def test_signed_interpolation_observation_binds_native_crop_and_coordinates(renders):
    _, edited = renders
    body = png(edited)
    sha = hashlib.sha256(body).hexdigest()
    measured = source.record(sha, body=body)
    assets, data = {}, {}
    for name, plate in source.plates(body, measured["region"]).items():
        ok, encoded = cv2.imencode(".png", plate)
        assert ok
        digest = hashlib.sha256(encoded).hexdigest()
        assets[name] = {"sha256": digest, "media_type": "image/png", "width": plate.shape[1], "height": plate.shape[0]}
        data[digest] = encoded.tobytes()
    review = {"record": measured, "artifacts": assets}
    commitment = canonical_hash(measured)
    document = {"schema": "chipmunk.image-evidence/1",
                "source": {"sha256": sha, "width": 2048, "height": 1536, "exif_orientation": 1, binding.COMMITMENT: commitment},
                binding.KEY: review}
    binding.validate(document, commitment)
    assert source.verify(body, review, data.__getitem__)
    original_crop = data[assets["crop"]["sha256"]]
    changed = cv2.imdecode(np.frombuffer(original_crop, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    changed[0, 0, 0] += 1
    ok, altered = cv2.imencode(".png", changed)
    assert ok
    data[assets["crop"]["sha256"]] = altered.tobytes()
    with pytest.raises(ValueError, match="differs from its original pixels"):
        source.verify(body, review, data.__getitem__)
