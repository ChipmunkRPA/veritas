"""Native-clock and source-image invariants for the optional matched-detail check."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import cv2
import numpy as np
import pytest

from app.core.canonical import canonical_hash
from app.detectors.video import face_matched_detail as detector
from app.detectors.video.decode import FrameTable
from app.evidence import face_matched_detail_binding as binding
from app.evidence import video_packets

START, END = 73, 76
SOURCE = {
    "width": 1280,
    "height": 720,
    "time_base": binding.TIME_BASE,
    "frame_count": 120,
    "timeline_origin_s": 0,
}
LANDMARKS = [
    [float(x).hex(), float(y).hex()] for x, y in ((475, 299), (584, 298), (530, 357), (487, 418), (574, 419))
]
BOX = {"x": 432, "y": 246, "w": 195, "h": 248}


def test_matched_detail_profile_keeps_native_scope():
    frames = FrameTable(binding.TIME_BASE, binding.TICKS, (0,), "I" + "P" * 119, 1280, 720, 1280, 720)
    assert binding.eligible(frames, binding.AVC, "mp4")
    for carrier, encoding in (("webm", binding.AVC), ("mp4", ("hevc", "yuv420p", "hvc1"))):
        assert not binding.eligible(frames, encoding, carrier)
    alternate = frames.to_json()
    alternate["pts"] = [i * 1001 for i in range(120)]
    assert not binding.eligible(FrameTable.from_json(alternate), binding.AVC, "mp4")
    with pytest.raises(ValueError, match="source clock"):
        binding.validate_clock(SOURCE | {"timeline_origin_s": 1}, binding.TICKS)


@pytest.fixture
def measured_record():
    y, x = np.indices((720, 1280))
    pattern = ((x // 15 + y // 15) % 2 * 160 + 40).astype(np.uint8)
    picture = cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB)
    softened = cv2.GaussianBlur(picture, (17, 17), 2)
    frames = {i: softened if START <= i < END else picture for i in range(START - 3, END + 3)}
    rgb = {
        i: detector.aligned_rgb(pixels, binding.face_landmarks({"landmarks_hex": LANDMARKS}))
        for i, pixels in frames.items()
    }
    _, target, detail = detector._match(rgb, START, END)
    rows = []
    for i, pixels in frames.items():
        middle = START <= i < END
        base = {"box": BOX, "landmarks_hex": LANDMARKS}
        box, landmarks = binding.interpolate(base, base, i, START, END) if middle else (BOX, LANDMARKS)
        # Produce the committed RGB with the exact interpolation arithmetic.
        crop = detector.aligned_rgb(pixels, binding.face_landmarks({"landmarks_hex": landmarks}))
        assert np.array_equal(crop, rgb[i])
        rows.append(
            {
                "index": i,
                "pts": binding.TICKS[i],
                "geometry": "interpolated" if middle else "observed",
                "box": box,
                "landmarks_hex": landmarks,
                "decoded_rgb_sha256": sha256(pixels).hexdigest(),
                "aligned_rgb_sha256": sha256(crop).hexdigest(),
                **detail[i],
            }
        )
    readings = (0.55, 0.59, 0.12, 0.10, 0.49, 0.50)
    response = min(readings[4], readings[5], min(readings[:2]) - max(readings[2:4]))
    record = {
        "method": binding.METHOD,
        "time_base": binding.TIME_BASE,
        "start_frame_index": START,
        "end_frame_index": END,
        "start_pts": binding.TICKS[START],
        "end_pts": binding.TICKS[END],
        "response_hex": response.hex(),
        "readings_hex": dict(zip(binding.READINGS, (v.hex() for v in readings), strict=True)),
        "proposal_hex": dict(zip(binding.SUPPORT, (v.hex() for v in (0.4, 0.3, 5.0)), strict=True)),
        "target_detail_hex": target.hex(),
        "views": rows,
        "scene_cuts": [32],
    }
    return record, frames


def test_matched_detail_replays_exact_crops_and_rejects_changed_source(measured_record):
    record, original = measured_record
    binding.validate_record(record, SOURCE, table_pts=binding.TICKS)
    assert binding.anchor_indices(record) == (72, 74, 76)
    assert any(row["sigma_hex"] != float(0).hex() for row in record["views"])
    detector.verify_crops(record, original)
    tampered = deepcopy(record)
    tampered["views"][4]["matched_rgb_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="reproduce the matched detail"):
        detector.verify_crops(tampered, original)
    tampered["views"][4]["landmarks_hex"][0][0] = float(477).hex()
    with pytest.raises(ValueError, match="geometry differs"):
        binding.validate_record(tampered, SOURCE)
    # A nearby scene cut makes the same native observations ineligible.
    with pytest.raises(ValueError, match="shot boundary"):
        binding.validate_record(record | {"scene_cuts": [72]}, SOURCE)


def test_matched_detail_packet_binds_coverage_and_original_views(measured_record, monkeypatch):
    record, original = measured_record
    state = binding.coverage("localized", proposed=2, usable=1)
    source = SOURCE | {
        "sha256": "a" * 64,
        "coordinate_space": "display_rgb24_pixels",
        binding.COMMITMENT: canonical_hash(record),
        binding.COVERAGE_COMMITMENT: canonical_hash(state),
    }
    anchors = {
        item["index"]: {
            "frame_index": item["index"],
            "pts": item["pts"],
            "time_base": binding.TIME_BASE,
            "decoded_rgb_sha256": item["decoded_rgb_sha256"],
            "width": SOURCE["width"],
            "height": SOURCE["height"],
        }
        for item in record["views"]
    }
    monkeypatch.setattr(video_packets, "_packet_image", lambda *args: sha256(args[-1].tobytes()).hexdigest())
    packet = video_packets._matched_detail_packet(None, None, None, "test", source, record, anchors, original)
    document = {
        "version": binding.MATCHED_DETAIL_AVC_VERSION,
        "source": source,
        "coverage": {binding.COVERAGE: state},
        "packets": [packet],
    }
    binding.validate(document, canonical_hash)
    assert [frame["interval_role"] for frame in packet["frames"]] == ["before", "during", "after"]
    assert packet["affects_overall_confidence"] is False
    assert len(packet["artifacts"]) == 9
    state["measured_pairs"] = 2
    with pytest.raises(ValueError, match="published commitments"):
        binding.validate(document, canonical_hash)
    state["measured_pairs"] = 1
    packet["frames"][1]["anchor"]["pts"] += 1000
    with pytest.raises(ValueError, match="differs from its native witness"):
        binding.validate(document, canonical_hash)
