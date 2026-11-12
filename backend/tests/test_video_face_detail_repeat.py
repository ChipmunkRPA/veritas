"""Exact same-window inputs, scope and lower-detail selection on the authored clock."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.core.canonical import canonical_hash
from app.detectors.video import face_appearance, face_detail_repeat
from app.detectors.video.decode import FrameTable
from app.evidence.image_packets import PacketIntegrityError
from app.evidence.video_packets import face_detail_coverage


class AppearanceFixture:
    def alignCrop(self, image, detection):
        assert detection.shape == (1, 15)
        return np.ascontiguousarray(image[:112, :112])

    def feature(self, image):
        feature = np.zeros((1, 128), dtype=np.float32)
        feature[0, int(image.std() > 20 or image.mean() > 170)] = 1
        return feature


def clock():
    return FrameTable("1/25", tuple(range(100, 220)), (0,), "I"+"P"*119, 640, 360, 640, 360)


def fixture(*, spatial_detail=True):
    table = clock()
    centers = {"before": [5, 10], "during": [15, 20, 25, 30], "after": [35, 40]}
    clear = np.full((360, 640, 3), 128, dtype=np.uint8)
    middle = clear.copy()
    checker = ((np.indices((112, 112)).sum(axis=0) % 2) * 255).astype(np.uint8)
    middle[:112, :112] = checker[:, :, None] if spatial_detail else 215
    layout = [[33., 45.], [72., 46.], [52., 62.], [40., 80.], [68., 81.]]
    aligned = face_appearance.cpp_alignment(layout)
    frames, inputs = [], []
    for role, group in centers.items():
        for center in group:
            for index in range(center-1, center+2):
                rgb = middle if role == "during" else clear
                frames.append((index, table.pts[index], rgb))
                inputs.append({"frame_index": index, "pts": table.pts[index],
                               "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                               "aligned_rgb_sha256": hashlib.sha256(rgb[:112, :112].tobytes()).hexdigest(),
                               "bbox": {"x": 0, "y": 0, "w": 112, "h": 112}, "landmarks": layout,
                               "alignment_matrix": aligned.tolist(),
                               "alignment_matrix_hex": [[float(value).hex() for value in row] for row in aligned],
                               "alignment_side": 112, "aligned_size": 112})
    scores = {"response": 1., "middle_to_before": 1., "middle_to_after": 1., "flank_change": 0.}
    start, stop = centers["during"][0], centers["during"][-1]
    record = {"method": face_appearance.METHOD, "track_id": "T1", "time_base": table.time_base,
              "centers": centers, "frame_inputs": inputs, "start_pts": table.pts[start], "end_pts": table.pts[stop],
              "start_stream_s": table.seconds(start), "end_stream_s": table.seconds(stop), **scores,
              "measurement_hex": {name: value.hex() for name, value in scores.items()}}
    return table, record, frames


@pytest.mark.parametrize(("spatial_detail", "expected"), [(True, "withheld"), (False, "retained")])
def test_repeat_uses_the_same_verified_triplets(spatial_detail, expected):
    table, coarse, frames = fixture(spatial_detail=spatial_detail)
    result = face_detail_repeat.measure(coarse, table, face_detail_repeat.ENCODING, reversed(frames), AppearanceFixture())
    assert result["status"] == expected
    assert result["candidate_sha256"] == canonical_hash(coarse)
    assert float.fromhex(result["original_response_hex"]) == 1.
    assert float.fromhex(result["measurements_hex"]["response"]) == (0. if spatial_detail else 1.)
    for measured, (index, pts, source) in zip(result["frame_inputs"], frames, strict=True):
        reduced = cv2.resize(cv2.resize(source[:112, :112], (56, 56), interpolation=cv2.INTER_AREA),
                             (112, 112), interpolation=cv2.INTER_LINEAR)
        assert measured == {"frame_index": index, "pts": pts, "lower_detail_rgb_sha256": hashlib.sha256(reduced.tobytes()).hexdigest()}
    face_detail_repeat.validate(result, coarse)


def test_scope_requires_the_exact_qualified_clock_and_encoding():
    table = clock()
    assert face_detail_repeat.supported(table, ("h264", "yuv420p"))
    for changed in (replace(table, time_base="1/24"), replace(table, pts=table.pts[:-1]), replace(table, rotation=90),
                    replace(table, truncated=True), replace(table, interpolated=(81,)), replace(table, height=352),
                    replace(table, pts=(*table.pts[:-1], table.pts[-1]+1))):
        assert not face_detail_repeat.supported(changed, ("h264", "yuv420p"))
    assert not face_detail_repeat.supported(table, ("hevc", "yuv420p"))
    assert not face_detail_repeat.supported(table, ("h264", "yuv444p"))


def test_repeat_rejects_incomplete_and_unverified_pictures_and_changed_alignment():
    table, record, frames = fixture()
    check = face_detail_repeat.measure
    encoding = face_detail_repeat.ENCODING
    with pytest.raises(ValueError, match="missing one or more selected"):
        check(record, table, encoding, frames[1:], AppearanceFixture())
    index, _, pixels = frames[0]
    with pytest.raises(ValueError, match="same verified source picture"):
        check(record, table, encoding, [(index, None, pixels), *frames[1:]], AppearanceFixture())
    with pytest.raises(ValueError, match="same verified source picture"):
        check(record, table, encoding, [frames[0], *frames], AppearanceFixture())
    _, pts, rgb = frames[0]
    different = rgb.copy()
    different[-1, -1, 0] ^= 1
    with pytest.raises(ValueError, match="same verified source picture"):
        check(record, table, encoding, [(index, pts, different), *frames[1:]], AppearanceFixture())
    corrupted = deepcopy(record)
    corrupted["frame_inputs"][0]["aligned_rgb_sha256"] = "0"*64
    with pytest.raises(ValueError, match="recorded aligned face crop"):
        check(corrupted, table, encoding, frames, AppearanceFixture())


def test_threshold_equality_withholds_and_tampering_is_rejected():
    table, record, frames = fixture()
    proof = face_detail_repeat.measure(record, table, face_detail_repeat.ENCODING, frames, AppearanceFixture())
    point = face_appearance.REVIEW_POINT.hex()
    for name in ("response", "middle_to_before", "middle_to_after"):
        proof["measurements_hex"][name] = point
    face_detail_repeat.validate(proof, record)
    altered = deepcopy(proof)
    altered["status"] = "retained"
    with pytest.raises(ValueError, match="inconsistent appearance measurements"):
        face_detail_repeat.validate(altered, record)
    altered = deepcopy(proof)
    altered["frame_inputs"][0]["pts"] += 1
    with pytest.raises(ValueError, match="same native picture inputs"):
        face_detail_repeat.validate(altered, record)
    altered = deepcopy(proof)
    altered["candidate_sha256"] = "0"*64
    with pytest.raises(ValueError, match="tracked appearance window"):
        face_detail_repeat.validate(altered, record)


def test_public_repeat_coverage_reuses_native_clock_and_rejects_changed_sources():
    table, record, frames = fixture()
    proof = face_detail_repeat.measure(record, table, face_detail_repeat.ENCODING, frames, AppearanceFixture())
    fields = ("start_pts", "end_pts", "time_base", "start_stream_s", "end_stream_s")
    repeat = {**{key: record[key] for key in fields}, "status": "withheld", "measurement": proof}
    details = {"face_sequence": None, "face_sequence_windows": [record], "face_detail_repeat": repeat}
    result = {"signals": [{"status": "ok", "effective_weight": 0., "details": details}]}
    anchors = {item["frame_index"]: item for item in record["frame_inputs"]}
    coverage = face_detail_coverage(result, table, anchors)
    assert coverage["status"] == "withheld" and coverage["affects_overall_confidence"] is False
    assert all(coverage[key] == record[key] for key in fields)
    assert "was withheld from the review ranges" in coverage["observation"]
    assert "measurement" not in coverage and "candidate_sha256" not in coverage
    assert face_detail_repeat.original_range(details) is record
    with pytest.raises(PacketIntegrityError, match="committed frame"):
        face_detail_coverage(result, table, {})
    repeat["end_pts"] += 1
    with pytest.raises(PacketIntegrityError, match="selected native range"):
        face_detail_coverage(result, table, anchors)


def test_deadline_check_is_never_converted_to_a_repeated_measurement():
    table, record, frames = fixture()

    def expired():
        raise TimeoutError("repeat budget expired")

    with pytest.raises(TimeoutError, match="repeat budget expired"):
        face_detail_repeat.measure(record, table, face_detail_repeat.ENCODING, frames, AppearanceFixture(), check=expired)
