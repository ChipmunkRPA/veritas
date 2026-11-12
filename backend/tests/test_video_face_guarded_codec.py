"""Guarded codec scope and source replay for short face-review evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.detectors.image.faces import Face
from app.detectors.video import face_return
from app.detectors.video.decode import FrameTable
from app.evidence import face_return_binding as binding
from app.reports.verify_video_source import _verify_guarded_scene

WEBM_HEADER = bytes.fromhex("1a45dfa39f4286810142f7810142f2810442f381084282847765626d4287810242858102")
MP4_HEADER = bytes.fromhex("0000001c6674797069736f6d0000020069736f6d69736f326d703431")


@pytest.fixture
def table():
    return FrameTable("1/1000", tuple(range(0, 4800, 40)), (0,), "I" + "P" * 119, 640, 360, 640, 360)


@pytest.mark.parametrize("encoding, carrier, header, other", [
    (("vp9", "yuv420p"), "webm", WEBM_HEADER, MP4_HEADER),
    (("hevc", "yuv420p"), "mp4", MP4_HEADER, WEBM_HEADER),
])
def test_guarded_schedule_requires_carrier_and_exact_120_picture_grid(tmp_path, table, encoding, carrier, header, other):
    path = tmp_path / "native"
    path.write_bytes(header)
    assert face_return.source_carrier(path) == carrier
    plan, chosen = face_return.guarded_dense_schedule(table, encoding, path)
    assert chosen == carrier and plan[1] == tuple(range(0, 120, 2))
    for disallowed in (replace(table, pts=table.pts[:-1]), replace(table, rotation=180),
                       replace(table, interpolated=(10,)), replace(table, time_base="1/900"),
                       replace(table, width=638), replace(table, truncated=True)):
        assert face_return.guarded_dense_schedule(disallowed, encoding, path) is None
    assert face_return.guarded_dense_schedule(table, ("h264", "yuv420p"), path) is None
    assert face_return.guarded_dense_schedule(table, (encoding[0], "yuv420p10le"), path) is None
    path.write_bytes(other)
    assert face_return.guarded_dense_schedule(table, encoding, path) is None
    path.write_bytes(MP4_HEADER[:8] + b"qt  " + MP4_HEADER[12:])
    assert face_return.source_carrier(path) is None


def record(table):
    before, during, after = list(range(46, 54, 2)), list(range(54, 60, 2)), list(range(60, 68, 2))
    edges = [52, 54, 58, 60]
    box = [170, 90, 280, 180]
    digest = "a" * 64
    matrix = [[1., 0., 0.], [0., 1., 0.]]

    def frame(i):
        return {"frame_index": i, "pts": table.pts[i], "decoded_rgb_sha256": digest}

    return {
        "method": binding.GUARDED_CODEC_METHOD, "time_base": table.time_base,
        "start_frame_index": 54, "end_frame_index": 60,
        "start_pts": table.pts[54], "end_pts": table.pts[60],
        "start_stream_s": table.seconds(54), "end_stream_s": table.seconds(60),
        "response": .15, "response_hex": .15.hex(),
        "scene_correlations": [.90, .94], "scene_correlations_hex": [.90.hex(), .94.hex()],
        "scene_lower_quartiles": [.75, .80], "scene_lower_quartiles_hex": [.75.hex(), .80.hex()],
        "anchors": {"before": 52, "during": 56, "after": 60},
        "scheduled_frame_indices": list(range(0, 120, 2)),
        "support_indices": {"before": before, "during": during, "after": after},
        "scene_frame_inputs": [{**frame(i), "face_box": box.copy()} for i in edges],
        "frame_inputs": [{**frame(i), "bbox": dict(zip(("x", "y", "w", "h"), box, strict=True)),
                          "aligned_size": 112, "alignment_side": 112, "aligned_rgb_sha256": digest,
                          "landmarks": [[0., 0.]] * 5, "alignment_matrix": matrix,
                          "alignment_matrix_hex": [[v.hex() for v in row] for row in matrix]}
                         for i in before + during + after],
        "source_encoding": ["vp9", "yuv420p"], "source_carrier": "webm",
    }


def test_structural_guard_rejects_wrong_carrier_below_cutoff_and_legacy_version(table):
    valid = record(table)
    source = {"width": 640, "height": 360, "frame_count": 120, "time_base": table.time_base}
    binding.validate_record(valid, source, table_pts=table.pts)
    with pytest.raises(ValueError, match="requires its measured VP9/WebM or HEVC/MP4"):
        binding.validate_record({**valid, "source_carrier": "mp4"}, source, table_pts=table.pts)
    with pytest.raises(ValueError, match="differs from its exact value"):
        binding.validate_record({**valid, "scene_lower_quartiles": [.59, .80],
                                 "scene_lower_quartiles_hex": [.59.hex(), .80.hex()]}, source)
    changed_box = deepcopy(valid)
    changed_box["scene_frame_inputs"][0]["face_box"][0] += 1
    with pytest.raises(ValueError, match="disagrees with the measured face box"):
        binding.validate_record(changed_box, source)
    packets = [{"face_return": valid, "semantics": binding.GUARDED_CODEC_METHOD}]
    binding.validate_set({"version": "video-evidence-16", "packets": packets})
    with pytest.raises(ValueError, match="evidence version disagrees"):
        binding.validate_set({"version": "video-evidence-15", "packets": packets})


def test_native_background_replay_rejects_plausible_altered_quartile(table):
    stored = record(table)
    rng = np.random.default_rng(190)
    texture = rng.integers(40, 215, size=(64, 96, 3), dtype=np.uint8)
    rgb = cv2.resize(texture, (640, 360), interpolation=cv2.INTER_NEAREST)
    changed = rgb.copy()
    changed[90:270, 170:450] = (240, 20, 10)
    frames = {52: rgb, 54: changed, 58: changed, 60: rgb}
    face = Face(170, 90, 280, 180, 0.)
    a = face_return.Sample(face, None, face_return.scene(rgb, face))
    b = face_return.Sample(face, None, face_return.scene(changed, face))
    median, quartile = face_return.scene_correlation(a, b), face_return.scene_lower_quartile(a, b)
    assert median is not None and quartile is not None and quartile > .99
    stored["scene_correlations_hex"] = [median.hex()] * 2
    stored["scene_lower_quartiles_hex"] = [quartile.hex()] * 2
    _verify_guarded_scene(stored, frames)
    stored["scene_lower_quartiles_hex"][0] = .75.hex()
    with pytest.raises(ValueError, match="Guarded background support differs"):
        _verify_guarded_scene(stored, frames)
