from __future__ import annotations

from dataclasses import replace

import pytest

from app.detectors.video import eye_local_hold, picture_hold
from app.detectors.video.decode import DecodedFrame, FrameTable
from app.evidence import eye_local_hold_binding as binding


@pytest.fixture
def table():
    return FrameTable("1/30000", tuple(range(150_000, 210_000, 1000)), (0,), "I" + "P" * 59,
                      480, 360, 480, 360)


def test_eye_observation_requires_exact_delivered_profile(table):
    assert binding.eligible(table, binding.ENCODING, "mp4")
    assert not binding.eligible(table, binding.ENCODING, "mkv")
    assert not binding.eligible(table, ("hevc", "yuv420p"), "mp4")
    assert not binding.eligible(table, ("h264", "yuv422p"), "mp4")
    for excluded in (replace(table, width=640), replace(table, rotation=180), replace(table, truncated=True),
                     replace(table, interpolated=(28,)), replace(table, pts=table.pts[:-1]),
                     replace(table, pts=(*table.pts[:-1], table.pts[-1] + 1)),
                     replace(table, time_base="1/25000")):
        assert not binding.eligible(excluded, binding.ENCODING, "mp4")


@pytest.mark.parametrize(("enabled", "verified", "scope", "status"), [
    (False, True, True, None),
    (True, True, False, "outside_scope"),
    (True, False, True, "unavailable"),
    (True, True, True, "no_match"),
])
def test_only_verified_eye_observations_are_measured(tmp_path, monkeypatch, table, enabled, verified, scope, status):
    import numpy as np

    source = tmp_path / "source.mp4"
    assert table.pts[0] > 0
    monkeypatch.setattr(picture_hold, "source_carrier", lambda _: "mp4")

    def decode(*_args, **_kwargs):
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        for i, pts in enumerate(table.pts):
            yield DecodedFrame(index=i, pts=pts, pts_verified=verified or i != 27, rgb=frame)

    calls = []

    def measure(frames, pts, hashes, **kwargs):
        assert len(frames) == binding.FRAME_COUNT and all(len(digest) == 64 for digest in hashes)
        calls.append(pts)
        return binding.record("no_match", pts=pts, hashes=hashes)

    monkeypatch.setattr(picture_hold, "decode_frames", decode)
    monkeypatch.setattr(eye_local_hold, "record", measure)
    if not scope:
        table = replace(table, time_base="1/25000")
    result = picture_hold.inspect(source, table, origin=5., motion_encoding=binding.ENCODING, eye_review=enabled)
    assert bool(calls) is (status == "no_match")
    review = result["coverage"].get(binding.KEY)
    if status is None:
        assert review is None
    else:
        assert review["status"] == status and review["packet_id"] is None and not review["affects_overall_confidence"]
        assert review["frames_scanned"] == (binding.FRAME_COUNT if calls else 0)
