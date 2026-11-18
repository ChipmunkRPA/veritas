import json
from fractions import Fraction
from types import SimpleNamespace

from app.core.canonical import canonical_json
from app.core.config import settings
from app.detectors.video import face_single_boundary as review
from app.detectors.video.decode import FrameTable
from app.engine.detector import Profile
from app.engine.stages import video


def test_single_boundary_records_provenance_and_profile_exclusions(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "video_face_single_boundary_review", True)
    ctx = SimpleNamespace(profile=Profile.FULL, state={}, outputs={})
    monkeypatch.setattr(video.provenance_gate, "stop", lambda _: True)
    assert video._single_boundary_review(ctx) == {review.COVERAGE: review.coverage("unavailable", "provenance")}

    monkeypatch.setattr(video.provenance_gate, "stop", lambda _: False)
    assert video._single_boundary_review(ctx) == {review.COVERAGE: review.coverage("unavailable", "unsupported_video")}

    path = tmp_path / "unsupported.webm"
    path.touch()
    ctx.outputs["probe"] = {"info": {"codec": "vp9", "pix_fmt": "yuv420p"}}
    ctx.state["frame_store"] = SimpleNamespace(path=path, table=None)
    assert video._single_boundary_review(ctx) == {review.COVERAGE: review.coverage("unavailable", "profile")}

    monkeypatch.setattr(settings, "video_face_single_boundary_review", False)
    assert video._single_boundary_review(ctx) == {}


def test_single_boundary_merges_source_pins_without_overwriting_a_conflict():
    table = FrameTable(time_base="1/25", pts=tuple(range(100)), key_frames=(0,), pict_types="I" + "P" * 99,
                       width=640, height=360, coded_width=640, coded_height=360)
    pins = {
        role: {"index": index, "pts": table.pts[index], "decoded_rgb_sha256": str(index) * 32}
        for role, index in (("before", 31), ("transition", 32), ("after", 43))
    }
    record = {"time_base": table.time_base, "source_frames": pins}
    disposition = review.coverage("localized", frames_scanned=100, measured_face_frames=50)
    detected = {review.KEY: record, review.COVERAGE: disposition}
    cached = {"frame_index": 32, "pts": 32, "time_base": table.time_base, "width": 640, "height": 360,
              "decoded_rgb_sha256": pins["transition"]["decoded_rgb_sha256"]}

    anchors, rendered = video._render_single_boundary(detected, [cached], table)
    assert [anchor["frame_index"] for anchor in anchors] == [31, 32, 43]
    assert anchors[1] == cached
    assert rendered == detected

    conflicting = cached | {"decoded_rgb_sha256": "0" * 64}
    anchors, rendered = video._render_single_boundary(detected, [conflicting], table)
    assert anchors == [conflicting]
    assert rendered == {review.COVERAGE: disposition | {"status": "unavailable", "reason": "native_measurement_unavailable"}}


def test_single_boundary_keeps_exact_crop_geometry_and_scores_after_canonical_rounding(monkeypatch):
    eyes = [[254.76583862304688, 169.1422882080078], [324.1093444824219, 169.707275390625]]
    readings = {"scene": .936256843, "persistent": .165750625, "edge_supported": .045168642,
                "jump_excess": .045168642, "fine_margin": .000000049, "fine_scene": .988200307}
    raw = readings | {"pts": 32000, "source_frames": {"transition": {"eyes": eyes}}}
    disposition = review.coverage("localized", frames_scanned=100, measured_face_frames=50)
    monkeypatch.setattr(review, "measure", lambda *args, **kwargs: (raw, disposition))
    record, status = review.inspect(None, SimpleNamespace(tb=Fraction(1, 25000)), None, None, check=None, timeout_s=None)
    saved = json.loads(canonical_json(record))

    assert status == disposition and saved["time_s"] == 1.28
    assert saved["fine_margin"] == 0 and raw["fine_margin"] > 0
    assert {name: float.fromhex(value) for name, value in saved["readings_hex"].items()} == readings
    frame = saved["source_frames"]["transition"]
    assert frame["eyes"] != eyes
    assert [[float.fromhex(value) for value in point] for point in frame["eyes_hex"]] == eyes
    assert raw["source_frames"]["transition"] == {"eyes": eyes}
