import json
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.video import face_single_boundary as review
from app.detectors.video.decode import FrameTable
from app.engine.detector import Profile
from app.engine.stages import video
from app.evidence import face_single_boundary_binding as binding


def test_single_boundary_records_provenance_and_profile_exclusions(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "video_face_single_boundary_review", True)
    monkeypatch.setattr(settings, "video_face_single_sparse_context_review", False)
    monkeypatch.setattr(settings, "video_face_single_boundary_native30_review", False)
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

    monkeypatch.setattr(settings, "video_face_single_sparse_context_review", True)
    assert video._single_boundary_review(ctx) == {review.COVERAGE: review.coverage("unavailable", "profile", sparse=True)}

    monkeypatch.setattr(settings, "video_face_single_boundary_review", False)
    assert video._single_boundary_review(ctx) == {}


def test_single_boundary_sparse_switch_invalidates_saved_detection(monkeypatch):
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "video_face_single_boundary_review", True)
    monkeypatch.setattr(settings, "video_face_single_sparse_context_review", False)
    monkeypatch.setattr(settings, "video_face_single_boundary_native30_review", False)
    monkeypatch.setattr(settings, "video_face_single_boundary_native2997_review", False)
    monkeypatch.setattr(video, "current_voice_detection_output", lambda _: True)
    monkeypatch.setattr(video, "same_frame_enabled", lambda: False)
    monkeypatch.setattr(video, "detail_enabled", lambda: False)
    monkeypatch.setattr(video, "local_return_enabled", lambda: False)
    old = {video.SINGLE_FACE_BOUNDARY_POLICY: True}
    assert video.current_detect_output(old)
    monkeypatch.setattr(settings, "video_face_single_sparse_context_review", True)
    assert not video.current_detect_output(old)
    assert video.current_detect_output(old | {video.SINGLE_FACE_SPARSE_POLICY: True})
    monkeypatch.setattr(settings, "video_face_single_boundary_native30_review", True)
    current_sparse = old | {video.SINGLE_FACE_SPARSE_POLICY: True}
    assert not video.current_detect_output(current_sparse)
    current_native = current_sparse | {video.SINGLE_FACE_NATIVE30_POLICY: True}
    assert video.current_detect_output(current_native)
    monkeypatch.setattr(settings, "video_face_single_boundary_native2997_review", True)
    previous_ntsc = current_native | {video.SINGLE_FACE_NATIVE2997_POLICY: True}
    assert not video.current_detect_output(previous_ntsc)
    assert video.current_detect_output(previous_ntsc | {video.SINGLE_FACE_NTSC_CARRIER_POLICY: True})


def _native30_table():
    return FrameTable(time_base="1/30000", pts=tuple(range(0, 120000, 1000)), key_frames=(0,),
                      pict_types="I" + "P" * 119, width=640, height=360, coded_width=640, coded_height=360)


def test_native30_review_is_opt_in_full_and_requires_the_exact_source(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "video_face_single_boundary_review", True)
    monkeypatch.setattr(settings, "video_face_single_sparse_context_review", True)
    monkeypatch.setattr(settings, "video_face_single_boundary_native30_review", False)
    monkeypatch.setattr(video.provenance_gate, "stop", lambda _: False)
    path = tmp_path / "source.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isommp41")
    table = _native30_table()
    encoding = ("h264", "yuv420p", "avc1")
    info = dict(zip(("codec", "pix_fmt", "codec_tag"), encoding, strict=True))
    ctx = SimpleNamespace(profile=Profile.FULL, outputs={"probe": {"info": info}}, state={
        "frame_store": SimpleNamespace(path=path, table=table),
        "registry": SimpleNamespace(manifest=SimpleNamespace(model=lambda _: None)),
    })
    assert binding.native30_source(table, encoding, "mp4")
    assert not binding.native30_source(replace(table, pts=tuple(1000 + v for v in table.pts)), encoding, "mp4")
    assert not binding.native30_source(replace(table, time_base="1/29970"), encoding, "mp4")
    assert not binding.native30_source(replace(table, interpolated=(17,)), encoding, "mp4")
    assert not binding.native30_source(replace(table, coded_width=656), encoding, "mp4")
    assert not binding.native30_source(table, encoding, "webm")
    assert video._single_boundary_review(ctx)[review.COVERAGE] == review.coverage("unavailable", "profile", sparse=True)

    monkeypatch.setattr(settings, "video_face_single_boundary_native30_review", True)
    assert video._single_boundary_review(ctx)[review.COVERAGE] == review.coverage("unavailable", "models_unavailable", native30=True)
    info["codec_tag"] = "avc3"
    assert video._single_boundary_review(ctx)[review.COVERAGE] == review.coverage("unavailable", "profile", native30=True)
    info["codec_tag"] = "avc1"
    ctx.profile = Profile.DEEP
    assert video._single_boundary_review(ctx)[review.COVERAGE] == review.coverage("unavailable", "profile", sparse=True)

    ctx.profile = Profile.FULL
    ctx.state["frame_store"].table = replace(table, time_base="1/25", pts=tuple(range(100)))
    assert video._single_boundary_review(ctx)[review.COVERAGE] == review.coverage("unavailable", "models_unavailable", sparse=True)


def test_native30_coverage_rejects_legacy_clock_and_incomplete_measurement():
    table = _native30_table()
    coverage = review.coverage("no_match", frames_scanned=120, measured_face_frames=46, native30=True)
    evidence = {"source": {"time_base": table.time_base, "frame_count": 120, "width": 640, "height": 360,
                           binding.COVERAGE_COMMITMENT: canonical_hash(coverage)},
                "packets": [], "coverage": {binding.COVERAGE: coverage}}
    binding.validate(evidence, canonical_hash, table_pts=table.pts)
    with pytest.raises(ValueError, match="30 fps source clock"):
        binding.validate(evidence, canonical_hash, table_pts=tuple(1000 + v for v in table.pts))
    for field, value in (("frames_scanned", 119), ("measured_face_frames", 45), ("method", binding.SPARSE_METHOD)):
        edited = coverage | {field: value}
        tampered = evidence | {"source": evidence["source"] | {binding.COVERAGE_COMMITMENT: canonical_hash(edited)},
                               "coverage": {binding.COVERAGE: edited}}
        with pytest.raises(ValueError, match="Invalid face transition coverage"):
            binding.validate(tampered, canonical_hash, table_pts=table.pts)


def test_native2997_review_requires_exact_native_frame_positions(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "video_face_single_boundary_review", True)
    monkeypatch.setattr(settings, "video_face_single_sparse_context_review", False)
    monkeypatch.setattr(settings, "video_face_single_boundary_native30_review", True)
    monkeypatch.setattr(settings, "video_face_single_boundary_native2997_review", False)
    monkeypatch.setattr(video.provenance_gate, "stop", lambda _: False)
    path = tmp_path / "source.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isommp41")
    pts = tuple(index * 1001 for index in range(120))
    table = replace(_native30_table(), pts=pts)
    profile = binding.NATIVE2997_METHOD
    encoding = ("h264", "yuv420p", "avc1")
    info = dict(zip(("codec", "pix_fmt", "codec_tag"), encoding, strict=True))
    ctx = SimpleNamespace(profile=Profile.FULL, outputs={"probe": {"info": info}}, state={
        "frame_store": SimpleNamespace(path=path, table=table),
        "registry": SimpleNamespace(manifest=SimpleNamespace(model=lambda _: None)),
    })
    assert binding.native30_source(table, encoding, "mp4", profile)
    assert not binding.native30_source(table, encoding, "mp4")
    for variant in (pts[1:], tuple(value + 1 for value in pts), pts[:40] + (pts[40] + 1,) + pts[41:]):
        assert not binding.native30_source(replace(table, pts=variant), encoding, "mp4", profile)
    assert video._single_boundary_review(ctx)[review.COVERAGE]["reason"] == "profile"
    monkeypatch.setattr(settings, "video_face_single_boundary_native2997_review", True)
    monkeypatch.setattr(review.mp4_source_clock, "verified", lambda *_: True)
    info["timeline_origin_s"] = 0
    assert video._single_boundary_review(ctx)[review.COVERAGE] == review.coverage(
        "unavailable", "models_unavailable", native30=True,
    ) | {"method": profile}

    coverage = review.coverage("no_match", frames_scanned=120, measured_face_frames=60, native30=True) | {"method": profile}
    evidence = {"source": {"time_base": table.time_base, "frame_count": 120, "width": 640, "height": 360,
                           binding.COVERAGE_COMMITMENT: canonical_hash(coverage)},
                "packets": [], "coverage": {binding.COVERAGE: coverage}}
    binding.validate(evidence, canonical_hash, table_pts=table.pts)
    with pytest.raises(ValueError, match="29.97 fps source clock"):
        binding.validate(evidence, canonical_hash, table_pts=_native30_table().pts)


@pytest.mark.parametrize(("incumbent", "alternatives", "localized", "status", "reason"), [
    (.4, (34, 40), 32, "localized", None),
    (.1, (34, 35), 34, "localized", None),
    (.1, (34, 40), None, "unavailable", "conflicting_boundary_context"),
    (.1, (), 32, "no_match", None),
])
def test_sparse_selection_preserves_incumbent_and_rejects_disagreement(monkeypatch, incumbent, alternatives, localized, status, reason):
    source = FrameTable(time_base="1/25", pts=tuple(range(100)), key_frames=(0,), pict_types="I" + "P" * 99,
                        width=640, height=360, coded_width=640, coded_height=360)
    shot = SimpleNamespace(frames_seen=100, boundaries=[], update=lambda *_: None)
    monkeypatch.setattr(review, "ShotDetector", lambda *_args, **_kwargs: shot)
    monkeypatch.setattr(review, "decode_frames", lambda *_args, **_kwargs: (
        SimpleNamespace(index=index, pts=index, pts_verified=True, rgb=None) for index in source.pts))
    monkeypatch.setattr(review.face_local_detail, "observe", lambda *_args, **_kwargs: (SimpleNamespace(vector=True), None, None, False))
    proposal = {"coarse": 32, "persistent": incumbent, "jump_excess": .2}
    monkeypatch.setattr(review, "proposals", lambda *_args, **_kwargs: [proposal])
    monkeypatch.setattr(review, "_refine", lambda *_args, **_kwargs: proposal | {"boundary": 32})

    def extra(*_args, **_kwargs):
        assert incumbent < review.POINT
        return [{"persistent": .4, "boundary": edge} for edge in alternatives]

    monkeypatch.setattr(review, "_sparse_readings", extra)
    reading, disposition = review.measure(None, source, None, None, check=lambda: None, timeout_s=lambda: 1., sparse=True)
    assert (reading["boundary"] if reading else None) == localized
    assert disposition == review.coverage(status, reason, frames_scanned=100, measured_face_frames=50, sparse=True)


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
