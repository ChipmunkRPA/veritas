"""Native-clock and source-image invariants for the optional matched-detail check."""

from __future__ import annotations

import shutil
import subprocess
from copy import deepcopy
from hashlib import sha256
from struct import pack_into
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.core.canonical import canonical_hash
from app.detectors.video import face_matched_detail as detector
from app.detectors.video import frames as video_frames
from app.detectors.video.adapters import face_appearance
from app.detectors.video.decode import FrameTable, probe_frames
from app.engine.boundary import Deadline
from app.engine.detector import Profile, StageOutputs
from app.engine.stages import video as video_stages
from app.evidence import face_matched_detail_binding as binding
from app.evidence import mp4_source_clock, video_packets
from app.evidence.mp4 import bounded_boxes

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
AAC48 = {"rate": 48000, "channels": 1, "start_pts": 0, "end_pts": 230400, "priming_samples": 1024}
AAC48_EXTENDED = AAC48 | {"priming_samples": 2112, "trailing_samples": 960}


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


def test_matched_detail_scope_distinguishes_exact_25_fps_clocks():
    profile = binding.PROFILE_25
    pts = tuple(i * profile.step for i in range(120))
    frames = FrameTable(profile.time_base, pts, (0,), "I" + "P" * 119, 1280, 720, 1280, 720)
    assert binding.input_profile(frames, binding.AVC, "mp4") == profile
    for change in ({"pts": [i * 3003 for i in range(120)]},
                   {"pts": [i * 1200 for i in range(120)], "time_base": "1/30000"},
                   {"pts": [i * profile.step + 1 for i in range(120)]}):
        assert binding.input_profile(FrameTable.from_json(frames.to_json() | change), binding.AVC, "mp4") is None


def test_matched_detail_hevc_scope_requires_hvc1_and_exact_native_geometry():
    profile = binding.PROFILE_HEVC25
    table = FrameTable(profile.time_base, tuple(i * profile.step for i in range(120)), (0,), "I" + "P" * 119,
                       1280, 720, 1280, 720)
    assert profile.point == binding.PROFILE_25.point
    assert binding.input_profile(table, binding.HEVC, "mp4") is profile
    assert binding.input_profile(table, binding.HEVC, "mp4", has_audio=True) is binding.PROFILE_HEVC25_AAC
    assert binding.input_profile(table, binding.AVC, "mp4", has_audio=True) is binding.PROFILE_25
    for encoding in (("hevc", "yuv420p", "hev1"), ("hevc", "yuv420p10le", "hvc1")):
        assert binding.input_profile(table, encoding, "mp4") is None
    for change in ({"rotation": 90}, {"width": 720}, {"time_base": "1/30000"},
                   {"pts": [1 + i * profile.step for i in range(120)]}):
        altered = FrameTable.from_json(table.to_json() | change)
        assert binding.input_profile(altered, binding.HEVC, "mp4") is None


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="requires FFmpeg source replay")
def test_matched_detail_hevc_rejects_changed_source_track_layout(tmp_path):
    common = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1"]
    encoders = subprocess.check_output([*common, "-encoders"], text=True, timeout=10)
    if "libx265" not in encoders:
        pytest.skip("requires an HEVC encoder for the native clock witness")
    exact = tmp_path / "video-only.mp4"
    other_tag = tmp_path / "other-tag.mp4"
    soundtrack = tmp_path / "audio-present.mp4"
    time_base = ["-video_track_timescale", "90000", "-movie_timescale", "90000", "-use_editlist", "1"]
    subprocess.run([*common, "-f", "lavfi", "-i", "color=black:size=1280x720:rate=25:duration=4.8",
                    "-c:v", "libx265", "-preset", "ultrafast", "-threads", "1", "-x265-params",
                    "pools=none:frame-threads=1:wpp=0:log-level=error", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
                    *time_base, str(exact)], check=True, timeout=30)
    subprocess.run([*common, "-i", str(exact), "-map", "0:v:0", "-c:v", "copy", "-tag:v", "hev1",
                    *time_base, str(other_tag)], check=True, timeout=20)
    subprocess.run([*common, "-i", str(exact), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=4.8",
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-tag:v", "hvc1", "-c:a", "aac",
                    *time_base, str(soundtrack)], check=True, timeout=20)
    profile = binding.PROFILE_HEVC25
    detector.verify_profile(exact, probe_frames(exact), profile)
    for changed in (other_tag, soundtrack):
        table = probe_frames(changed)
        assert table.pts == probe_frames(exact).pts
        assert mp4_source_clock.verified(changed, table, profile.step)
        with pytest.raises(ValueError, match="recorded matched-detail profile"):
            detector.verify_profile(changed, table, profile)
    with_audio = binding.PROFILE_HEVC25_AAC
    audio_table = probe_frames(soundtrack)
    assert detector.verify_profile(soundtrack, audio_table, with_audio) == AAC48
    assert mp4_source_clock.aac_track_schedule(soundtrack, audio_table.tb * with_audio.step * len(audio_table), 1024) == (
        48000, 230400, [1024] * 226)
    state = binding.coverage("no_match", proposed=1, usable=1, profile=with_audio, soundtrack_clock=AAC48)
    detector.verify_coverage_source(soundtrack, audio_table, state)
    with pytest.raises(ValueError, match="soundtrack clock differs from the original"):
        detector.verify_coverage_source(soundtrack, audio_table, state | {binding.SOUNDTRACK_CLOCK: AAC48 | {"channels": 2}})
    with pytest.raises(ValueError, match="recorded matched-detail profile"):
        detector.verify_profile(exact, probe_frames(exact), with_audio)

    delayed = tmp_path / "offset-audio.mp4"
    subprocess.run([*common, "-i", str(exact), "-itsoffset", "0.12", "-i", str(soundtrack), "-map", "0:v:0",
                    "-map", "1:a:0", "-c", "copy", *time_base, str(delayed)], check=True, timeout=20)
    delayed_table = probe_frames(delayed)
    assert delayed_table.pts == audio_table.pts and mp4_source_clock.verified(delayed, delayed_table, with_audio.step)
    with pytest.raises(ValueError, match="recorded matched-detail profile"):
        detector.verify_profile(delayed, delayed_table, with_audio)

    # A platform-independent test file with complete 1024-sample AAC packets. The edit selects the
    # exact native picture span after the first 2112 samples and excludes the final 960 samples.
    full_packets = tmp_path / "full-aac-packets.mp4"
    subprocess.run([*common, "-i", str(exact), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                    "-af", "atrim=end_sample=232448", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    *time_base, str(full_packets)], check=True, timeout=20)
    patched = bytearray(full_packets.read_bytes())
    with full_packets.open("rb") as stream:
        size, _, boxes, one, payload = bounded_boxes(stream)

        def children(box):
            return boxes(box.start, box.end)

        movie = children(one(boxes(0, size), b"moov"))
        track = children([b for b in movie if b.kind == b"trak"][1])
        mvhd, tkhd, elst = one(movie, b"mvhd"), one(track, b"tkhd"), one(children(one(track, b"edts")), b"elst")
        assert all(payload(b)[0] == 0 for b in (mvhd, tkhd, elst))
        for offset in (mvhd.start + 16, tkhd.start + 20, elst.start + 8):
            pack_into(">I", patched, offset, 432000)
        pack_into(">i", patched, elst.start + 12, 2112)
    extended = tmp_path / "exact-extended-priming.mp4"
    extended.write_bytes(patched)
    profile_extended = binding.PROFILE_HEVC25_AAC_EXTENDED
    extended_table = probe_frames(extended)
    assert extended_table == audio_table
    assert detector.verify_profile(extended, extended_table, profile_extended) == AAC48_EXTENDED
    assert detector.verify_profile(extended, extended_table, with_audio, allow_extended_aac=True) == AAC48_EXTENDED
    assert detector.verify_profile(soundtrack, audio_table, with_audio, allow_extended_aac=True) == AAC48
    with pytest.raises(ValueError, match="recorded matched-detail profile"):
        detector.verify_profile(extended, extended_table, with_audio)
    with pytest.raises(ValueError, match="recorded matched-detail profile"):
        detector.verify_profile(soundtrack, audio_table, profile_extended)
    extended_state = binding.coverage("no_match", proposed=1, usable=1, profile=profile_extended,
                                      soundtrack_clock=AAC48_EXTENDED)
    detector.verify_coverage_source(extended, extended_table, extended_state)
    with pytest.raises(ValueError, match="soundtrack clock differs from the original"):
        detector.verify_coverage_source(extended, extended_table, extended_state | {
            binding.SOUNDTRACK_CLOCK: AAC48_EXTENDED | {"trailing_samples": 959},
        })
    wrong_edit = tmp_path / "wrong-extended-priming.mp4"
    pack_into(">i", patched, elst.start + 12, 2048)
    wrong_edit.write_bytes(patched)
    with pytest.raises(ValueError, match="recorded matched-detail profile"):
        detector.verify_profile(wrong_edit, extended_table, with_audio, allow_extended_aac=True)


@pytest.mark.parametrize(("hevc_enabled", "avc_enabled", "encode", "has_audio", "expected"), [
    (False, False, binding.HEVC, False, None),
    (False, True, binding.HEVC, False, "profile_avc"),
    (False, True, binding.HEVC, True, "profile_avc"),
    (True, False, binding.HEVC, False, binding.PROFILE_HEVC25),
    (True, False, binding.HEVC, True, binding.PROFILE_HEVC25_AAC),
    (True, False, binding.HEVC, True, binding.PROFILE_HEVC25_AAC_EXTENDED),
    (True, False, binding.AVC, False, None),
    (True, True, binding.AVC, True, binding.PROFILE_25),
    (True, False, ("hevc", "yuv420p", "hev1"), False, "profile_hevc"),
    (True, False, ("hevc", "yuv420p", "hev1"), True, "profile_hevc_aac"),
])
def test_matched_detail_hevc_uses_separate_opt_in_and_preserves_avc(monkeypatch, hevc_enabled, avc_enabled, encode, has_audio, expected):
    settings = face_appearance.settings
    for name, value in {
        "environment": "dev", "face_min_size": 48,
        "video_face_matched_detail_hevc_review": hevc_enabled,
        "video_face_matched_detail_avc_review": avc_enabled,
        "video_face_very_brief_avc_review": avc_enabled,
        "video_face_short_avc_review": avc_enabled,
    }.items():
        monkeypatch.setattr(settings, name, value)
    profile = binding.PROFILE_HEVC25
    table = FrameTable(profile.time_base, tuple(i * profile.step for i in range(120)), (0,), "I" + "P" * 119,
                       1280, 720, 1280, 720)
    inputs = SimpleNamespace(table=table, encoding=encode[:2], codec_tag=encode[2], has_audio=has_audio,
                             store=SimpleNamespace(path="source"))
    ctx = SimpleNamespace(profile=Profile.FULL, deadline=Deadline(30))
    result = SimpleNamespace(status="ok", details={"incumbent": "retained"})
    verify = []
    soundtrack = (AAC48_EXTENDED if expected == binding.PROFILE_HEVC25_AAC_EXTENDED
                  else AAC48 if expected == binding.PROFILE_HEVC25_AAC else None)

    def verify_profile(path, frame_table, received, **kwargs):
        assert path == "source" and frame_table is table
        verify.append(received)
        assert kwargs["allow_extended_aac"] is True
        return soundtrack

    monkeypatch.setattr(detector, "verify_profile", verify_profile)
    monkeypatch.setattr(detector, "inspect", lambda *args, profile: (None, binding.coverage("no_match", proposed=1, usable=1, profile=profile)))
    assert face_appearance._with_matched_detail(result, inputs, None, None, ctx) is result
    assert result.details["incumbent"] == "retained"
    if isinstance(expected, binding.NativeProfile):
        assert verify == [binding.PROFILE_HEVC25_AAC if expected == binding.PROFILE_HEVC25_AAC_EXTENDED else expected]
        assert result.details[binding.COVERAGE] == binding.coverage("no_match", proposed=1, usable=1, profile=expected,
            soundtrack_clock=soundtrack)
    elif expected is not None:
        assert not verify
        assert result.details[binding.COVERAGE] == binding.coverage("unassessed", expected)
    else:
        assert not verify and binding.COVERAGE not in result.details


@pytest.mark.parametrize(("encoding", "analysis_profile", "scanned", "expected"), [
    (binding.HEVC, Profile.FULL, 120, (67,)),
    (binding.HEVC, Profile.FULL, 119, None),
    (binding.HEVC, Profile.DEEP, 120, None),
    (("hevc", "yuv420p", "hev1"), Profile.FULL, 120, None),
    (binding.AVC, Profile.FULL, 120, None),
])
def test_matched_detail_hevc_receives_shot_context_without_avc(monkeypatch, encoding, analysis_profile, scanned, expected):
    settings = video_stages.settings
    for name, value in {
        "environment": "dev", "video_face_matched_detail_hevc_review": True,
        "video_face_matched_detail_avc_review": False, "video_face_very_brief_avc_review": False,
        "video_face_short_avc_review": False,
    }.items():
        monkeypatch.setattr(settings, name, value)
    profile = binding.PROFILE_HEVC25
    table = FrameTable(profile.time_base, tuple(i * profile.step for i in range(120)), (0,), "I" + "P" * 119,
                       1280, 720, 1280, 720)
    ctx = SimpleNamespace(profile=analysis_profile, state={
        "frame_store": SimpleNamespace(table=table),
        "registry": SimpleNamespace(enabled=lambda *_: [video_stages.APPEARANCE_REVIEW]),
    }, outputs={"shots": {"frames_scanned": scanned, "boundaries": [{"frame_index": 67}]}})
    seen = []

    def run_detectors(*args, outputs, **kwargs):
        if outputs.face_review is not None:
            seen.append(outputs.face_review.shot_boundaries)
        return {"detectors": []}

    probe = dict(zip(("codec", "pix_fmt", "codec_tag"), encoding, strict=True))
    monkeypatch.setattr(video_stages, "_outputs", lambda _: StageOutputs(probe=probe))
    monkeypatch.setattr(video_stages, "_single_boundary_review", lambda _: {})
    monkeypatch.setattr(video_stages.common, "run_detectors", run_detectors)
    monkeypatch.setattr(video_stages.common, "stage_exhibit_list", lambda *_: [])
    video_stages.detect(ctx)
    assert seen == [expected]


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="requires FFmpeg source replay")
def test_matched_detail_native25_rejects_non_square_encoded_pixels(tmp_path, monkeypatch):
    square = tmp_path / "ordinary-square.mp4"
    stretched = tmp_path / "anamorphic.mp4"
    common = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1"]
    time_base = ["-video_track_timescale", "90000", "-movie_timescale", "90000", "-use_editlist", "1"]
    subprocess.run([*common, "-f", "lavfi", "-i", "color=black:size=1280x720:rate=25", "-vf", "setsar=0/1", "-frames:v", "120",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", *time_base, str(square)],
                   check=True, timeout=20)
    subprocess.run([*common, "-i", str(square), "-c:v", "copy", "-bsf:v",
                    "h264_metadata=sample_aspect_ratio=4/3", *time_base, str(stretched)], check=True, timeout=20)
    profile = binding.PROFILE_25
    original = probe_frames(square)
    detector.verify_profile(square, original, profile)
    changed = probe_frames(stretched)
    assert video_frames.probe(stretched, include_sample_aspect_ratio=True)["sample_aspect_ratio"] == "4:3"
    assert binding.input_profile(changed, binding.AVC, "mp4") == profile
    assert mp4_source_clock.verified(stretched, changed, profile.step)
    with pytest.raises(ValueError, match="recorded matched-detail profile"):
        detector.verify_profile(stretched, changed, profile)

    metadata = video_frames.probe(square, include_codec_tag=True, include_sample_aspect_ratio=True)
    monkeypatch.setattr(video_frames, "probe", lambda *_args, **_kw: metadata | {"timeline_origin_s": -0.04})
    with pytest.raises(ValueError, match="recorded matched-detail profile"):
        detector.verify_profile(square, original, profile)


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


@pytest.mark.parametrize("profile", (binding.PROFILE_30, binding.PROFILE_25, binding.PROFILE_HEVC25,
                                     binding.PROFILE_HEVC25_AAC, binding.PROFILE_HEVC25_AAC_EXTENDED))
def test_matched_detail_packet_binds_coverage_and_original_views(measured_record, monkeypatch, profile):
    record, original = measured_record
    record.update(method=profile.method, time_base=profile.time_base, start_pts=START * profile.step,
                  end_pts=END * profile.step)
    for view in record["views"]:
        view["pts"] = view["index"] * profile.step
    soundtrack_clock = (AAC48_EXTENDED if profile == binding.PROFILE_HEVC25_AAC_EXTENDED
                        else AAC48 if profile == binding.PROFILE_HEVC25_AAC else None)
    if soundtrack_clock:
        record[binding.SOUNDTRACK_CLOCK] = soundtrack_clock
    state = binding.coverage("localized", proposed=2, usable=1, profile=profile, soundtrack_clock=soundtrack_clock)
    source = SOURCE | {
        "time_base": profile.time_base,
        "sha256": "a" * 64,
        "coordinate_space": "display_rgb24_pixels",
        binding.COMMITMENT: canonical_hash(record),
        binding.COVERAGE_COMMITMENT: canonical_hash(state),
    }
    anchors = {
        item["index"]: {
            "frame_index": item["index"],
            "pts": item["pts"],
            "time_base": profile.time_base,
            "decoded_rgb_sha256": item["decoded_rgb_sha256"],
            "width": SOURCE["width"],
            "height": SOURCE["height"],
        }
        for item in record["views"]
    }
    monkeypatch.setattr(video_packets, "_packet_image", lambda *args: sha256(args[-1].tobytes()).hexdigest())
    packet = video_packets._matched_detail_packet(None, None, None, "test", source, record, anchors, original)
    document = {
        "version": profile.version,
        "source": source,
        "coverage": {binding.COVERAGE: state},
        "packets": [packet],
    }
    binding.validate(document, canonical_hash)
    assert [frame["interval_role"] for frame in packet["frames"]] == ["before", "during", "after"]
    assert [frame["anchor"]["playback_time_s"] for frame in packet["frames"]] == [
        index / (30 if profile == binding.PROFILE_30 else 25) for index in (72, 74, 76)
    ]
    assert packet["affects_overall_confidence"] is False
    assert len(packet["artifacts"]) == 9
    if soundtrack_clock:
        state[binding.SOUNDTRACK_CLOCK] = soundtrack_clock | {"channels": 2}
        source[binding.COVERAGE_COMMITMENT] = canonical_hash(state)
        with pytest.raises(ValueError, match="confidence-neutral source interval"):
            binding.validate(document, canonical_hash)
        state[binding.SOUNDTRACK_CLOCK] = soundtrack_clock
        source[binding.COVERAGE_COMMITMENT] = canonical_hash(state)
        if profile == binding.PROFILE_HEVC25_AAC_EXTENDED:
            state[binding.SOUNDTRACK_CLOCK] = soundtrack_clock | {"trailing_samples": 224}
            source[binding.COVERAGE_COMMITMENT] = canonical_hash(state)
            with pytest.raises(ValueError, match="soundtrack clock differs from its verified profile"):
                binding.validate(document, canonical_hash)
            state[binding.SOUNDTRACK_CLOCK] = soundtrack_clock
            source[binding.COVERAGE_COMMITMENT] = canonical_hash(state)
    state["measured_pairs"] = 2
    with pytest.raises(ValueError, match="published commitments"):
        binding.validate(document, canonical_hash)
    state["measured_pairs"] = 1
    packet["frames"][1]["anchor"]["pts"] += 1000
    with pytest.raises(ValueError, match="differs from its native witness"):
        binding.validate(document, canonical_hash)
    packet["frames"][1]["anchor"]["pts"] -= 1000
    record["readings_hex"]["entry"] = record["response_hex"] = 0.15.hex()
    source[binding.COMMITMENT] = canonical_hash(record)
    if profile != binding.PROFILE_30:
        with pytest.raises(ValueError, match="response differs"):
            binding.validate(document, canonical_hash)
    else:
        binding.validate(document, canonical_hash)
