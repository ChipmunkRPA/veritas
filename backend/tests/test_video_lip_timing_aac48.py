# SPDX-License-Identifier: MIT
import hashlib
import subprocess
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from app.detectors.audio.decode import decode_audio
from app.detectors.audio.phone_events import MODEL_BLOBS
from app.detectors.video import viseme_review, viseme_stereo
from app.detectors.video.decode import FrameTable
from app.evidence import video_lip_agreement as agreement
from app.evidence import video_viseme_binding as binding


def _aac_mapping(tmp_path, rate):
    path = tmp_path / f"sound-{rate}.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=400:sample_rate={rate}:duration=2.1", "-c:a", "aac", str(path)],
                   check=True, timeout=20)
    return decode_audio(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()).mapping


@pytest.fixture
def aac_mapping(tmp_path):
    return _aac_mapping(tmp_path, 48_000)


def test_441khz_dropout_requires_a_verified_mono_clock(tmp_path):
    mapping = _aac_mapping(tmp_path, 44_100)
    assert mapping[agreement.AAC_44100_CLOCK] is True
    assert mapping["source_duration_pts"] == 92_610
    assert mapping["samples"] > 33_600
    assert binding.presented_samples(mapping) == 33_600
    assert binding.presented_samples(mapping | {"source_duration_pts": 92_609}) == 33_599
    assert binding.audio_origin(mapping) == 0
    assert binding.presented_samples(mapping | {agreement.AAC_44100_CLOCK: False}) is None
    assert binding.presented_samples(mapping | {"source_profile": "HE-AAC"}) is None
    assert binding.presented_samples(mapping | {"source_start_pts": 1}) is None
    assert binding.audio_origin(mapping | {"source_channels": 2}) is None
    assert binding.audio_origin(mapping | {"timeline": [
        {"start_sample": 0, "end_sample": 500, "source_pts": 0},
        {"start_sample": 500, "end_sample": mapping["samples"], "source_pts": 501},
    ]}) is None

    # This encoding is admitted to the separate amplitude and geometry route.
    table = FrameTable(time_base="1/30", pts=tuple(range(63)), key_frames=(0,),
                       pict_types="I" + "P" * 62, width=640, height=360, coded_width=640, coded_height=360)
    assert viseme_review.supported_shots(table, mapping, [[0, 63]])
    assert not viseme_review.supported(table, mapping, [[0, 63]])
    with pytest.raises(ValueError, match="no supported continuous source clock"):
        binding.measured({"frames": [], "phones": []}, timestamps=[], time_base="1/30", source={}, mapping=mapping)


def test_aac441_motion_has_separate_qualified_picture_and_audio_clocks(tmp_path):
    mapping = _aac_mapping(tmp_path, 44_100)
    table = FrameTable(time_base="1/24", pts=tuple(range(51)), key_frames=(0,),
                       pict_types="I" + "P" * 50, width=720, height=480, coded_width=720, coded_height=480)
    assert viseme_review.supported_aac441_motion(table, mapping, [[0, 51]], 51)
    assert not viseme_review.supported(table, mapping, [[0, 51]])
    assert not viseme_review.supported_shots(table, mapping, [[0, 51]])
    assert not viseme_review.supported_aac441_motion(table, mapping, [[0, 25], [25, 51]], 51)
    assert not viseme_review.supported_aac441_motion(table, mapping, [[0, 51]], 50)
    assert not viseme_review.supported_aac441_motion(replace(table, coded_width=768), mapping, [[0, 51]], 51)
    assert not viseme_review.supported_aac441_motion(replace(table, time_base="1/30"), mapping, [[0, 51]], 51)
    assert not viseme_review.supported_aac441_motion(table, mapping | {"audio_stream_indices": [1, 2]}, [[0, 51]], 51)
    assert not viseme_review.supported_aac441_motion(table, mapping | {agreement.AAC_44100_CLOCK: False}, [[0, 51]], 51)

    frames = [{"index": i, "pts": i, "gap_units": 20_000, "mouth_box": [1, 1, 3, 3],
               "rgb_sha256": "1" * 64, "mouth_crop_sha256": "2" * 64} for i in table.pts]
    source = {"width": table.width, "height": table.height}
    scope = {"timestamps": table.pts, "time_base": "1/24", "source": source, "mapping": mapping}
    base = {"frames": frames, "phones": [], agreement.AAC_44100_MOTION_STATUS: "measured"}
    comparison = agreement.encode({"span": [12667, 16667], "picture_span": [19, 25],
                                   "bracket_frames": [18, 21, 26], "lag_samples": 0,
                                   "agreement": [0.75, 0.14, 0.70], "score": 0.56})
    raw = base | {agreement.KEY: comparison}
    result = binding.measured(raw, **scope)
    assert result["schema"] == binding.SCHEMA_WITH_AAC441_MOTION
    assert result["measurement_basis"] == agreement.KEY and result["affects_overall_confidence"] is False
    assert result["witness"]["picture_frames"] == {"before": 18, "during": 21, "after": 26}
    assert result["witness"]["audio_source_pts"] == [12667, 16667]
    assert result["inputs"]["samples"] == 33_600
    assert binding.measured(result["inputs"], **scope) == result
    for status, state in (("coverage", "insufficient_events"), ("whole_track_delay", "ambiguous_change"), ("no_local_dip", "no_match")):
        unresolved = binding.measured(base | {agreement.AAC_44100_MOTION_STATUS: status}, **scope)
        assert unresolved["status"] == state and unresolved["witness"] is None
        assert binding.measured(unresolved["inputs"], **scope) == unresolved
        if status == "whole_track_delay":
            assert "soundtrack offset" in unresolved["observation"] and "inconclusive" in unresolved["observation"]
    with pytest.raises(ValueError, match="qualified input"):
        binding.measured(raw | {"phones": [["closure", 200, 0.9]]}, **scope)
    with pytest.raises(ValueError, match="qualified input"):
        binding.measured(raw, **(scope | {"source": {"width": 640, "height": 360}}))
    with pytest.raises(ValueError, match="native timestamps"):
        binding.measured(raw | {agreement.KEY: comparison | {"span": [12668, 16667]}}, **scope)


def test_aac441_motion_routes_without_phone_checkpoint(tmp_path, monkeypatch):
    mapping = _aac_mapping(tmp_path, 44_100)
    table = FrameTable(time_base="1/24", pts=tuple(range(51)), key_frames=(0,),
                       pict_types="I" + "P" * 50, width=720, height=480, coded_width=720, coded_height=480)
    geometry = tmp_path / "geometry.task"
    geometry.touch()
    for name in ("vision", "shared"):
        (tmp_path / f"{name}.safetensors").touch()
    for field, value in {"video_lip_aac441_motion_review": False, "video_lip_timing_model_dir": None,
                         "video_lip_timing_runtime": None, "video_lip_motion_sound_models": tmp_path,
                         "video_lip_timing_face_asset": geometry, "video_lip_silent_mouth_review": True,
                         "video_lip_aac48_articulation_review": True, "video_lip_articulation_review": True}.items():
        monkeypatch.setattr(viseme_review.settings, field, value)
    ctx = SimpleNamespace(deadline=SimpleNamespace(remaining=lambda: 50), store=SimpleNamespace(path_for=lambda _: tmp_path / "sound.wav"))
    store = SimpleNamespace(table=table, path=tmp_path / "video.mov")
    args = (ctx, store, mapping, {"wav_sha256": "0" * 64}, [[0, 51]])
    assert viseme_review.assess(*args, shot_frames=51) == binding.skipped("unsupported")

    def compare(request, _):
        assert request["motion_only"] is True and request["checkpoint"] is None
        assert not request["articulation"] and not request["ranked"] and request["shot_ranges"] is None
        assert request["motion_models"] == str(tmp_path.resolve())
        return {"frames": [{"index": i, "pts": i, "gap_units": None} for i in table.pts],
                "phones": [], agreement.AAC_44100_MOTION_STATUS: "coverage"}

    monkeypatch.setattr(viseme_review.settings, "video_lip_aac441_motion_review", True)
    monkeypatch.setattr(viseme_review, "_observe", compare)
    result = viseme_review.assess(*args, shot_frames=51)
    assert result["schema"] == binding.SCHEMA_WITH_AAC441_MOTION and result["status"] == "insufficient_events"
    assert result["witness"] is None and result["inputs"]["phones"] == []


def test_48khz_timing_stops_at_the_presented_end(aac_mapping):
    assert aac_mapping["source_duration_pts"] == 100_800
    assert aac_mapping["samples"] > 33_600
    assert binding.audio_origin(aac_mapping) == 0
    assert binding.presented_samples(aac_mapping) == 33_600

    with pytest.raises(ValueError, match="invalid speech event"):
        binding.measured({"frames": [], "phones": [["closure", 33_800, 0.9]]}, timestamps=[],
                         time_base="1/24", source={}, mapping=aac_mapping)


def test_articulation_uses_presented_extent_on_48khz_aac(aac_mapping):
    raw = {"frames": [], "phones": [], "articulation_phones": [["lower_lip", 33_480, 0.9]]}
    result = binding.measured(raw, timestamps=[], time_base="1/24", source={}, mapping=aac_mapping)
    assert result["schema"] == binding.SCHEMA_WITH_ARTICULATION
    assert result["inputs"]["samples"] == 33_600

    raw["articulation_phones"][0][1] = 33_800
    with pytest.raises(ValueError, match="invalid speech event"):
        binding.measured(raw, timestamps=[], time_base="1/24", source={}, mapping=aac_mapping)


def test_local_mouth_motion_has_verified_frames_and_stays_outside_the_estimate(aac_mapping):
    frames = [{"index": i, "pts": i, "gap_units": 20_000, "mouth_box": [1, 1, 3, 3],
               "rgb_sha256": "1" * 64, "mouth_crop_sha256": "2" * 64} for i in range(50)]
    comparison = agreement.encode({"span": [12160, 16000], "picture_span": [19, 25],
                                   "bracket_frames": [18, 21, 25], "lag_samples": 0,
                                   "agreement": [0.75, 0.14, 0.70], "score": 0.56})
    scope = {"timestamps": range(50), "time_base": "1/25", "source": {"width": 4, "height": 4},
             "mapping": aac_mapping}
    raw = {"frames": frames, "phones": [], agreement.KEY: comparison}
    observed = binding.measured(raw, **scope)
    assert observed["schema"] == binding.SCHEMA_WITH_MOTION_SOUND
    assert observed["affects_overall_confidence"] is False
    assert observed["witness"]["picture_frames"] == {"before": 18, "during": 21, "after": 25}
    assert observed["witness"]["audio_source_pts"] == [12160, 16000]
    assert "0.760–1.000 s" in binding.observation(observed)
    assert [row["value"] for row in binding.motion_measurements(observed["witness"])] == [.75, .14, .70]

    with pytest.raises(ValueError, match="native timestamps"):
        binding.measured(raw | {agreement.KEY: comparison | {"span": [12161, 16000]}}, **scope)
    with pytest.raises(ValueError, match="unsupported source clock"):
        binding.measured(raw, **(scope | {"mapping": aac_mapping | {"audio_stream_indices": [1, 2]}}))


def test_shot_evidence_keeps_multiple_intervals_and_the_unassessed_cut(aac_mapping):
    from app.evidence import video_silent_mouth as silent

    samples = 125 * 640
    mapping = aac_mapping | {
        "samples": samples + 4, "source_duration_pts": samples * 3,
        "timeline": [{"start_sample": 0, "end_sample": samples + 4, "source_pts": 0}],
    }
    frames = [{"index": i, "pts": i, "gap_units": 20_000 + 200_000 * (i % 3), "mouth_box": [1, 1, 3, 3],
               "rgb_sha256": "1" * 64, "mouth_crop_sha256": "2" * 64} for i in range(125)]
    shots = [[0, 50], [50, 65], [65, 125]]
    audible = [500, silent.HOP * 500**2]
    levels = [audible for _ in range(samples // silent.HOP)]
    for start, stop in ((12160, 16000), (60000, 64000), (30400, 33600)):
        levels[start // silent.HOP:stop // silent.HOP] = [[0, 0]] * ((stop - start) // silent.HOP)
    scope = {"timestamps": range(125), "time_base": "1/25", "source": {"width": 4, "height": 4}, "mapping": mapping}
    raw = {"frames": frames, "phones": [], "sound_levels": levels, silent.SHOT_KEY: shots}
    record = binding.measured(raw, **scope)
    assert record["schema"] == binding.SCHEMA_WITH_SHOTS and not record["affects_overall_confidence"]
    assert [row["exhibit"] for row in record["shot_coverage"]] == ["VLT01", None, "VLT02"]
    by_id = dict(binding.exhibits(record))
    assert by_id["VLT01"]["witness"]["audio_source_pts"] == [12160, 16000]
    assert by_id["VLT02"]["witness"]["audio_source_pts"] == [60000, 64000]
    assert all(65 <= frame < 125 for frame in by_id["VLT02"]["witness"]["picture_frames"].values())
    with pytest.raises(ValueError, match="partition"):
        binding.measured(raw | {silent.SHOT_KEY: [[0, 50], [49, 65], [65, 125]]}, **scope)


@pytest.mark.parametrize(("time_base", "timestamps"), [
    ("1/1000", [round(Fraction(i * 1001, 30)) for i in range(63)]),
    ("1/30000", [i * 1000 for i in range(63)]),
])
def test_single_shot_dropout_tracks_original_fast_picture_clock(aac_mapping, time_base, timestamps):
    from app.evidence import video_silent_mouth as silent

    frames = [{"index": i, "pts": pts, "gap_units": 20_000 + 200_000 * (i % 3), "mouth_box": [1, 1, 3, 3],
               "rgb_sha256": "1" * 64, "mouth_crop_sha256": "2" * 64} for i, pts in enumerate(timestamps)]
    levels = [[0, 0] if 120 <= i < 160 else [200, silent.HOP * 200**2] for i in range(33_600 // silent.HOP)]
    raw = {"frames": frames, "phones": [], "sound_levels": levels, silent.SHOT_KEY: [[0, 63]]}
    review = binding.measured(raw, timestamps=timestamps, time_base=time_base,
                              source={"width": 4, "height": 4}, mapping=aac_mapping)
    assert review["shot_coverage"][0]["audio_samples"] == [0, 33_600]
    [(_, finding)] = binding.exhibits(review)
    assert finding["witness"]["audio_source_pts"] == [9600, 12800]
    assert finding["affects_overall_confidence"] is False
    brackets = finding["witness"]["picture_frames"]
    picture_times = {role: timestamps[index] * Fraction(time_base) for role, index in brackets.items()}
    assert picture_times["before"] < Fraction(3, 5) <= picture_times["during"] < Fraction(4, 5) <= picture_times["after"]

    dropped_picture = timestamps[:30] + timestamps[31:]
    with pytest.raises(ValueError, match="continuous picture timing"):
        agreement.shot_audio_ranges(dropped_picture, time_base, 33_600, [[0, 62]])


def test_near_silent_range_needs_changing_mouth_frames_and_audible_flanks(aac_mapping):
    from app.evidence import video_silent_mouth as silent

    fixed_gap = 200_000
    frames = [{"index": i, "pts": i, "gap_units": fixed_gap + (180_000 if i in {16, 18} else 0),
               "mouth_box": [0, 0, 2, 2], "rgb_sha256": "1" * 64, "mouth_crop_sha256": "2" * 64}
              for i in range(52)]
    levels = [[200, silent.HOP * 200**2] if not 120 <= i < 160 else [0, 0] for i in range(33_600 // silent.HOP)]
    scope = {"timestamps": range(52), "time_base": "1/25", "source": {"width": 2, "height": 2}, "mapping": aac_mapping}
    inputs = {"frames": frames, "phones": [], "sound_levels": levels}

    flagged = binding.measured(inputs, **scope)
    assert flagged["schema"] == binding.SCHEMA_WITH_SILENCE
    assert flagged["witness"]["audio_source_pts"] == [9600, 12800]
    assert flagged["witness"]["picture_frames"] == {"before": 14, "during": 17, "after": 20}
    assert flagged["affects_overall_confidence"] is False
    baseline = binding.measured({"frames": frames, "phones": []}, **scope)
    sound_tail = [[0, 0]] * (len(levels) - 160)
    assert binding.measured(inputs | {"sound_levels": levels[:160] + sound_tail}, **scope) == baseline
    still = [row | {"gap_units": fixed_gap} for row in frames]
    assert not binding.measured(inputs | {"frames": still}, **scope)["witness"]


@pytest.mark.parametrize("change", [
    {"source_duration_pts": None},
    {"source_duration_pts": 99_000},
    {"source_duration_pts": 110_000},
    {"source_start_pts": 3},
    {"source_time_base": "1/90000"},
    {"source_channels": 2},
    {"timeline": [{"start_sample": 0, "end_sample": 32000, "source_pts": 0}]},
])
def test_48khz_timing_declines_unsupported_or_incomplete_clocks(aac_mapping, change):
    assert binding.audio_origin(aac_mapping | change) is None


def test_timing_reports_unavailable_assets_when_the_checkpoint_drive_is_missing(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in MODEL_BLOBS:
        (checkpoint / name).touch()
    (checkpoint / "pytorch_model.bin").symlink_to(tmp_path / "disconnected" / "pytorch_model.bin")
    geometry = tmp_path / "geometry.task"
    geometry.touch()
    monkeypatch.setattr(viseme_review.settings, "video_lip_timing_model_dir", checkpoint)
    monkeypatch.setattr(viseme_review.settings, "video_lip_timing_face_asset", geometry)
    monkeypatch.setattr(viseme_review, "supported", lambda *_: True)

    result = viseme_review.assess(None, SimpleNamespace(table=None), {}, {}, [])

    assert result == binding.skipped("assets")


@pytest.mark.parametrize(("scale", "admitted"), [(1, True), (-1, False)])
def test_stereo_48khz_timing_checks_both_channels_and_the_entire_clock(tmp_path, monkeypatch, scale, admitted):
    path = tmp_path / "two-channels.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=400:sample_rate=48000:duration=2.1", "-af", f"pan=stereo|c0=c0|c1={scale}*c0",
                    "-c:a", "aac", str(path)], check=True, timeout=20)
    decoded = decode_audio(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), inspect_stereo=True, retain_stereo=True)
    mapping = decoded.mapping
    scope = viseme_stereo.capture(decoded)
    assert scope["blocks"][-1][1] == 33600
    assert binding.audio_origin(mapping) is None
    assert binding.audio_origin(mapping, scope) == (0 if admitted else None)
    assert binding.audio_origin(mapping, scope | {"blocks": scope["blocks"][:-1]}) is None
    assert binding.audio_origin(mapping | {"pcm_sha256": "9" * 64}, scope) is None
    if admitted:
        record = binding.measured({"frames": [], "phones": []}, timestamps=[], time_base="1/24", source={},
                                  mapping=mapping, stereo_scope=scope)
        assert record["schema"] == binding.SCHEMA_WITH_STEREO and record["stereo_scope"] == scope
        assert record["affects_overall_confidence"] is False
    else:
        monkeypatch.setattr(viseme_review, "use_stereo", lambda: True)
        record = viseme_review.assess(None, None, mapping, {}, [], stereo_scope=scope)
        assert record == binding.skipped("stereo_unresolved")
