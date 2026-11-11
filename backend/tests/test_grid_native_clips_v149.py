import copy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import grid_native_clips_v149 as clips
from scripts import grid_portrait_frames_v146 as portrait
from scripts import grid_source_speech_v147 as speech


def source_fixture():
    pts = [3600 * i for i in range(75)]
    pictures = [{"frame_index": i, "pts": p, "file": f"/authored/{i:04d}.png"} for i, p in enumerate(pts)]
    row = {
        "status": clips.planning.reviewed.AVAILABLE,
        "source_sha256": "12" * 32,
        "native_video_table": {
            "time_base": "1/90000",
            "pts": pts,
            "height": 4,
            "width": 5,
            "interpolated": [],
            "truncated": False,
        },
        "pictures": pictures,
        "video_origin_seconds": "0",
        "audio_origin_seconds": "0",
        "pcm_mapping": {
            "timeline_time_base": "1/16000",
            "sample_rate": 16000,
            "channels": 1,
            "sample_format": "s16le",
            "samples": 47648,
            "timeline": [{"start_sample": 0, "end_sample": 47648, "source_pts": 0}],
        },
    }
    frames = [
        {
            "frame_index": i,
            "status": "ready",
            "source_picture": pictures[i],
            "earlier_picture": pictures[i - 8],
            "source_time_seconds": str(Fraction(i, 25)),
            "earlier_time_seconds": str(Fraction(i - 8, 25)),
        }
        for i in clips.FRAMES
    ]
    specs = clips.planning.case_specs(row, frames, {"status": "ready"}, {"status": "ready"})
    return {
        "ordinal": 0,
        "source_slot": 0,
        "component_index": 0,
        "role": "pilot",
        "input": row,
        "cases": specs,
        "frames": frames,
    }


def bundle_fixture():
    source = source_fixture()
    geometry = [{"frame_index": i, "status": "geometry_available"} for i in clips.FRAMES]
    frames = [
        {"condition": mode, "frame_index": i, "rendered": True} for mode in clips.MODES for i in clips.FRAMES
    ]
    waves = {
        name: {
            "status": "encoded",
            "waveform": {
                "samples": 47648,
                "changed_samples": int(name == "delayed_audio_control"),
                "output_clock_mapping": {"authored": "clock"},
                "input_output_sample_mapping": [{"authored": "values"}],
            },
        }
        for name in ("original_pcm", "delayed_audio_control")
    }
    return {
        "source": source,
        "liveportrait_geometry": {"pictures": copy.deepcopy(geometry)},
        "liveportrait_frames": {"frames": copy.deepcopy(frames)},
        "musetalk_frames": {"geometry": geometry, "frames": frames},
        "waveforms": waves,
    }


def test_source_clock_uses_every_exact_native_timestamp_and_separate_audio_origin():
    row = source_fixture()["input"]
    row["audio_origin_seconds"] = "3/1000"
    row["pcm_mapping"]["timeline"][0]["source_pts"] = 48
    clock, reason = clips.source_clock(row)
    assert reason is None and clock["pts_ms"] == [40 * i for i in range(75)]
    assert clock["pts"][27] == Fraction(27, 25) and clock["pts"][47] == Fraction(47, 25)
    assert clock["audio_origin"] == Fraction(3, 1000)
    assert clock["constructed_last_picture_duration_ms"] == 40


@pytest.mark.parametrize("which", ["video", "audio"])
def test_unsupported_fractional_millisecond_clock_is_preserved_without_rounding(which):
    row = source_fixture()["input"]
    if which == "video":
        row["native_video_table"]["pts"][1] += 1
        row["pictures"][1]["pts"] += 1
    else:
        row["pcm_mapping"]["timeline"][0]["source_pts"] = 1
        row["audio_origin_seconds"] = "1/16000"
    clock, reason = clips.source_clock(row)
    assert clock is None and "millisecond" in reason


def test_missing_picture_and_wrong_pts_are_rejected_in_available_native_bank():
    row = source_fixture()["input"]
    row["pictures"].pop()
    with pytest.raises(ValueError):
        clips.source_clock(row)
    row = source_fixture()["input"]
    row["pictures"][35]["pts"] += 1
    with pytest.raises(ValueError):
        clips.source_clock(row)


def test_numeric_names_are_complete_unique_stable_and_condition_opaque():
    source = source_fixture()
    numbers = clips.case_numbers(source)
    assert sorted(numbers.values()) == list(range(11))
    assert [numbers[name] for name in clips.planning.CASES] != list(range(11))
    assert numbers == clips.case_numbers(source)
    assert all(f"{i:02d}.mkv".split(".")[0].isdigit() for i in numbers.values())


def test_partial_producer_case_stays_unavailable_while_independent_controls_remain_ready():
    bundle = bundle_fixture()
    bundle["musetalk_frames"]["frames"][-1]["rendered"] = False
    geo = {branch: clips.geometry_rows(bundle, branch) for branch in clips.BRANCHES}
    renders = {branch: clips.render_rows(bundle, branch) for branch in clips.BRANCHES}
    reasons = {
        spec["id"]: clips.case_reason(spec, None, geo, renders, bundle["waveforms"])
        for spec in bundle["source"]["cases"]
    }
    assert len(reasons) == 11 and [name for name, reason in reasons.items() if reason] == [
        "musetalk_earlier_320ms"
    ]
    assert "twenty-picture" in reasons["musetalk_earlier_320ms"]
    bundle["musetalk_frames"]["frames"].pop()
    with pytest.raises(ValueError):
        clips.render_rows(bundle, "musetalk")


def test_branch_geometry_failure_cannot_block_genuine_cut_and_audio_delay():
    bundle = bundle_fixture()
    bundle["musetalk_frames"]["geometry"][0]["status"] = "unavailable_no_face"
    geo = {branch: clips.geometry_rows(bundle, branch) for branch in clips.BRANCHES}
    renders = {branch: clips.render_rows(bundle, branch) for branch in clips.BRANCHES}
    reasons = {
        spec["id"]: clips.case_reason(spec, None, geo, renders, bundle["waveforms"])
        for spec in bundle["source"]["cases"]
    }
    assert sum(reason is not None for reason in reasons.values()) == 4
    assert reasons["genuine_frame_retime"] is None and reasons["audio_delay_320ms"] is None


def test_genuine_cut_uses_complete_actual_earlier_pictures_at_output_frame_indices():
    source = source_fixture()
    originals = [np.full((4, 5, 3), i, dtype=np.uint8) for i in range(75)]
    spec = next(c for c in source["cases"] if c["id"] == "genuine_frame_retime")
    pts = [Fraction(i, 25) for i in range(75)]
    result = clips.case_pictures(spec, source, originals, pts, {}, {}, None, None)
    assert list(result) == list(range(27, 47))
    assert np.array_equal(result[27][0], originals[19]) and (result[46][1] == 255).all()
    pts[19] += Fraction(1, 1000)
    with pytest.raises(ValueError, match="earlier picture"):
        clips.case_pictures(spec, source, originals, pts, {}, {}, None, None)


def test_branch_resampling_uses_512_linear_only_for_liveportrait(monkeypatch):
    original = np.zeros((4, 5, 3), np.uint8)
    geometry = {"crop": np.full((256, 256, 3), 91, np.uint8), "matrix": np.eye(2, 3, dtype=np.float64)}
    shapes = []

    def inverse(value, matrix, shape):
        shapes.append(value.shape)
        assert matrix is geometry["matrix"]
        return np.resize(np.asarray([90, 155, 244], dtype=np.uint8), shape)

    monkeypatch.setattr(clips, "inverse_warp", inverse)
    a = clips.ordinary_pixels(original, geometry, "liveportrait", True)
    b = clips.ordinary_pixels(original, geometry, "musetalk", False)
    assert shapes == [(512, 512, 3), (256, 256, 3)]
    assert a[0, 0].tolist() == [99, 170, 255] and b[0, 0].tolist() == [90, 155, 244]


def test_exact_mask_refuses_omitted_changed_pixel_and_outside_alpha():
    original = np.zeros((3, 4, 3), np.uint8)
    edited = original.copy()
    edited[1, 2] = 7
    mask = np.zeros((3, 4), np.uint8)
    with pytest.raises(ValueError, match="every actually changed"):
        clips.exact_mask(original, edited, mask)
    mask[1, 2] = 255
    assert clips.exact_mask(original, edited, mask) == 1
    with pytest.raises(ValueError, match="outside exact branch alpha"):
        clips.exact_mask(original, edited, mask, np.zeros((3, 4), np.float32))


def test_neural_case_rebuilds_native_composite_and_edge_opacity_from_saved_pixels(tmp_path):
    folder = tmp_path / "clip-00"
    folder.mkdir()
    original = np.full((4, 5, 3), 20, np.uint8)
    generated = np.full((512, 512, 3), 100, np.uint8)
    alpha = np.zeros((4, 5), np.float32)
    alpha[1, 2] = 1
    geometry = {"alpha": alpha, "matrix": np.eye(2, 3, dtype=np.float64), "proof": {"source": "bound"}}
    edited = original.copy()
    edited[1, 2] = 60  # Frame 27 uses half opacity.
    mask = np.zeros((4, 5), np.uint8)
    mask[1, 2] = 255
    refs = {
        kind: portrait.save_png(folder / f"source_time-0027-{kind}.png", value)
        for kind, value in (("crop", generated), ("native", edited), ("mask", mask))
    }
    record = {
        "frame_index": 27,
        "condition": "source_time",
        "rendered": True,
        "opacity": 0.5,
        "outside_alpha_changed_pixels": 0,
        "changed_native_pixels": 1,
        "source": {"source": "bound"},
        "files": refs,
    }
    actual, actual_mask = clips.neural_picture(
        {"ordinal": 0}, original, geometry, "liveportrait", record, SimpleNamespace(CACHE=tmp_path), portrait
    )
    assert np.array_equal(actual, edited) and np.array_equal(actual_mask, mask)
    record["source"] = {"source": "wrong"}
    with pytest.raises(ValueError, match="source proof"):
        clips.neural_picture(
            {"ordinal": 0},
            original,
            geometry,
            "liveportrait",
            record,
            SimpleNamespace(CACHE=tmp_path),
            portrait,
        )


def test_roundtrip_consumer_requires_complete_payload_proof_with_mock_mux(tmp_path):
    clock, _ = clips.source_clock(source_fixture()["input"])
    frames = [np.full(clock["shape"], i, np.uint8) for i in range(75)]
    pcm = np.arange(640, dtype="<i2")
    calls = []

    def encode(path, rgb, pts, shape, audio, origin):
        calls.append("encode")
        return {
            "rgb_sha256": [clips.fingerprint(f) for f in rgb],
            "picture_pts_ms": pts,
            "constructed_last_picture_duration_ms": 40,
            "audio_start_seconds": str(origin),
            "audio_samples": len(audio),
            "pcm_s16_sha256": clips.fingerprint(audio),
        }

    def verify(path, expected, shape):
        calls.append("verify")
        return clips.roundtrip_record(640)

    mux = SimpleNamespace(encode=encode, verify_media=verify)
    clips.build_case(tmp_path / "05.mkv", frames, pcm, clock, mux, lambda _: calls.append("header"))
    assert calls == ["encode", "verify", "header"]
    mux.verify_media = lambda *_: {**clips.roundtrip_record(640), "pcm_exact": False}
    with pytest.raises(ValueError, match="lossless decode"):
        clips.build_case(tmp_path / "05.mkv", frames, pcm, clock, mux, lambda _: None)
    mux.encode = lambda *args: {**encode(*args), "constructed_last_picture_duration_ms": 41}
    with pytest.raises(ValueError, match="input proof"):
        clips.build_case(tmp_path / "05.mkv", frames, pcm, clock, mux, lambda _: None)


def test_whole_delayed_audio_has_exact_values_and_both_payload_hashes(tmp_path, monkeypatch):
    source = source_fixture()
    source["audio_delay"] = {
        "status": "ready",
        "shift_samples": 5120,
        "whole_input_reencoding_required": True,
        "preserve_length_origins_and_outside_samples": True,
        "padding_or_wraparound": False,
        "output_samples_half_open": [17280, 30080],
        "source_samples_half_open": [12160, 24960],
    }
    cache = tmp_path / "cache"
    folder = cache / "batch-00/clip-00"
    folder.mkdir(parents=True)
    monkeypatch.setattr(speech, "CACHE", cache)
    original_pcm = np.arange(47648, dtype=np.int32).astype("<i2")
    original = original_pcm.astype("<f4") / np.float32(32768)
    delayed, changed = speech.delayed_pcm(original, source["audio_delay"])
    name = "delayed_audio_control"
    waveform = speech.save_waveform(folder, name, delayed, source, changed)
    record = {"waveform_id": name, "waveform": waveform, "batch": 0}
    recovered = clips.read_waveform(source, record, original, speech)
    assert len(recovered) == len(original_pcm)
    assert np.array_equal(recovered[17280:30080], original_pcm[12160:24960])
    assert np.array_equal(recovered[:17280], original_pcm[:17280])
    assert np.array_equal(recovered[30080:], original_pcm[30080:])
    waveform["wav_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="whole-waveform bytes changed"):
        clips.read_waveform(source, record, original, speech)


def test_blind_container_metadata_refuses_labels_and_nonuniform_codec():
    video = SimpleNamespace(
        type="video", codec_context=SimpleNamespace(name="ffv1"), metadata={"DURATION": "00:00:03"}
    )
    audio = SimpleNamespace(
        type="audio",
        codec_context=SimpleNamespace(name="pcm_s16le", sample_rate=16000, channels=1),
        metadata={},
    )
    container = SimpleNamespace(
        format=SimpleNamespace(name="matroska,webm"), streams=[video, audio], metadata={"ENCODER": "Lavf"}
    )
    clips.header_contract(container)
    container.metadata["title"] = "neural case"
    with pytest.raises(ValueError, match="labels"):
        clips.header_contract(container)
    container.metadata.pop("title")
    video.codec_context.name = "h264"
    with pytest.raises(ValueError, match="uniform"):
        clips.header_contract(container)


def test_no_case_may_be_promoted_from_unavailable_without_actual_payload():
    spec = source_fixture()["cases"][0]
    spec.update(status="unavailable", unavailable_reason="Source picture missing")
    assert clips.case_reason(spec, None, {}, {}, {}) == "Source picture missing"
    spec["unavailable_reason"] = None
    with pytest.raises(ValueError):
        clips.case_reason(spec, None, {}, {}, {})


def test_constructor_routes_only_owned_fixed_clip_bind_worker(monkeypatch):
    calls = []
    monkeypatch.setattr(clips.subprocess, "Popen", lambda argv, **kwargs: calls.append(argv))
    path = str(clips.TRIAL / "clip-03/bind-00/output")
    argv = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", path, "0"]
    clips.routed_popen(3, argv)
    assert calls[0][3:] == ["scripts.grid_native_clips_v149", "construct", path, "3"]
    with pytest.raises(ValueError):
        clips.routed_popen(2, argv)


def test_pending_musetalk_context_creates_no_freeze_or_stage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(clips, "BASE", tmp_path)
    monkeypatch.setattr(clips, "PANEL", tmp_path)
    monkeypatch.setattr(clips, "TRIAL", tmp_path / "new")
    clips.freeze()
    assert "pending_prerequisite" in capsys.readouterr().out and not clips.TRIAL.exists()
