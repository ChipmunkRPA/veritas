import copy
import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.video.decode import FrameTable
from scripts import grid_native_mouth_features_v150 as stage


def source_clock():
    table = FrameTable("1/1000", tuple(i * 40 for i in range(75)), (0,), "I" + "P" * 74, 64, 48, 64, 48)
    source = {
        "ordinal": 3,
        "source_slot": 1,
        "component_index": 1,
        "input": {
            "native_video_table": {
                "pts": [i * 3600 for i in range(75)],
                "time_base": "1/90000",
                "width": 64,
                "height": 48,
            },
            "native_audio_clock": {"time_base": "1/44100", "sample_rate": 44100},
        },
    }
    return table, source


def constructor_case():
    table, source = source_clock()
    original = source["input"]["native_video_table"]
    case = {
        "status": "ready",
        "file": "/private/authored.mkv",
        "file_sha256": "authored",
        "source_video_time_base": original["time_base"],
        "source_video_pts": original["pts"],
        "audio_output_clock_mapping": {"samples": 48000},
        "waveform_id": "original_pcm",
        "expected": {"picture_pts_ms": list(table.pts), "rgb_sha256": ["authored"] * 75},
        "edit_proofs": [{"private": "truth"}],
        "edit_frames_half_open": [27, 47],
        "planned_neural_provenance": True,
    }
    return table, source, case


def test_media_projection_and_observed_native_clock():
    table, source, case = constructor_case()
    selected = stage.case_inputs(case)
    assert not {"edit_proofs", "edit_frames_half_open", "planned_neural_provenance"}.intersection(selected)
    stage.verify_table(table, selected, source)


@pytest.mark.parametrize(
    "problem", ["original_timebase", "shift", "interpolated", "truncated", "missing_frame", "dimension"]
)
def test_false_or_incomplete_decode_clock_cannot_feed_features(problem):
    table, source, case = constructor_case()
    if problem == "original_timebase":
        table = replace(table, time_base="1/90000")
    elif problem == "shift":
        table = replace(table, pts=tuple(p + 1 for p in table.pts))
        case["expected"]["picture_pts_ms"] = list(table.pts)
    elif problem == "interpolated":
        table = replace(table, interpolated=(30,))
    elif problem == "truncated":
        table = replace(table, truncated=True)
    elif problem == "missing_frame":
        table = replace(table, pts=table.pts[:-1])
    else:
        table = replace(table, width=32)
    with pytest.raises(ValueError):
        stage.verify_table(table, stage.case_inputs(case), source)


def test_full_output_time_route_retains_unavailable_edges_and_uses_current_picture():
    table, _ = source_clock()
    contexts = stage.speech_contexts(table, stage.Fraction(0), 48000, stage.Fraction(2, 44100))
    assert len(contexts) == 75
    assert [r["frame_index"] for r in contexts if r["status"] == "supported"] == list(range(2, 73))
    route = contexts[27]["plan"]
    assert route["source_time_seconds"] == "27/25"
    assert route["tick_indices"] == list(range(50, 60))
    assert contexts[0]["status"] == contexts[74]["status"] == "unavailable"
    shifted_origin = stage.speech_contexts(table, stage.Fraction(1, 50), 48000, stage.Fraction(0))
    assert shifted_origin[27]["plan"]["tick_indices"] == list(range(49, 59))


def test_original_and_delayed_whole_states_use_the_same_output_clock(tmp_path):
    from scripts import grid_source_speech_v147 as speech

    _, replay = stage.feature_modules()
    table, _ = source_clock()
    contexts = stage.speech_contexts(table, stage.Fraction(0), 48000, stage.Fraction(0))
    speech_cache = tmp_path / "state-cache"
    folder = speech_cache / "batch-00/clip-03"
    folder.mkdir(parents=True)
    consumer = SimpleNamespace(CACHE=speech_cache, validate_hidden=speech.validate_hidden)
    hidden = np.broadcast_to(np.arange(1500, dtype=np.float32)[:, None, None], (1500, 5, 384)).copy()
    output = []
    for waveform_id, values in (("original_pcm", hidden), ("delayed_audio_control", hidden + np.float32(20))):
        path = folder / f"{waveform_id}.hidden.npy"
        np.save(path, values, allow_pickle=False)
        record = {
            "waveform_id": waveform_id,
            "batch": 0,
            "source_ordinal": 3,
            "hidden_file": str(path),
            "hidden_file_sha256": stage.sha(path),
            "proof": {"full_hidden_sha256": stage.fingerprint(values), "layer_readout_matches": True},
            "waveform": {"pcm_sha256": waveform_id},
        }
        ref = stage.speech_features(tmp_path, record, contexts, consumer, replay)
        data = replay.array_file(ref, np.float64)
        output.append((ref, data[ref["indices"].index(27)]))
    assert output[0][0]["contexts"] == output[1][0]["contexts"]
    assert output[0][1][0] == 54.5 and output[1][1][0] == 74.5
    assert output[0][1][1920] == output[1][1][1920] == np.std(np.arange(50, 60, dtype=np.float64))


def test_exact_cache_requires_bytes_and_shape_and_fixed_picture_can_bypass():
    calls = []
    memo = {}

    def observe(rgb):
        calls.append(rgb.shape)
        return {"status": "unassessed", "reason": "Authored fixture"}, None

    pixels = np.zeros((4, 8, 3), np.uint8)
    assert stage.observe_picture(pixels, memo, observe)[1] is False
    assert stage.observe_picture(pixels.copy(), memo, observe)[1] is True
    assert stage.observe_picture(pixels.reshape(8, 4, 3), memo, observe)[1] is False
    changed = pixels.copy()
    changed[-1, -1, -1] = 1
    assert stage.observe_picture(changed, memo, observe)[1] is False
    assert stage.observe_picture(pixels, memo, observe, fresh=True)[1] is False
    assert calls == [(4, 8, 3), (8, 4, 3), (4, 8, 3), (4, 8, 3)]


def test_repeat_checks_canonical_metadata_and_exact_vector_bytes():
    vector = np.zeros(768, np.float32)
    meta = {
        "status": "ok",
        "face": {"landmarks": [[0.0, 1.0]]},
        "crop_rgb_sha256": "a",
        "model_input_float32_sha256": "b",
    }
    assert stage.same_observation((meta, vector), (copy.deepcopy(meta), vector.copy()))
    negative_zero = copy.deepcopy(meta)
    negative_zero["face"]["landmarks"][0][0] = -0.0
    assert not stage.same_observation((meta, vector), (negative_zero, vector))
    changed = vector.copy()
    changed[-1] = np.nextafter(np.float32(0), np.float32(1))
    assert not stage.same_observation((meta, vector), (meta, changed))
    assert not stage.same_observation((meta, vector), (meta, None))


def test_v125_point_features_consumes_v150_shapes_and_indices(tmp_path):
    from scripts import fit_native_mouth_heads_v125 as downstream

    table, _ = source_clock()
    pixels = np.zeros(768, np.float32)
    pixels[0] = pixels[384] = 1
    image = stage.save_array(tmp_path / "case-00.npy", [pixels] * 3, 768, np.float32)
    image["file_sha256"] = image.pop("sha256")
    values = np.arange(3840, dtype=np.float64)
    speech = {
        **stage.save_array(tmp_path / "speech-original_pcm.npy", [values], 3840, np.float64),
        "indices": [27],
    }
    frames = [
        {
            "frame_index": i,
            "status": "ok" if i in (26, 27, 28) else "unassessed",
            "feature_row": i - 26 if 26 <= i <= 28 else None,
        }
        for i in range(75)
    ]
    record = {**image, "status": "extracted", "table": table.to_json(), "frames": frames, "speech": speech}
    indices, visual, audio = downstream.point_features(record)
    assert indices == [27] and visual.shape == (1, 2304) and np.array_equal(audio[0], values)


def test_late_bad_pixel_blocks_entire_case_before_feature_processing(tmp_path, monkeypatch):
    from app.detectors.audio import decode as audio_decode
    from app.detectors.video import decode as video_decode

    table, source, case = constructor_case()
    features, _ = stage.feature_modules()
    path = tmp_path / "authored.mkv"
    path.write_bytes(b"authored mock decoder fixture")
    case.update(file=str(path), file_sha256=stage.sha(path))
    rgb = np.zeros((48, 64, 3), np.uint8)
    case["expected"]["rgb_sha256"] = [stage.fingerprint(rgb)] * 75
    monkeypatch.setattr(video_decode, "probe_frames", lambda *a, **k: table)
    observed = []

    def frames(*args, **kwargs):
        for i in range(75):
            observed.append(i)
            yield SimpleNamespace(index=i, pts=table.pts[i], rgb=rgb if i < 74 else np.ones_like(rgb))

    monkeypatch.setattr(video_decode, "decode_frames", frames)

    def invalid_audio(*args, **kwargs):
        raise AssertionError("A failed complete RGB proof must stop the case")

    monkeypatch.setattr(audio_decode, "decode_audio", invalid_audio)
    with pytest.raises(ValueError, match="RGB"):
        stage.verified_case(stage.case_inputs(case), source, {}, tmp_path, features, None)
    assert observed == list(range(75))


def test_actual_case_pcm_must_match_bound_whole_state(tmp_path, monkeypatch):
    from app.detectors.audio import decode as audio_decode
    from app.detectors.video import decode as video_decode
    from scripts import grid_source_speech_v147 as speech

    table, source, case = constructor_case()
    features, _ = stage.feature_modules()
    path = tmp_path / "authored.mkv"
    path.write_bytes(b"authored mock decoder fixture")
    case.update(file=str(path), file_sha256=stage.sha(path))
    rgb = np.zeros((48, 64, 3), np.uint8)
    pcm = np.arange(48000, dtype=np.int16).astype(np.float32) / np.float32(32768)
    case["expected"].update(
        rgb_sha256=[stage.fingerprint(rgb)] * 75,
        audio_samples=len(pcm),
        audio_start_seconds="0",
        pcm_s16_sha256=hashlib.sha256(speech.pcm_lattice(pcm)).hexdigest(),
    )
    state = {
        "waveform_id": "original_pcm",
        "waveform": {
            "id": "original_pcm",
            "samples": len(pcm),
            "audio_origin_seconds": "0",
            "pcm_sha256": stage.fingerprint(pcm),
            "output_clock_mapping": case["audio_output_clock_mapping"],
        },
    }
    monkeypatch.setattr(video_decode, "probe_frames", lambda *a, **k: table)
    monkeypatch.setattr(
        video_decode,
        "decode_frames",
        lambda *a, **k: (SimpleNamespace(index=i, pts=table.pts[i], rgb=rgb) for i in range(75)),
    )
    monkeypatch.setattr(audio_decode, "decode_audio", lambda *a, **k: SimpleNamespace(pcm=pcm, mapping={}))
    monkeypatch.setattr(audio_decode, "source_ranges", lambda *a, **k: [{"source_start_pts": 0}])
    actual, frames, origin = stage.verified_case(
        stage.case_inputs(case), source, state, tmp_path, features, speech
    )
    assert actual == table and len(frames) == 75 and origin == 0
    state["waveform"]["pcm_sha256"] = "other whole waveform"
    with pytest.raises(ValueError, match="PCM"):
        stage.verified_case(stage.case_inputs(case), source, state, tmp_path, features, speech)


def test_wholly_unavailable_clip_retains_all_eleven_rows_and_zero_neural_calls(tmp_path, monkeypatch, capsys):
    _, source = source_clock()
    cases = [
        {
            "condition": name,
            "status": "unavailable",
            "reason": "Authored fixture unavailable",
            "waveform_id": "original_pcm",
        }
        for name in stage.planning.CASES
    ]
    monkeypatch.setattr(stage, "input_binding", lambda ordinal: (source, cases, {}, {}, "authored-journal"))
    monkeypatch.setattr(stage, "upstream", lambda: (None, None))
    monkeypatch.setattr(stage, "feature_modules", lambda: (None, None))
    cache = tmp_path / "feature-cache"
    monkeypatch.setattr(stage, "CACHE", cache)
    public = tmp_path / "fixture.json"
    public.write_text("{}\n")
    monkeypatch.setattr(stage, "PUBLIC", public)
    out = tmp_path / "output"
    out.mkdir()
    stage.extract(3, out)
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    rows = stage.planning.reviewed.lines(cache / "clip-03/feature-cases.jsonl")
    stage.validate_inventory(rows, cases, source, {}, cache / "clip-03", "authored-journal")
    assert result["planned_cases"] == result["unavailable_cases"] == 11
    assert (
        result["native_pictures"] == result["new_neural_calls"] == result["strict_encoder_tensor_count"] == 0
    )
    assert result["fixed_original_repeat_bitwise_equal"] is None
    assert all("file" not in r and "speech" not in r for r in rows)


def test_historical_feature_import_restores_the_caller_path():
    before = sys.path[:]
    feature, review = stage.feature_modules()
    assert review.features is feature
    assert sys.path == before


def test_fixed_routing_preserves_authored_command_and_adds_only_ordinal(monkeypatch):
    calls = []
    monkeypatch.setattr(stage.subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs)))
    argv = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", "/private/output"]
    stage.routed_popen(argv, 3, env={"FIXTURE": "1"})
    assert calls == [
        (
            [
                "python",
                "-B",
                "-m",
                "scripts.grid_native_mouth_features_v150",
                "authored",
                "/private/output",
                "3",
            ],
            {"env": {"FIXTURE": "1"}},
        )
    ]
    with pytest.raises(ValueError):
        stage.routed_popen([*argv[:4], "extract", argv[-1]], 3)
