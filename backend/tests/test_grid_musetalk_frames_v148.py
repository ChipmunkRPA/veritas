import copy
import json
import subprocess
import sys
import types

import numpy as np
import pytest

from scripts import grid_musetalk_frames_v148 as stage


def source_fixture():
    source = {
        "ordinal": 3,
        "role": "pilot",
        "input": {
            "native_video_table": {
                "pts": [3600 * i for i in range(75)],
                "time_base": "1/90000",
                "width": 128,
                "height": 128,
            },
            "audio_origin_seconds": "0",
            "video_origin_seconds": "0",
            "pcm_mapping": {"samples": 48000, "pcm_sha256": "authored"},
            "pictures": [],
        },
        "frames": [],
    }
    for i in stage.FRAMES:
        source["frames"].append(
            {
                "frame_index": i,
                "source_time_conditioning": stage.clock.frame_plan(
                    i, str(stage.Fraction(i, 25)), "0", "0", 48000, "0"
                ),
                "earlier_conditioning": stage.clock.frame_plan(
                    i, str(stage.Fraction(i - 8, 25)), "0", "0", 48000, "0"
                ),
            }
        )
    return source


def route_fixture():
    source = source_fixture()
    hidden = np.broadcast_to(
        (
            np.arange(150, dtype=np.float32)[:, None, None] * 100
            + np.arange(5, dtype=np.float32)[None, :, None]
        ),
        (150, 5, 384),
    ).copy()
    routes = []
    for frame in source["frames"]:
        for view in ("source_time_conditioning", "earlier_conditioning"):
            plan = frame[view]
            selected = np.concatenate([hidden[tick] for tick in plan["tick_indices"]], axis=0)
            routes.append(
                {
                    "frame_index": frame["frame_index"],
                    "view": view,
                    "status": "verified",
                    "conditioning_plan": copy.deepcopy(plan),
                    "chunk_sha256": stage.fingerprint(selected),
                }
            )
    return source, {"routing": routes}, hidden


def test_both_conditions_use_fixed_native_picture_with_exact_ten_tick_five_layer_selection():
    source, record, hidden = route_fixture()
    jobs, chunks = stage.conditioning(source, record, hidden)
    assert len(jobs) == len(chunks) == 40
    assert jobs[0]["frame_index"] == jobs[20]["frame_index"] == 27
    assert jobs[0]["seed"] == jobs[20]["seed"] == stage.seed(3, 27)
    assert jobs[1]["seed"] != jobs[0]["seed"]
    assert chunks["source_time", 27][[0, 4, 5, 49], 0].tolist() == [5000, 5004, 5100, 5904]
    assert chunks["earlier_320ms", 27][[0, 4, 5, 49], 0].tolist() == [3400, 3404, 3500, 4304]
    assert stage.expected_roots(41) == {"vae_encode": 82, "vae_decode": 41, "unet": 41, "speech_encoder": 0}
    assert 21 + sum(stage.expected_roots(41).values()) == 185


def test_one_unavailable_route_does_not_hide_other_verified_pairs():
    source, record, hidden = route_fixture()
    record["routing"][1]["status"] = "unavailable"
    jobs, chunks = stage.conditioning(source, record, hidden)
    assert len(jobs) == 40 and len(chunks) == 39
    assert jobs[20]["status"] == "unavailable" and jobs[0]["status"] == "available"
    assert ("earlier_320ms", 27) not in chunks


@pytest.mark.parametrize("problem", ["duplicate", "offset", "evidence_index", "chunk_sha", "audio_origin"])
def test_route_substitution_or_unbound_chunk_fails_before_rendering(problem):
    source, record, hidden = route_fixture()
    if problem == "duplicate":
        record["routing"][-1] = record["routing"][0]
    elif problem in {"offset", "evidence_index"}:
        key = "source_time_seconds" if problem == "offset" else "video_frame_index"
        value = "7/10" if problem == "offset" else 19
        source["frames"][0]["earlier_conditioning"][key] = value
        record["routing"][1]["conditioning_plan"][key] = value
    elif problem == "chunk_sha":
        record["routing"][1]["chunk_sha256"] = "wrong"
    else:
        source["input"]["audio_origin_seconds"] = "1/1000"
    with pytest.raises(ValueError):
        stage.conditioning(source, record, hidden)


def states_fixture(tmp_path):
    source = source_fixture()
    folder = tmp_path / "batch-00/clip-03"
    folder.mkdir(parents=True)
    hidden = np.zeros((1500, 5, 384), np.float32)
    pcm = np.arange(48000, dtype=np.int32).astype(np.int16).astype(np.float32) / np.float32(32768)
    hidden_file, pcm_file = folder / "original_pcm.hidden.npy", folder / "original_pcm.pcm.npy"
    np.save(hidden_file, hidden, allow_pickle=False)
    np.save(pcm_file, pcm, allow_pickle=False)
    mapping = {"samples": len(pcm), "pcm_sha256": stage.fingerprint(pcm)}
    source["input"]["pcm_mapping"] = mapping
    record = {
        "source_ordinal": 3,
        "waveform_id": "original_pcm",
        "role": "pilot",
        "batch": 0,
        "status": "encoded",
        "whole_input_encoded": True,
        "hidden_shape": [1500, 5, 384],
        "hidden_dtype": "float32",
        "hidden_file": str(hidden_file),
        "hidden_file_sha256": stage.sha(hidden_file),
        "proof": {"full_hidden_sha256": stage.fingerprint(hidden), "layer_readout_matches": True},
        "waveform": {
            "id": "original_pcm",
            "changed_samples": 0,
            "audio_origin_seconds": "0",
            "original_pcm_mapping": copy.deepcopy(mapping),
            "samples": len(pcm),
            "pcm_sha256": mapping["pcm_sha256"],
            "pcm_file": str(pcm_file),
            "pcm_file_sha256": stage.sha(pcm_file),
        },
    }
    return source, record, hidden, pcm


def test_exact_original_cached_pcm_and_hidden_payload_are_verified(tmp_path):
    source, record, hidden, _ = states_fixture(tmp_path)
    observed = stage.source_states(source, record, tmp_path)
    assert observed.dtype == np.float32 and observed.shape == (1500, 5, 384)
    assert np.array_equal(observed, hidden)


@pytest.mark.parametrize(
    "problem", ["delayed_id", "delayed_path", "raw_hidden_hash", "wrong_shape", "nonfinite", "changed_pcm"]
)
def test_delayed_or_corrupted_state_payload_cannot_become_generator_input(problem, tmp_path):
    source, record, hidden, pcm = states_fixture(tmp_path)
    if problem == "delayed_id":
        record["waveform_id"] = "delayed_audio_control"
    elif problem == "delayed_path":
        path = tmp_path / "batch-00/clip-03/delayed_audio_control.hidden.npy"
        np.save(path, hidden, allow_pickle=False)
        record["hidden_file"] = str(path)
    elif problem == "raw_hidden_hash":
        record["proof"]["full_hidden_sha256"] = "wrong"
    elif problem in {"wrong_shape", "nonfinite"}:
        if problem == "wrong_shape":
            hidden = hidden[:-1]
        else:
            hidden[0, 0, 0] = np.nan
        np.save(record["hidden_file"], hidden, allow_pickle=False)
        record["hidden_file_sha256"] = stage.sha(record["hidden_file"])
        record["proof"]["full_hidden_sha256"] = stage.fingerprint(hidden)
    else:
        pcm[0] = np.float32(0.125)
        np.save(record["waveform"]["pcm_file"], pcm, allow_pickle=False)
        record["waveform"]["pcm_file_sha256"] = stage.sha(record["waveform"]["pcm_file"])
    with pytest.raises(ValueError):
        stage.source_states(source, record, tmp_path)


def face_fixture():
    return types.SimpleNamespace(
        x=8,
        y=6,
        w=110,
        h=112,
        score=0.91002004,
        landmarks=[(30.0, 32.0), (78.0, 32.0), (54.0, 53.0), (38.0, 77.0), (68.0, 78.0)],
    )


def authored_crop(rgb, face, mode):
    assert mode == "landmark-box"
    return (
        np.zeros((256, 256, 3), np.uint8),
        np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5]]),
        {"mode": mode, "bbox": [0, 0, 128, 128]},
    )


def authored_alpha(shape, face):
    alpha = np.zeros(shape[:2], np.float32)
    alpha[64:85, 32:72] = 1
    return alpha


@pytest.mark.parametrize("n", [0, 2])
def test_no_or_multiple_faces_are_observed_without_selecting_a_replacement(n):
    rgb = np.zeros((128, 128, 3), np.uint8)
    events = []

    def unexpected(*args):
        raise AssertionError("A missing/ambiguous face cannot choose geometry")

    row, data = stage.observe_geometry(
        rgb, lambda _: [face_fixture()] * n, unexpected, unexpected, events.append
    )
    assert data is None
    assert row["status"] == ("unavailable_no_face" if n == 0 else "unavailable_multiple_faces")
    assert events == [{"event": "face_begin"}, {"event": "face_end", "observed_faces": n}]


def test_geometry_repeat_binds_full_face_values_and_support_fingerprints():
    rgb = np.zeros((128, 128, 3), np.uint8)
    face = face_fixture()
    first, _ = stage.observe_geometry(rgb, lambda _: [face], authored_crop, authored_alpha, lambda _: None)
    repeated, _ = stage.observe_geometry(
        rgb.copy(), lambda _: [face], authored_crop, authored_alpha, lambda _: None
    )
    assert stage.geometry_repeat(first, repeated) is True
    face.score += 0.000000001
    changed, _ = stage.observe_geometry(rgb, lambda _: [face], authored_crop, authored_alpha, lambda _: None)
    assert stage.geometry_repeat(first, changed) is False
    assert stage.geometry_repeat(first, {"status": "unavailable_no_face"}) is False
    assert stage.geometry_repeat({"status": "unavailable_no_face"}, repeated) is None


@pytest.mark.parametrize(
    "problem", ["nonfinite_face", "nonfinite_matrix", "empty_alpha", "nonfinite_alpha", "bad_crop"]
)
def test_invalid_native_geometry_is_unassessed_instead_of_a_replacement(problem):
    rgb = np.zeros((128, 128, 3), np.uint8)
    face = face_fixture()
    if problem == "nonfinite_face":
        face.landmarks[0] = (float("nan"), 32.0)

    def crop_fn(value, found, mode):
        crop, matrix, geometry = authored_crop(value, found, mode)
        if problem == "nonfinite_matrix":
            matrix[0, 2] = np.inf
        if problem == "bad_crop":
            crop = crop[:-1]
        return crop, matrix, geometry

    def alpha_fn(shape, found):
        alpha = authored_alpha(shape, found)
        if problem == "empty_alpha":
            alpha[:] = 0
        if problem == "nonfinite_alpha":
            alpha[0, 0] = np.nan
        return alpha

    row, values = stage.observe_geometry(rgb, lambda _: [face], crop_fn, alpha_fn, lambda _: None)
    assert row["status"] == "unavailable_geometry" and values is None


def test_overlay_is_restored_after_nested_runtime_import_side_effect_or_error(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "OVERLAY", tmp_path / "retained-runtime")
    before = sys.path[:]
    with pytest.raises(RuntimeError, match="authored"):
        with stage.isolated_overlay():
            assert sys.path[0] == str(stage.OVERLAY)
            sys.path.insert(0, str(stage.OVERLAY))
            with stage.isolated_overlay():
                assert sys.path[0] == str(stage.OVERLAY)
            assert sys.path[:2] == [str(stage.OVERLAY)] * 2
            raise RuntimeError("authored")
    assert sys.path == before


def test_pixels_alone_do_not_prove_repeat_or_same_conditioning_noise():
    pixels = np.zeros((256, 256, 3), np.uint8)
    proof = {
        "seed": 1,
        "masked_input_f32_sha256": "a",
        "original_input_f32_sha256": "b",
        "sampled_latents_f32_sha256": "c",
        "condition_with_position_f32_sha256": "d",
    }
    shifted = {**proof, "condition_with_position_f32_sha256": "e"}
    assert stage.same_noise(proof, shifted) and not stage.render_repeat(pixels, pixels, proof, shifted)
    assert stage.render_repeat(pixels, pixels.copy(), proof, proof.copy())
    assert not stage.same_noise(proof, {**shifted, "sampled_latents_f32_sha256": "different"})


def test_authored_no_picture_or_speech_case_records_all_unavailable_without_loading(
    tmp_path, monkeypatch, capsys
):
    source = source_fixture()
    record = {"source_ordinal": 3, "waveform_id": "original_pcm", "role": "pilot", "status": "unavailable"}
    monkeypatch.setattr(stage, "input_binding", lambda ordinal: (source, record, {}))
    monkeypatch.setattr(
        stage, "dependencies", lambda: (None, types.SimpleNamespace(CACHE=tmp_path / "speech"))
    )
    monkeypatch.setattr(stage, "CACHE", tmp_path / "authored-cache")
    public = tmp_path / "authored-freeze.json"
    public.write_text("{}\n")
    monkeypatch.setattr(stage, "PUBLIC", public)
    output = tmp_path / "output"
    output.mkdir()
    stage.build(3, output)
    result = json.loads(capsys.readouterr().out)
    assert result["planned_condition_frames"] == result["skipped_condition_frames"] == 40
    assert result["executed_condition_frames"] == result["new_neural_calls"] == result["renderer_inputs"] == 0
    assert result["geometry_repeat_bitwise"] is result["renderer_repeat_bitwise"] is None
    cache = stage.CACHE / "clip-03"
    records = json.loads((cache / "outputs.json").read_text())
    stage.check_output_inventory(records, result, source, cache)
    assert all(not row["rendered"] for row in records["frames"])
    assert len(records["geometry"]) == 20 and records["render_repeat"] is None
    assert (
        (output / "root-progress.jsonl").read_bytes() == (output / "face-progress.jsonl").read_bytes() == b""
    )
    records["frames"][0]["files"] = {"native": {"path": "invented"}}
    with pytest.raises(ValueError):
        stage.check_output_inventory(records, result, source, cache)


def test_original_png_bytes_and_raw_rgb_are_both_bound(tmp_path):
    source = source_fixture()
    rgb = np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3)
    file = stage.save_png(tmp_path / "authored.png", rgb)
    picture = {
        "file": file["path"],
        "file_sha256": file["sha256"],
        "rgb_sha256": file["raw_sha256"],
        "frame_index": 27,
        "pts": 27 * 3600,
    }
    assert np.array_equal(stage.source_rgb(picture, source), rgb)
    picture["rgb_sha256"] = "changed"
    with pytest.raises(ValueError):
        stage.source_rgb(picture, source)


def authored_inventory(cache):
    source = source_fixture()
    picture = {"frame_index": 27, "pts": 27 * 3600, "file": "/private/authored-picture.png"}
    source["input"]["pictures"] = [picture]

    def ref(path):
        return {"path": str(cache / path), "sha256": "0" * 64, "raw_sha256": "1" * 64}

    geometry = [{"frame_index": i, "status": "unavailable_picture"} for i in stage.FRAMES]
    geometry[0].update(
        status="geometry_available",
        picture=picture,
        source_to_crop_matrix_hex=[[float(v).hex() for v in row] for row in ((2, 0, 0.5), (0, 2, 0.5))],
        crop=ref("geometry-0027/crop.png"),
        alpha=ref("geometry-0027/alpha.npy"),
    )
    frames = []
    for mode in stage.CONDITIONS:
        for i in stage.FRAMES:
            row = {
                "condition": mode,
                "frame_index": i,
                "seed": stage.seed(3, i),
                "status": "unavailable",
                "reason": "Missing authored picture",
                "rendered": False,
            }
            if i == 27:
                row.update(
                    status="rendered",
                    rendered=True,
                    files={kind: ref(f"{mode}-0027-{kind}.png") for kind in ("crop", "native", "mask")},
                    proof={"seed": row["seed"], "output_rgb_sha256": "1" * 64},
                    conditioning_plan=source["frames"][0][
                        "source_time_conditioning" if mode == "source_time" else "earlier_conditioning"
                    ],
                    outside_alpha_changed_pixels=0,
                    changed_native_pixels=0 if mode == "source_time" else 12,
                )
            frames.append(row)
    records = {
        "schema": "chipmunk.grid-musetalk-frames-private/148",
        "clip_ordinal": 3,
        "planned": source,
        "frames": frames,
        "geometry": geometry,
        "render_repeat": {
            "frame_index": 27,
            "repeat_bitwise": True,
            "proof": copy.deepcopy(frames[0]["proof"]),
            "files": {
                kind: ref(f"source_time-repeat-0027-{kind}.png") for kind in ("crop", "native", "mask")
            },
        },
    }
    roots = stage.expected_roots(3)
    result = {
        "planned_condition_frames": 40,
        "executed_condition_frames": 2,
        "available_condition_frames": 2,
        "skipped_condition_frames": 38,
        "geometry_available_frames": 1,
        "executed_render_repeat_frames": 1,
        "renderer_repeat_bitwise": True,
        "geometry_repeat_bitwise": True,
        "renderer_inputs": 3,
        "top_level_module_forwards_started": roots,
        "top_level_module_forwards_completed": roots,
        "face_inputs_started": 2,
        "face_inputs_completed": 2,
        "new_neural_calls": 14,
        "new_speech_encoder_inputs": 0,
        "zero_effect_native_outputs": 1,
    }
    return records, result, source


def test_completed_partial_inventory_retains_zero_effect_and_fixed_repeat(tmp_path):
    stage.check_output_inventory(*authored_inventory(tmp_path), tmp_path)


@pytest.mark.parametrize("problem", ["path", "repeat", "counts"])
def test_completed_inventory_rejects_output_substitution_or_invented_counts(problem, tmp_path):
    records, result, source = authored_inventory(tmp_path)
    if problem == "path":
        records["frames"][0]["files"]["native"]["path"] = str(tmp_path.parent / "other-clip.png")
    elif problem == "repeat":
        records["render_repeat"]["files"]["mask"]["raw_sha256"] = "2" * 64
    else:
        result["new_neural_calls"] += 1
    with pytest.raises(ValueError):
        stage.check_output_inventory(records, result, source, tmp_path)


def test_fixed_cli_routing_cannot_start_other_worker_commands(monkeypatch):
    calls = []
    monkeypatch.setattr(stage.subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs)))
    stage.routed_popen(
        ["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", "/private/output"],
        3,
        env={"FIXTURE": "1"},
    )
    assert calls == [
        (
            ["python", "-B", "-m", "scripts.grid_musetalk_frames_v148", "authored", "/private/output", "3"],
            {"env": {"FIXTURE": "1"}},
        )
    ]
    with pytest.raises(ValueError):
        stage.routed_popen(
            ["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract", "/private/output"], 3
        )


def test_import_has_no_pretrained_runtime_or_geometry_side_effect():
    check = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; from scripts import grid_musetalk_frames_v148; assert not {'torch', 'cv2', 'mediapipe', 'transformers', 'diffusers'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
