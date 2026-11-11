import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from scripts import grid_portrait_frames_v146 as stage


def hexes(value):
    return np.vectorize(lambda x: float(x).hex(), otypes=[object])(np.asarray(value)).tolist()


def fixture():
    pictures = [
        {
            "frame_index": i,
            "pts": i * 3600,
            "file": f"/private/fixture/source-{i:04d}.png",
            "file_sha256": "png",
            "rgb_sha256": "rgb",
        }
        for i in range(75)
    ]
    source = {
        "ordinal": 2,
        "role": "pilot",
        "input": {
            "pictures": pictures,
            "native_video_table": {
                "pts": [p["pts"] for p in pictures],
                "time_base": "1/90000",
                "height": 48,
                "width": 64,
            },
        },
        "frames": [
            {
                "frame_index": i,
                "source_picture": pictures[i],
                "earlier_picture": pictures[i - 8],
                "source_time_seconds": str(stage.Fraction(i, 25)),
                "earlier_time_seconds": str(stage.Fraction(i - 8, 25)),
            }
            for i in stage.FRAMES
        ],
    }
    mesh = np.linspace(0.25, 0.75, 478 * 3).reshape(478, 3)
    mesh[78, :2], mesh[308, :2] = (0.3, 0.52), (0.7, 0.52)
    mesh[13, :2], mesh[14, :2] = (0.5, 0.50), (0.5, 0.54)
    matrix = np.array([[2.0, 0, 4], [0, 2.0, 6]])
    rows = [
        {
            "frame_index": p["frame_index"],
            "pts": p["pts"],
            "time_base": "1/90000",
            "source_file": p["file"],
            "source_file_sha256": p["file_sha256"],
            "source_rgb_sha256": p["rgb_sha256"],
            "status": "geometry_available",
            "mesh_xyz_hex": hexes(mesh),
            "source_to_256_matrix_hex": hexes(matrix),
            "source_to_512_matrix_hex": hexes(stage.upscale_matrix(matrix)),
            "source_ratio_hex": (0.125).hex(),
            "crop": {"path": "/private/fixture/crop.png", "sha256": "crop"},
            "alpha": {"path": "/private/fixture/alpha.npy", "sha256": "alpha"},
        }
        for p in pictures[19:47]
    ]
    records = {
        "schema": "chipmunk.grid-portrait-geometry-records/144",
        "clip_ordinal": 2,
        "planned": copy.deepcopy(source),
        "pictures": rows,
        "repeat": copy.deepcopy(rows[8]),
        "repeat_bitwise": True,
        "geometry_qualified": True,
    }
    return source, records


def test_full_plan_has_forty_native_pairs_and_five_roots_per_render():
    source, records = fixture()
    jobs, _ = stage.jobs_for(source, records)
    assert len(jobs) == sum(j["status"] == "available" for j in jobs) == 40
    assert jobs[0]["frame_index"] == jobs[0]["driver_frame_index"] == 27
    assert jobs[20]["frame_index"] == 27 and jobs[20]["driver_frame_index"] == 19
    assert jobs[-1]["frame_index"] == 46 and jobs[-1]["driver_frame_index"] == 38
    assert stage.expected_roots(41) == dict(zip(stage.ROOT_NAMES, (41, 82, 41, 41, 0), strict=True))
    assert sum(stage.expected_roots(41).values()) == 205


def test_unrelated_geometry_failure_does_not_exclude_supported_pairs():
    source, records = fixture()
    records["geometry_qualified"] = False
    records["pictures"][0]["status"] = "unavailable_no_face"
    jobs, _ = stage.jobs_for(source, records)
    assert sum(j["status"] == "available" for j in jobs) == 39
    assert [j["condition"] for j in jobs if j["status"] == "unavailable"] == ["earlier_320ms"]
    assert jobs[0]["status"] == "available"


def test_unavailable_repeat_picture_is_not_replaced_by_another_frame():
    source, records = fixture()
    records["geometry_qualified"] = False
    records["repeat_bitwise"] = None
    records["pictures"][8]["status"] = "unavailable_multiple_faces"
    jobs, _ = stage.jobs_for(source, records)
    assert jobs[0]["status"] == "unavailable"
    assert sum(j["status"] == "available" for j in jobs) == 37
    assert sum(stage.expected_roots(37).values()) == 185


def test_wholly_unavailable_geometry_preserves_all_planned_pairs_without_forwards():
    source, records = fixture()
    records["geometry_qualified"], records["repeat_bitwise"] = False, None
    for row in records["pictures"]:
        row["status"] = "unavailable_no_face"
    jobs, _ = stage.jobs_for(source, records)
    assert len(jobs) == 40 and all(j["status"] == "unavailable" for j in jobs)
    assert sum(stage.expected_roots(0).values()) == 0


def test_unavailable_whole_request_set_has_zero_model_inputs_and_forty_skips():
    source, records = fixture()
    source["geometry"] = {
        "status": "unavailable",
        "pictures": [],
        "unavailable_reason": "Native context missing",
    }
    records.update(
        planned=copy.deepcopy(source), pictures=[], repeat=None, repeat_bitwise=None, geometry_qualified=False
    )
    jobs, rows = stage.jobs_for(source, records)
    assert rows == {} and len(jobs) == 40 and all(job["status"] == "unavailable" for job in jobs)
    assert sum(stage.expected_roots(0).values()) == 0


def test_empty_geometry_cannot_hide_available_planned_requests():
    source, records = fixture()
    source["geometry"] = {"status": "ready", "pictures": source["input"]["pictures"][19:47]}
    records.update(
        planned=copy.deepcopy(source), pictures=[], repeat=None, repeat_bitwise=None, geometry_qualified=False
    )
    with pytest.raises(ValueError):
        stage.jobs_for(source, records)


@pytest.mark.parametrize("empty_request_set", [False, True])
def test_unavailable_authored_worker_retains_outputs_without_loading_models(
    empty_request_set, tmp_path, monkeypatch, capsys
):
    import types

    source, records = fixture()
    records.update(repeat_bitwise=None, geometry_qualified=False)
    if empty_request_set:
        source["geometry"] = {
            "status": "unavailable",
            "pictures": [],
            "unavailable_reason": "Missing original context",
        }
        records.update(planned=copy.deepcopy(source), pictures=[], repeat=None)
    else:
        for row in records["pictures"]:
            row["status"] = "unavailable_no_face"
    jobs, by_index = stage.jobs_for(source, records)
    monkeypatch.setattr(stage, "input_binding", lambda ordinal: (source, records, jobs, by_index, {}))
    cache = tmp_path / "authored-cache"
    monkeypatch.setattr(stage, "CACHE", cache)
    monkeypatch.setattr(
        stage, "dependencies", lambda: (None, types.SimpleNamespace(CACHE=tmp_path / "geometry"))
    )
    public = tmp_path / "authored-freeze.json"
    public.write_text("{}\n")
    monkeypatch.setattr(stage, "PUBLIC", public)
    output = tmp_path / "output"
    output.mkdir()
    stage.build(2, output)
    result = json.loads(capsys.readouterr().out)
    assert result["planned_condition_frames"] == result["skipped_condition_frames"] == 40
    assert result["available_condition_frames"] == result["executed_condition_frames"] == 0
    assert (
        result["renderer_inputs"]
        == result["strict_exact_checkpoint_loads"]
        == result["new_neural_calls"]
        == 0
    )
    assert result["executed_repeat_frames"] == 0 and result["renderer_repeat_bitwise"] is None
    references = json.loads((cache / "clip-02/outputs.json").read_text())
    assert len(references["frames"]) == 40 and all(not row["rendered"] for row in references["frames"])
    assert references["repeat"] is None and references["state_loads"] == []
    assert (
        (output / "root-progress.jsonl").read_bytes()
        == (output / "operator-progress.jsonl").read_bytes()
        == b""
    )


@pytest.mark.parametrize(
    "problem",
    [
        "failed_repeat",
        "unknown_status",
        "source_png",
        "wrong_pts",
        "earlier_offset",
        "driver_substitution",
        "missing_picture",
        "noncanonical",
        "wrong_512",
        "nonfinite_mesh",
    ],
)
def test_corrupt_routes_and_geometry_are_integrity_failures(problem):
    source, records = fixture()
    row = records["pictures"][0]
    if problem == "failed_repeat":
        records["repeat_bitwise"] = False
    elif problem == "unknown_status":
        row["status"] = "original"
    elif problem == "source_png":
        row["source_file_sha256"] = "substitute"
    elif problem == "wrong_pts":
        row["pts"] += 1
    elif problem in {"earlier_offset", "driver_substitution"}:
        if problem == "earlier_offset":
            source["frames"][0]["earlier_time_seconds"] = "79/100"
        else:
            source["frames"][0]["earlier_picture"] = source["input"]["pictures"][18]
        records["planned"] = copy.deepcopy(source)
    elif problem == "missing_picture":
        records["pictures"].pop()
    elif problem == "noncanonical":
        row["source_to_256_matrix_hex"][0][0] = "2"
    elif problem == "wrong_512":
        row["source_to_512_matrix_hex"][0][2] = (8.0).hex()
    else:
        row["mesh_xyz_hex"][0][0] = float("nan").hex()
    with pytest.raises(ValueError):
        stage.jobs_for(source, records)


def test_full_resolution_pixel_centres_have_the_required_half_pixel_translation():
    matrix = np.array([[0.75, 0.1, 8.0], [-0.1, 0.75, -3.0]])
    point = np.array([31.25, 29.75, 1.0])
    assert np.array_equal(stage.upscale_matrix(matrix) @ point, 2 * (matrix @ point) + 0.5)


def test_native_support_contains_exact_pixels_and_zero_effect_is_retained():
    source = np.full((8, 10, 3), 10, np.uint8)
    transformed = np.full_like(source, 110)
    alpha = np.zeros((8, 10), np.float32)
    alpha[2:4, 3:6] = 1
    edited, mask = stage.composite(source, transformed, alpha, stage.opacity(27))
    assert np.all(edited[2:4, 3:6] == 60)
    assert np.array_equal(edited[alpha == 0], source[alpha == 0])
    assert np.count_nonzero(mask) == 6 and np.all(mask[alpha == 0] == 0)
    same, empty = stage.composite(source, source.copy(), alpha, 1.0)
    assert np.array_equal(same, source) and np.count_nonzero(empty) == 0
    assert [stage.opacity(i) for i in stage.FRAMES] == [0.5] + [1.0] * 18 + [0.5]


@pytest.mark.parametrize(
    "value",
    [np.zeros((256, 256, 3), np.uint8), np.zeros((512, 512, 3), np.float32), np.zeros((512, 512), np.uint8)],
)
def test_partial_or_wrong_precision_output_is_rejected(value):
    with pytest.raises(ValueError):
        stage.generated_contract(value)


def test_png_round_trip_is_exact_and_exclusive(tmp_path):
    rgb = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    path = tmp_path / "authored.png"
    record = stage.save_png(path, rgb)
    assert record["raw_sha256"] == stage.fingerprint(rgb)
    with pytest.raises(FileExistsError):
        stage.save_png(path, np.zeros_like(rgb))


def input_fixture(tmp_path):
    import cv2

    source, records = fixture()
    row = records["pictures"][8]
    rgb = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    source_path = tmp_path / "source.png"
    record = stage.save_png(source_path, rgb)
    picture = source["input"]["pictures"][27]
    picture.update(file=str(source_path), file_sha256=record["sha256"], rgb_sha256=record["raw_sha256"])
    row.update(
        source_file=picture["file"],
        source_file_sha256=picture["file_sha256"],
        source_rgb_sha256=picture["rgb_sha256"],
    )
    mesh, matrix, _ = stage.row_contract(row, source)
    crop = cv2.warpAffine(rgb, matrix, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    row["crop"] = stage.save_png(tmp_path / "crop.png", crop)
    alpha_path = tmp_path / "alpha.npy"
    with alpha_path.open("xb") as stream:
        np.save(stream, stage.alpha_from_mesh(mesh, rgb.shape), allow_pickle=False)
    row["alpha"] = {"path": str(alpha_path), "sha256": stage.sha(alpha_path)}
    return source, row, rgb


def test_actual_cached_crop_alpha_matrix_and_native_rgb_are_reconstructed(tmp_path):
    source, row, expected = input_fixture(tmp_path)
    rgb, crop, alpha, matrix, proof = stage.load_frame(row, source, tmp_path)
    assert np.array_equal(rgb, expected) and crop.shape == (256, 256, 3)
    assert alpha.shape == expected.shape[:2] and matrix.shape == (2, 3)
    assert proof["matrix256_f64_sha256"] != proof["matrix512_f64_sha256"]


@pytest.mark.parametrize(
    "problem", ["changed_alpha", "crop_correspondence", "matrix_correspondence", "outside_cache", "raw_rgb"]
)
def test_authored_payload_substitution_is_detected_even_with_updated_file_hash(problem, tmp_path):
    source, row, rgb = input_fixture(tmp_path)
    if problem == "changed_alpha":
        with open(row["alpha"]["path"], "wb") as stream:
            np.save(stream, np.zeros(rgb.shape[:2], np.float32), allow_pickle=False)
        row["alpha"]["sha256"] = stage.sha(row["alpha"]["path"])
    elif problem == "crop_correspondence":
        path = tmp_path / "different.png"
        row["crop"] = stage.save_png(path, np.zeros((256, 256, 3), np.uint8))
    elif problem == "matrix_correspondence":
        matrix = stage.hex_array(row["source_to_256_matrix_hex"], (2, 3))
        matrix[0, 2] += 5
        row["source_to_256_matrix_hex"] = hexes(matrix)
        row["source_to_512_matrix_hex"] = hexes(stage.upscale_matrix(matrix))
    elif problem == "outside_cache":
        row["crop"]["path"] = str(tmp_path.parent / "outside.png")
    else:
        row["source_rgb_sha256"] = "changed"
        source["input"]["pictures"][27]["rgb_sha256"] = "changed"
    with pytest.raises((ValueError, FileNotFoundError)):
        stage.load_frame(row, source, tmp_path)


def test_only_the_fixed_authored_cli_is_routed(monkeypatch):
    calls = []
    monkeypatch.setattr(stage.subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs)))
    argv = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", "/private/fixed/output"]
    stage.routed_popen(argv, 2, env={"FIXTURE": "1"})
    assert calls == [
        (
            [
                "python",
                "-B",
                "-m",
                "scripts.grid_portrait_frames_v146",
                "authored",
                "/private/fixed/output",
                "2",
            ],
            {"env": {"FIXTURE": "1"}},
        )
    ]
    with pytest.raises(ValueError):
        stage.routed_popen([*argv, "unexpected"], 2)


def qualified_metadata_fixture(tmp_path, monkeypatch):
    root = tmp_path
    base, panel = root / "data", root / "panel"
    folder = base / "portrait-expression-qualifier-v140"
    output = folder / "authored/output"
    output.mkdir(parents=True)
    panel.mkdir()
    code = {}
    for name in stage.FROZEN_CODE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("authored historical source fixture\n")
        code[name] = stage.sha(path)
    frozen = {"code_sha256": code}
    stage.write(folder / "freeze.json", frozen)
    stage.write(panel / "portrait-expression-qualifier-freeze-v140.json", frozen)
    result = {
        "schema": "chipmunk.portrait-expression-qualifier-result/140",
        "status": "complete",
        "freeze_sha256": stage.sha(folder / "freeze.json"),
        "strict_exact_checkpoint_loads": 5,
        "renderer_inputs": 3,
        "new_face_task_inputs": 0,
        "new_neural_calls": 15,
        "top_level_module_forwards_started": dict(zip(stage.ROOT_NAMES, (3, 6, 3, 3, 0), strict=True)),
        "top_level_module_forwards_completed": dict(zip(stage.ROOT_NAMES, (3, 6, 3, 3, 0), strict=True)),
        "renderer_repeat_bitwise": True,
        "renderer_response_qualified": True,
        "native_coordinate_containment_qualified": True,
    }
    for name, key in [
        ("private-render-records.json", "private_render_records_sha256"),
        ("private-input-files.json", "private_input_files_sha256"),
        ("render-records.jsonl", "private_render_journal_sha256"),
        ("forward-progress.jsonl", "private_forward_progress_sha256"),
    ]:
        (output / name).write_text("{}\n")
        result[key] = stage.sha(output / name)
    stage.write(output / "result.json", result)
    profile = {
        "child_exit_code": 0,
        "guard": None,
        "sample_errors": 0,
        "new_neural_calls": 15,
        "worker_receipt_sha256": stage.sha(output / "result.json"),
    }
    stage.write(folder / "authored/profile.json", profile)
    stage.write(
        panel / "portrait-expression-qualifier-result-v140.json",
        {
            "status": "complete",
            "measurement": result,
            "profile": profile,
            "result_sha256": stage.sha(output / "result.json"),
            "profile_sha256": stage.sha(folder / "authored/profile.json"),
        },
    )
    monkeypatch.setattr(stage, "ROOT", root)
    monkeypatch.setattr(stage, "BASE", base)
    monkeypatch.setattr(stage, "PANEL", panel)
    return folder, output


def test_actual_complete_renderer_metadata_is_bound_without_media_or_model_imports(tmp_path, monkeypatch):
    _, output = qualified_metadata_fixture(tmp_path, monkeypatch)
    binding = stage.completed_renderer()
    assert binding[str((output / "render-records.jsonl").relative_to(tmp_path))] == stage.sha(
        output / "render-records.jsonl"
    )


@pytest.mark.parametrize("problem", ["journal", "historical_code", "worker_receipt"])
def test_a_stale_success_receipt_does_not_qualify_changed_artifacts(problem, tmp_path, monkeypatch):
    folder, output = qualified_metadata_fixture(tmp_path, monkeypatch)
    if problem == "journal":
        (output / "render-records.jsonl").write_text("changed\n")
    elif problem == "historical_code":
        (tmp_path / stage.FROZEN_CODE[0]).write_text("changed\n")
    else:
        path = folder / "authored/profile.json"
        record = json.loads(path.read_text())
        record["worker_receipt_sha256"] = "stale"
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        stage.completed_renderer()


def test_import_has_no_pretrained_runtime_or_geometry_side_effect():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; from scripts import grid_portrait_frames_v146; assert not {'torch', 'mediapipe', 'transformers', 'diffusers'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
