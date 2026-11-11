import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import grid_portrait_geometry_v144 as geometry


def planned(ordinal=0):
    pictures = [
        {
            "frame_index": i,
            "pts": i * 3600 + 100,
            "file": f"/authored/frame-{i:04d}.png",
            "file_sha256": "png",
            "rgb_sha256": "rgb",
        }
        for i in range(75)
    ]
    source = {
        "pictures": pictures,
        "native_video_table": {"pts": [p["pts"] for p in pictures], "time_base": "1/90000"},
    }
    return {
        "ordinal": ordinal,
        "source_slot": ordinal // 2,
        "within_source_ordinal": ordinal % 2,
        "role": "pilot",
        "input": source,
        "geometry": {
            "status": "ready",
            "frame_indices": list(range(19, 47)),
            "repeat_frame_index": 27,
            "pictures": pictures[19:47],
        },
    }


def face():
    return [SimpleNamespace(x=float(i / 1000), y=0.5, z=0.0) for i in range(478)]


def crop_fn(rgb, mesh):
    return np.zeros((256, 256, 3), np.uint8), np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]), 0.25


def alpha_fn(shape, mesh):
    return np.ones(shape[:2], dtype=np.float32)


def observe(faces=None, crop=crop_fn, alpha=alpha_fn):
    events = []
    picture = planned()["geometry"]["pictures"][0]
    result = geometry.observe_geometry(
        picture,
        "1/90000",
        np.zeros((80, 100, 3), np.uint8),
        lambda rgb: SimpleNamespace(face_landmarks=[face()] if faces is None else faces),
        crop,
        alpha,
        events.append,
    )
    return result, events


def test_all28_original_native_requests_and_exact_repeat27_are_fixed():
    row = planned()
    requests = geometry.requests(row)
    assert [p["frame_index"] for p in requests] == [*range(19, 47), 27]
    assert requests[-1] is row["input"]["pictures"][27]
    row["geometry"]["status"] = "unavailable"
    assert geometry.requests(row) == []


@pytest.mark.parametrize("damage", ["reserve", "drop", "repeat", "pts", "order"])
def test_no_reserve_substitution_or_geometry_request_changes(damage):
    row = copy.deepcopy(planned())
    if damage == "reserve":
        row["role"] = "reserve"
    elif damage == "drop":
        row["geometry"]["pictures"].pop()
    elif damage == "repeat":
        row["geometry"]["repeat_frame_index"] = 28
    elif damage == "pts":
        row["geometry"]["pictures"][0]["pts"] += 1
    else:
        row["geometry"]["pictures"].reverse()
    with pytest.raises(ValueError):
        geometry.requests(row)


def test_hex_geometry_and_full512_pixel_centre_matrix_are_exact():
    (row, crop, alpha), events = observe()
    assert row["status"] == geometry.AVAILABLE and crop.shape == (256, 256, 3) and alpha.dtype == np.float32
    matrix = geometry.decode_hex(row["source_to_256_matrix_hex"], (2, 3))
    full = geometry.decode_hex(row["source_to_512_matrix_hex"], (2, 3))
    np.testing.assert_array_equal(full, [[2.0, 0.0, 4.5], [0.0, 2.0, 6.5]])
    assert row["source_ratio_hex"] == (0.25).hex()
    assert events == [
        {"event": "face_task_begin", "frame_index": 19},
        {"event": "face_task_end", "frame_index": 19, "observed_faces": 1},
    ]
    assert geometry.canonical_hex(matrix, (2, 3)) == row["source_to_256_matrix_hex"]


@pytest.mark.parametrize(
    "faces,status",
    [
        ([], "unavailable_no_face"),
        ([face(), face()], "unavailable_multiple_faces"),
        ([face()[:477]], "unavailable_mesh"),
    ],
)
def test_missing_ambiguous_or_incomplete_mesh_is_retained_after_exact_task_call(faces, status):
    (row, crop, alpha), events = observe(faces)
    assert row["status"] == status and crop is alpha is None
    assert [e["event"] for e in events] == ["face_task_begin", "face_task_end"]


def test_nonfinite_mesh_is_unavailable_and_never_used_for_cropping():
    values = face()
    values[3].x = float("nan")

    def forbidden(*args):
        pytest.fail("Invalid mesh must not reach geometry helpers")

    (row, _, _), _events = observe([values], crop=forbidden)
    assert row["status"] == "unavailable_mesh"


@pytest.mark.parametrize(
    "damage", ["singular", "crop_size", "alpha_dtype", "alpha_empty", "alpha_nan", "ratio"]
)
def test_invalid_crop_support_or_matrix_is_unavailable(damage):
    def crop(rgb, mesh):
        value, matrix, ratio = crop_fn(rgb, mesh)
        if damage == "singular":
            matrix[:] = 0
        elif damage == "crop_size":
            value = value[:128]
        elif damage == "ratio":
            ratio = float("inf")
        return value, matrix, ratio

    def alpha(shape, mesh):
        value = alpha_fn(shape, mesh)
        if damage == "alpha_dtype":
            value = value.astype(np.float64)
        elif damage == "alpha_empty":
            value[:] = 0
        elif damage == "alpha_nan":
            value[0, 0] = float("nan")
        return value

    (row, crop, alpha), _events = observe(crop=crop, alpha=alpha)
    assert row["status"] == "unavailable_geometry" and crop is alpha is None


def test29_calls_continue_after_an_unavailable_face_without_replacement():
    events, seen, records = [], [], []

    def detect(rgb):
        seen.append(len(seen))
        return SimpleNamespace(face_landmarks=[] if len(seen) == 1 else [face()])

    for picture in geometry.requests(planned()):
        row, _crop, _alpha = geometry.observe_geometry(
            picture, "1/90000", np.zeros((80, 100, 3), np.uint8), detect, crop_fn, alpha_fn, events.append
        )
        records.append(row)
    assert len(seen) == 29 and records[0]["status"] == "unavailable_no_face"
    assert [r["frame_index"] for r in records] == [*range(19, 47), 27]
    assert (
        sum(e["event"] == "face_task_begin" for e in events)
        == sum(e["event"] == "face_task_end" for e in events)
        == 29
    )


def test_task_runtime_error_stays_failed_with_a_flushed_begin_not_unavailable():
    events = []

    def detect(_rgb):
        raise RuntimeError("authored task fault")

    with pytest.raises(RuntimeError, match="task fault"):
        geometry.observe_geometry(
            planned()["geometry"]["pictures"][0],
            "1/90000",
            np.zeros((80, 100, 3), np.uint8),
            detect,
            crop_fn,
            alpha_fn,
            events.append,
        )
    assert [e["event"] for e in events] == ["face_task_begin"]


def test_own_repeat_checks_mesh_mapping_aperture_crop_and_exact_float_alpha():
    (row, _, _), _events = observe()
    row.update(crop={"sha256": "crop"}, alpha={"sha256": "alpha"})
    assert geometry.repeat_check(row, copy.deepcopy(row)) is True
    for field in (
        "mesh_xyz_hex",
        "source_to_256_matrix_hex",
        "source_to_512_matrix_hex",
        "source_ratio_hex",
        "crop",
        "alpha",
    ):
        changed = copy.deepcopy(row)
        changed[field] = {"sha256": "changed"} if field in {"crop", "alpha"} else "changed"
        assert geometry.repeat_check(row, changed) is False
    assert geometry.repeat_check(row, {"status": "unavailable_no_face"}) is None


@pytest.mark.parametrize("value", ["nan", "inf", "0x1p+0"])
def test_hex_cache_rejects_nonfinite_or_noncanonical_float_strings(value):
    with pytest.raises(ValueError):
        geometry.decode_hex([[value]], (1, 1))


def completion_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(geometry, "TRIAL", tmp_path / "trial")
    monkeypatch.setattr(geometry, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(geometry, "PUBLIC", tmp_path / "public.json")
    monkeypatch.setattr(geometry, "verify", lambda: None)
    geometry.TRIAL.mkdir()
    geometry.PUBLIC.write_text("{}")
    selected = planned()
    geometry.write(geometry.TRIAL / "plan.json", {"pilots": [selected]})
    stage = geometry.TRIAL / "clip-00/authored"
    out = stage / "output"
    out.mkdir(parents=True)
    records, progress = [], []
    for i, picture in enumerate(geometry.requests(selected)):
        (template, _, _), _events = observe()
        row = {**template, **geometry.base_record(picture, "1/90000")}
        cache = (
            geometry.CACHE / "clip-00" / ("repeat-0027" if i == 28 else f"frame-{picture['frame_index']:04d}")
        )
        row.update(
            crop={"path": str(cache / "crop.png"), "sha256": "crop"},
            alpha={"path": str(cache / "alpha.npy"), "sha256": "alpha"},
        )
        records.append(row)
        progress.extend(
            [
                {"event": "face_task_begin", "frame_index": picture["frame_index"]},
                {"event": "face_task_end", "frame_index": picture["frame_index"], "observed_faces": 1},
                {
                    "event": "geometry_record",
                    "request_ordinal": i,
                    "frame_index": picture["frame_index"],
                    "status": geometry.AVAILABLE,
                },
            ]
        )
    body = {
        "schema": "chipmunk.grid-portrait-geometry-records/144",
        "clip_ordinal": 0,
        "planned": selected,
        "pictures": records[:28],
        "repeat": records[-1],
        "repeat_bitwise": True,
        "geometry_qualified": True,
    }
    geometry.write(out / "geometry-records.json", body)
    cache_records = geometry.CACHE / "clip-00/geometry-records.json"
    cache_records.parent.mkdir(parents=True)
    geometry.write(cache_records, body)
    geometry.write(out / "repeat-check.json", {"repeat_bitwise": True, "geometry_qualified": True})
    (out / "face-progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in progress))
    resources = [
        {
            "elapsed_s": i / 4,
            "tree_rss_bytes": i * 1024,
            "ram_available_bytes": 7 * 1024**3,
            "disk_free_bytes": 4 * 1024**3,
            "archive_free_disk_bytes": 4 * 1024**3,
            "violations": [],
        }
        for i in (0, 1)
    ]
    (stage / "resources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in resources))
    result = {
        "schema": "chipmunk.grid-portrait-geometry-result/144",
        "status": "complete",
        "clip_ordinal": 0,
        "task_delegate": "CPU",
        "running_mode": "IMAGE",
        "num_faces": 2,
        "freeze_sha256": geometry.sha(geometry.PUBLIC),
        "defaults_changed": False,
        "generator_calls": 0,
        "encoder_calls": 0,
        "detector_calls": 0,
        "private_geometry_records_sha256": geometry.sha(out / "geometry-records.json"),
        "private_geometry_cache_sha256": geometry.sha(cache_records),
        "private_face_progress_sha256": geometry.sha(out / "face-progress.jsonl"),
        "private_repeat_check_sha256": geometry.sha(out / "repeat-check.json"),
        "new_neural_calls": 29,
        "face_task_inputs_started": 29,
        "face_task_inputs_completed": 29,
        "planned_face_task_inputs": 29,
        "retained_distinct_picture_records": 28,
        "geometry_qualified": True,
        "repeat_bitwise": True,
        "availability": {geometry.AVAILABLE: 29},
    }
    geometry.write(out / "result.json", result)
    profile = {
        "schema": "chipmunk.video-appearance-resource-profile/108",
        "command": "authored",
        "batch": None,
        "child_started": True,
        "child_exit_code": 0,
        "guard": None,
        "sample_errors": 0,
        "error_types": [],
        "resource_limit_observations": [],
        "worker_receipt_sha256": geometry.sha(out / "result.json"),
        "resource_journal_sha256": geometry.sha(stage / "resources.jsonl"),
        "runtime_sha256": geometry.sha(geometry.__file__),
        "new_neural_calls": 29,
        "duration_s": 0.3,
        "sampled_tree_rss_lower_bound_bytes": 1024,
        "minimum_available_ram_bytes": 7 * 1024**3,
    }
    geometry.write(stage / "profile.json", profile)
    return out, body, progress, result, profile


def rewrite_completion(out, body, progress, result, profile):
    (out / "geometry-records.json").write_text(json.dumps(body))
    cache_records = geometry.CACHE / "clip-00/geometry-records.json"
    cache_records.write_text(json.dumps(body))
    (out / "face-progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in progress))
    result.update(
        private_geometry_records_sha256=geometry.sha(out / "geometry-records.json"),
        private_geometry_cache_sha256=geometry.sha(cache_records),
        private_face_progress_sha256=geometry.sha(out / "face-progress.jsonl"),
    )
    (out / "result.json").write_text(json.dumps(result))
    profile["worker_receipt_sha256"] = geometry.sha(out / "result.json")
    (out.parent / "profile.json").write_text(json.dumps(profile))


def test_completed_metadata_admission_never_reads_saved_crops_or_alpha(tmp_path, monkeypatch):
    out, body, *_ = completion_fixture(tmp_path, monkeypatch)
    assert geometry.load_completed(0) == body
    assert not list(geometry.CACHE.rglob("*.png")) and not list(geometry.CACHE.rglob("*.npy"))
    assert geometry.completed(0) and not geometry.completed(1)


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_status",
        "missing_end",
        "wrong_request_ordinal",
        "wrong_clip",
        "repeat_protocol",
        "outside_cache",
        "wrong_faces",
        "duplicate_call",
    ],
)
def test_rehashed_completion_still_rejects_protocol_corruption(tmp_path, monkeypatch, damage):
    out, body, progress, result, profile = completion_fixture(tmp_path, monkeypatch)
    if damage == "unknown_status":
        body["pictures"][0]["status"] = "original"
    elif damage == "missing_end":
        progress.pop(1)
    elif damage == "wrong_request_ordinal":
        progress[2]["request_ordinal"] = 1
    elif damage == "wrong_clip":
        body["clip_ordinal"] = 1
    elif damage == "repeat_protocol":
        body["repeat"]["frame_index"] = 28
    elif damage == "outside_cache":
        body["pictures"][0]["crop"]["path"] = str(tmp_path / "elsewhere.png")
    elif damage == "wrong_faces":
        progress[1]["observed_faces"] = 2
    else:
        progress[2:2] = copy.deepcopy(progress[:2])
    rewrite_completion(out, body, progress, result, profile)
    with pytest.raises(ValueError):
        geometry.load_completed(0)


def test_completed_partial_geometry_remains_available_for_supported_pairs(tmp_path, monkeypatch):
    out, body, progress, result, profile = completion_fixture(tmp_path, monkeypatch)
    body["pictures"][0] = {
        **geometry.base_record(body["planned"]["geometry"]["pictures"][0], "1/90000"),
        "status": "unavailable_no_face",
    }
    body["geometry_qualified"] = result["geometry_qualified"] = False
    result["availability"] = {geometry.AVAILABLE: 28, "unavailable_no_face": 1}
    progress[1]["observed_faces"] = 0
    progress[2]["status"] = "unavailable_no_face"
    rewrite_completion(out, body, progress, result, profile)
    assert geometry.load_completed(0)["geometry_qualified"] is False
    assert geometry.load_completed(0)["pictures"][1]["status"] == geometry.AVAILABLE


@pytest.mark.parametrize("which", ["source", "repeat", "both"])
def test_decode_unavailable_repeat_is_unassessed_with_safe_calls_retained(tmp_path, monkeypatch, which):
    out, body, progress, result, profile = completion_fixture(tmp_path, monkeypatch)
    positions = ([8] if which == "source" else [28]) if which != "both" else [8, 28]
    for position in positions:
        row = {
            **geometry.base_record(body["planned"]["input"]["pictures"][27], "1/90000"),
            "status": "unavailable_picture_decode",
        }
        if position == 28:
            body["repeat"] = row
        else:
            body["pictures"][8] = row
        for event in progress:
            if event["event"] == "geometry_record" and event["request_ordinal"] == position:
                event["status"] = "unavailable_picture_decode"
    progress = [
        event
        for i, event in enumerate(progress)
        if not (i // 3 in positions and event["event"] in {"face_task_begin", "face_task_end"})
    ]
    count = 29 - len(positions)
    body["geometry_qualified"] = result["geometry_qualified"] = False
    body["repeat_bitwise"] = result["repeat_bitwise"] = None
    result["availability"] = {geometry.AVAILABLE: count, "unavailable_picture_decode": len(positions)}
    result["new_neural_calls"] = result["face_task_inputs_started"] = result[
        "face_task_inputs_completed"
    ] = profile["new_neural_calls"] = count
    (out / "repeat-check.json").write_text(json.dumps({"repeat_bitwise": None, "geometry_qualified": False}))
    result["private_repeat_check_sha256"] = geometry.sha(out / "repeat-check.json")
    rewrite_completion(out, body, progress, result, profile)
    assert geometry.load_completed(0)["repeat_bitwise"] is None


def test_available_repeat_failure_cannot_be_admitted_by_rehashing(tmp_path, monkeypatch):
    out, body, progress, result, profile = completion_fixture(tmp_path, monkeypatch)
    body["repeat"]["crop"]["sha256"] = "different-crop"
    body["repeat_bitwise"] = result["repeat_bitwise"] = False
    body["geometry_qualified"] = result["geometry_qualified"] = False
    (out / "repeat-check.json").write_text(json.dumps({"repeat_bitwise": False, "geometry_qualified": False}))
    result["private_repeat_check_sha256"] = geometry.sha(out / "repeat-check.json")
    rewrite_completion(out, body, progress, result, profile)
    with pytest.raises(ValueError, match="own repeat"):
        geometry.load_completed(0)


@pytest.mark.parametrize("damage", ["float_bytes", "nan_elapsed", "ram", "guard", "rss"])
def test_original_neural_resource_limits_are_replayed(tmp_path, monkeypatch, damage):
    out, body, progress, result, profile = completion_fixture(tmp_path, monkeypatch)
    path = out.parent / "resources.jsonl"
    resources = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == "float_bytes":
        resources[0]["disk_free_bytes"] = float(resources[0]["disk_free_bytes"])
    elif damage == "nan_elapsed":
        resources[0]["elapsed_s"] = float("nan")
    elif damage == "ram":
        resources[0]["ram_available_bytes"] = 5 * 1024**3
    elif damage == "guard":
        profile["guard"] = "resource_limit"
    else:
        resources[1]["tree_rss_bytes"] = 7 * 1024**3
    path.write_text("".join(json.dumps(r) + "\n" for r in resources))
    profile["resource_journal_sha256"] = geometry.sha(path)
    rewrite_completion(out, body, progress, result, profile)
    with pytest.raises(ValueError):
        geometry.load_completed(0)


def test_only_fixed_exclusive_per_pilot_neural_argv_is_dispatched(monkeypatch):
    calls = []
    monkeypatch.setattr(geometry.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    argv = [
        "python",
        "-B",
        "-m",
        "scripts.video_appearance_native_v108",
        "authored",
        str(geometry.TRIAL / "clip-00/authored/output"),
    ]
    geometry.routed_popen(0, argv)
    assert calls[0] == [
        "python",
        "-B",
        "-m",
        "scripts.grid_portrait_geometry_v144",
        "authored",
        argv[-1],
        "0",
    ]
    with pytest.raises(ValueError):
        geometry.routed_popen(1, argv)
    with pytest.raises(ValueError):
        geometry.routed_popen(0, [*argv[:4], "bind", *argv[5:]])
