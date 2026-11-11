import copy
import hashlib
import json
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import grid_mouth_observer_v157 as observer

SHAPE = [100, 120, 3]


def points():
    native = {
        1: (40, 32),
        13: (40, 42),
        14: (40, 47),
        33: (20, 20),
        78: (29, 45),
        82: (36, 43),
        87: (36, 46),
        133: (30, 20),
        263: (60, 20),
        308: (51, 45),
        312: (44, 43),
        317: (44, 46),
        362: (50, 20),
    }
    return np.asarray([native[i] for i in observer.POINTS], np.float64)


def face(values=None):
    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    if values is None:
        values = points()
    for index, (x, y) in zip(observer.POINTS, values, strict=True):
        landmarks[index] = SimpleNamespace(x=x / SHAPE[1], y=y / SHAPE[0], z=0.0)
    return landmarks


def inventory():
    pts = [9000 + i * 3600 for i in range(75)]
    source = {
        "ordinal": 3,
        "role": "pilot",
        "source_slot": 1,
        "component_index": 2,
        "input": {
            "native_video_table": {"pts": pts, "time_base": "1/90000", "height": SHAPE[0], "width": SHAPE[1]}
        },
    }
    original = [hashlib.sha256(f"original:{i}".encode()).hexdigest() for i in range(75)]
    cases = []
    for i, name in enumerate(observer.planning.CASES):
        if name in {"original", "audio_delay_320ms"}:
            hashes = original.copy()
        elif name == "genuine_frame_retime":
            hashes = [original[n - 8] if 27 <= n < 47 else original[n] for n in range(75)]
        else:
            hashes = [hashlib.sha256(f"{name}:{n}".encode()).hexdigest() for n in range(75)]
        cases.append(
            {
                "asset_index": 3,
                "source_slot": 1,
                "component_index": 2,
                "study_role": "pilot",
                "case_number": 10 - i,
                "condition": name,
                "status": "ready",
                "reason": None,
                "source_video_pts": pts,
                "source_video_time_base": "1/90000",
                "expected": {"picture_pts_ms": [p // 90 for p in pts], "rgb_sha256": hashes},
            }
        )
    return source, cases


def test_fixed_schedule_preserves_native_context_and_anonymous_case_numbers():
    source, cases = inventory()
    plan = observer.schedule(source, cases)
    requests = plan["requests"]
    assert plan["source_clock"]["time_base"] == "1/90000" and plan["picture_shape"] == SHAPE
    assert len(requests) == 285 and [r["request_ordinal"] for r in requests] == list(range(285))
    assert [r["frame_index"] for r in requests[:44]] == list(range(15, 59))
    assert [r["frame_index"] for r in requests[44:68]] == list(range(25, 49))
    assert requests[0]["case_number"] == 10
    assert requests[-1] == {**requests[12], "request_ordinal": 284, "repeat": True}
    assert Fraction(requests[12]["source_time_seconds"]) == Fraction(59, 50)
    assert all("condition" not in r for r in requests)


@pytest.mark.parametrize("damage", ["case_drop", "clock", "component", "retime", "audio", "reserve"])
def test_schedule_refuses_wrong_cohort_clock_or_exact_control(damage):
    source, cases = inventory()
    if damage == "case_drop":
        cases.pop()
    elif damage == "clock":
        cases[1]["expected"]["picture_pts_ms"][27] += 1
    elif damage == "component":
        cases[1]["component_index"] = 4
    elif damage == "retime":
        cases[-2]["expected"]["rgb_sha256"][27] = cases[0]["expected"]["rgb_sha256"][27]
    elif damage == "audio":
        cases[-1]["expected"]["rgb_sha256"][35] = "0" * 64
    else:
        source["role"] = "reserve"
    with pytest.raises(ValueError):
        observer.schedule(source, cases)


def test_unavailable_case_retains_all_requests_without_fabricated_decode_values():
    source, cases = inventory()
    cases[2].update(status="unavailable", reason="authored absence")
    del cases[2]["expected"]
    plan = observer.schedule(source, cases)
    selected = [r for r in plan["requests"] if r["case_number"] == cases[2]["case_number"]]
    assert len(plan["requests"]) == 285 and len(selected) == 24
    assert all(r["pts"] is r["time_base"] is r["rgb_sha256"] is None for r in selected)
    assert all(observer.unavailable(r)["status"] == "unavailable_case" for r in selected)


def test_descriptor_removes_native_similarity_transform():
    native = points()
    first = observer.describe(native, SHAPE)
    theta = 0.35
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    second = observer.describe(0.8 * native @ rotation.T + [25.0, 5.0], SHAPE)

    def decode(row):
        return [float.fromhex(x) for x in row["descriptor_hex"]]

    np.testing.assert_allclose(decode(first), [3 / 30, 5 / 30, 3 / 30, 22 / 30], rtol=0, atol=1e-12)
    np.testing.assert_allclose(decode(first), decode(second), rtol=0, atol=1e-12)
    assert float.fromhex(second["eye_roll_radians_hex"]) == pytest.approx(theta)
    assert float.fromhex(second["eye_distance_px_hex"]) == pytest.approx(24.0)


@pytest.mark.parametrize("damage", ["out_of_bounds", "nonfinite", "eye_scale", "width", "crossing", "nose"])
def test_invalid_native_geometry_has_explicit_guard(damage):
    native = points()
    index = observer.POINTS.index
    if damage == "out_of_bounds":
        native[index(1), 0] = SHAPE[1]
    elif damage == "nonfinite":
        native[index(13), 1] = float("nan")
    elif damage == "eye_scale":
        native[[index(362), index(263)], 0] -= 24
    elif damage == "width":
        native[index(308), 0] = 35
    elif damage == "crossing":
        native[index(14), 1] = 40
    else:
        native[index(1), 1] = 10
    with pytest.raises(ValueError):
        observer.describe(native, SHAPE)


def fixture_picture():
    rgb = np.zeros(SHAPE, np.uint8)
    request = {
        "request_ordinal": 0,
        "case_number": 10,
        "frame_index": 15,
        "repeat": False,
        "case_status": "ready",
        "pts": 700,
        "rgb_sha256": observer.rgb_commitment(rgb),
    }
    return request, SimpleNamespace(rgb=rgb, index=15, pts=700)


@pytest.mark.parametrize("damage", ["pts", "index", "rgb", "dtype"])
def test_bad_decoded_binding_never_reaches_observer(damage):
    request, frame = fixture_picture()
    if damage == "pts":
        frame.pts += 1
    elif damage == "index":
        frame.index += 1
    elif damage == "rgb":
        frame.rgb[0, 0, 0] = 1
    else:
        frame.rgb = frame.rgb.astype(np.float32)
    events = []

    def forbidden(_rgb):
        pytest.fail("Mismatched native input reached the observer")

    with pytest.raises(ValueError, match="before face task"):
        observer.observe_picture(request, frame, SHAPE, forbidden, events.append)
    assert events == []


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("none", "unavailable_no_face"),
        ("multiple", "unavailable_multiple_faces"),
        ("short", "unavailable_mesh"),
        ("bad_geometry", "unavailable_geometry"),
        ("finite", observer.AVAILABLE),
    ],
)
def test_face_task_protocol_records_observed_missing_geometry(kind, expected):
    request, frame = fixture_picture()
    values, events = face(), []
    if kind == "bad_geometry":
        values[1].y = 0.05
    faces = {
        "none": [],
        "multiple": [values, values],
        "short": [values[:477]],
        "bad_geometry": [values],
        "finite": [values],
    }[kind]

    def fake(rgb):
        rgb[0, 0, 0] = 255  # The observer receives a copy of the verified input.
        return SimpleNamespace(face_landmarks=faces)

    row = observer.observe_picture(request, frame, SHAPE, fake, events.append)
    assert row["status"] == expected and row["observed_faces"] == len(faces)
    if expected == observer.AVAILABLE:
        assert row["reason"] is None
    else:
        assert row["reason"]
    assert [event["event"] for event in events] == ["face_task_begin", "face_task_end"]
    assert events[-1]["observed_faces"] == len(faces) and frame.rgb[0, 0, 0] == 0


def test_task_runtime_exception_preserves_only_the_started_event():
    request, frame = fixture_picture()
    events = []

    def fail(_rgb):
        raise RuntimeError("authored runtime failure")

    with pytest.raises(RuntimeError, match="authored runtime failure"):
        observer.observe_picture(request, frame, SHAPE, fail, events.append)
    assert [event["event"] for event in events] == ["face_task_begin"]


def test_eye_patch_drift_requires_equal_native_eye_pixels():
    native = points()
    shifted = native.copy()
    shifted[[observer.POINTS.index(i) for i in observer.EYES], 0] += 2
    original = {"status": observer.AVAILABLE, **observer.describe(native, SHAPE)}
    variant = {"status": observer.AVAILABLE, **observer.describe(shifted, SHAPE)}
    a = np.zeros(SHAPE, np.uint8)
    b = a.copy()
    b[45, 40, 0] = 1
    matched = observer.eye_drift(original, variant, a, b, SHAPE)
    assert matched == {"status": "identical_eye_pixels", "rms_px_hex": (2.0).hex()}
    b[20, 20, 0] = 1
    assert observer.eye_drift(original, variant, a, b, SHAPE)["status"] == "different_eye_pixels"


def authored_receipt(source=None, cases=None):
    plan = observer.schedule(*(inventory() if source is None else (source, cases)))
    plan["bindings"] = {"constructor": "authored"}
    rows, events = [], []
    values = observer.describe(points(), SHAPE)
    for request in plan["requests"]:
        ready = request["case_status"] == "ready"
        row = (
            {**request, "status": observer.AVAILABLE, "observed_faces": 1, "reason": None, **values}
            if ready
            else observer.unavailable(request)
        )
        if request["case_number"] != plan["cases"][0]["case_number"]:
            row["eye_patch_drift"] = (
                {"status": "identical_eye_pixels", "rms_px_hex": (0.0).hex()}
                if ready
                else {"status": "unavailable_geometry", "rms_px_hex": None}
            )
        rows.append(row)
        if ready:
            events.extend(
                [
                    observer.task_event("face_task_begin", request),
                    observer.task_event("face_task_end", request, 1),
                ]
            )
    return plan, rows, events


def test_completed_protocol_keeps_unavailable_control_rows_without_task_events():
    source, cases = inventory()
    cases[-1].update(status="unavailable", reason="authored audio control absence")
    del cases[-1]["expected"]
    plan, rows, events = authored_receipt(source, cases)
    assert len(rows) == 285 and observer.validate_observations(plan, rows, events) == 261
    controls = observer.control_checks(plan, rows)
    assert controls["audio_delay_320ms"]["unavailable_pairs"] == 24
    assert controls["genuine_frame_retime"]["matching_geometry_pairs"] == 24


@pytest.mark.parametrize("damage", ["drop", "event", "order", "descriptor", "false_control_drift", "type"])
def test_observation_receipt_replays_exact_schedule_events_geometry_and_control_drift(damage):
    plan, rows, events = authored_receipt()
    assert observer.validate_observations(plan, rows, events) == 285
    if damage == "drop":
        rows.pop()
    elif damage == "event":
        events.pop(1)
    elif damage == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif damage == "descriptor":
        rows[1]["descriptor_hex"] = [(1.0).hex()] * 4
    elif damage == "false_control_drift":
        audio = plan["cases"][-1]["case_number"]
        row = next(r for r in rows if r["case_number"] == audio)
        row["eye_patch_drift"] = {"status": "different_eye_pixels", "rms_px_hex": None}
    else:
        rows[0]["request_ordinal"] = False
    with pytest.raises(ValueError):
        observer.validate_observations(plan, rows, events)


def test_controls_compare_their_own_bound_references_and_preserve_missing_scope():
    plan, rows, _events = authored_receipt()
    controls = observer.control_checks(plan, rows)
    assert controls["fixed_repeat"]["matching_geometry_pairs"] == 1
    assert controls["audio_delay_320ms"]["matching_geometry_pairs"] == 24
    assert controls["genuine_frame_retime"]["matching_geometry_pairs"] == 24
    original = plan["cases"][0]["case_number"]
    retime = plan["cases"][-2]["case_number"]
    # Frame 19 in the original is the retime reference for output frame 27.
    for number, index in ((original, 19), (retime, 27)):
        row = next(r for r in rows if r["case_number"] == number and r["frame_index"] == index)
        row.update(status="unavailable_no_face", observed_faces=0, reason="authored absence")
        for field in observer.GEOMETRY_FIELDS:
            del row[field]
    controls = observer.control_checks(plan, rows)
    assert controls["genuine_frame_retime"]["matching_unavailable_geometry_pairs"] == 1
    assert controls["genuine_frame_retime"]["mismatched_pairs"] == 0
    rows[-1]["descriptor_hex"] = [(0.0).hex()] * 4
    assert observer.control_checks(plan, rows)["fixed_repeat"]["mismatched_pairs"] == 1


def test_completed_loader_validates_actual_style_bound_receipts_without_media(tmp_path, monkeypatch):
    plan, rows, events = authored_receipt()
    trial, cache, public = tmp_path / "trial", tmp_path / "cache", tmp_path / "public.json"
    monkeypatch.setattr(observer, "TRIAL", trial)
    monkeypatch.setattr(observer, "CACHE", cache)
    monkeypatch.setattr(observer, "PUBLIC", public)
    monkeypatch.setattr(observer, "verify", lambda: None)
    monkeypatch.setattr(observer, "input_binding", lambda ordinal: (plan, []))
    folder, stage = cache / "clip-03", trial / "clip-03/authored"
    out = stage / "output"
    folder.mkdir(parents=True)
    out.mkdir(parents=True)
    observer.write(public, {})
    controls = observer.control_checks(plan, rows)
    observer.write(folder / "input-binding.json", plan)
    observer.write(folder / "control-checks.json", controls)
    for path, values in ((folder / "observations.jsonl", rows), (out / "face-progress.jsonl", events)):
        path.write_text("".join(json.dumps(v) + "\n" for v in values))
    resources = [
        {
            "elapsed_s": 0.25 * n,
            "tree_rss_bytes": n * 1024,
            "ram_available_bytes": 7 * 1024**3,
            "disk_free_bytes": 4 * 1024**3,
            "archive_free_disk_bytes": 4 * 1024**3,
            "violations": [],
        }
        for n in (0, 1)
    ]
    (stage / "resources.jsonl").write_text("".join(json.dumps(v) + "\n" for v in resources))
    result = {
        "schema": "chipmunk.grid-mouth-observer-result/157",
        "status": "complete",
        "clip_ordinal": 3,
        "defaults_changed": False,
        "freeze_sha256": observer.sha(public),
        **observer.observed_counts(rows, 285, controls),
        "private_input_binding_sha256": observer.sha(folder / "input-binding.json"),
        "private_observation_journal_sha256": observer.sha(folder / "observations.jsonl"),
        "private_control_checks_sha256": observer.sha(folder / "control-checks.json"),
        "private_face_progress_sha256": observer.sha(out / "face-progress.jsonl"),
    }
    observer.write(out / "result.json", result)
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
        "runtime_sha256": observer.sha(observer.__file__),
        "worker_receipt_sha256": observer.sha(out / "result.json"),
        "new_neural_calls": 285,
        "resource_journal_sha256": observer.sha(stage / "resources.jsonl"),
        "duration_s": 0.3,
        "sampled_tree_rss_lower_bound_bytes": 1024,
        "minimum_available_ram_bytes": 7 * 1024**3,
    }
    observer.write(stage / "profile.json", profile)
    body = observer.load_completed(3)
    assert body["observations"] == rows and body["controls"] == controls and "requests" not in body
    assert not list(tmp_path.rglob("*.mkv"))
    changed = copy.deepcopy(result)
    changed["new_neural_calls"] = 284
    (out / "result.json").write_text(json.dumps(changed))
    profile["new_neural_calls"] = 284
    profile["worker_receipt_sha256"] = observer.sha(out / "result.json")
    (stage / "profile.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="outcomes or counters"):
        observer.load_completed(3)


def test_fixed_route_dispatches_only_one_exclusive_observer(monkeypatch):
    seen = []
    monkeypatch.setattr(observer.subprocess, "Popen", lambda argv, **kw: seen.append(argv))
    expected = str(observer.TRIAL / "clip-03/authored/output")
    argv = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", expected]
    observer.routed_popen(3, argv)
    assert seen == [["python", "-B", "-m", "scripts.grid_mouth_observer_v157", "observe", expected, "3"]]
    with pytest.raises(ValueError):
        observer.routed_popen(4, argv)
    with pytest.raises(ValueError):
        observer.routed_popen(3, [*argv[:4], "bind", expected])


@pytest.mark.parametrize("state", ["missing", "finished", "detached_stdout", "detached_stderr", "attached"])
def test_worker_requires_live_monitor_and_both_log_descriptors_before_creating_output(
    tmp_path, monkeypatch, state
):
    monkeypatch.setattr(observer, "TRIAL", tmp_path)
    stage = tmp_path / "clip-03/authored"
    stage.mkdir(parents=True)
    output, log = stage / "output", stage / "worker.log"
    log.touch()
    if state != "missing":
        (stage / "resources.jsonl").write_text('{"authored":true}\n')
    if state == "finished":
        (stage / "profile.json").touch()
    unrelated = tmp_path / "unrelated"
    unrelated.touch()
    fake_stats = {1: log.stat(), 2: log.stat()}
    if state.startswith("detached"):
        fake_stats[1 if state == "detached_stdout" else 2] = unrelated.stat()
    seen = []
    real_fstat = observer.os.fstat
    monkeypatch.setattr(
        observer.os, "fstat", lambda fd: fake_stats[fd] if fd in fake_stats else real_fstat(fd)
    )
    monkeypatch.setattr(observer, "verify", lambda: seen.append("verify"))
    monkeypatch.setattr(observer, "observe", lambda ordinal, path: seen.append((ordinal, path)))
    if state == "attached":
        observer.worker(3, output)
        assert output.is_dir() and seen == ["verify", (3, output)]
    else:
        with pytest.raises(ValueError, match="Active outer monitor"):
            observer.worker(3, output)
        assert not output.exists() and not seen
