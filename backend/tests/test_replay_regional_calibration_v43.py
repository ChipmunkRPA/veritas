"""Source/map identity, actual screening-map replay and unknown denominators."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import replay_regional_calibration_v43 as replay
from scripts.image_regional_inference_v28 import encoded_view, read, sha, write


@pytest.mark.parametrize("name", ["load_map", "pair_metrics", "phase", "outcome_counts", "seal"])
def test_frozen_replay_and_selection_math_has_only_version_label_changes(name):
    root = Path(__file__).resolve().parents[2] / "backend/scripts"
    def function(path):
        text = path.read_text()
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(text, node)
    expected = function(root / "replay_regional_calibration_v30.py").replace("v30", "v43").replace('/30"', '/43"')
    actual = function(root / "replay_regional_calibration_v43.py")
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))


def test_producer_without_fixed_monitor_identity_cannot_enter_readout(tmp_path):
    folder = tmp_path / "candidate-native-000"
    profile = tmp_path / "candidate-native-000-profile"
    folder.mkdir()
    profile.mkdir()
    write(folder / "completed.json", {})
    write(profile / "profile.json", {"module": "scripts.image_regional_inference_v28"})
    with pytest.raises(ValueError, match="Fixed-monitor provenance"):
        replay.producer_rows(tmp_path, "candidate", "native", 0, [])


def pair_fixture(tmp_path, *, fmt="native", kind="ai"):
    root = tmp_path / "maps"
    root.mkdir()
    rng = np.random.default_rng(123)
    image = Image.fromarray(rng.integers(50, 206, (192, 256, 3), dtype=np.uint8))
    data, decoded = encoded_view(image, fmt)
    pixels = np.asarray(decoded)
    region = np.zeros((192, 256), bool)
    region[48:144, 60:180] = True
    truth = region if kind == "ai" else np.zeros_like(region)
    common = {
        "source_id": 21,
        "component": "fixture-component",
        "role": "calibration",
        "kind": kind,
        "format": fmt,
        "status": "ok",
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "source_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "source_size": [256, 192],
        "positive_truth_png_sha256": "fixture-source-truth",
        "negative_truth_is_zero": kind != "ai",
    }
    result = []
    for arm, raw in (
        ("candidate", region.astype(np.float32) * np.float32(0.95)),
        ("baseline", np.zeros((384, 384) if fmt == "native" else (512, 512), np.float32)),
    ):
        path = root / (arm + ".npy")
        np.save(path, raw, allow_pickle=False)
        result.append(
            {
                **common,
                "arm": arm,
                "raw_map_file": str(path),
                "raw_map_sha256": sha(path),
                "raw_map_shape": list(raw.shape),
                "image_gate": None if arm == "candidate" else False,
                "image_score": None if fmt == "native" else 0.1,
                "pixel_threshold": 0.7 if fmt == "native" else 0.5,
            }
        )
    return (*result, root, data, pixels, truth)


@pytest.mark.parametrize("fmt", ["native", "jpeg75"])
def test_actual_map_pipeline_preserves_truth_positions_without_loading_models(tmp_path, monkeypatch, fmt):
    from app.detectors.image import fused, splice
    from scripts.score_splice_localizers import score_map

    def prohibited(*args, **kwargs):
        raise AssertionError("Map replay must not load a neural model")

    monkeypatch.setattr(fused, "load_model", prohibited)
    monkeypatch.setattr(splice, "load_model", prohibited)
    a, b, root, data, pixels, truth = pair_fixture(tmp_path, fmt=fmt)
    result = replay.pair_metrics(a, b, root, data, pixels, truth, "fixture-source-truth")
    assert result["baseline"]["metrics"]["predicted_positive_pixels"] == 0
    assert len(result["candidate"]) == 28
    expected = score_map(np.load(a["raw_map_file"], allow_pickle=False), True, 0.9, data, pixels, truth)[
        "metrics"
    ]
    chosen = result["candidate"]["p0.90-a0.01000"]["metrics"]
    assert chosen == expected
    assert chosen["reference_positive_pixels"] == 96 * 120
    assert chosen["iou"] > 0.9 and chosen["recall"] > 0.9


def test_ordinary_control_has_zero_ai_truth_even_when_a_response_flags_it(tmp_path):
    a, b, root, data, pixels, truth = pair_fixture(tmp_path, kind="tone", fmt="jpeg75")
    result = replay.pair_metrics(a, b, root, data, pixels, truth, "fixture-source-truth")
    metrics = result["candidate"]["p0.90-a0.01000"]["metrics"]
    assert metrics["reference_positive_pixels"] == 0
    assert metrics["predicted_positive_pixels"] > 0
    assert metrics["false_positive_pixels"] == metrics["predicted_positive_pixels"]
    assert metrics["recall"] is None


@pytest.mark.parametrize("damage", ["dtype", "shape", "nan", "path", "bytes"])
def test_malformed_or_unbound_maps_are_not_clear_observations(tmp_path, damage):
    a, b, root, data, pixels, truth = pair_fixture(tmp_path)
    path = root / "candidate.npy"
    value = np.load(path, allow_pickle=False)
    if damage == "dtype":
        np.save(path, value.astype(np.float64), allow_pickle=False)
    elif damage == "shape":
        np.save(path, np.zeros((3, 4), np.float32), allow_pickle=False)
        a["raw_map_shape"] = [3, 4]
    elif damage == "nan":
        value[0, 0] = np.nan
        np.save(path, value, allow_pickle=False)
    elif damage == "path":
        path = tmp_path / "unrelated.npy"
        np.save(path, value, allow_pickle=False)
        a["raw_map_file"] = str(path)
    else:
        path.write_bytes(path.read_bytes() + b"uncommitted")
    if damage != "bytes":
        a["raw_map_sha256"] = sha(path)
    with pytest.raises(ValueError):
        replay.pair_metrics(a, b, root, data, pixels, truth, "fixture-source-truth")


@pytest.mark.parametrize("damage", ["failed", "input", "truth", "baseline_gate"])
def test_failed_or_differently_bound_pairs_never_get_metrics(tmp_path, damage):
    a, b, root, data, pixels, truth = pair_fixture(tmp_path)
    if damage == "failed":
        a["status"] = "skipped"
    elif damage == "input":
        b["source_rgb_sha256"] = "different-photo"
    elif damage == "truth":
        a["positive_truth_png_sha256"] = "different-mask"
    else:
        b["image_gate"] = True
    with pytest.raises(ValueError):
        replay.pair_metrics(a, b, root, data, pixels, truth, "fixture-source-truth")


def test_metadata_coverage_retains_all_missing_and_failed_cases(tmp_path, monkeypatch):
    study, trial = tmp_path / "study", tmp_path / "trial"
    study.mkdir()
    trial.mkdir()
    rows = [{"source_id": i, "component": f"group{min(i, 31):03d}", "role": "calibration"} for i in range(39)]
    write(study / "plan.json", {"rows": rows})
    write(trial / "private-paths.json", {"study": str(study)})
    monkeypatch.setattr(replay, "check_score_admission", lambda *args: None)
    tasks = replay.calibration_tasks({"rows": rows})
    part = trial / "candidate-native-000"
    part.mkdir()
    observed = [
        {**tasks[i], "arm": "candidate", "format": "native", "status": status}
        for i, status in enumerate(("failed", "unavailable", "skipped"))
    ]
    (part / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in observed))
    replay.producer_coverage(trial, trial / "coverage.json")
    report = read(trial / "coverage.json")
    candidate = report["arm_formats"]["candidate-native"]
    assert candidate["planned_unique_views"] == 156
    assert candidate["recorded_status_counts"] == {
        "failed": 1,
        "unavailable": 1,
        "skipped": 1,
        "missing": 153,
    }
    assert candidate["not_recorded_ok"] == 156
    assert all(r["planned_unique_views"] == 156 for r in report["arm_formats"].values())
    assert report["whole_file_originality_established"] is report["grants_calibration_admission"] is False


def test_guarded_readout_never_selects_a_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "admission", lambda *args: None)
    part, profile = tmp_path / "readout-v43-native-000", tmp_path / "readout-v43-native-000-profile"
    part.mkdir()
    profile.mkdir()
    write(part / "completed.json", {})
    write(
        profile / "profile.json",
        {
            "module": "scripts.replay_regional_calibration_v43",
            "arguments": ["phase", str(tmp_path), "native", "0"],
            "child_exit_code": -15,
            "guard": "recorded_resource_limit",
            "sample_errors": 0,
            "resource_limit_observations": [],
        },
    )
    with pytest.raises(ValueError, match="Incomplete or failed"):
        replay.seal(tmp_path)
    assert not (tmp_path / "calibration-v43-summary.json").exists()


def test_readout_monitor_refuses_to_launch_without_headroom(tmp_path, monkeypatch):
    from scripts import regional_calibration_runtime_v42 as runtime

    maps = tmp_path / "maps"
    maps.mkdir()
    write(tmp_path / "private-paths.json", {"raw_map_root": str(maps)})
    called = []
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: called.append(args))
    monkeypatch.setattr(
        runtime.limits,
        "observe",
        lambda *args, **kwargs: {
            "elapsed_s": 0.0,
            "tree_rss_bytes": 0,
            "ram_available_bytes": 0,
            "disk_free_bytes": 8 * 1024**3,
        },
    )
    with pytest.raises(RuntimeError, match="Failed/guarded calibration"):
        runtime.run("phase", [str(tmp_path), "native", "0"])
    assert not called
    profile = read(tmp_path / "readout-v43-native-000-profile/profile.json")
    assert profile["guard"] == "initial_headroom_unavailable" and profile["child_started"] is False


def test_outcome_report_separates_missed_regions_from_control_false_alarms(tmp_path):
    a, b, root, data, pixels, truth = pair_fixture(tmp_path)
    positive = replay.pair_metrics(a, b, root, data, pixels, truth, "fixture-source-truth")
    missed = copy.deepcopy(positive)
    missed["source_id"] = 22
    missed["component"] = "another-fixture"
    policy = "p0.90-a0.01000"
    missed["candidate"][policy]["metrics"] = copy.deepcopy(missed["baseline"]["metrics"])
    control_a, control_b = copy.deepcopy(a), copy.deepcopy(b)
    for row in (control_a, control_b):
        row["kind"] = "tone"
        row["negative_truth_is_zero"] = True
    control = replay.pair_metrics(
        control_a, control_b, root, data, pixels, np.zeros_like(truth), "fixture-source-truth"
    )
    report = replay.outcome_counts([positive, missed, control], policy)
    assert report["outcomes"]["edited_photos"] == 2
    assert report["outcomes"]["missed_region_fraction"] == 0.5
    assert report["outcomes"]["not_usefully_localized_fraction"] == 0.5
    assert report["outcomes"]["controls"]["tone"]["flagged_fraction"] == 1
    assert report["whole_file_originality_established"] is False


def test_unselected_policy_has_no_favorable_zero_outcomes():
    report = replay.outcome_counts([], None)
    assert report["outcomes"] is report["selected_policy"] is None
    assert report["whole_file_originality_established"] is False
