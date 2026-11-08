"""Admission, cohort completeness and exact map persistence; no neural scoring."""

import json

import numpy as np
import pytest
from PIL import Image

from scripts import image_regional_inference_v28 as producer


def plan():
    return {
        "rows": [
            {"source_id": i, "component": f"group{min(i, 31):03d}", "role": "calibration"} for i in range(39)
        ]
        + [{"source_id": 100, "component": "reserved-confirmation", "role": "confirmation"}]
    }


def test_all_calibration_views_are_preserved_in_complete_photo_chunks():
    tasks = producer.calibration_tasks(plan())
    assert len(tasks) == 156
    chunks = [producer.phase_tasks(tasks, n) for n in range(7)]
    assert [len(c) for c in chunks] == [24] * 6 + [12]
    assert [r for chunk in chunks for r in chunk] == tasks
    assert all({r["kind"] for r in tasks if r["source_id"] == i} == set(producer.KINDS) for i in range(39))
    assert all(r["source_id"] != 100 and r["role"] == "calibration" for r in tasks)
    with pytest.raises(ValueError):
        producer.phase_tasks(tasks, 7)


@pytest.mark.parametrize("number", [-1, True, 0.5])
def test_phase_identity_is_not_coerced(number):
    with pytest.raises(ValueError):
        producer.phase_tasks(producer.calibration_tasks(plan()), number)


def test_missing_source_photo_cannot_shrink_the_cohort():
    value = plan()
    value["rows"].pop(0)
    with pytest.raises(ValueError, match="39 photos required"):
        producer.calibration_tasks(value)


def test_encoding_preserves_coordinates_and_matches_frozen_training_format():
    from scripts.image_regional_training_v26 import compressed

    rng = np.random.default_rng(123)
    pixels = rng.integers(0, 256, (31, 47, 3), dtype=np.uint8)
    source = Image.fromarray(pixels)
    for fmt, magic in (("native", b"\x89PNG"), ("jpeg75", b"\xff\xd8")):
        data, decoded = producer.encoded_view(source, fmt)
        assert data.startswith(magic) and decoded.size == (47, 31)
        assert np.array_equal(np.asarray(decoded), np.asarray(compressed(source, fmt)))
    assert np.array_equal(np.asarray(source), pixels)


def test_unfinished_training_never_authorizes_scoring(tmp_path, monkeypatch):
    study = tmp_path / "training"
    study.mkdir()
    producer.write(study / "plan.json", {"schedule": [None] * 1278})
    producer.write(
        study / "freeze.json", {"private_plan_sha256": producer.sha(study / "plan.json"), "code_sha256": {}}
    )
    with pytest.raises(ValueError, match="All training phases must finish"):
        producer.audit_training(study)
    called = []
    monkeypatch.setattr(producer, "load_predictor", lambda *args: called.append(args))
    with pytest.raises(FileNotFoundError):
        producer.score(tmp_path, "candidate", "native", 0)
    assert not called and not (tmp_path / "candidate-native-000").exists()


def test_remote_freeze_does_not_admit_an_incomplete_checkpoint(tmp_path):
    producer.write(
        tmp_path / "freeze.json", {"complete_training_updates": 64, "reserved_source_forwards_at_freeze": 0}
    )
    producer.write(
        tmp_path / "execution-start.json",
        {
            "public_freeze_sha256": producer.sha(tmp_path / "freeze.json"),
            "remote_verified_before_first_reserved_forward": True,
            "remote_freeze_blob_sha": "a" * 40,
        },
    )
    with pytest.raises(ValueError, match="Completed training"):
        producer.lease(tmp_path)


def test_model_setup_failure_retains_the_full_planned_denominator(tmp_path, monkeypatch):
    (tmp_path / "maps").mkdir()
    producer.write(tmp_path / "private-paths.json", {"raw_map_root": str(tmp_path / "maps")})
    monkeypatch.setattr(producer, "lease", lambda trial: ({}, {}, plan()))

    def unavailable(*args):
        raise RuntimeError("Local checkpoint unavailable")

    monkeypatch.setattr(producer, "load_predictor", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        producer.score(tmp_path, "candidate", "native", 0)
    rows = [
        json.loads(line) for line in (tmp_path / "candidate-native-000/rows.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 24 and all(r["status"] == "unavailable" for r in rows)
    assert not (tmp_path / "candidate-native-000/completed.json").exists()


def test_complete_writer_repeats_first_input_and_preserves_exact_maps(tmp_path, monkeypatch):
    from scripts import image_regional_training_v26 as training

    (tmp_path / "maps").mkdir()
    (tmp_path / "pairs/qualification").mkdir(parents=True)
    producer.write(tmp_path / "private-paths.json", {"raw_map_root": str(tmp_path / "maps")})
    monkeypatch.setattr(producer, "lease", lambda trial: ({}, {}, plan()))
    monkeypatch.setattr(producer, "sha", lambda path: "fixture-hash")
    monkeypatch.setattr(training, "PAIRS", tmp_path / "pairs")
    with (tmp_path / "pairs/qualification/pairs.jsonl").open("w") as stream:
        for i in range(39):
            stream.write(json.dumps({"source_id": i, "truth_png_sha256": "fixture-truth"}) + "\n")
    image = Image.fromarray(np.full((7, 11, 3), 73, np.uint8))
    monkeypatch.setattr(
        training,
        "source_views",
        lambda *args: (
            image,
            {kind: image for kind in ("original", "tone", "blur")},
            np.zeros((7, 11), np.uint8),
        ),
    )
    calls = []

    def predict(value):
        calls.append(value.size)
        return np.full((7, 11), 0.25, np.float32), {"image_gate": None, "image_score": None}

    monkeypatch.setattr(producer, "load_predictor", lambda *args: predict)
    producer.score(tmp_path, "candidate", "native", 0)
    assert len(calls) == 25
    rows = [
        json.loads(line) for line in (tmp_path / "candidate-native-000/rows.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 24
    for row in rows:
        assert row["source_size"] == [11, 7] and row["raw_map_shape"] == [7, 11]
        assert row["negative_truth_is_zero"] == (row["kind"] != "ai")
        assert np.array_equal(
            np.load(row["raw_map_file"], allow_pickle=False), np.full((7, 11), 0.25, np.float32)
        )
    receipt = producer.read(tmp_path / "candidate-native-000/completed.json")
    assert receipt["completed_model_forwards"] == receipt["attempted_model_forwards"] == 25
    assert receipt["exact_repeat_extra_forwards"] == 1
    assert receipt["serving_admitted"] is False


def test_resource_preflight_never_starts_a_worker_without_headroom(tmp_path, monkeypatch):
    from scripts import image_regional_inference_runtime_v28 as runtime

    raw = tmp_path / "raw-maps"
    raw.mkdir()
    producer.write(tmp_path / "private-paths.json", {"raw_map_root": str(raw)})
    calls = []
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
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
    with pytest.raises(RuntimeError, match="Failed/guarded inference"):
        runtime.run(tmp_path, "candidate", "native", 0)
    assert not calls
    profile = producer.read(tmp_path / "candidate-native-000-profile/profile.json")
    assert profile["child_started"] is False
    assert profile["guard"] == "initial_headroom_unavailable"
    assert "ram_reserve" in profile["resource_limit_observations"][0]["violations"]
    with pytest.raises(FileExistsError):
        runtime.run(tmp_path, "candidate", "native", 0)


@pytest.mark.parametrize("arm,fmt", [("other", "native"), ("candidate", "other"), ("../", "jpeg75")])
def test_monitor_rejects_unknown_arms_before_creating_output(tmp_path, arm, fmt):
    from scripts import image_regional_inference_runtime_v28 as runtime

    with pytest.raises(ValueError, match="Explicit inference"):
        runtime.run(tmp_path, arm, fmt, 0)
    assert not list(tmp_path.iterdir())
