"""Audit/supplement contracts using fixtures; never load or score a model."""

import copy

import pytest

from scripts import prepare_regional_calibration_v29 as prepare
from scripts.image_regional_inference_v28 import read, sha, write


def test_active_training_refuses_preparation_before_checkpoint_hashing(tmp_path, monkeypatch):
    study, trial, maps = tmp_path / "study", tmp_path / "trial", tmp_path / "maps"
    study.mkdir()
    maps.mkdir()
    called = []
    monkeypatch.setattr(prepare, "audit_training", lambda *args: called.append(args))
    with pytest.raises(ValueError, match="Final training phase has not completed"):
        prepare.prepare(study, trial, maps)
    assert not trial.exists() and not called


@pytest.mark.parametrize("occupied", ["trial", "maps"])
def test_preparation_never_reuses_occupied_destinations(tmp_path, occupied):
    study, trial, maps = tmp_path / "study", tmp_path / "trial", tmp_path / "maps"
    study.mkdir()
    maps.mkdir()
    if occupied == "trial":
        trial.mkdir()
    else:
        (maps / "existing-map.npy").write_bytes(b"retained")
    with pytest.raises(ValueError):
        prepare.prepare(study, trial, maps)


def test_supplement_has_full_task_counts_and_hides_private_paths(tmp_path, monkeypatch):
    study, trial, maps = tmp_path / "study", tmp_path / "trial", tmp_path / "maps"
    study.mkdir()
    maps.mkdir()
    for part in ("phase-039", "phase-039-profile"):
        (study / part).mkdir()
    write(study / "phase-039/completed.json", {})
    write(study / "phase-039-profile/profile.json", {})
    rows = [{"source_id": i, "component": f"group{min(i, 31):03d}", "role": "calibration"} for i in range(39)]
    write(study / "plan.json", {"rows": rows})
    write(study / "freeze.json", {})
    chain = {
        "completed_updates": 1278,
        "phases": [{} for _ in range(40)],
        "final_checkpoint_sha256": "fixture-weight-hash",
    }
    monkeypatch.setattr(prepare, "audit_training", lambda path: copy.deepcopy(chain))
    monkeypatch.setattr(prepare, "implementations", lambda path: {})
    monkeypatch.setattr(prepare, "ASSETS", {})
    monkeypatch.setattr(prepare, "runtime_versions", lambda: {"fixture": "1"})
    result = prepare.prepare(study, trial, maps)
    assert result["unique_model_calls_planned"] == 624 and result["repeat_calls_planned"] == 4
    assert result["complete_training_updates"] == 1278 and result["complete_training_phases"] == 40
    assert result["reserved_source_forwards_at_freeze"] == 0
    assert str(tmp_path) not in (trial / "freeze.json").read_text()
    assert read(trial / "private-paths.json")["study"] == str(study)
    assert result["private_paths_sha256"] == sha(trial / "private-paths.json")
    assert len(read(trial / "calibration-tasks.json")) == 156


def admission_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "runtime_versions", lambda: {"fixture": "1"})
    study, trial, maps, profile = (
        tmp_path / name for name in ("study", "trial", "maps", "trial-preparation-profile")
    )
    for path in (study, trial, maps, profile):
        path.mkdir()
    (tmp_path / "core.py").write_text("# frozen fixture\n")
    (profile / "resource-samples.jsonl").write_text("{}\n")
    write(
        profile / "profile.json",
        {
            "module": "scripts.prepare_regional_calibration_v29",
            "arguments": ["prepare", str(study), str(trial), str(maps)],
            "child_exit_code": 0,
            "guard": None,
            "sample_errors": 0,
            "resource_limit_observations": [],
            "resource_journal_sha256": sha(profile / "resource-samples.jsonl"),
        },
    )
    write(
        trial / "private-paths.json",
        {
            "study": str(study),
            "raw_map_root": str(maps),
            "preparation_profile": str(profile / "profile.json"),
        },
    )
    write(
        trial / "freeze.json",
        {
            "private_paths_sha256": sha(trial / "private-paths.json"),
            "runtime_versions": {"fixture": "1"},
            "implementation_sha256": {"core.py": sha(tmp_path / "core.py")},
        },
    )
    write(
        trial / "execution-start.json",
        {
            "public_freeze_sha256": sha(trial / "freeze.json"),
            "remote_verified_before_first_reserved_forward": True,
            "remote_freeze_blob_sha": "b" * 40,
            "preparation_profile_sha256": sha(profile / "profile.json"),
        },
    )
    return trial, profile


def test_verified_complete_preparation_admits_only_the_fixed_runtime(tmp_path, monkeypatch):
    trial, _profile = admission_fixture(tmp_path, monkeypatch)
    assert prepare.check_score_admission(trial)["runtime_versions"] == {"fixture": "1"}
    from scripts import image_regional_inference_runtime_v28 as runtime

    calls = []
    monkeypatch.setattr(runtime, "run", lambda *args: calls.append(args))
    prepare.score(trial, "candidate", "native", 0)
    assert calls == [(trial, "candidate", "native", 0)]


@pytest.mark.parametrize(
    "damage", ["runtime", "implementation", "publication", "observations", "private_paths"]
)
def test_changed_preparation_cannot_start_inference(tmp_path, monkeypatch, damage):
    trial, profile = admission_fixture(tmp_path, monkeypatch)
    if damage == "runtime":
        monkeypatch.setattr(prepare, "runtime_versions", lambda: {"fixture": "2"})
    elif damage == "implementation":
        (tmp_path / "core.py").write_text("# changed\n")
    elif damage == "publication":
        value = read(trial / "execution-start.json")
        value["remote_verified_before_first_reserved_forward"] = False
        (trial / "execution-start.json").unlink()
        write(trial / "execution-start.json", value)
    elif damage == "observations":
        (profile / "resource-samples.jsonl").write_text("changed\n")
    else:
        (trial / "private-paths.json").write_text("{}\n")
    with pytest.raises(ValueError):
        prepare.check_score_admission(trial)


@pytest.mark.parametrize("damage", ["failed", "guarded", "wrong_audit"])
def test_terminal_audit_success_cannot_be_replaced_by_another_receipt(tmp_path, monkeypatch, damage):
    trial, profile = admission_fixture(tmp_path, monkeypatch)
    value = read(profile / "profile.json")
    if damage == "failed":
        value["child_exit_code"] = 1
    elif damage == "guarded":
        value["guard"] = "recorded_resource_limit"
    else:
        value["arguments"][1] = "another-study"
    (profile / "profile.json").unlink()
    write(profile / "profile.json", value)
    start = read(trial / "execution-start.json")
    start["preparation_profile_sha256"] = sha(profile / "profile.json")
    (trial / "execution-start.json").unlink()
    write(trial / "execution-start.json", start)
    with pytest.raises(ValueError):
        prepare.check_score_admission(trial)


def test_audit_monitor_refuses_insufficient_memory_without_launching_worker(tmp_path, monkeypatch):
    from scripts import regional_calibration_preparation_runtime_v29 as runtime

    maps = tmp_path / "maps"
    maps.mkdir()
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
    with pytest.raises(RuntimeError, match="Failed/guarded audit"):
        runtime.run(tmp_path / "study", tmp_path / "trial", maps)
    assert not called
    profile = read(tmp_path / "trial-preparation-profile/profile.json")
    assert profile["child_started"] is False and profile["guard"] == "initial_headroom_unavailable"
    assert not (tmp_path / "trial").exists()
