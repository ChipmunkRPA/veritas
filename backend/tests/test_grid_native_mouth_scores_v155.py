import hashlib
import json

import pytest

from scripts import grid_native_mouth_scores_v155 as continuation

original = continuation.original


def test_worker_uses_the_frozen_score_code_with_only_output_bindings_changed():
    worker = continuation.score_function()
    assert worker.__code__ is original.score.__code__
    assert worker.__globals__["CACHE"] == continuation.CACHE
    assert worker.__globals__["PUBLIC"] == continuation.PUBLIC
    assert {
        name for name, value in original.score.__globals__.items() if worker.__globals__[name] is not value
    } == {"CACHE", "PUBLIC"}


def test_reader_retains_frozen_validation_and_rebinds_only_receipt_context():
    def check():
        raise AssertionError("A function identity check must not execute the reader")

    reader = continuation.tail_reader(check)
    assert reader.__code__ is original.load_completed.__code__
    assert reader.__globals__["verify"] is check
    assert reader.__globals__["planning"].reviewed.stage is continuation.tail_stage
    assert reader.__globals__["planning"].reviewed.lines is original.planning.reviewed.lines
    assert reader.__globals__["input_binding"] is original.input_binding
    assert reader.__globals__["visual_invariance"] is original.visual_invariance


def test_receipt_caller_is_rebound_only_for_tail_stage(tmp_path, monkeypatch):
    public = tmp_path / "freeze.json"
    public.write_text("{}\n")
    monkeypatch.setattr(continuation, "PUBLIC", public)
    monkeypatch.setattr(continuation, "TRIAL", tmp_path / "v155")
    seen = []
    monkeypatch.setattr(original.planning.reviewed, "stage", lambda *args: seen.append(args))
    digest = continuation.sha(public)
    folder = continuation.stage_path(continuation.TRIAL, 1)
    continuation.tail_stage(folder, digest, continuation.ORIGINAL_CALLER, 0, {})
    assert seen == [(folder, digest, continuation.CALLER, 0, {})]
    with pytest.raises(ValueError, match="caller"):
        continuation.tail_stage(
            continuation.stage_path(continuation.TRIAL, 0), digest, continuation.ORIGINAL_CALLER, 0, {}
        )
    with pytest.raises(ValueError, match="caller"):
        continuation.tail_stage(folder, digest, "other.py", 0, {})


def test_launcher_preserves_data_only_stage_and_rejects_reuse_or_unsafe_route(monkeypatch):
    calls = []
    monkeypatch.setattr(continuation.subprocess, "Popen", lambda *a, **kw: calls.append((a, kw)))
    output = continuation.stage_path(continuation.TRIAL, 1) / "output"
    command = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", str(output), "0"]
    continuation.routed_popen(1, command, env={"FIXTURE": "1"})
    assert calls == [
        (
            (["python", "-B", "-m", "scripts.grid_native_mouth_scores_v155", "score", str(output), "1"],),
            {"env": {"FIXTURE": "1"}},
        )
    ]
    for ordinal in (0, 16, True):
        with pytest.raises(ValueError, match="tail ordinal"):
            continuation.routed_popen(ordinal, command)
    with pytest.raises(ValueError, match="exclusive"):
        continuation.routed_popen(1, [*command[:-2], str(output.parent.parent), "0"])


@pytest.mark.parametrize(
    "problem", ["missing_log", "missing_resources", "terminal_profile", "unattached_log"]
)
def test_worker_requires_live_monitor_artifacts_before_output(problem, tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "TRIAL", tmp_path)
    stage = continuation.stage_path(tmp_path, 1)
    stage.mkdir(parents=True)
    if problem != "missing_log":
        (stage / "worker.log").touch()
    if problem != "missing_resources":
        (stage / "resources.jsonl").write_text("{}\n")
    if problem == "terminal_profile":
        (stage / "profile.json").write_text("{}\n")

    def unexpected():
        raise AssertionError("Rejected entry must stop before verification")

    monkeypatch.setattr(continuation, "verify", unexpected)
    with pytest.raises(ValueError, match="outer monitor"):
        continuation.worker(1, stage / "output")
    assert not (stage / "output").exists()


def test_failure_history_binds_actual_shape_without_fabricating_a_score(tmp_path, monkeypatch):
    trial, cache = tmp_path / "old-trial", tmp_path / "old-cache"
    monkeypatch.setattr(original, "TRIAL", trial)
    monkeypatch.setattr(original, "CACHE", cache)
    stage = continuation.stage_path(trial, 1)
    stage.mkdir(parents=True)
    log, resources = stage / "worker.log", stage / "resources.jsonl"
    log.touch()
    resources.write_text("{}\n{}\n")
    continuation.write(
        stage / "profile.json",
        {
            "schema": "chipmunk.video-appearance-resource-profile/108",
            "command": "bind",
            "batch": 0,
            "runtime_sha256": continuation.sha(
                continuation.ROOT / "backend/scripts" / continuation.ORIGINAL_CALLER
            ),
            "child_started": True,
            "child_exit_code": -15,
            "guard": "monitor_unavailable",
            "sample_errors": 1,
            "error_types": ["PermissionError"],
            "resource_limit_observations": [],
            "worker_receipt_sha256": None,
            "resource_journal_sha256": continuation.sha(resources),
        },
    )
    failure = continuation.failure_history()
    assert failure["worker_receipt_present"] is False
    assert failure["profile_sha256"] == continuation.sha(stage / "profile.json")
    (stage / "output").mkdir()
    with pytest.raises(ValueError, match="absence of worker output"):
        continuation.failure_history()


def test_aggregate_reuses_clip0_and_commits_the_full_ordered_collection(tmp_path, monkeypatch):
    new_public, old_public = tmp_path / "new.json", tmp_path / "old.json"
    new_public.write_text("{}\n")
    old_public.write_text("{}\n")
    monkeypatch.setattr(continuation, "PUBLIC", new_public)
    monkeypatch.setattr(original, "PUBLIC", old_public)
    monkeypatch.setattr(continuation, "verify", lambda: None)
    monkeypatch.setattr(original.planning, "load_completed", lambda: "authored plan")
    seen = []

    def rows(ordinal):
        seen.append(ordinal)
        return [ordinal] * 11

    def completed(ordinal):
        assert ordinal == 0
        return rows(ordinal)

    def tail(ordinal):
        assert ordinal in range(1, 16)
        return rows(ordinal)

    def aggregate(plan, scores):
        assert plan == "authored plan"
        assert scores == [i for i in range(16) for _ in range(11)]
        return {
            "planned_cases": len(scores),
            "source_slot_paired_aggregates": ["private"],
            "component_paired_aggregates": ["private"],
        }

    monkeypatch.setattr(original, "load_completed", completed)
    monkeypatch.setattr(continuation, "tail_reader", lambda _check: tail)
    monkeypatch.setattr(original, "aggregate_records", aggregate)
    monkeypatch.setattr(continuation, "commitments", lambda _trial, _cache, i: {"fixture": i})
    public = continuation.aggregate_completed()
    assert seen == list(range(16))
    assert public["planned_cases"] == 176 and public["retained_v151_clips"] == 1
    assert "source_slot_paired_aggregates" not in public and "component_paired_aggregates" not in public
    assert "private_stage_commitments" not in public
    private = continuation.aggregate_completed(include_private=True)
    assert private["source_slot_paired_aggregates"] == ["private"]
    collection = json.dumps(
        private["private_stage_commitments"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert public["receipt_collection_sha256"] == hashlib.sha256(collection).hexdigest()
