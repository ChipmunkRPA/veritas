"""Full synthetic denominators and source/support refusal; no accuracy claims."""
from __future__ import annotations

import copy
import json
import math
import subprocess

import numpy as np
import pytest

from scripts import run_video_demamba_v13 as runner
from scripts import video_demamba_clock_v13 as clock


def support():
    plan = clock.sample_plan(24., 240)
    _, trace = clock.preprocess(np.zeros((240, 320, 3), np.uint8))
    return {"schema": "chipmunk.video-demamba-native-input/13", "plan": plan,
            "source_frame_count": 240, "decoded_clock_entries": 240, "synthetic_padding_frames": 0,
            "tensor_shape": [1, 8, 3, 224, 224], "tensor_sha256": "0" * 64, "first_decoded_pts_s": 0.,
            "frames": [{"native_index": i, "pts_s": i / 24, "relative_to_first_decoded_pts_s": i / 24, **trace}
                       for i in plan["selected_indices"]]}


def matrix(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text("{}\n")
    inputs, baseline, candidate = [], [], []
    for dataset in runner.DATASETS:
        for i in range(72):
            for variant in ("original", "matched"):
                unavailable = dataset == "deepaction" and i >= 70
                source = {"dataset": dataset, "id": f"source-{i}", "group": f"{dataset}:{i // 2}",
                          "label": i % 2, "split": "calibration" if i < 24 else "test", "family": "test-family",
                          "variant": variant, "sha256": None if unavailable else "0" * 64,
                          "relative_path": None if unavailable else f"{dataset}/{i}-{variant}.mp4",
                          "status": "unavailable" if unavailable else "ready"}
                inputs.append(source)
                score = .75 if source["label"] else .25
                baseline.append({**source, "status": "unavailable" if unavailable else "ok",
                                 "score": None if unavailable else score,
                                 "checks": {} if unavailable else {name: {"status": "ok", "prob": score} for name in runner.CHECKS}})
                raw = 2. if source["label"] else -2.
                candidate.append({**source, "status": "unavailable" if unavailable else "ok",
                                  "score": None if unavailable else 1 / (1 + math.exp(-raw)),
                                  "logit": None if unavailable else raw, "support": None if unavailable else support()})
    paths = {}
    for name, rows in [("baseline", baseline), ("candidate", candidate)]:
        path = tmp_path / (name + ".jsonl")
        path.write_text(json.dumps({"system": name, "lock_sha256": runner.digest(lock)}) + "\n" + "".join(json.dumps(r) + "\n" for r in rows))
        paths[name] = path
    return lock, {"rows": inputs}, paths, baseline, candidate


@pytest.mark.parametrize("p", [None, float("nan"), float("inf"), True, -1., 2., .5])
def test_missing_or_invalid_response_never_becomes_original(p):
    assert runner.direction(p) == "inconclusive"


def test_complete_sources_and_expected_unavailable_inputs_are_retained(tmp_path):
    lock, body, paths, _, _ = matrix(tmp_path)
    rows = runner.read_scores(paths["candidate"], lock, body, "candidate")
    assert len(rows) == 288 and sum(r["status"] == "unavailable" for r in rows) == 4
    assert len(runner.native_plan(body["rows"])) == 16


def test_baseline_collector_preserves_inapplicability_and_failed_status():
    from app.bench.score_v2 import DETECTOR_STAGES_BY_KIND

    stage = DETECTOR_STAGES_BY_KIND["video"][0]
    keys = {name: name + "@test" for name in runner.CHECKS}
    entries = [{"detector_id": name, "status": "not_applicable", "reason": "Native frame too small"} for name in runner.CHECKS]
    rows = runner.baseline_rows({stage: {"detectors": entries}}, keys)
    assert all(r["status"] == "not_applicable" and r["prob"] is None for r in rows.values())
    entries[0].update(status="failed", reason="Execution failed")
    rows = runner.baseline_rows({stage: {"detectors": entries}}, keys)
    assert rows[keys[runner.CHECKS[0]]]["status"] == "failed"
    assert rows[keys[runner.CHECKS[0]]]["prob"] is None


@pytest.mark.parametrize("fault", ["omitted", "wrong_truth", "boolean_truth", "score_nan", "unsupported_vote", "missing_frame", "padding", "changed_clock", "failed_check"])
def test_actual_complete_journal_alterations_are_refused(tmp_path, fault):
    lock, body, paths, old, new = matrix(tmp_path)
    system, rows = "candidate", new
    if fault == "omitted":
        rows.pop()
    elif fault == "wrong_truth":
        rows[0]["label"] = 1
    elif fault == "boolean_truth":
        rows[0]["label"] = False
    elif fault == "score_nan":
        rows[0]["score"] = float("nan")
    elif fault == "unsupported_vote":
        rows[0].update(status="not_applicable", score=.01, logit=None)
    elif fault == "missing_frame":
        rows[0]["support"]["frames"].pop()
    elif fault == "padding":
        rows[0]["support"]["synthetic_padding_frames"] = 1
    elif fault == "changed_clock":
        rows[0]["support"]["frames"][0]["pts_s"] += .1
    else:
        system, rows = "baseline", old
        rows[0].update(status="not_applicable", score=None)
        rows[0]["checks"] = {name: {"status": "failed", "prob": None} for name in runner.CHECKS}
    paths[system].write_text(json.dumps({"system": system, "lock_sha256": runner.digest(lock)}) + "\n" + "".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        runner.read_scores(paths[system], lock, body, system)


def test_synthetic_full_comparison_preserves_all192_development_rows(tmp_path, monkeypatch):
    lock, body, paths, _, _ = matrix(tmp_path)
    monkeypatch.setattr(runner, "check", lambda _: copy.deepcopy(body))
    output = tmp_path / "summary.json"
    runner.compare(lock, paths["baseline"], paths["candidate"], output)
    result = json.loads(output.read_text())
    for system in ("baseline", "candidate"):
        assert result["totals"][system]["planned"] == 192
        assert result["totals"][system]["inconclusive"] == 4
        assert len(result["calibrated_development"][system]) == 192
    assert result["native_audit_pending"] and not result["advance_to_fresh_confirmation"]
    assert not result["deployment_eligible"]


def test_native_plan_is_response_independent_and_spans_both_roles(tmp_path):
    _, body, _, _, _ = matrix(tmp_path)
    first = runner.native_plan(body["rows"])
    altered = [{**r, "score": 1. if r["label"] == 0 else .01} for r in reversed(body["rows"])]
    assert runner.native_plan(altered) == first
    chosen = {tuple(k) for k in first}
    assert {r["split"] for r in body["rows"] if runner.key(r) in chosen} == {"calibration", "test"}


def test_separate_lossless_native_capture_binds_pixels_and_timestamps(tmp_path):
    # Generated color patterns, not a human/AI evaluation recording.
    pixels = np.zeros((240, 64, 80, 3), np.uint8)
    pixels[:, :, :, 0] = np.arange(240, dtype=np.uint8)[:, None, None]
    pixels[:, :, :, 1] = np.arange(80, dtype=np.uint8)[None, None, :]
    path = tmp_path / "synthetic-clock.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "80x64", "-r", "24", "-i", "pipe:0",
                    "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path)], input=pixels.tobytes(), capture_output=True, check=True, timeout=30)
    data, support = clock.read_native(path)
    assert np.array_equal(data, runner.replay_pixels(path, support))
    changed = copy.deepcopy(support)
    changed["frames"][0]["native_rgb_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="RGB correspondence"):
        runner.replay_pixels(path, changed)


@pytest.mark.parametrize("fault", ["threshold", "native", "code", "runtime", "installed", "source", "preparation", "already_scored", "admitted"])
def test_prospective_freeze_alterations_are_refused_before_inference(tmp_path, monkeypatch, fault):
    _, fixture, _, _, _ = matrix(tmp_path)
    body = {"schema": runner.SCHEMA, "method": copy.deepcopy(runner.METHOD), "implementation_sha256": {}, "runtime": {}, "installed_sha256": {},
            "base": str(tmp_path), "rows": fixture["rows"], "native_cases": runner.native_plan(fixture["rows"]),
            "preparation_sha256": "original", "recordings_scored": 0, "calibration_fitted": False, "deployment_eligible": False}
    monkeypatch.setattr(runner, "implementation", lambda: {})
    monkeypatch.setattr(runner, "runtime", lambda: {})
    monkeypatch.setattr(runner, "installed", lambda: {})
    monkeypatch.setattr(runner, "source_rows", lambda _: (None, fixture["rows"]))
    monkeypatch.setattr(runner, "preparation", lambda: "original")
    path = tmp_path / "prospective-lock.json"
    path.write_text(json.dumps(body))
    assert runner.check(path, assets=False)["recordings_scored"] == 0
    if fault == "threshold":
        body["method"]["research_reference"] = .6
    elif fault == "native":
        body["native_cases"].pop()
    elif fault == "code":
        body["implementation_sha256"]["extra.py"] = "x"
    elif fault in {"runtime", "installed"}:
        body["runtime" if fault == "runtime" else "installed_sha256"]["changed"] = True
    elif fault == "source":
        body["rows"] = copy.deepcopy(body["rows"])
        body["rows"][0]["label"] = 1
    elif fault == "preparation":
        body["preparation_sha256"] = "changed"
    elif fault == "already_scored":
        body["recordings_scored"] = 1
    else:
        body["deployment_eligible"] = True
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        runner.check(path, assets=False)
