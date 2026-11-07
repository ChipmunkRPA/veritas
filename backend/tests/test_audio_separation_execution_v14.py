"""Whole-source/estimate clocks, immutable roles and missing-probe decisions."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts import audio_separation_features_v14 as features
from scripts import run_audio_separation_v14 as run


def example():
    reference = {"head_weight": [0.] * 1024, "head_bias": 0.}
    source = {"status": "ok", "raw": .8, "pcm_sha256": "a" * 64}
    stem = {"index": 0, "samples": 64000, "pcm_sha256": "b" * 64, "raw": .5,
            "embedding": [0.] * 1024, "embedding_sha256": features.pcm_sha(np.zeros(1024, np.float32))}
    record = {"status": "ok", "raw": .8, "input_pcm_sha256": "a" * 64,
              "stems": [stem, {**copy.deepcopy(stem), "index": 1, "pcm_sha256": "c" * 64}]}
    return record, source, reference


@pytest.mark.parametrize("change", ["clock", "clock_type", "index", "missing", "wave_hash", "vector_hash",
                                    "bad_raw", "nonfinite", "partial_vector", "source_pcm"])
def test_estimates_are_complete_source_bound_and_match_the_frozen_head(change):
    record, source, reference = example()
    features.validate(record, source, reference)
    if change == "clock":
        record["stems"][1]["samples"] = 63999
    elif change == "clock_type":
        record["stems"][1]["samples"] = 64000.
    elif change == "index":
        record["stems"][1]["index"] = 0
    elif change == "missing":
        record["stems"].pop()
    elif change == "wave_hash":
        record["stems"][0]["pcm_sha256"] = "not a digest"
    elif change == "vector_hash":
        record["stems"][0]["embedding_sha256"] = "0" * 64
    elif change == "bad_raw":
        record["stems"][0]["raw"] = .7
    elif change == "nonfinite":
        record["stems"][0]["embedding"][0] = float("nan")
    elif change == "partial_vector":
        record["stems"][0]["embedding"].pop()
    else:
        record["input_pcm_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        features.validate(record, source, reference)


def test_unassessed_cannot_acquire_estimates_or_change_scope():
    _, _, reference = example()
    source = {"status": "not_applicable", "raw": None, "pcm_sha256": "a" * 64}
    record = {"status": "not_applicable", "raw": None, "input_pcm_sha256": "a" * 64, "stems": None}
    features.validate(record, source, reference)
    for altered in ({**record, "raw": .01}, {**record, "stems": []}, {**record, "status": "ok"}):
        with pytest.raises(ValueError):
            features.validate(altered, source, reference)


def fake_source():
    return {**dict.fromkeys(run.prior.META), "id": "synthetic-unassessed", "role": "development",
            "status": "not_applicable", "raw": None}


def test_whole_unassessed_flow_cannot_load_models_or_issue_original(tmp_path, monkeypatch):
    source = fake_source()
    baseline = {source["id"]: {"original_p": None, "original_decision": "inconclusive"}}
    monkeypatch.setattr(run, "feature_rows", lambda *_: ({}, ({}, {}, [], baseline, {}), [source]))
    head = {"fitted": {arm: {} for arm in run.method.ARMS}, "guards": {arm: {"0": [], "1": []} for arm in run.method.ARMS}}
    monkeypatch.setattr(run, "read_head", lambda *_: head)
    monkeypatch.setattr(run.method, "predict", lambda *_: None)
    monkeypatch.setattr(run.prior, "load_models", lambda *_: pytest.fail("Unassessed case must not run a model"))
    monkeypatch.setattr(run, "response_header", lambda *_: {"synthetic_test": True})
    output = tmp_path / "responses.jsonl"
    run.compare(tmp_path / "lock", tmp_path / "receipt", tmp_path / "head", output)
    result = json.loads(output.read_text().splitlines()[1])
    assert result["status"] == "not_applicable"
    assert all(c["p"] is None and c["decision"] == "inconclusive" for c in result["candidates"].values())
    assert result["probe"]["status"] == "not_required"


def test_missing_required_probe_is_refused_before_a_directional_result(monkeypatch):
    source = fake_source()
    source.update(status="ok", raw=.1)
    old = {"original_p": .1, "original_decision": "original"}
    record = {**{k: source[k] for k in run.prior.META}, "status": "ok", **old,
              "probe": {"status": "not_required", "raw": None, "stems": None}}
    head = {"fitted": {a: {} for a in run.method.ARMS}, "guards": {a: {} for a in run.method.ARMS}}
    monkeypatch.setattr(run.method, "predict", lambda *_: .1)
    monkeypatch.setattr(run, "initial", lambda *_: "original")
    with pytest.raises(ValueError, match="stability probe"):
        run.validate_response(record, source, old, head, {}, None, {})


def test_lock_refuses_changed_sources_roles_runtime_and_deployment(monkeypatch):
    monkeypatch.setattr(run, "implementation", lambda: {"method": "fixed"})
    monkeypatch.setattr(run, "runtime", lambda: {"threads": 4})
    monkeypatch.setattr(run, "installed_implementation", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(run, "structural_audit", lambda: "synthetic-audit")
    monkeypatch.setattr(run.serving, "load", lambda: {"service": "fixed"})
    lock = {"schema": run.SCHEMA, "method": run.method.METHOD, "implementation_sha256": run.implementation(),
        "runtime": run.runtime(), "installed_implementation_sha256": run.installed_implementation(),
        "source_sha256": run.prior.prior.original.SOURCE_SHA, "variants_sha256": run.prior.prior.original.VARIANTS_SHA,
        "role_counts": run.COUNTS, "planned": 4248, "expected_unassessed": 62,
        "structural_audit_sha256": run.structural_audit(), "upstream_assets": run.upstream.WHEELS,
        "serving_calibration_sha256": run.serving.SHA256, "serving_bundle": run.serving.load(),
        "original_receipt_sha256": run.prior.prior.reuse.RECEIPT_SHA,
        "original_responses_sha256": run.prior.prior.reuse.RESPONSES_SHA,
        "reserved_confirmation_sha256": run.prior.prior.original.CONFIRMATION_SHA,
        "project_forwards_started": False, "fitting_started": False, "deployment_eligible": False}
    run.validate_lock(lock)
    for key, value in (("planned", 4247), ("expected_unassessed", 0), ("role_counts", {"fit": 4248}),
        ("implementation_sha256", {}), ("installed_implementation_sha256", {}), ("runtime", {}),
        ("source_sha256", "wrong"), ("structural_audit_sha256", "changed"), ("upstream_assets", {}),
        ("reserved_confirmation_sha256", "changed"), ("project_forwards_started", True),
        ("fitting_started", True), ("deployment_eligible", True)):
        altered = copy.deepcopy(lock)
        altered[key] = value
        with pytest.raises(ValueError):
            run.validate_lock(altered)
