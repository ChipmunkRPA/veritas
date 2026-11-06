"""Fresh-role, published-freeze and real-response parity safeguards."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts import fused_crop_confirmation_v6 as protocol
from scripts import run_fused_crop_confirmation_v6 as runner
from scripts.localization_benchmark import sha


def panel():
    groups, items = {}, []
    for family in ("firefly", "sd2", "flux-fill"):
        groups[family] = []
        for i in range(16):
            source = f"{family}-{i}"
            pair = {"source_id": source, "family": family,
                    "original": ["https://example.org/a", source + "-0"],
                    "image": ["https://example.org/a", source + "-1"]}
            groups[family].append(pair)
            for label in (0, 1):
                digest = sha(f"{source}/{label}".encode())
                role = "image" if label else "original"
                items.append({"id": f"{source}-{label}", "source_id": source, "label": label,
                              "family": family, "image_sha256": digest,
                              "archive_source": {"archive": pair[role][0], "member": pair[role][1], "sha256": digest}})
    return {"items": items}, {"groups": groups}


def test_exact_new_pairs_have_complete_truth_and_provenance():
    manifest, plan = panel()
    assert len(protocol.paired_items(manifest, plan, {"past"}, {"old-digest"})) == 48


@pytest.mark.parametrize("damage", ["missing", "duplicate", "past_id", "past_bytes", "family", "role", "label", "digest"])
def test_source_binding_refuses_reuse_or_changed_truth(damage):
    manifest, plan = panel()
    ids, digests = set(), set()
    if damage == "missing":
        manifest["items"].pop()
    elif damage == "duplicate":
        manifest["items"][-1] = manifest["items"][0]
    elif damage == "past_id":
        ids.add(manifest["items"][0]["source_id"])
    elif damage == "past_bytes":
        digests.add(manifest["items"][0]["image_sha256"])
    elif damage == "family":
        manifest["items"][0]["family"] = "other"
    elif damage == "role":
        manifest["items"][0]["archive_source"]["member"] = "edited-instead-of-original"
    elif damage == "label":
        manifest["items"][0]["label"] = True
    else:
        manifest["items"][0]["archive_source"]["sha256"] = "changed"
    with pytest.raises(ValueError):
        protocol.paired_items(manifest, plan, ids, digests)


def response():
    crop = {"box": [0, 0, 320, 320], "image_score": .6, "rgb_sha256": "a" * 64}
    row = {"status": "ok", "input_sha256": "input", "baseline": {"intersection": 0},
           "baseline_image_score": .4, "crops": [copy.deepcopy(crop) for _ in range(4)]}
    arrays = {n: np.zeros((3, 4), np.float32) for n in ("baseline", "crop0", "crop1", "crop2", "crop3")}
    return row, arrays


def test_parity_preserves_decisions_even_inside_float_tolerance():
    old, expected = response()
    actual = copy.deepcopy(old)
    actual["crops"][0]["image_score"] = .6000001
    measured = runner.parity(actual, expected, old, expected, .60000005)
    assert measured["max_score_error"] < 2e-6
    assert not measured["same_crop_gate_decisions"] and not measured["passed"]


@pytest.mark.parametrize("damage", ["failed", "old_failed", "missing", "geometry", "dtype", "nan", "large_error", "input", "box", "rgb", "metric", "baseline_gate"])
def test_parity_cannot_hide_missing_or_changed_neural_evidence(damage):
    old, expected = response()
    actual, arrays = copy.deepcopy(old), {k: v.copy() for k, v in expected.items()}
    if damage == "failed":
        actual["status"] = "failed"
    elif damage == "old_failed":
        old["status"] = "failed"
    elif damage == "missing":
        arrays.pop("crop0")
    elif damage == "geometry":
        arrays["crop0"] = np.zeros((2, 4), np.float32)
    elif damage == "dtype":
        arrays["crop0"] = arrays["crop0"].astype(np.float64)
    elif damage == "nan":
        arrays["crop0"][0, 0] = np.nan
    elif damage == "large_error":
        arrays["crop0"][0, 0] = .01
    elif damage == "input":
        actual["input_sha256"] = "other"
    elif damage == "box":
        actual["crops"][0]["box"][0] = 1
    elif damage == "rgb":
        actual["crops"][0]["rgb_sha256"] = "b" * 64
    elif damage == "metric":
        actual["baseline"]["intersection"] = 1
    else:
        actual["baseline_image_score"] = .7
    assert not runner.parity(actual, arrays, old, expected, .5)["passed"]


def native_receipt(path):
    cases = [{"id": f"{family}-{label}", "family": family, "label": label,
              "condition": condition, "passed": True}
             for family in ("firefly", "sd2", "flux-fill") for label in (0, 1) for condition in ("native", "jpeg75")]
    d = {"schema": "chipmunk.fused-crop-v6-native-parity/1", "freeze_sha256": "freeze",
         "development_predictions_sha256": protocol.DEVELOPMENT_PREDICTIONS_SHA256,
         "cases": cases, "expected_cases": 12, "fresh_inputs_scored": 0, "passed": True}
    path.write_text(json.dumps(d))
    return d


@pytest.mark.parametrize("damage", ["missing", "duplicate", "failed", "fresh", "family", "freeze", "transfer"])
def test_native_requirement_refuses_coherent_but_incomplete_receipts(tmp_path, damage):
    p = tmp_path / "native.json"
    d = native_receipt(p)
    original_hash = sha(p.read_bytes())
    if damage == "missing":
        d["cases"].pop()
    elif damage == "duplicate":
        d["cases"][-1] = d["cases"][0]
    elif damage == "failed":
        d["cases"][0]["passed"] = False
    elif damage == "fresh":
        d["fresh_inputs_scored"] = 1
    elif damage == "family":
        d["cases"][0]["family"] = "other"
    elif damage == "freeze":
        d["freeze_sha256"] = "changed"
    else:
        d["unbound"] = "changed transfer"
    p.write_text(json.dumps(d))
    expected = original_hash if damage == "transfer" else sha(p.read_bytes())
    with pytest.raises(ValueError):
        runner.require_native(p, expected, "freeze")


def test_failed_native_receipt_prevents_model_loading_and_output_creation(tmp_path, monkeypatch):
    p = tmp_path / "native.json"
    d = native_receipt(p)
    d["passed"] = False
    p.write_text(json.dumps(d))
    monkeypatch.setattr(protocol, "check", lambda *_: ({}, {}))
    monkeypatch.setattr(runner, "models", lambda *_: pytest.fail("Fresh models must not load"))
    output = tmp_path / "fresh"
    with pytest.raises(ValueError):
        runner.run(tmp_path / "lock", tmp_path, "freeze", p, sha(p.read_bytes()), tmp_path, output)
    assert not output.exists()


def test_native_plan_uses_source_identity_instead_of_previous_responses():
    manifest, _ = panel()
    chosen = runner.native_plan(manifest["items"])
    reordered = list(reversed(manifest["items"]))
    assert {(i["source_id"], i["label"]) for i in chosen} == {
        (i["source_id"], i["label"]) for i in runner.native_plan(reordered)}
    assert len(chosen) == 6


def test_published_freeze_hash_is_checked_before_any_manifest_access(tmp_path):
    path = tmp_path / "lock.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="published commitment"):
        protocol.check(path, tmp_path / "missing", "0" * 64)


def test_original_development_method_cannot_change_for_confirmation(monkeypatch):
    monkeypatch.setattr(protocol, "development_code_pins", lambda: {"changed": "0" * 64})
    with pytest.raises(ValueError, match="development method changed"):
        protocol.code_pins()
