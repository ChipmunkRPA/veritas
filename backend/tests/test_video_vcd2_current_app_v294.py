import copy
import hashlib
import json
import subprocess
import sys

import pytest

from scripts import video_vcd2_current_app_v294 as study


def test_inventory_keeps_every_control_and_rejects_changed_receipt_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "CAP", 8)
    slots = [{"slot": slot} for slot in study.SLOTS]
    hashes = []
    for slot in study.SLOTS:
        folder = tmp_path / f"slot-{slot:02d}-attempt-01"
        folder.mkdir()
        cases = {}
        for role in study.ROLES:
            path = folder / f"{role}.mp4"
            size = 9 if slot == 8 and role == "recompression_only" else 8
            path.write_bytes(b"fixture\n" + b"." * (size - 8))
            cases[role] = {"file": path.name, "sha256": "a" * 64, "verified_frames": 4}
        receipt = folder / "qualification.json"
        receipt.write_text(json.dumps({"slot": slot, "width": 1280, "height": 720, "cases": cases}))
        hashes.append(study.sha(receipt))

    summary = {
        "schema": "chipmunk.video-vcd2-matched-controls-result/290",
        "status": "complete",
        "freeze_sha256": "freeze",
        "source_groups": 8,
        "controls": dict.fromkeys(study.ROLES, 8),
        "verified_frames": dict.fromkeys(study.ROLES, 32),
        "model_calls": 0,
        "private_receipt_bundle_sha256": hashlib.sha256(json.dumps(hashes).encode()).hexdigest(),
    }

    def receipt_plan(entry, document):
        assert document["slot"] == entry["slot"]

    plan = study.inventory(slots, tmp_path.resolve(), summary, "freeze", receipt_plan)
    census = study.census(plan)
    assert tuple(plan) == study.CONDITIONS
    assert plan["slot-08-recompression_only"]["status"] == "unsupported_upload_size"
    assert census["upload_eligible"] == 23
    assert census["upload_ineligible"] == 1
    assert census["complete_upload_eligible_triplets"] == 7
    assert census["by_resolution"]["1280x720"]["maximum_bytes"] == 9
    with pytest.raises(ValueError, match="Exact complete v290"):
        study.inventory(
            slots,
            tmp_path.resolve(),
            {**summary, "private_receipt_bundle_sha256": "changed"},
            "freeze",
            receipt_plan,
        )


def appearance_evidence():
    roles = ("before", "during", "after")
    frames = [
        {
            "frame_index": index,
            "bbox": {"w": 10.0},
            "landmarks": [[0.0, 0.0], [2.0 + index // 3, 0.0]],
        }
        for index in range(9)
    ]
    sequence = {
        "method": "face-appearance-change/1",
        "centers": dict(zip(roles, ([1], [4], [7]), strict=True)),
        "frame_inputs": frames,
    }
    measurements = [
        {"label": f"Median eye spacing {role} the range", "value": value, "unit": "% of face-box width"}
        for role, value in zip(roles, (20.0, 30.0, 40.0), strict=True)
    ]
    return {
        "version": "video-evidence-8",
        "status": "ready",
        "packets": [{"face_sequence": sequence, "measurements": measurements}, {}],
    }


def test_independently_checks_each_recorded_context_role_and_current_version():
    evidence = appearance_evidence()
    assert study.check_current_evidence(evidence, 6) == 1
    bad = copy.deepcopy(evidence)
    bad["packets"][0]["face_sequence"]["frame_inputs"][3]["landmarks"][1][0] = 8.0
    bad["packets"][0]["face_sequence"]["frame_inputs"][4]["landmarks"][1][0] = 8.0
    with pytest.raises(ValueError, match="differs from source-bound native landmarks"):
        study.check_current_evidence(bad, 6)
    with pytest.raises(ValueError, match="current video evidence version 8"):
        study.check_current_evidence({**evidence, "version": "video-evidence-7", "packets": []}, 6)


def test_worker_handoff_scopes_only_the_current_reference_and_owned_paths(tmp_path, monkeypatch):
    observed = {}

    def inspect_worker(arm, output):
        context = globals()
        observed.update(arm=arm, output=output, trial=context["TRIAL"], work=context["WORK"])
        observed["verification"] = context["verify"]()
        observed["input_validator"] = (
            context["prior"].worker.__globals__["record_case"].__globals__["verify_input"]
        )

    condition = study.CONDITIONS[0]
    monkeypatch.delitem(sys.modules, "app.core.config", raising=False)
    monkeypatch.setattr(study, "verify", lambda case: case)
    monkeypatch.setattr(study.raw.prior.runtime, "worker", inspect_worker)
    study.worker(condition, tmp_path)
    assert observed == dict(
        arm="reference",
        output=tmp_path,
        trial=study.TRIAL / condition,
        work=study.WORK / condition,
        verification=condition,
        input_validator=study.verify_input,
    )
    assert study.raw.prior.ARM == "batch-one"
    assert study.BASELINE.SETTINGS["offline"] == "true"
    assert study.BASELINE.SETTINGS["billing_mode"] == "disabled"


def test_worker_import_is_fresh_and_oversize_control_has_no_worker(tmp_path, monkeypatch):
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from scripts.video_vcd2_current_app_v294 import raw; raw.require_fresh()",
        ],
        cwd=study.ROOT / "backend",
        check=True,
    )
    monkeypatch.setattr(study, "PREPARED", tmp_path / "summary.json")
    assert study.preflight()["status"] == "waiting_for_v290_seal"
    condition = study.CONDITIONS[0]
    monkeypatch.setattr(
        study, "verify_plan", lambda: ({}, {condition: {"status": "unsupported_upload_size"}})
    )
    monkeypatch.setattr(study.later, "run", None)
    assert study.run(condition) == {"status": "unsupported_upload_size"}


def test_unavailable_audit_keeps_budget_skip_and_admits_before_read_only_hook(monkeypatch):
    observed = []

    def verify_plan():
        observed.append("admitted")
        return "sealed"

    def enforce_read_only():
        observed.append("read-only")

    def verify_case(condition):
        assert globals()["verify_plan"]() == "sealed"
        return condition

    def audit_case(condition):
        assert globals()["verify"](condition) == condition
        return {"status": "evidence_unavailable"}

    monkeypatch.setattr(study, "verify_plan", verify_plan)
    monkeypatch.setattr(study.raw, "verify", verify_case)
    monkeypatch.setattr(study.raw.edge, "enforce_read_only", enforce_read_only)
    monkeypatch.setattr(study.raw, "audit_case", audit_case)
    monkeypatch.setattr(
        study, "read", lambda _: {"target_status": "skipped", "target_reason": "memory budget"}
    )
    result = study.audit_case(study.CONDITIONS[0])
    assert observed == ["admitted", "read-only"]
    assert result["status"] == "evidence_unavailable"
    assert result["legacy_frame_budget_skip"] is True
