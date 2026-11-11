"""Test the one-case replay commitments without running models."""

import subprocess
import sys

import pytest

from scripts import video_vcd2_verify_cache_replay_v251 as replay


def test_import_in_fresh_process():
    subprocess.run(
        [sys.executable, "-B", "-c", f"from {replay.MODULE} import raw; raw.require_fresh()"], check=True
    )


def test_source_identity_allows_only_exact_patch_at_run_time():
    old = {replay.FETCHERS: replay.OLD_FETCHERS, "backend/app/other.py": "unchanged"}
    new = {**old, replay.FETCHERS: replay.PATCHED_FETCHERS}
    replay.check_app_sources(old, old, patched=False)
    replay.check_app_sources(old, new, patched=True)
    with pytest.raises(ValueError, match="historical application source"):
        replay.check_app_sources(old, new, patched=False)
    for changed in (
        old,
        {**new, replay.FETCHERS: "unreviewed fix"},
        {**new, "backend/app/other.py": "different"},
        {**new, "backend/app/added.py": "additional"},
        {replay.FETCHERS: replay.PATCHED_FETCHERS},
    ):
        with pytest.raises(ValueError, match="patched application source"):
            replay.check_app_sources(old, changed, patched=True)


def test_archive_commitment_binds_member_names_and_hashes():
    first = {"receipt_sha256": "first-receipt", "archive_sha256": "first-archive"}
    second = {"receipt_sha256": "second-receipt", "archive_sha256": "second-archive"}
    records = {"slot-a": first, "slot-b": second}
    original = replay.archive_set_commitment(records)
    assert replay.archive_set_commitment({"slot-b": second, "slot-a": first}) == original
    assert replay.archive_set_commitment({"slot-a": first}) != original
    assert replay.archive_set_commitment({"slot-a": second, "slot-b": first}) != original
    assert (
        replay.archive_set_commitment({**records, "slot-b": {**second, "archive_sha256": "other"}})
        != original
    )


def test_upload_proof_needs_matching_workspace_content():
    original = {"kind": "file", "path": "work/data/objects/input", "size": 10, "sha256": "exact"}
    assert replay.has_uploaded_bytes([original], 10, "exact")
    for altered in (
        {**original, "path": "stage/input"},
        {**original, "size": 9},
        {**original, "sha256": "wrong"},
        {**original, "kind": "directory"},
    ):
        assert not replay.has_uploaded_bytes([altered], 10, "exact")


def test_runtime_requires_frozen_bindings_and_case(tmp_path, monkeypatch):
    trial, public = tmp_path / "trial", tmp_path / "public.json"
    stage = trial / replay.CONDITION
    stage.mkdir(parents=True)
    row = {"sha256": "exact", "bytes": 10}
    copied = stage / "inputs.json"
    manifest = stage / f"{replay.ARM}.yaml"
    replay.write(copied, row)
    manifest.write_text("frozen manifest")
    frozen = {
        "private_input_record_sha256": replay.sha(copied),
        "model_manifest_sha256": replay.sha(manifest),
    }
    for target in (public, trial / "freeze.json", stage / "freeze.json"):
        replay.write(target, frozen)
    replay.write(trial / "private-plan.json", {replay.CONDITION: row})
    monkeypatch.setattr(replay, "TRIAL", trial)
    monkeypatch.setattr(replay, "PUBLIC", public)

    def context(*, patched):
        assert patched is True
        return frozen, row, tmp_path / "video", {}

    monkeypatch.setattr(replay, "context", context)
    assert replay.verify() == (frozen, tmp_path / "video", {"file_sha256": "exact", "bytes": 10})
    with pytest.raises(ValueError, match="fixed historical case"):
        replay.verify("different-case")
    manifest.write_text("changed manifest")
    with pytest.raises(ValueError, match="patched replay freeze"):
        replay.verify()


def test_public_status_keeps_route_outcome_without_private_details():
    visible = {"status": "scored", "customer_direction": "unavailable", "appearance_route_status": "failed"}
    assert replay.statuses({**visible, "source": "private", "detector": "private"}) == visible
    assert replay.collision_log("FileNotFoundError: '.veritas-verified.json.tmp' -> '.veritas-verified.json'")
    assert not replay.collision_log("FileNotFoundError: other")
    assert not replay.collision_log("Found .veritas-verified.json.tmp")
