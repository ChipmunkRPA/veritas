from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import video_vcd2_delivery_app_v232 as study


def fixtures():
    pts = [0, 3, 9, 11, 15, 18]
    source = "e" * 64
    old_baseline = ["a" * 64] * 6
    old_images = {"unchanged": old_baseline, "chroma_conversion_only": old_baseline.copy(),
                  "recompression_only": old_baseline.copy()}
    old_images["chroma_conversion_only"][3] = "b" * 64
    old_images["recompression_only"][3] = "c" * 64
    new_images = {role: ["d" * 64] * 6 for role in study.ROLES}
    new_images["chroma_conversion_only"][3] = "1" * 64
    new_images["recompression_only"][3] = "2" * 64
    def indexes(a, b):
        return [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]

    def diffs(ids):
        return [{"index": i, "pts": pts[i], "rgb_mae": 1.0} for i in ids]
    base = {"slot": 1, "source_sha256": source, "frame_count": 6, "relative_pts": pts, "time_base": "1/3",
            "frame_interval": [2, 5], "width": 100, "height": 50, "model_calls": 0, "source_origin_pts": 0,
            "last_duration": 3, "source_color": {"sample_aspect_ratio": "1/1"}}
    old = {**base, "cases": {}}
    new = {**base, "schema": "chipmunk.video-vcd2-delivery-qualification/206", "delivery_attempt": 1,
           "freeze_sha256": "freeze", "v200_qualification_sha256": "rawreceipt", "cases": {},
           "lossy_codec_changed_indices": [3], "suppressed_lossy_codec_change_indices": []}
    selection = {"attempt": "slot-01-attempt-01", "qualification_sha256": "rawreceipt", "cases": {}}
    for i, role in enumerate(study.ROLES):
        input_sha, output_sha = str(4 + i) * 64, str(7 + i) * 64
        old["cases"][role] = {"file": f"{role}.mp4", "sha256": input_sha, "rgb_sha256": old_images[role],
            "verified_frames": 6, "changed_frames": diffs(indexes(old_images["unchanged"], old_images[role]))}
        new["cases"][role] = {"file": f"{role}.mp4", "sha256": output_sha, "input_sha256": input_sha,
            "bytes": 1024, "status": "prepared", "rgb_sha256": new_images[role], "verified_frames": 6,
            "changed_frames": diffs(indexes(new_images["unchanged"], new_images[role])),
            "delivery_encode_changed_frames": diffs(indexes(old_images[role], new_images[role])),
            "suppressed_pair_change_indices": []}
        selection["cases"][role] = {"sha256": input_sha, "bytes": 1000}
    return {"slot": 1, "selected": {"source_sha256": source}}, selection, old, new


def test_import_stays_fresh_for_isolated_worker():
    env = {**os.environ, "PYTHONPATH": str(study.ROOT / "backend"), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-B", "-c", "from scripts import video_vcd2_delivery_app_v232 as s; "
                           "s.raw.require_fresh()"], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_qualification_binds_both_delivery_and_raw_differences(monkeypatch):
    entry, selection, old, new = fixtures()
    monkeypatch.setattr(study.raw, "direct_file", lambda p: p)
    monkeypatch.setattr(study, "read", lambda p: old if p.is_relative_to(study.raw.PREPARATION) else new)
    monkeypatch.setattr(study, "sha", lambda p: "rawreceipt" if p.is_relative_to(study.raw.PREPARATION) else "deliveryreceipt")
    assert study.qualification(1, entry, selection, "freeze") == (new, "deliveryreceipt")
    broken = copy.deepcopy(new)
    broken["cases"]["chroma_conversion_only"]["delivery_encode_changed_frames"].pop()
    monkeypatch.setattr(study, "read", lambda p: old if p.is_relative_to(study.raw.PREPARATION) else broken)
    with pytest.raises(ValueError, match="per-frame delivery"):
        study.qualification(1, entry, selection, "freeze")
    broken = copy.deepcopy(new)
    broken["cases"]["chroma_conversion_only"]["suppressed_pair_change_indices"] = [3]
    monkeypatch.setattr(study, "read", lambda p: old if p.is_relative_to(study.raw.PREPARATION) else broken)
    with pytest.raises(ValueError, match="preserve the paired"):
        study.qualification(1, entry, selection, "freeze")
    broken = copy.deepcopy(new)
    broken["cases"]["unchanged"]["input_sha256"] = "f" * 64
    monkeypatch.setattr(study, "read", lambda p: old if p.is_relative_to(study.raw.PREPARATION) else broken)
    with pytest.raises(ValueError, match="delivery input"):
        study.qualification(1, entry, selection, "freeze")


def test_expected_case_uses_delivery_not_raw_rgb(monkeypatch):
    _, _, _, delivered = fixtures()
    monkeypatch.setattr(study.raw, "direct_file", lambda p: p)
    monkeypatch.setattr(study, "sha", lambda _: "digest")
    monkeypatch.setattr(study, "read", lambda _: delivered)
    result = study.expected_case({"slot": 1, "role": "chroma_conversion_only", "qualification_sha256": "digest"})
    assert result["rgb"] == delivered["cases"]["chroma_conversion_only"]["rgb_sha256"]
    assert result["changed"] == [3]


def test_nested_runtime_and_audit_scope_bind_isolated_case(monkeypatch):
    seen = []
    def capture(function, overrides):
        seen.append((function, overrides))
        return lambda *args: "called"
    monkeypatch.setattr(study, "scoped", capture)
    assert study.worker("slot-01-unchanged", Path("/tmp/output")) == "called"
    overrides = seen[-1][1]
    assert overrides["TRIAL"] == study.TRIAL and overrides["WORK"] == study.WORK
    assert overrides["verify"] is study.verify and overrides["verify_input"] is study.verify_input
    assert study.audit_case("slot-01-unchanged") == "called"
    overrides = seen[-1][1]
    assert overrides["qualification"] is study.qualification_run and overrides["expected_case"] is study.expected_case
    assert overrides["TRIAL"] == study.TRIAL and overrides["verify"] is study.verify
    assert study.seal("slot-01-unchanged") == "called"
    overrides = seen[-1][1]
    assert overrides["SEALED"] == study.SEALED and overrides["work_dir"] is study.work_dir
    assert overrides["archive_paths"] is study.archive_paths and overrides["active_case"] is study.active_case
