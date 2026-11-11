"""Bind all source files before any archived app-record imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import video_webcam_second_flag_v241 as original
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = original.ROOT, original.BASE, original.PANEL
TRIAL = BASE / "video-webcam-second-flag-resume-v242"
PUBLIC = PANEL / "video-webcam-second-flag-resume-freeze-v242.json"
UNAVAILABLE = PANEL / "video-webcam-second-flag-unavailable-v241.json"
read, sha, require = original.read, original.sha, original.require


def identity():
    source, _, _ = original.context()
    require(
        read(original.TRIAL / "freeze.json") == read(original.PUBLIC) == source, "Exact v241 freeze required"
    )
    require(
        read(UNAVAILABLE)
        == {
            "schema": "chipmunk.video-webcam-second-flag-unavailable/241",
            "status": "unavailable",
            "freeze_sha256": sha(original.PUBLIC),
            "exit_code": 1,
            "reason": "fresh_process_source_verification_failed_after_archive_read",
            "aggregate_written": False,
            "model_calls": 0,
        },
        "Preserve and bind the actual v241 failure",
    )
    return {
        "schema": "chipmunk.video-webcam-second-flag-resume-freeze/242",
        "v241_freeze_sha256": sha(original.PUBLIC),
        "v241_unavailable_sha256": sha(UNAVAILABLE),
        "code_sha256": sha(Path(__file__)),
        "source_verification_order": "all_cases_before_reading_archived_app_records",
        "model_calls_permitted": False,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve earlier attempts")
    locked = identity()
    TRIAL.mkdir(mode=0o700)
    original.raw.write(TRIAL / "freeze.json", locked)
    original.raw.write(PUBLIC, locked)
    return locked


def readout():
    require(read(TRIAL / "freeze.json") == read(PUBLIC) == identity(), "Exact v242 freeze required")
    raw = original.raw
    verified = {condition: raw.verify(condition) for condition in original.CONDITIONS}
    binding = SimpleNamespace(
        edge=raw.edge,
        archive_paths=raw.archive_paths,
        direct_file=raw.direct_file,
        authored=raw.authored,
        work_dir=raw.work_dir,
        verify=verified.__getitem__,
        expected_case=raw.expected_case,
    )
    result = scoped(original.readout, {"raw": binding})()
    return {
        **result,
        "schema": "chipmunk.video-webcam-second-flag-resume-result/242",
        "freeze_sha256": sha(PUBLIC),
        "v241_unavailable_sha256": sha(UNAVAILABLE),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "readout"))
    result = {"freeze": freeze, "readout": readout}[parser.parse_args().command]()
    print(json.dumps(result, indent=2, allow_nan=False))
