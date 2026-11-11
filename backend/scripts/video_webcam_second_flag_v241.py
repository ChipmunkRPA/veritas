"""Repeat the source-only webcam diagnostic on the second flagged recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile

from scripts import video_webcam_source_geometry_v231 as geometry

audit = geometry.prior
raw = audit.prior
ROOT, BASE, PANEL = audit.ROOT, audit.BASE, audit.PANEL
TRIAL = BASE / "video-webcam-second-flag-v241"
PUBLIC = PANEL / "video-webcam-second-flag-freeze-v241.json"
COHORT = PANEL / "video-vcd2-full-app-six-triplets-v202.json"
FIRST = PANEL / "video-webcam-source-geometry-complete-v231.json"
OBSERVED_SLOTS = (1, 3, 4, 5, 7, 8)
CONDITIONS = tuple(f"slot-07-{role}" for role in raw.ROLES)
read, sha, require = audit.read, audit.sha, audit.require


def context():
    _, plan = raw.verify_plan()
    observed = read(COHORT)
    require(
        observed["freeze_sha256"] == sha(raw.PUBLIC)
        and observed["complete_matched_source_groups"] == 6
        and observed["source_groups_with_any_face_review_interval"] == 2,
        "The published six-recording negative-control comparison is required",
    )
    archives = {}
    for condition, case in plan.items():
        if case["slot"] not in OBSERVED_SLOTS:
            continue
        _, path = raw.archive_paths(condition)
        record = read(raw.direct_file(path))
        archives[condition] = {"receipt_sha256": sha(path), "archive_sha256": record["archive_sha256"]}
    commitment = hashlib.sha256(json.dumps(archives, sort_keys=True).encode()).hexdigest()
    require(
        commitment == observed["archive_set_commitment_sha256"]
        and len(archives) == observed["sealed_cases"] == 18
        and all(case in archives and plan[case]["status"] == "prepared" for case in CONDITIONS),
        "Exact published archive set required",
    )
    code = (
        "backend/scripts/video_webcam_second_flag_v241.py",
        "backend/scripts/video_webcam_source_geometry_v231.py",
        "backend/scripts/video_webcam_confound_resume_v227.py",
        "backend/scripts/video_webcam_confound_v224.py",
        "backend/scripts/video_face_confound_v222.py",
        "docs/video-webcam-second-flag-v241-method.md",
    )
    lock = {
        "schema": "chipmunk.video-webcam-second-flag-freeze/241",
        "six_group_aggregate_sha256": sha(COHORT),
        "first_flag_geometry_sha256": sha(FIRST),
        "published_archive_commitment_sha256": commitment,
        "selected_archive_set_sha256": hashlib.sha256(
            json.dumps([archives[case] for case in CONDITIONS], sort_keys=True).encode()
        ).hexdigest(),
        "code_sha256": {name: sha(ROOT / name) for name in code},
        "roles": list(raw.ROLES),
        "new_neural_calls_permitted": False,
    }
    return lock, plan, archives


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve earlier attempts")
    lock, _, _ = context()
    TRIAL.mkdir(mode=0o700)
    raw.write(TRIAL / "freeze.json", lock)
    raw.write(PUBLIC, lock)
    return lock


def readout():
    lock, plan, archives = context()
    require(read(TRIAL / "freeze.json") == read(PUBLIC) == lock, "Exact diagnostic freeze required")
    raw.edge.enforce_read_only()
    results, selections, ranges = {}, {}, []
    for condition in CONDITIONS:
        source = plan[condition]
        role = source["role"]
        tar, receipt_path = raw.archive_paths(condition)
        receipt = read(raw.direct_file(receipt_path))
        audit.archive.verify_archive(tar, receipt)
        require(
            sha(receipt_path) == archives[condition]["receipt_sha256"]
            and receipt["archive_sha256"] == archives[condition]["archive_sha256"]
            and receipt["stage_path"] == str(raw.authored(condition))
            and receipt["work_path"] == str(raw.work_dir(condition)),
            "Exact archived case required",
        )
        with tarfile.open(tar, "r:") as stream:
            observed = audit.document(stream, "stage/audit.json")
            [flag] = observed["face_review_intervals"]["face-appearance-change/1"]
            ranges.append((flag["start_frame"], flag["last_supported_frame"]))
            record, table, unavailable = geometry.resume.select_record(stream, condition, source)
        if unavailable:
            results[role] = {"status": "unavailable", "reason": unavailable}
            continue
        _, file, _ = raw.verify(condition)
        result = geometry.assess(record, table, raw.expected_case(source), file)
        require(sha(file) == source["sha256"], "Source changed during decoding")
        results[role] = result
        if result["status"] == "measured":
            selections[role] = audit.role_indices(record)
    require(len(set(ranges)) == 1, "The three archived flag boundaries differ")
    return {
        "schema": "chipmunk.video-webcam-second-flag-result/241",
        "freeze_sha256": sha(PUBLIC),
        "same_flag_bounds_across_roles": True,
        "roles": results,
        "paired_window_comparability": audit.comparability(results, selections),
        "model_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "readout"))
    result = {"freeze": freeze, "readout": readout}[parser.parse_args().command]()
    print(json.dumps(result, indent=2, allow_nan=False))
