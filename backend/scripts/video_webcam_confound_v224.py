"""Read-only v222 diagnostic of the first frozen v202 matched webcam triplet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tarfile
from fractions import Fraction
from pathlib import Path

from scripts import video_face_confound_v222 as diagnostic
from scripts import video_vcd2_case_archive_v202 as archive
from scripts import video_vcd2_full_app_v202 as prior

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
TRIAL = BASE / "video-webcam-confound-v224"
PUBLIC = PANEL / "video-webcam-confound-freeze-v224.json"
CONDITIONS = tuple(f"slot-01-{role}" for role in prior.ROLES)
CODE = (
    "backend/scripts/video_webcam_confound_v224.py",
    "backend/tests/test_video_webcam_confound_v224.py",
    "backend/scripts/video_face_confound_v222.py",
    "backend/tests/test_video_face_confound_v222.py",
    "docs/video-webcam-confound-v224-method.md",
)
read, sha, require = prior.read, prior.sha, prior.require


def member(stream, name, limit=128 * 1024 * 1024):
    item = stream.getmember(name)
    require(item.isfile() and 0 <= item.size <= limit, "Bounded regular archived member required")
    with stream.extractfile(item) as file:
        require(file is not None, "Archived member unavailable")
        body = file.read(limit + 1)
    require(len(body) == item.size, "Archived member size differs")
    return body


def document(stream, name):
    return json.loads(member(stream, name, 16 * 1024 * 1024))


def content(stream, digest):
    require(prior.SHA.fullmatch(digest) is not None, "Content digest required")
    body = member(stream, f"work/data/objects/sha256/{digest[:2]}/{digest[2:4]}/{digest}", 16 * 1024 * 1024)
    require(hashlib.sha256(body).hexdigest() == digest, "Archived content differs")
    return json.loads(body)


def identity():
    locked, plan = read(prior.PUBLIC), read(prior.TRIAL / "private-plan.json")
    require(
        locked == read(prior.TRIAL / "freeze.json")
        and locked["schema"] == "chipmunk.video-vcd2-full-app-freeze/202"
        and locked["plan_sha256"] == hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        and locked["all_face_swap_negatives"] is True
        and locked["serving_change_permitted"] is False
        and all(sha(ROOT / path) == digest for path, digest in locked["code_sha256"].items()),
        "Exact frozen v202 plan and code required",
    )
    roster = prior.prepared_roster()
    require(
        sha(prior.PREPARATION / "freeze.json") == locked["preparation_freeze_sha256"], "v200 freeze differs"
    )
    entry = roster[0]
    require(entry["slot"] == 1 and entry["status"] == "qualified", "First v200 source unavailable")
    source = prior.PREPARATION / "slot-01-attempt-01"
    qualification = read(prior.direct_file(source / "qualification.json"))
    prior.qualify_slot(entry, qualification)
    decoder = shutil.which("ffmpeg")
    require(decoder is not None, "Frozen local ffmpeg required")
    tool = Path(decoder).resolve(strict=True)
    require(
        sha(tool) == read(prior.PREPARATION / "freeze.json")["identity"]["tool_sha256"]["ffmpeg"],
        "Exact v200 decoder required",
    )
    states = {}
    for condition in CONDITIONS:
        row = plan[condition]
        require(
            row["slot"] == 1
            and condition == f"slot-01-{row['role']}"
            and row["qualification_sha256"] == sha(source / "qualification.json")
            and row["sha256"] == qualification["cases"][row["role"]]["sha256"],
            "Case plan differs",
        )
        tar, receipt_path = prior.archive_paths(condition)
        if row["status"] != "prepared":
            states[condition] = {"status": row["status"]}
        elif not tar.exists() and not receipt_path.exists():
            states[condition] = {"status": "archive_unavailable"}
        elif not tar.is_file() or not receipt_path.is_file():
            states[condition] = {"status": "archive_incomplete"}
        else:
            receipt = read(prior.direct_file(receipt_path))
            archive.verify_archive(tar, receipt)
            require(
                receipt["stage_path"] == str(prior.authored(condition))
                and receipt["work_path"] == str(prior.work_dir(condition)),
                "Archive belongs to another case",
            )
            states[condition] = {
                "status": "archived",
                "receipt_sha256": sha(receipt_path),
                "archive_sha256": receipt["archive_sha256"],
            }
    return {
        "schema": "chipmunk.video-webcam-confound-freeze/224",
        "v202_freeze_sha256": sha(prior.PUBLIC),
        "v200_qualification_sha256": sha(source / "qualification.json"),
        "ffmpeg_sha256": sha(tool),
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "cases": states,
        "model_calls_permitted": False,
        "serving_change_permitted": False,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve all earlier attempts")
    frozen = identity()
    TRIAL.mkdir(mode=0o700)
    for path in (TRIAL / "freeze.json", PUBLIC):
        with path.open("x") as stream:
            json.dump(frozen, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    return frozen


def one(database, query, args):
    rows = database.execute(query, args).fetchall()
    require(len(rows) == 1, "One exact archived analysis record required")
    return rows[0]


def select_record(stream, condition, source_row):
    """Bind audit, guard, source and archived analysis without extracting the tar."""
    audit = document(stream, "stage/audit.json")
    require(audit["condition"] == condition and audit["role"] == source_row["role"], "Archived case differs")
    if audit["status"] != "scored":
        return None, None, "analysis_unavailable"
    profile = document(stream, "stage/profile.json")
    worker, case = document(stream, "stage/output/result.json"), document(stream, "work/case.json")
    locked, row = document(stream, "stage/freeze.json"), document(stream, "stage/inputs.json")
    require(
        locked == read(prior.PUBLIC)
        and row == source_row
        and audit["condition"] == condition
        and audit["role"] == row["role"]
        and profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "authored"
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["error_types"]
        and not profile["resource_limit_observations"]
        and profile["runtime_sha256"] == locked["code_sha256"]["backend/scripts/video_vcd2_full_app_v202.py"]
        and hashlib.sha256(member(stream, "stage/resources.jsonl")).hexdigest()
        == profile["resource_journal_sha256"]
        and hashlib.sha256(member(stream, "stage/output/result.json")).hexdigest()
        == profile["worker_receipt_sha256"]
        and worker["status"] == "complete"
        and worker["freeze_sha256"] == sha(prior.PUBLIC)
        and worker["manifest_sha256"] == locked["app_manifest_sha256"]
        and worker["private_case_sha256"] == hashlib.sha256(member(stream, "work/case.json")).hexdigest()
        and worker["private_runtime_sha256"]
        == hashlib.sha256(member(stream, "work/runtime.json")).hexdigest(),
        "Archived audit and guarded analysis binding differs",
    )
    if not audit["appearance_selected_window"]:
        return None, None, "no_selected_appearance_window"
    names = {entry.name: entry for entry in stream.getmembers()}
    require(
        all(
            name not in names or names[name].size == 0
            for name in ("work/data/veritas.db-wal", "work/data/veritas.db-journal")
        ),
        "Archived database must be checkpointed",
    )
    db = sqlite3.connect(":memory:")
    try:
        db.deserialize(member(stream, "work/data/veritas.db"))
        db.execute("PRAGMA query_only=ON")
        analysis_id, source_sha, status, result_hash, revision = one(
            db, "SELECT id, sha256, status, result_hash, revision FROM analyses", ()
        )
        require(
            source_sha == row["sha256"]
            and status == "done"
            and result_hash == case["result_hash"]
            and case["analysis_status"] == case["pipeline_status"] == "done",
            "Archived analysis differs",
        )
        (raw_digest,) = one(
            db,
            "SELECT a.sha256 FROM detector_runs d JOIN artefacts a ON a.id=d.raw_output_artefact_id "
            "WHERE d.analysis_id=? AND d.detector_id=? AND d.status='ok'",
            (analysis_id, "vid.review.face_appearance"),
        )
        details = content(stream, raw_digest)["details"]
        record = details["face_sequence"]
        require(
            details["appearance_check_status"] == "ok"
            and record is not None
            and len(details["face_sequence_windows"]) == audit["appearance_measured_windows"]
            and record in details["face_sequence_windows"],
            "Selected appearance record differs from audit",
        )
        evidence_rows = db.execute(
            "SELECT id,sha256 FROM artefacts WHERE analysis_id=? AND role='exhibit_payload' "
            "AND media_type='application/json' AND created_by LIKE 'video-evidence-packets/%' ORDER BY created_at DESC,id",
            (analysis_id,),
        ).fetchall()
        require(bool(evidence_rows), "Committed video evidence unavailable")
        evidence_id, evidence_sha = evidence_rows[0]
        evidence = content(stream, evidence_sha)
        require(
            evidence["id"] == evidence_id
            and evidence["analysis_id"] == analysis_id
            and evidence["revision"] == revision
            and evidence["original_result_hash"] == result_hash
            and evidence["source"]["sha256"] == source_sha
            and evidence["status"] == "ready",
            "Evidence source differs",
        )
        packets = [p for p in evidence["packets"] if p["semantics"] == "face-appearance-change/1"]
        require(
            len(packets) == 1 and packets[0]["face_sequence"] == record, "Selected evidence packet differs"
        )
        (probe_sha,) = one(
            db,
            "SELECT a.sha256 FROM analysis_stages s JOIN artefacts a ON a.id=s.output_artefact_id "
            "WHERE s.analysis_id=? AND s.stage='probe' AND s.status='done'",
            (analysis_id,),
        )
        table = content(stream, probe_sha)["frame_table"]
        return record, table, None
    finally:
        db.close()


def role_indices(record):
    groups = {role: [i + k for i in record["centers"][role] for k in (-1, 0, 1)] for role in diagnostic.ROLES}
    indices = [i for role in diagnostic.ROLES for i in groups[role]]
    if indices != sorted(set(indices)):
        return None
    return groups


def assess(record, table_doc, expected, source_file):
    from app.detectors.video.decode import FrameTable, decode_frames
    from app.evidence.face_interval_binding import validate_sequence_record

    table = FrameTable.from_json(table_doc)
    require(
        list(table.pts) == expected["pts"]
        and Fraction(table.time_base) == Fraction(expected["time_base"])
        and (table.width, table.height) == expected["size"]
        and table.rotation == 0
        and not table.interpolated
        and not table.truncated,
        "Archived native picture clock differs",
    )
    validate_sequence_record(
        record, {"time_base": table.time_base, "frame_count": len(table)}, table_pts=table.pts
    )
    by_index = {row["frame_index"]: row for row in record["frame_inputs"]}
    groups = role_indices(record)
    if groups is None:
        return {"status": "unavailable", "reason": "overlapping_or_unordered_role_frames"}
    indices = [i for role in diagnostic.ROLES for i in groups[role]]
    result = {}
    for frame in decode_frames(source_file, table, indices, timeout_s=90):
        row = by_index[frame.index]
        require(
            frame.pts == row["pts"]
            and hashlib.sha256(frame.rgb.tobytes()).hexdigest()
            == row["decoded_rgb_sha256"]
            == expected["rgb"][frame.index],
            "Native RGB or clock differs",
        )
        result[frame.index] = {
            "frame_index": frame.index,
            "rgb": frame.rgb,
            "bbox": row["bbox"],
            "landmarks": row["landmarks"],
        }
    require(list(result) == indices, "Every selected source frame must be decoded")
    return diagnostic.measure({role: [result[index] for index in groups[role]] for role in diagnostic.ROLES})


def comparability(results, selections):
    """Only call two measured arms frame-matched when their role indices agree."""
    result = {}
    for role in ("chroma_conversion_only", "recompression_only"):
        if results["unchanged"]["status"] != "measured" or results[role]["status"] != "measured":
            result[role] = "measurement_unavailable"
        elif selections["unchanged"] == selections[role]:
            result[role] = "same_role_frame_indices"
        else:
            result[role] = "different_selected_frames"
    return result


def readout():
    frozen = identity()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == frozen, "Exact frozen diagnostic required")
    plan = read(prior.TRIAL / "private-plan.json")
    sys.dont_write_bytecode = True
    prior.edge.enforce_read_only()
    results, selections = {}, {}
    for condition in CONDITIONS:
        state, row = frozen["cases"][condition], plan[condition]
        if state["status"] != "archived":
            results[row["role"]] = {"status": "unavailable", "reason": state["status"]}
            continue
        tar, _ = prior.archive_paths(condition)
        try:
            with tarfile.open(tar, "r:") as stream:
                record, table, unavailable = select_record(stream, condition, row)
            require(sha(tar) == state["archive_sha256"], "Archive changed during inspection")
            if unavailable:
                results[row["role"]] = {"status": "unavailable", "reason": unavailable}
                continue
            source = prior.PREPARATION / "slot-01-attempt-01" / f"{row['role']}.mp4"
            require(
                prior.direct_file(source).stat().st_size == row["bytes"] and sha(source) == row["sha256"],
                "Source bytes differ",
            )
            result = assess(record, table, prior.expected_case(row), source)
            require(sha(source) == row["sha256"], "Source changed during decoding")
            results[row["role"]] = result
            if result["status"] == "measured":
                selections[row["role"]] = role_indices(record)
        except (ValueError, KeyError, OSError, sqlite3.DatabaseError, tarfile.TarError):
            results[row["role"]] = {"status": "unavailable", "reason": "source_or_archive_binding_failed"}
    return {
        "schema": "chipmunk.video-webcam-confound-result/224",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "source_groups": 1,
        "roles": results,
        "paired_window_comparability": comparability(results, selections),
        "model_calls": 0,
        "serving_defaults_changed": False,
        "scope": "One consumed matched webcam source. Descriptive measurements are not edit or origin decisions.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "readout"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "readout": readout}[args.action](), indent=2, allow_nan=False))
