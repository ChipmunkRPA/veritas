"""Read-only audit of selected and other retained full-app landmark windows."""

from __future__ import annotations

import json
import sqlite3
import tarfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

from app.evidence.face_interval_binding import APPEARANCE_METHOD, validate_packet, validate_sequence_record
from scripts import video_face_confound_v222 as geometry
from scripts import video_faceswapai_edge_review_v194 as edge
from scripts import video_landmark_support as support
from scripts import video_vcd2_case_archive_v202 as archives
from scripts import video_vcd2_remaining_webcams_v253 as webcam
from scripts import video_webcam_confound_v224 as archived
from scripts.video_webcam_confound_resume_v227 import memory_connect

swap = edge.prior
read, sha, require = edge.read, edge.sha, edge.require
WEB_RESULT = webcam.PANEL / f"{webcam.NAME}-complete.json"
WEB_RESULT_SHA = "d8410ea6b3b9d7775b3a689125290a42eb22bd3ffc7b9a2aee5d5ba7303ea7fe"
EDGE_FREEZE_SHA = "345369984c9a1c8a91b7e0f36559f5320ca70eda458405ff381600f0fec794b7"


def windows(db, load, digest, case, expected):
    analysis_id, source_sha, status, result, revision, profile = archived.one(
        db, "SELECT id,sha256,status,result_hash,revision,profile FROM analyses", ()
    )
    require(
        source_sha == digest
        and status == case["analysis_status"] == case["pipeline_status"] == "done"
        and result == case["result_hash"]
        and profile == "full",
        "Exact committed full analysis required",
    )
    load(result)
    (raw_sha,) = archived.one(
        db,
        "SELECT a.sha256 FROM detector_runs d JOIN artefacts a ON a.id=d.raw_output_artefact_id "
        "WHERE d.analysis_id=? AND d.detector_id='vid.review.face_appearance' AND d.status='ok'",
        (analysis_id,),
    )
    (probe_sha,) = archived.one(
        db,
        "SELECT a.sha256 FROM analysis_stages s JOIN artefacts a ON a.id=s.output_artefact_id "
        "WHERE s.analysis_id=? AND s.stage='probe' AND s.status='done'",
        (analysis_id,),
    )
    details = load(raw_sha)["details"]
    table = load(probe_sha)["frame_table"]
    source = {key: table[key] for key in ("time_base", "width", "height")}
    source["frame_count"] = len(table["pts"])
    require(
        table["pts"] == expected["pts"]
        and Fraction(table["time_base"]) == Fraction(expected["time_base"])
        and (table["width"], table["height"]) == expected["size"]
        and table["rotation"] == 0
        and not table["truncated"]
        and not table["interpolated"],
        "Frozen native picture clock and geometry required",
    )

    evidence_rows = db.execute(
        "SELECT id,sha256 FROM artefacts WHERE analysis_id=? AND role='exhibit_payload' "
        "AND media_type='application/json' AND created_by LIKE 'video-evidence-packets/%' "
        "ORDER BY created_at DESC,id LIMIT 1",
        (analysis_id,),
    ).fetchall()
    require(len(evidence_rows) == 1, "Exact committed video evidence required")
    evidence_id, evidence_sha = evidence_rows[0]
    evidence = load(evidence_sha)
    require(
        evidence["id"] == evidence_id
        and evidence["analysis_id"] == analysis_id
        and evidence["revision"] == revision
        and evidence["original_result_hash"] == result
        and evidence["source"]["sha256"] == digest
        and evidence["status"] == "ready"
        and all(evidence["source"][key] == value for key, value in source.items()),
        "Evidence belongs to a different source or clock",
    )
    packets = [item for item in evidence["packets"] if item["semantics"] == APPEARANCE_METHOD]
    selected, retained = details["face_sequence"], details["face_sequence_windows"]
    require(
        details["appearance_check_status"] == "ok"
        and 0 < len(retained) <= 12
        and len(packets) == int(selected is not None),
        "Appearance window availability differs from the stored evidence",
    )
    if selected is not None:
        require(
            packets[0]["face_sequence"] == selected and retained.count(selected) == 1,
            "Selected window differs",
        )
        validate_packet(packets[0], evidence["source"], table_pts=table["pts"])
    measured = []
    for record in retained:
        validate_sequence_record(record, source, table_pts=table["pts"], require_flag=record == selected)
        require(
            record["method"] == APPEARANCE_METHOD
            and all(
                row["decoded_rgb_sha256"] == expected["rgb"][row["frame_index"]]
                for row in record["frame_inputs"]
            ),
            "A retained window references different source pixels",
        )
        measured.append((record == selected, support.measure(record)))
    return measured


def swap_cases():
    require(sha(edge.PUBLIC) == EDGE_FREEZE_SHA, "Exact database commitment required")
    frozen = read(edge.PUBLIC)
    require(
        sha(swap.PUBLIC) == frozen["prior_freeze_sha256"]
        and sha(edge.PRIOR_RESULT) == frozen["prior_result_sha256"],
        "Exact v190 source panel required",
    )
    manifest = read(swap.MANIFEST)
    require(sha(swap.MANIFEST) == read(swap.PUBLIC)["private_manifest_sha256"], "Source manifest differs")
    results = {
        row["condition"]: row["appearance_review_coverage"] for row in read(edge.PRIOR_RESULT)["cases"]
    }
    require(tuple(results) == swap.CONDITIONS, "Exact v190 comparison roster required")
    for row in manifest["cases"]:
        condition = row["condition"]
        work = swap.WORK / condition / swap.ARM
        lock, case_file = frozen["cases"][condition], work / "case.json"
        source_file = swap.ARCHIVE / f"{condition}.mp4"
        db_file = edge.database_path(work)
        require(
            row == read(swap.TRIAL / condition / "inputs.json")
            and sha(db_file) == lock["database_sha256"]
            and sha(case_file) == lock["case_sha256"]
            and source_file.stat().st_size == row["bytes"]
            and sha(source_file) == row["sha256"] == lock["input_sha256"],
            "Exact v190 media and saved analysis required",
        )
        from app.storage.objects import ObjectStore

        store = ObjectStore(work / "data/objects")

        def load(digest, store=store):
            store.verify(digest)
            return json.loads(store.read_bytes(digest))

        table = row["frame_table"]
        expected = {**table, "size": (table["width"], table["height"]), "rgb": row["rgb_sha256"]}
        db = sqlite3.connect(f"{db_file.as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            measured = windows(db, load, row["sha256"], read(case_file), expected)
        finally:
            db.close()
        coverage = results[condition]
        require(
            len(measured) == coverage["measured_windows"]
            and any(selected for selected, _ in measured) == coverage["selected_window"],
            "Saved v190 coverage differs from the published audit",
        )
        yield ("known_swap_renders" if condition.startswith("swap_") else "matched_no_swap_renders"), measured


def webcam_cases():
    require(sha(WEB_RESULT) == WEB_RESULT_SHA, "Exact v253 result required")
    published, frozen = read(WEB_RESULT), read(webcam.PUBLIC)
    plan = read(webcam.TRIAL / "private-plan.json")
    require(
        sha(webcam.PUBLIC) == published["freeze_sha256"]
        and read(webcam.TRIAL / "freeze.json") == frozen
        and tuple(plan) == webcam.CONDITIONS
        and webcam.replay.archive_set_commitment(plan) == frozen["plan_sha256"],
        "Exact six webcam inputs required",
    )
    receipts = {}
    for condition in webcam.CONDITIONS:
        path, receipt_path = webcam.archive_paths(condition)
        receipt = read(webcam.raw.direct_file(receipt_path))
        receipts[condition] = (path, receipt_path, receipt)
    require(
        webcam.replay.archive_set_commitment(
            {
                condition: {"receipt_sha256": sha(receipt_path), "archive_sha256": receipt["archive_sha256"]}
                for condition, (_, receipt_path, receipt) in receipts.items()
            }
        )
        == published["archive_set_commitment_sha256"],
        "Exact published webcam archive set required",
    )
    for condition, (path, _, receipt) in receipts.items():
        row = plan[condition]
        archives.verify_archive(path, receipt)
        require(
            webcam.verified_seal(condition, receipt, row) == (True, False),
            "Exact archived webcam upload required",
        )
        with tarfile.open(path, "r:") as stream:
            audit = archived.document(stream, "stage/audit.json")
            require(
                audit["status"] == "scored"
                and audit["condition"] == condition
                and audit["appearance_route_status"] == audit["appearance_check_status"] == "ok",
                "Exact scored webcam route required",
            )
            members = {entry.name: entry for entry in stream.getmembers()}
            require(
                all(
                    name not in members or members[name].size == 0
                    for name in ("work/data/veritas.db-wal", "work/data/veritas.db-journal")
                ),
                "Archived webcam database must be checkpointed",
            )
            db = memory_connect(":memory:")
            try:
                db.deserialize(archived.member(stream, "work/data/veritas.db"))
                db.execute("PRAGMA query_only=ON")
                measured = windows(
                    db,
                    lambda digest: archived.content(stream, digest),
                    row["sha256"],
                    archived.document(stream, "work/case.json"),
                    webcam.raw.expected_case(row),
                )
            finally:
                db.close()
            require(
                len(measured) == audit["appearance_measured_windows"]
                and any(selected for selected, _ in measured) == audit["appearance_selected_window"],
                "Saved webcam coverage differs from the sealed audit",
            )
        yield "genuine_webcam_renders", measured


def bounds(values):
    values = list(values)
    return [round(min(values), 6), round(max(values), 6)] if values else None


def aggregate(rows):
    good = [row for row in rows if row["status"] == "measured"]
    return {
        "windows": len(rows),
        "measured": len(good),
        "unavailable": dict(Counter(row["reason"] for row in rows if row["status"] == "unavailable")),
        "roles": {
            role: {
                "frame_observations": sum(row["role_frame_counts"][role] for row in good),
                "eye_box_width_median_range": bounds(row["eye_box_width"][role]["median"] for row in good),
                "eye_box_width_minimum_range": bounds(row["eye_box_width"][role]["minimum"] for row in good),
                "out_of_box_observations": sum(
                    row["eye_box_width"][role]["outside_box_frames"] for row in good
                ),
            }
            for role in support.ROLES
        },
        "during_minus_flank_eye_box_width_range": bounds(
            row["during_minus_flank_eye_box_width"] for row in good
        ),
        "geometry_rms_eye_units_ranges": {
            key: bounds(row["geometry_rms_eye_units"][key] for row in good)
            for key in ("before_to_during", "after_to_during", "before_to_after")
        },
    }


def readout():
    cases = [*swap_cases(), *webcam_cases()]
    cohorts = {}
    for name, source_groups in (
        ("known_swap_renders", 1),
        ("matched_no_swap_renders", 1),
        ("genuine_webcam_renders", 2),
    ):
        rows = [windows for group, windows in cases if group == name]
        cohorts[name] = {
            "source_groups": source_groups,
            "renders": len(rows),
            "selected": aggregate([row for windows in rows for selected, row in windows if selected]),
            "other_retained": aggregate(
                [row for windows in rows for selected, row in windows if not selected]
            ),
        }
    return {
        "schema": "chipmunk.video-saved-landmark-support/254",
        "status": "complete",
        "v190_result_sha256": sha(edge.PRIOR_RESULT),
        "v194_database_freeze_sha256": EDGE_FREEZE_SHA,
        "v253_result_sha256": WEB_RESULT_SHA,
        "code_sha256": {
            Path(file).name: sha(Path(file)) for file in (__file__, support.__file__, geometry.__file__)
        },
        "cohorts": cohorts,
        "independent_known_swap_interventions": 1,
        "decoded_frames": 0,
        "model_calls": 0,
        "serving_changed": False,
    }


if __name__ == "__main__":
    edge.enforce_read_only()
    print(json.dumps(readout(), indent=2, allow_nan=False, sort_keys=True))
