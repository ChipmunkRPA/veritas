"""Replay frozen webcam measurements with complete native frame geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction

from scripts import video_webcam_confound_resume_v227 as resume
from scripts.video_appearance_corrected_v112 import scoped

prior = resume.prior
ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
TRIAL = BASE / "video-webcam-source-geometry-v231"
PUBLIC = PANEL / "video-webcam-source-geometry-freeze-v231.json"
FAILED = PANEL / "video-webcam-confound-unavailable-v227.json"
CODE = (
    "backend/scripts/video_webcam_source_geometry_v231.py",
    "backend/tests/test_video_webcam_source_geometry_v231.py",
    "docs/video-webcam-source-geometry-v231-method.md",
)
read, sha, require = prior.read, prior.sha, prior.require


def verified_geometry(table, expected):
    require(
        list(table.pts) == expected["pts"]
        and Fraction(table.time_base) == Fraction(expected["time_base"])
        and (table.width, table.height) == expected["size"]
        and table.rotation == 0
        and not table.interpolated
        and not table.truncated,
        "Archived native picture clock and geometry differ",
    )
    return {
        "time_base": table.time_base,
        "frame_count": len(table),
        "width": table.width,
        "height": table.height,
    }


def assess(record, table_doc, expected, source_file):
    from app.detectors.video.decode import FrameTable, decode_frames
    from app.evidence.face_interval_binding import validate_sequence_record

    table = FrameTable.from_json(table_doc)
    validate_sequence_record(record, verified_geometry(table, expected), table_pts=table.pts)
    groups = prior.role_indices(record)
    if groups is None:
        return {"status": "unavailable", "reason": "overlapping_or_unordered_role_frames"}
    indices = [index for role in prior.diagnostic.ROLES for index in groups[role]]
    bound = {row["frame_index"]: row for row in record["frame_inputs"]}
    measured = {}
    for frame in decode_frames(source_file, table, indices, timeout_s=90):
        row = bound[frame.index]
        digest = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
        require(
            frame.pts == row["pts"] and digest == row["decoded_rgb_sha256"] == expected["rgb"][frame.index],
            "Native RGB or clock differs",
        )
        measured[frame.index] = {
            "frame_index": frame.index,
            "rgb": frame.rgb,
            "bbox": row["bbox"],
            "landmarks": row["landmarks"],
        }
    require(list(measured) == indices, "Every selected source frame must be decoded")
    return prior.diagnostic.measure(
        {role: [measured[index] for index in groups[role]] for role in prior.diagnostic.ROLES}
    )


def identity():
    require(
        read(resume.PUBLIC) == read(resume.TRIAL / "freeze.json") == resume.identity(),
        "Exact v227 freeze required",
    )
    failed = read(FAILED)
    require(
        failed["schema"] == "chipmunk.video-webcam-confound-resume-result/227"
        and failed["status"] == "complete"
        and failed["freeze_sha256"] == sha(resume.PUBLIC)
        and failed["model_calls"] == 0
        and set(failed["roles"]) == set(prior.prior.ROLES)
        and all(
            row == {"status": "unavailable", "reason": "source_or_archive_binding_failed"}
            for row in failed["roles"].values()
        ),
        "Preserve and bind the original v227 readout",
    )
    return {
        "schema": "chipmunk.video-webcam-source-geometry-freeze/231",
        "v227_freeze_sha256": sha(resume.PUBLIC),
        "v227_failed_readout_sha256": sha(FAILED),
        "sequence_source_geometry_fields": ["width", "height"],
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "model_calls_permitted": False,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve previous attempts")
    locked = identity()
    TRIAL.mkdir(mode=0o700)
    for path in (TRIAL / "freeze.json", PUBLIC):
        with path.open("x") as stream:
            json.dump(locked, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return locked


def readout():
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == identity(), "Exact source geometry freeze required"
    )
    result = scoped(prior.readout, {"select_record": resume.select_record, "assess": assess})()
    return {
        **result,
        "schema": "chipmunk.video-webcam-source-geometry-result/231",
        "freeze_sha256": sha(PUBLIC),
        "v227_failed_readout_sha256": sha(FAILED),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "readout"))
    print(
        json.dumps(
            {"freeze": freeze, "readout": readout}[parser.parse_args().action](), indent=2, allow_nan=False
        )
    )
