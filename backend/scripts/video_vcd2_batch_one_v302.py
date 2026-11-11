"""Evaluate one-face batching against the current app on the fixed camera controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

from scripts import video_vcd2_current_app_v294 as current
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL, ARCHIVE = current.ROOT, current.BASE, current.PANEL, current.ARCHIVE
read, write, sha, require = current.read, current.write, current.sha, current.require
NAME = "video-vcd2-batch-one-v302"
MODULE = "scripts.video_vcd2_batch_one_v302"
TRIAL, WORK, SEALED = BASE / NAME, BASE / f"{NAME}-work", ARCHIVE / NAME
PUBLIC = PANEL / f"{NAME}-freeze.json"
REFERENCE = PANEL / "video-vcd2-current-app-v294-complete.json"
REFERENCE_FREEZE_SHA = "476e1e980ab0d6df14b437a96bcde4aca03482f196e6410344892cfccbdb02bd"
PRIOR = {
    "video-frame-full-app-complete-v171.json": "d78fc77278f16ca6359812dc8755cca1cdd7bbcf671161c9e3dd3366ca1c0130",
    "video-frame-full-app-complete-v175.json": "2e664c41669277a724ebc8f8b54429307d00b404d6a483fbece96b0a729709a0",
}
ARM = "batch-one"
ADAPTER = ROOT / "backend/scripts/video_frame_check_batch_one.py"
CODE = (
    Path(__file__),
    ADAPTER,
    Path(current.APP.__file__),
    ROOT / "backend/tests/test_video_vcd2_batch_one_v302.py",
    ROOT / f"docs/{NAME}-method.md",
)


def identity():
    reference, plan = current.verify_plan()
    require(sha(current.PUBLIC) == REFERENCE_FREEZE_SHA, "Exact v294 app freeze required")
    require(
        all(sha(PANEL / name) == digest for name, digest in PRIOR.items()),
        "Exact prior batching trials required",
    )
    require(
        all(case["status"] == "prepared" for case in plan.values()), "All 24 original inputs must be eligible"
    )
    manifest = current.APP.candidate_manifest(current.MANIFEST.read_bytes())
    return (
        {
            **reference,
            "schema": "chipmunk.video-vcd2-batch-one-freeze/302",
            "reference_freeze_sha256": REFERENCE_FREEZE_SHA,
            "reference_manifest_sha256": reference["app_manifest_sha256"],
            "app_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "code_sha256": {
                **reference["code_sha256"],
                **{str(path.relative_to(ROOT)): sha(path) for path in CODE},
            },
            "prior_batch_trials": PRIOR,
            "batch_faces": 1,
        },
        plan,
        manifest,
    )


def freeze():
    require(not any(path.exists() for path in (TRIAL, WORK, SEALED, PUBLIC)), "Preserve prior attempts")
    frozen, plan, manifest = identity()
    TRIAL.mkdir(mode=0o700)
    WORK.mkdir(mode=0o700)
    write(TRIAL / "private-plan.json", plan)
    for condition, row in plan.items():
        folder = TRIAL / condition
        folder.mkdir(mode=0o700)
        write(folder / "inputs.json", row)
        write(folder / "freeze.json", frozen)
        (folder / f"{ARM}.yaml").write_bytes(manifest)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    return {"status": "frozen", "freeze_sha256": sha(PUBLIC), "planned_cases": len(plan)}


def verify_plan():
    frozen, plan, _ = identity()
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == frozen and read(TRIAL / "private-plan.json") == plan,
        "Exact frozen batching candidate and input plan required",
    )
    return frozen, plan


verify = scoped(current.verify, {"TRIAL": TRIAL, "ARM": ARM, "verify_plan": verify_plan})
authored = scoped(current.authored, {"TRIAL": TRIAL, "ARM": ARM})
work_dir = scoped(current.work_dir, {"WORK": WORK, "ARM": ARM})
audit_path = scoped(current.audit_path, {"authored": authored})
archive_paths = scoped(current.archive_paths, {"SEALED": SEALED})
active_case = scoped(current.active_case, {"TRIAL": TRIAL})
qualification = scoped(
    current.qualification, {"TRIAL": TRIAL, "WORK": WORK, "ARM": ARM, "__file__": __file__}
)
evidence_version = scoped(current.evidence_version, {"work_dir": work_dir})
worker = scoped(current.worker, {"TRIAL": TRIAL, "WORK": WORK, "ARM": ARM, "verify": verify})
review = scoped(
    current.review,
    {"TRIAL": TRIAL, "verify": verify, "authored": authored, "audit_path": audit_path, "MODULE": MODULE},
)
seal = scoped(
    current.seal,
    {
        "TRIAL": TRIAL,
        "SEALED": SEALED,
        "ARM": ARM,
        "verify": verify,
        "authored": authored,
        "audit_path": audit_path,
        "archive_paths": archive_paths,
        "active_case": active_case,
        "work_dir": work_dir,
    },
)
release = scoped(
    current.release,
    {
        "verify": verify,
        "authored": authored,
        "archive_paths": archive_paths,
        "active_case": active_case,
        "work_dir": work_dir,
    },
)
run_case = scoped(
    current.run,
    {
        "verify_plan": verify_plan,
        "verify": verify,
        "active_case": active_case,
        "WORK": WORK,
        "ARM": ARM,
        "authored": authored,
        "archive_paths": archive_paths,
        "MODULE": MODULE,
        "__file__": __file__,
    },
)


def coverage(case):
    faces = case["input"]["face_frame_indices"]
    scored = case["target_scored_frames"]
    require(
        len(faces) == len(set(faces)) and 0 <= scored <= len(faces),
        "Frame coverage exceeds native face frames",
    )
    linked = all(
        case[key]
        for key in (
            "target_rows_bound",
            "target_raw_verified",
            "target_signal_published",
            "target_series_published",
        )
    )
    if case["target_status"] == "ok":
        require(
            scored == len(faces) > 0 and linked, "Successful frame coverage must be fully linked to evidence"
        )
    else:
        require(scored == 0, "Skipped or failed frame checks must not supply scores")
    return {
        "face_frames": len(faces),
        "scored_frames": scored,
        "status": case["target_status"] or "not_run",
        "budget_skip": case["target_status"] == "skipped" and case["target_reason"] == "memory budget",
    }


def audit_case(condition):
    result = scoped(
        current.audit_case,
        {
            "TRIAL": TRIAL,
            "verify_plan": verify_plan,
            "verify": verify,
            "qualification": qualification,
            "work_dir": work_dir,
            "evidence_version": evidence_version,
        },
    )(condition)
    path = work_dir(condition) / "case.json"
    if path.is_file():
        result["frame_coverage"] = coverage(read(current.raw.direct_file(path)))
    return result


def reference_seals():
    outcome = read(current.raw.direct_file(REFERENCE))
    require(
        outcome["schema"] == "chipmunk.video-vcd2-current-app-result/294"
        and outcome["status"] == "complete"
        and outcome["freeze_sha256"] == REFERENCE_FREEZE_SHA
        and outcome["sealed_cases"] == len(current.CONDITIONS)
        and 0 <= outcome["verified_sealed_uploads"] <= outcome["sealed_cases"],
        "Complete verified reference panel required",
    )
    seals = {}
    for condition in current.CONDITIONS:
        path, adjacent = current.archive_paths(condition)
        receipt = read(current.raw.direct_file(adjacent))
        require(
            receipt["archive_path"] == str(path)
            and receipt["stage_path"] == str(current.authored(condition)),
            "Reference archive ownership differs",
        )
        seals[condition] = {"receipt_sha256": sha(adjacent), "archive_sha256": receipt["archive_sha256"]}
    require(current.digest(seals) == outcome["archive_set_commitment_sha256"], "Reference seal set differs")
    return seals


def run(condition):
    require(
        not (current.TRIAL / "active.lock").exists(),
        "Wait for the reference worker to release its local lease",
    )
    reference_seals()
    return run_case(condition)


def preflight():
    frozen, plan, _ = identity()
    return {
        "status": "ready_for_candidate"
        if REFERENCE.is_file() and reference_seals()
        else "waiting_for_complete_reference",
        "planned_cases": len(plan),
        "reference_manifest_sha256": frozen["reference_manifest_sha256"],
        "candidate_manifest_sha256": frozen["app_manifest_sha256"],
        "batch_faces": frozen["batch_faces"],
        "new_neural_calls": 0,
    }


def sealed_json(paths, member_name):
    archive, receipt_path = paths
    receipt = read(current.raw.direct_file(receipt_path))
    entries = [row for row in receipt["entries"] if row["path"] == member_name]
    require(
        len(entries) == 1 and entries[0]["kind"] == "file" and 0 < entries[0]["size"] <= 8 * 1024**2,
        "Exact bounded sealed case record required",
    )
    with tarfile.open(current.raw.direct_file(archive), "r:") as stream:
        entry = stream.getmember(member_name)
        require(entry.isfile() and entry.size == entries[0]["size"], "Unexpected sealed JSON header")
        with stream.extractfile(entry) as payload:
            data = payload.read(entry.size + 1)
    require(
        len(data) == entry.size and hashlib.sha256(data).hexdigest() == entries[0]["sha256"],
        "Sealed JSON differs from the committed bytes",
    )
    return json.loads(data)


def matched(reference, candidate):
    return all(
        reference[key] == candidate[key]
        for key in ("input", "condition_profile", "fuser_sha256", "fusion_mode", "non_target_runs")
    )


def compare_cases():
    reference_seals()
    counts, directions, review_flags = Counter(), Counter(), Counter()
    for condition in current.CONDITIONS:
        paths = archive_paths(condition)
        if not paths[1].is_file():
            continue
        counts["candidate_sealed"] += 1
        audit = sealed_json(paths, "stage/audit.json")
        if audit["status"] != "scored":
            counts["candidate_unavailable"] += 1
            continue
        old_paths = current.archive_paths(condition)
        old_audit = sealed_json(old_paths, "stage/audit.json")
        if old_audit["status"] != "scored":
            counts["reference_unavailable"] += 1
            continue
        reference, candidate = (sealed_json(pair, "work/case.json") for pair in (old_paths, paths))
        old, new = coverage(reference), coverage(candidate)
        require(new == audit["frame_coverage"], "Sealed candidate coverage differs")
        counts["matched_inputs_and_other_checks"] += matched(reference, candidate)
        counts["reference_face_frames"] += old["face_frames"]
        counts["candidate_face_frames"] += new["face_frames"]
        counts["reference_scored_frames"] += old["scored_frames"]
        counts["candidate_scored_frames"] += new["scored_frames"]
        counts["restored_after_budget_skip"] += old["budget_skip"] and new["status"] == "ok"
        left, right = (record["customer_assessment"]["direction"] for record in (reference, candidate))
        directions[f"{left}_to_{right}"] += 1
        before, after = (
            "flagged" if item.get("face_review_intervals") else "clear" for item in (old_audit, audit)
        )
        if all(item.get("appearance_check_status") == "ok" for item in (old_audit, audit)):
            review_flags[f"{before}_to_{after}"] += 1
        else:
            review_flags["appearance_unassessed"] += 1
    return {
        "coverage": dict(counts),
        "customer_direction_transitions": dict(directions),
        "paired_face_review_transitions": dict(review_flags),
    }


def readout():
    report = scoped(
        current.readout,
        {
            "verify_plan": verify_plan,
            "archive_paths": archive_paths,
            "audit_path": audit_path,
            "authored": authored,
            "work_dir": work_dir,
            "ARM": ARM,
            "PUBLIC": PUBLIC,
        },
    )()
    return {
        **report,
        "schema": "chipmunk.video-vcd2-batch-one-result/302",
        "lab_batch_override": True,
        "reference_result_sha256": sha(REFERENCE),
        **compare_cases(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("preflight", "freeze", "readout"):
        commands.add_parser(action)
    for action in ("run", "review", "seal", "release", "_worker", "_audit"):
        command = commands.add_parser(action)
        command.add_argument("condition", choices=current.CONDITIONS)
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        worker(args.condition, args.output)
    else:
        function = audit_case if args.command == "_audit" else globals()[args.command]
        result = function(args.condition) if hasattr(args, "condition") else function()
        if result is not None:
            print(json.dumps(result, indent=2, allow_nan=False))
