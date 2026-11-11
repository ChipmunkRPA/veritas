"""Full-app comparison for nine fixed v206 delivered webcam controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts import video_vcd2_full_app_v202 as raw
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = raw.ROOT, raw.BASE, raw.PANEL
TRIAL, WORK = BASE / "video-vcd2-delivery-app-v232", BASE / "video-vcd2-delivery-app-work-v232"
SEALED = raw.ARCHIVE / "video-vcd2-delivery-app-v232"
DELIVERY = raw.ARCHIVE / "video-vcd2-delivery-v206"
PREP_PUBLIC = PANEL / "video-vcd2-delivery-freeze-v206.json"
PREP_COMPLETE = PANEL / "video-vcd2-delivery-complete-v206.json"
RAW_RESULT = PANEL / "video-vcd2-full-app-first-triplet-v202.json"
PUBLIC = PANEL / "video-vcd2-delivery-app-freeze-v232.json"
MODULE = "scripts.video_vcd2_delivery_app_v232"
SLOTS, ROLES, ARM = (1, 2, 6), raw.ROLES, raw.ARM
CONDITIONS = tuple(f"slot-{slot:02d}-{role}" for slot in SLOTS for role in ROLES)
CODE = ("backend/scripts/video_vcd2_delivery_app_v232.py", "backend/tests/test_video_vcd2_delivery_app_v232.py",
        "docs/video-vcd2-delivery-app-v232-method.md")
read, write, sha, require = raw.read, raw.write, raw.sha, raw.require


def changes(left, right):
    require(len(left) == len(right), "Equal source frame counts required")
    return [i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]


def check_changes(records, indices, pts):
    require([row["index"] for row in records] == indices and all(
        type(row["index"]) is int and row["pts"] == pts[row["index"]] and 0 < row["rgb_mae"] <= 255
        for row in records), "Exact per-frame delivery difference receipt required")


def qualification(slot, entry, selection, freeze_hash):
    """Validate raw and derivative source/clock and all receipt-level RGB relations."""
    folder = DELIVERY / f"slot-{slot:02d}-attempt-01"
    path = raw.direct_file(folder / "qualification.json")
    doc = read(path)
    raw_path = raw.PREPARATION / selection["attempt"] / "qualification.json"
    original = read(raw.direct_file(raw_path))
    raw.qualify_slot(entry, original)
    raw.qualify_slot(entry, doc)
    require(doc["schema"] == "chipmunk.video-vcd2-delivery-qualification/206"
            and doc["delivery_attempt"] == 1 and doc["freeze_sha256"] == freeze_hash
            and doc["v200_qualification_sha256"] == selection["qualification_sha256"] == sha(raw_path)
            and all(doc[k] == original[k] for k in ("slot", "source_sha256", "frame_count", "width", "height",
                "source_origin_pts", "time_base", "relative_pts", "last_duration", "frame_interval", "source_color"))
            and set(doc["cases"]) == set(ROLES), "Exact delivery/source qualification required")
    cases, original_cases, pts = doc["cases"], original["cases"], doc["relative_pts"]
    baseline = cases["unchanged"]["rgb_sha256"]
    for role in ROLES:
        case, old = cases[role], original_cases[role]
        raw_hashes, hashes = old["rgb_sha256"], case["rgb_sha256"]
        require(case["input_sha256"] == old["sha256"] == selection["cases"][role]["sha256"]
                and case["bytes"] > 0 and case["status"] in {"prepared", "unsupported_upload_size"}
                and (case["status"] == "prepared") == (case["bytes"] <= raw.MAX_APP_VIDEO_BYTES),
                "Exact delivery input or upload eligibility differs")
        paired = changes(baseline, hashes)
        prior_paired = changes(original_cases["unchanged"]["rgb_sha256"], raw_hashes)
        check_changes(case["changed_frames"], paired, pts)
        check_changes(case["delivery_encode_changed_frames"], changes(raw_hashes, hashes), pts)
        require(paired == prior_paired and (role == "unchanged" or bool(paired))
                and case["suppressed_pair_change_indices"] == [],
                "Delivery did not preserve the paired-control relation")
    old_lossy = changes(original_cases[ROLES[1]]["rgb_sha256"], original_cases[ROLES[2]]["rgb_sha256"])
    new_lossy = changes(cases[ROLES[1]]["rgb_sha256"], cases[ROLES[2]]["rgb_sha256"])
    require(bool(old_lossy) and new_lossy == old_lossy and doc["lossy_codec_changed_indices"] == new_lossy
            and doc["suppressed_lossy_codec_change_indices"] == [],
            "Delivery codec contrast differs")
    return doc, sha(path)


def source_plan(*, verify_raw_archives=False):
    """Bind all v206 published receipt commitments and the nine planned files."""
    raw.require_fresh()
    v202_lock, raw_plan = raw.verify_plan()
    publication, completed, locked = read(PREP_PUBLIC), read(PREP_COMPLETE), read(DELIVERY / "freeze.json")
    require(completed["schema"] == "chipmunk.video-vcd2-delivery-completion/206"
            and completed["status"] == "complete" and completed["source_recordings"] == 12
            and completed["prepared_controls"] == completed["within_upload_limit"] == 36
            and completed["new_neural_calls"] == 0 and sha(DELIVERY / "freeze.json")
            == completed["private_freeze_sha256"] == publication["private_freeze_sha256"]
            and locked["identity"]["schema"] == "chipmunk.video-vcd2-delivery/206"
            and locked["identity"]["v200_freeze_sha256"] == sha(raw.PREPARATION / "freeze.json")
            == v202_lock["preparation_freeze_sha256"]
            and locked["identity"]["v200_publication_sha256"] == sha(PANEL / "video-vcd2-controls-freeze-v200.json")
            and locked["identity"]["tool_sha256"] == read(raw.PREPARATION / "freeze.json")["identity"]["tool_sha256"]
            and all(publication.get(k) == v for k, v in locked["identity"].items())
            and all(sha(ROOT / name) == digest for name, digest in locked["identity"]["code_sha256"].items())
            and [entry["slot"] for entry in locked["slots"]] == list(range(1, 13)),
            "Exact complete v206 publication and frozen method required")
    receipt_hashes = [sha(raw.direct_file(DELIVERY / f"slot-{i:02d}-attempt-01/qualification.json"))
                      for i in range(1, 13)]
    commitment = hashlib.sha256(json.dumps(receipt_hashes, separators=(",", ":")).encode()).hexdigest()
    require(commitment == completed["ordered_private_qualification_commitments_sha256"],
            "Published ordered delivery receipt commitment differs")
    roster = raw.prepared_roster()
    plan = {}
    for slot in SLOTS:
        entry, frozen_slot = roster[slot - 1], locked["slots"][slot - 1]
        require(entry["slot"] == slot and frozen_slot["slot"] == slot and frozen_slot["status"] == "prepared"
                and frozen_slot["selected"] is not None, "Frozen delivery source unavailable")
        doc, digest = qualification(slot, entry, frozen_slot["selected"], completed["private_freeze_sha256"])
        require(digest == receipt_hashes[slot - 1], "Delivery receipt differs from published commitment")
        for role in ROLES:
            item = doc["cases"][role]
            path = raw.direct_file(DELIVERY / f"slot-{slot:02d}-attempt-01" / item["file"])
            require(path.stat().st_size == item["bytes"] and item["status"] == "prepared", "Delivered upload unavailable")
            condition = f"slot-{slot:02d}-{role}"
            plan[condition] = {"slot": slot, "role": role, "status": "prepared", "qualification_sha256": digest,
                "sha256": item["sha256"], "bytes": item["bytes"], "v200_input_sha256": item["input_sha256"]}
    # Verify the full raw archives at freeze and readout. Workers only bind the
    # published receipts and audits because their inputs are the delivered files.
    from scripts.video_vcd2_case_archive_v202 import verify_archive
    anchors = {}
    for condition in CONDITIONS[:3]:
        tar, receipt = raw.archive_paths(condition)
        sealed, audited = read(raw.direct_file(receipt)), read(raw.direct_file(raw.audit_path(condition)))
        if verify_raw_archives:
            verify_archive(tar, sealed)
        require(raw.direct_file(tar) == tar and sealed["archive_path"] == str(tar)
                and sealed["stage_path"] == str(raw.authored(condition)) and audited["condition"] == condition
                and audited["status"] == "scored" and audited["role"] == raw_plan[condition]["role"]
                and raw_plan[condition]["sha256"] == plan[condition]["v200_input_sha256"]
                and [r["sha256"] for r in sealed["entries"] if r["path"] == "stage/audit.json"] == [sha(raw.audit_path(condition))],
                "Exact sealed slot-one raw anchor required")
        anchors[condition] = {"receipt_sha256": sha(receipt), "archive_sha256": sealed["archive_sha256"],
                              "audit_sha256": sha(raw.audit_path(condition))}
    raw_result = read(RAW_RESULT)
    raw_seals = {
        condition: {key: anchor[key] for key in ("receipt_sha256", "archive_sha256")}
        for condition, anchor in anchors.items()
    }
    raw_commitment = hashlib.sha256(json.dumps(raw_seals, sort_keys=True).encode()).hexdigest()
    require(raw_result["schema"] == "chipmunk.video-vcd2-full-app-result/202"
            and raw_result["sealed_cases"] == len(anchors) == 3
            and raw_result["freeze_sha256"] == sha(raw.PUBLIC)
            and raw_result["archive_set_commitment_sha256"] == raw_commitment, "Published raw anchor differs")
    raw.require_fresh()
    return v202_lock, plan, anchors


def identity(plan, anchors):
    old = read(raw.PUBLIC)
    return {"schema": "chipmunk.video-vcd2-delivery-app-freeze/232", "planned_cases": 9, "planned_slots": 3,
            "roles": list(ROLES), "v202_freeze_sha256": sha(raw.PUBLIC), "v202_raw_anchor_sha256": sha(RAW_RESULT),
            "v206_freeze_sha256": sha(PREP_PUBLIC), "v206_complete_sha256": sha(PREP_COMPLETE),
            "raw_anchors_sha256": hashlib.sha256(json.dumps(anchors, sort_keys=True).encode()).hexdigest(),
            "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
            "code_sha256": {path: sha(ROOT / path) for path in CODE},
            "app_manifest_sha256": old["app_manifest_sha256"], "model_manager_budget_bytes": old["model_manager_budget_bytes"],
            "device": old["device"], "all_face_swap_negatives": True, "serving_change_permitted": False}


def freeze():
    require(not any(path.exists() for path in (TRIAL, WORK, PUBLIC)), "Preserve every earlier attempt")
    _, plan, anchors = source_plan(verify_raw_archives=True)
    frozen = identity(plan, anchors)
    TRIAL.mkdir(mode=0o700)
    WORK.mkdir(mode=0o700)
    write(TRIAL / "private-plan.json", plan)
    for condition, row in plan.items():
        folder = TRIAL / condition
        folder.mkdir(mode=0o700)
        write(folder / "inputs.json", row)
        write(folder / "freeze.json", frozen)
        (folder / f"{ARM}.yaml").write_bytes((raw.prior.runtime.TRIAL / f"{ARM}.yaml").read_bytes())
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    return {"status": "frozen", "planned_cases": len(plan)}


def verify_plan(*, verify_raw_archives=False):
    _, plan, anchors = source_plan(verify_raw_archives=verify_raw_archives)
    frozen = identity(plan, anchors)
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == frozen and read(TRIAL / "private-plan.json") == plan,
            "Exact frozen delivery-app plan required")
    return frozen, plan


def verify(condition):
    require(condition in CONDITIONS, "Exact delivery case required")
    frozen, plan = verify_plan()
    row, folder = plan[condition], TRIAL / condition
    require(read(folder / "inputs.json") == row and read(folder / "freeze.json") == frozen
            and sha(folder / f"{ARM}.yaml") == frozen["app_manifest_sha256"], "Case or serving manifest differs")
    file = raw.direct_file(DELIVERY / f"slot-{row['slot']:02d}-attempt-01" / f"{row['role']}.mp4")
    require(file.stat().st_size == row["bytes"] and sha(file) == row["sha256"], "Delivered upload bytes differ")
    return frozen, file, {"file_sha256": row["sha256"], "bytes": row["bytes"]}


def expected_case(row):
    path = raw.direct_file(DELIVERY / f"slot-{row['slot']:02d}-attempt-01/qualification.json")
    require(sha(path) == row["qualification_sha256"], "Delivery qualification differs")
    doc = read(path)
    case = doc["cases"][row["role"]]
    return {"time_base": doc["time_base"], "pts": doc["relative_pts"], "size": (doc["width"], doc["height"]),
            "rgb": case["rgb_sha256"], "ordinary": doc["frame_interval"],
            "changed": [item["index"] for item in case["changed_frames"]]}


def verify_input(outputs, row, digest):
    return scoped(raw.verify_input, {"expected_case": expected_case})(outputs, row, digest)


def worker(condition, output):
    return scoped(raw.worker, {"TRIAL": TRIAL, "WORK": WORK, "verify": verify, "verify_input": verify_input})(condition, output)


def qualification_run(condition):
    return scoped(raw.qualification, {"TRIAL": TRIAL, "WORK": WORK, "__file__": __file__})(condition)


def authored(condition):
    return TRIAL / condition / ARM / "authored"


def work_dir(condition):
    return WORK / condition / ARM


def audit_path(condition):
    return authored(condition) / "audit.json"


def archive_paths(condition):
    path = SEALED / f"{condition}.tar"
    return path, path.with_suffix(".tar.receipt.json")


def active_case():
    return scoped(raw.active_case, {"TRIAL": TRIAL})()


def run(condition):
    frozen, plan = verify_plan()
    require(condition in CONDITIONS and frozen["planned_cases"] == 9, "Exact delivery case required")
    row = plan[condition]
    with active_case():
        require(not any(WORK.glob(f"*/{ARM}")), "Archive and release the existing case first")
        free, needed = shutil.disk_usage(BASE).free, raw.disk_required(row["bytes"])
        if free < needed:
            return {"status": "pending_disk_headroom", "free_bytes": free, "required_bytes": needed}
        verify(condition)
        stage = authored(condition)
        stage.parent.mkdir(exist_ok=True)

        def launch(argv, **kwargs):
            output = stage / "output"
            require(argv[3:] == ["scripts.video_appearance_native_v108", "authored", str(output)], "Exact worker required")
            return subprocess.Popen([*argv[:3], MODULE, "_worker", condition, str(output)], **kwargs)

        scoped(raw.prior.original.runtime.run, {"TRIAL": stage.parent, "verify": lambda: verify(condition),
               "prerequisites": lambda *_: True, "subprocess": SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT,
               TimeoutExpired=subprocess.TimeoutExpired), "__file__": __file__})("authored")


def audit_case(condition):
    return scoped(raw.audit_case, {"TRIAL": TRIAL, "verify": verify, "qualification": qualification_run,
        "expected_case": expected_case, "verify_input": verify_input})(condition)


def review(condition):
    return scoped(raw.review, {"verify": verify, "authored": authored, "audit_path": audit_path,
                              "TRIAL": TRIAL, "MODULE": MODULE})(condition)


def seal(condition):
    return scoped(raw.seal, {"verify": verify, "authored": authored, "audit_path": audit_path,
        "active_case": active_case, "archive_paths": archive_paths, "work_dir": work_dir,
        "SEALED": SEALED, "TRIAL": TRIAL})(condition)


def release(condition):
    return scoped(raw.release, {"verify": verify, "authored": authored, "active_case": active_case,
        "archive_paths": archive_paths, "work_dir": work_dir})(condition)


def readout():
    verify_plan(verify_raw_archives=True)
    summarize = scoped(raw.readout, {"verify_plan": verify_plan, "archive_paths": archive_paths,
        "audit_path": audit_path, "authored": authored, "SLOTS": SLOTS, "PUBLIC": PUBLIC})()
    raw_rows = [read(raw.audit_path(condition)) for condition in CONDITIONS[:3]]
    delivered = []
    for condition in CONDITIONS[:3]:
        _, receipt = archive_paths(condition)
        delivered.append(read(audit_path(condition)) if receipt.is_file() else None)
    paired, by_role = Counter(), {}
    for old, new in zip(raw_rows, delivered, strict=True):
        if new is None or new["status"] != "scored":
            paired["comparison_unavailable"] += 1
            by_role[old["role"]] = {"status": "comparison_unavailable"}
        else:
            a, b = bool(old["face_review_intervals"]), bool(new["face_review_intervals"])
            transition = ("flagged" if a else "clear") + "_to_" + ("flagged" if b else "clear")
            paired[transition] += 1
            by_role[old["role"]] = {"status": "compared", "face_review_transition": transition,
                "raw_customer_direction": old["customer_direction"], "delivered_customer_direction": new["customer_direction"]}
    return {**summarize, "schema": "chipmunk.video-vcd2-delivery-app-result/232",
            "raw_slot_one_to_delivered_flag_changes": dict(paired), "raw_slot_one_comparison_by_role": by_role,
            "delivery_cohort_separate": True,
            "scope": "Nine delivered controls from three consumed webcam recordings; no swap sensitivity or population claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "readout"):
        sub.add_parser(name)
    for name in ("run", "review", "seal", "release", "_worker", "_audit"):
        command = sub.add_parser(name)
        command.add_argument("condition", choices=CONDITIONS)
        if name == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        worker(args.condition, args.output)
        return
    if args.command == "_audit":
        raw.edge.enforce_read_only()
        result = audit_case(args.condition)
    else:
        command = globals()[args.command]
        result = command(args.condition) if hasattr(args, "condition") else command()
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
