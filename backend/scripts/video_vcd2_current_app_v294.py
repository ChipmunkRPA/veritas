"""Run the current full app on the sealed v290 ordinary video controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts import video_vcd2_full_app_v202 as raw
from scripts import video_vcd2_remaining_webcams_v253 as later
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL, ARCHIVE = raw.ROOT, raw.BASE, raw.PANEL, raw.ARCHIVE
NAME = "video-vcd2-current-app-v294"
MODULE = "scripts.video_vcd2_current_app_v294"
TRIAL, WORK, SEALED = BASE / NAME, BASE / f"{NAME}-work", ARCHIVE / NAME
PUBLIC = PANEL / f"{NAME}-freeze.json"
PREPARATION_NAME = "video-vcd2-matched-controls-v290"
PREPARATION, PREPARATION_FREEZE = ARCHIVE / PREPARATION_NAME, PANEL / f"{PREPARATION_NAME}-freeze.json"
PREPARED = PREPARATION / "summary.json"
ARM = "reference"
SLOTS, ROLES = tuple(range(1, 9)), raw.ROLES
CONDITIONS = tuple(raw.case_name(slot, role) for slot in SLOTS for role in ROLES)
APP, BASELINE = raw.prior.original, raw.prior.original.baseline
MANIFEST = ROOT / "backend/app/models.yaml"
CAP = raw.MAX_APP_VIDEO_BYTES
PACKET_VERSION = "video-evidence-8"
BASELINE_SHA = "ec9032d3b79daad0997cb9c64a2a4d39522f7a4f34ae01f9abfe10a786467194"
HISTORICAL_SHA = "d8410ea6b3b9d7775b3a689125290a42eb22bd3ffc7b9a2aee5d5ba7303ea7fe"
HISTORICAL = PANEL / "video-vcd2-remaining-webcams-v253-complete.json"
read, write, sha, require = raw.read, raw.write, raw.sha, raw.require


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inventory(slots, directory, summary, prep_freeze_sha, receipt_plan):
    require([row["slot"] for row in slots] == list(SLOTS), "Exact eight prepared source groups required")
    plan, qualifications, frames = {}, [], Counter()
    for entry in slots:
        slot = entry["slot"]
        folder = directory / f"slot-{slot:02d}-attempt-01"
        path = raw.direct_file(folder / "qualification.json")
        document = read(path)
        receipt_plan(entry, document)
        qualifications.append(sha(path))
        for role in ROLES:
            item = document["cases"][role]
            size = raw.direct_file(folder / item["file"]).stat().st_size
            require(size > 0, "Empty raw control")
            frames[role] += item["verified_frames"]
            plan[raw.case_name(slot, role)] = {
                "slot": slot,
                "role": role,
                "status": "prepared" if size <= CAP else "unsupported_upload_size",
                "qualification_sha256": qualifications[-1],
                "sha256": item["sha256"],
                "bytes": size,
                "resolution": f"{document['width']}x{document['height']}",
            }
    require(
        tuple(plan) == CONDITIONS
        and summary["schema"] == "chipmunk.video-vcd2-matched-controls-result/290"
        and summary["status"] == "complete"
        and summary["freeze_sha256"] == prep_freeze_sha
        and summary["source_groups"] == len(SLOTS)
        and summary["controls"] == dict.fromkeys(ROLES, len(SLOTS))
        and summary["verified_frames"] == frames
        and summary["model_calls"] == 0
        and summary["private_receipt_bundle_sha256"]
        == hashlib.sha256(json.dumps(qualifications).encode()).hexdigest(),
        "Exact complete v290 roster and receipt bundle required",
    )
    return plan


def inspect_preparation():
    # The historical receipt tools import the decoder and its settings at module load.
    # Keep that import out of both the parent guard and the isolated app worker.
    from scripts import video_vcd2_matched_controls_v290 as preparation

    require(
        (preparation.TRIAL, preparation.PUBLIC, preparation.SLOTS)
        == (PREPARATION, PREPARATION_FREEZE, SLOTS),
        "Exact v290 preparation paths and slots required",
    )
    prepared = preparation.plan()
    preparation.admit(prepared)
    summary = read(raw.direct_file(PREPARED))
    return inventory(
        prepared["slots"], PREPARATION, summary, sha(PREPARATION_FREEZE), preparation.delivery.receipt_plan
    )


def case_inventory():
    return json.loads(
        subprocess.check_output(
            [sys.executable, "-B", "-m", MODULE, "_inventory"],
            cwd=ROOT / "backend",
            text=True,
            timeout=60,
        )
    )


def census(plan):
    sizes, states, groups = defaultdict(list), defaultdict(Counter), defaultdict(set)
    for row in plan.values():
        resolution = row["resolution"]
        sizes[resolution].append(row["bytes"])
        states[resolution][row["status"]] += 1
        if row["status"] != "prepared":
            groups[resolution].add(row["slot"])
    outcomes = Counter(row["status"] for row in plan.values())
    return {
        "planned_controls": len(plan),
        "max_upload_bytes": CAP,
        "upload_eligible": outcomes["prepared"],
        "upload_ineligible": outcomes["unsupported_upload_size"],
        "complete_upload_eligible_triplets": len(SLOTS) - sum(map(len, groups.values())),
        "by_resolution": {
            resolution: {
                "controls": len(sizes[resolution]),
                "maximum_bytes": max(sizes[resolution]),
                "upload_eligible": states[resolution]["prepared"],
                "upload_ineligible": states[resolution]["unsupported_upload_size"],
                "triplets_with_ineligible_controls": len(groups[resolution]),
            }
            for resolution in sorted(sizes)
        },
    }


def identity(plan):
    raw.require_fresh()
    old = read(BASELINE.PUBLIC)
    history = read(later.PUBLIC)
    require(
        sha(BASELINE.PUBLIC) == BASELINE_SHA
        and sha(HISTORICAL) == HISTORICAL_SHA
        and read(HISTORICAL)["freeze_sha256"] == sha(later.PUBLIC)
        and sha(APP.INVENTORY) == old["asset_inventory_sha256"] == history["model_inventory_sha256"]
        and sha(MANIFEST) == old["manifest_sha256"]
        and BASELINE.package_versions() == old["runtime_packages"],
        "Exact reference models, configuration, and offline runtime dependencies required",
    )
    BASELINE.checked_programs(old)
    code = (
        Path(__file__),
        Path(raw.__file__),
        Path(later.__file__),
        ROOT / "backend/scripts/video_faceswapai_full_app_v187.py",
        ROOT / "backend/scripts/video_vcd2_matched_controls_v290.py",
        Path(APP.__file__),
        Path(raw.prior.__file__),
        Path(raw.prior.runtime.__file__),
        Path(raw.prior.original.runtime.__file__),
        Path(raw.edge.__file__),
        Path(later.replay.__file__),
        Path(APP.serving.__file__),
        ROOT / "backend/scripts/video_vcd2_case_archive_v202.py",
        ROOT / "backend/scripts/video_appearance_corrected_v112.py",
        ROOT / "backend/scripts/local_probe_limits_v24.py",
        ROOT / "backend/scripts/audio_tree_sampler_v64.py",
        ROOT / "backend/scripts/grid_source_speech_v147.py",
        ROOT / "backend/tests/test_video_vcd2_current_app_v294.py",
        ROOT / f"docs/{NAME}-method.md",
    )
    return {
        "schema": "chipmunk.video-vcd2-current-app-freeze/294",
        "v290_freeze_sha256": sha(PREPARATION_FREEZE),
        "v290_complete_sha256": sha(PREPARED),
        "plan_sha256": digest(plan),
        "planned_cases": len(CONDITIONS),
        "planned_slots": len(SLOTS),
        "census": census(plan),
        "baseline_freeze_sha256": BASELINE_SHA,
        "historical_reference_sha256": HISTORICAL_SHA,
        "model_inventory_sha256": sha(APP.INVENTORY),
        "app_code_sha256": BASELINE.code_identity(),
        "app_manifest_sha256": sha(MANIFEST),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code},
        "evidence_packet_version": PACKET_VERSION,
        "device": history["device"],
        "model_manager_budget_bytes": history["model_manager_budget_bytes"],
        "resource_limits": history["resource_limits"],
        "profile": "full",
        "all_face_swap_negatives": True,
        "billing_mode": "disabled",
        "offline": True,
    }


def preflight():
    if not PREPARED.is_file():
        return {"status": "waiting_for_v290_seal", "planned_controls": len(CONDITIONS), "new_neural_calls": 0}
    plan = case_inventory()
    frozen = identity(plan)
    free = shutil.disk_usage(BASE).free
    return {
        "status": "ready_for_freeze",
        **frozen["census"],
        "available_internal_bytes": free,
        "prepared_fitting_current_disk": sum(
            row["status"] == "prepared" and free >= raw.disk_required(row["bytes"]) for row in plan.values()
        ),
        "new_neural_calls": 0,
    }


def freeze():
    require(
        not any(path.exists() for path in (TRIAL, WORK, SEALED, PUBLIC)), "Preserve every earlier attempt"
    )
    plan = case_inventory()
    frozen = identity(plan)
    manifest = MANIFEST.read_bytes()
    require(
        hashlib.sha256(manifest).hexdigest() == frozen["app_manifest_sha256"], "Customer manifest changed"
    )
    TRIAL.mkdir(mode=0o700)
    WORK.mkdir(mode=0o700)
    write(TRIAL / "private-plan.json", plan)
    for condition, row in plan.items():
        if row["status"] == "prepared":
            folder = TRIAL / condition
            folder.mkdir(mode=0o700)
            write(folder / "inputs.json", row)
            write(folder / "freeze.json", frozen)
            (folder / f"{ARM}.yaml").write_bytes(manifest)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    return {"status": "frozen", **frozen["census"]}


def verify_plan():
    plan = case_inventory()
    frozen = identity(plan)
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == frozen and read(TRIAL / "private-plan.json") == plan,
        "Exact current-app and source freeze required",
    )
    return frozen, plan


def verify(condition):
    return scoped(
        raw.verify, {"TRIAL": TRIAL, "ARM": ARM, "PREPARATION": PREPARATION, "verify_plan": verify_plan}
    )(condition)


def expected_case(row):
    return scoped(raw.expected_case, {"PREPARATION": PREPARATION})(row)


def verify_input(outputs, row, source_sha):
    return scoped(raw.verify_input, {"expected_case": expected_case})(outputs, row, source_sha)


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
    _, plan = verify_plan()
    if plan[condition]["status"] != "prepared":
        return {"status": plan[condition]["status"]}
    return scoped(
        later.run,
        {
            "verify": verify,
            "active_case": active_case,
            "WORK": WORK,
            "ARM": ARM,
            "CONDITIONS": CONDITIONS,
            "authored": authored,
            "archive_paths": archive_paths,
            "MODULE": MODULE,
            "__file__": __file__,
        },
    )(condition)


def worker(condition, output):
    # v190 supplies the nested isolated app initialization. Only the arm and new source are scoped.
    return scoped(
        raw.prior.worker,
        {"TRIAL": TRIAL, "WORK": WORK, "ARM": ARM, "verify": verify, "verify_input": verify_input},
    )(condition, output)


def qualification(condition):
    from scripts import video_faceswapai_full_app_v187 as first_app

    return scoped(first_app.qualification, {"TRIAL": TRIAL, "WORK": WORK, "ARM": ARM, "__file__": __file__})(
        condition
    )


def check_current_evidence(evidence, decimals):
    require(
        evidence["version"] == PACKET_VERSION and evidence["status"] == "ready",
        "The current app must commit current video evidence version 8",
    )
    verified = 0
    roles = ("before", "during", "after")
    keys = [f"Median eye spacing {role} the range" for role in roles]
    for packet in evidence["packets"]:
        sequence = packet.get("face_sequence") or {}
        measurements = packet.get("measurements", [])
        require(
            isinstance(measurements, list) and all(isinstance(row, dict) for row in measurements),
            "Current measurements must be recorded as a list",
        )
        tracked = [row for row in measurements if row.get("label") in keys]
        if sequence.get("method") != "face-appearance-change/1":
            require(not tracked, "Only appearance sequences can report this tracking context")
            continue
        by_index = {row["frame_index"]: row for row in sequence["frame_inputs"]}
        expected = []
        for label, role in zip(keys, roles, strict=True):
            samples = []
            for index in sorted(
                {center + delta for center in sequence["centers"][role] for delta in (-1, 0, 1)}
            ):
                frame = by_index[index]
                left, right = (tuple(round(v, decimals) for v in eye) for eye in frame["landmarks"][:2])
                width = round(frame["bbox"]["w"], decimals)
                require(width > 0, "Native tracking context requires a positive face width")
                samples.append(math.dist(left, right) / width)
            expected.append(
                {"label": label, "value": round(100 * median(samples), 2), "unit": "% of face-box width"}
            )
        require(tracked == expected, "Current tracking context differs from source-bound native landmarks")
        verified += 1
    return verified


def evidence_version(condition):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.core.canonical import FLOAT_DP
    from app.evidence.service import load_packet_set
    from app.storage.db import Analysis
    from app.storage.objects import ObjectStore

    work = work_dir(condition)
    db_path = raw.edge.database_path(work)
    before = sha(db_path)
    engine = create_engine(f"sqlite:///{db_path.as_uri()}?mode=ro&immutable=1&uri=true")
    try:
        with Session(engine) as db:
            analysis = db.scalars(select(Analysis)).one()
            packet_set = load_packet_set(db, analysis, ObjectStore(work / "data/objects"))
            require(packet_set is not None, "Expected the exact committed current packet set")
            count = check_current_evidence(packet_set, FLOAT_DP)
    finally:
        engine.dispose()
    require(sha(db_path) == before, "Current evidence audit changed the database")
    return count


def audit_case(condition):
    frozen_plan = verify_plan()

    def verify_frozen_case(case):
        return scoped(verify, {"verify_plan": lambda: frozen_plan})(case)

    raw.edge.enforce_read_only()
    result = scoped(
        raw.audit_case,
        {
            "TRIAL": TRIAL,
            "verify": verify_frozen_case,
            "qualification": qualification,
            "expected_case": expected_case,
            "verify_input": verify_input,
            "inspect_refinement": lambda *_: {"status": "not_requested", "crop_forwards": 0},
        },
    )(condition)
    case = read(work_dir(condition) / "case.json")
    result["legacy_frame_status"] = case["target_status"] or "not_run"
    result["legacy_frame_budget_skip"] = (
        case["target_status"] == "skipped" and case["target_reason"] == "memory budget"
    )
    if result["status"] == "scored":
        result["verified_alignment_context_packets"] = evidence_version(condition)
        result["evidence_packet_version"] = PACKET_VERSION
    return result


def review(condition):
    return scoped(
        raw.review,
        {"TRIAL": TRIAL, "verify": verify, "authored": authored, "audit_path": audit_path, "MODULE": MODULE},
    )(condition)


def seal(condition):
    return scoped(
        raw.seal,
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
    )(condition)


def release(condition):
    return scoped(
        raw.release,
        {
            "verify": verify,
            "authored": authored,
            "archive_paths": archive_paths,
            "active_case": active_case,
            "work_dir": work_dir,
        },
    )(condition)


def readout():
    frozen, plan = verify_plan()
    checked = {}

    def verified_audit(condition):
        _, path = archive_paths(condition)
        uploaded, collision = scoped(
            later.verified_seal,
            {
                "ARM": ARM,
                "PUBLIC": PUBLIC,
                "authored": authored,
                "audit_path": audit_path,
                "work_dir": work_dir,
            },
        )(condition, read(raw.direct_file(path)), plan[condition])
        row = read(audit_path(condition))
        require(
            row["status"] != "scored" or row["evidence_packet_version"] == PACKET_VERSION,
            "An audited scored case must retain current evidence validation",
        )
        checked[condition] = row, uploaded, collision
        return audit_path(condition)

    summary = scoped(
        raw.readout,
        {
            "verify_plan": lambda: (frozen, plan),
            "archive_paths": archive_paths,
            "audit_path": verified_audit,
            "authored": authored,
            "SLOTS": SLOTS,
            "PUBLIC": PUBLIC,
            "paired_flag_changes": later.paired_flag_changes,
        },
    )()
    return {
        **summary,
        "schema": "chipmunk.video-vcd2-current-app-result/294",
        "raw_upload_census": frozen["census"],
        "verified_sealed_uploads": sum(uploaded for _, uploaded, _ in checked.values()),
        "historical_cache_collision_signatures": sum(collision for _, _, collision in checked.values()),
        "evidence_version_8_verified": sum(
            row.get("evidence_packet_version") == PACKET_VERSION for row, _, _ in checked.values()
        ),
        "verified_alignment_context_packets": sum(
            row.get("verified_alignment_context_packets", 0) for row, _, _ in checked.values()
        ),
        "observed_legacy_frame_route_statuses": dict(
            Counter(
                row["legacy_frame_status"] for row, _, _ in checked.values() if "legacy_frame_status" in row
            )
        ),
        "legacy_frame_budget_skips": sum(
            row.get("legacy_frame_budget_skip", False) for row, _, _ in checked.values()
        ),
        "lab_batch_override": False,
        "additional_edge_replay_enabled": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("preflight", "freeze", "readout", "_inventory"):
        commands.add_parser(action)
    for action in ("run", "review", "seal", "release", "_worker", "_audit"):
        command = commands.add_parser(action)
        command.add_argument("condition", choices=CONDITIONS)
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        worker(args.condition, args.output)
    else:
        function = {"_audit": audit_case, "_inventory": inspect_preparation}.get(args.command) or globals()[
            args.command
        ]
        result = function(args.condition) if hasattr(args, "condition") else function()
        if result is not None:
            print(json.dumps(result, indent=2, allow_nan=False))
