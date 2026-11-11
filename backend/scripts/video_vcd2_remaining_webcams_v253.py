"""Evaluate the two previously unrun webcam groups with the verified cache fix."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts import video_vcd2_verify_cache_replay_v251 as replay
from scripts.video_appearance_corrected_v112 import scoped

raw = replay.raw
ROOT, BASE, PANEL, ARM = raw.ROOT, raw.BASE, raw.PANEL, raw.ARM
read, sha, require = raw.read, raw.sha, raw.require
NAME = "video-vcd2-remaining-webcams-v253"
MODULE = "scripts.video_vcd2_remaining_webcams_v253"
TRIAL, WORK = BASE / NAME, BASE / "video-vcd2-remaining-webcams-work-v253"
SEALED = raw.ARCHIVE / NAME
PUBLIC = PANEL / f"{NAME}-freeze.json"
REPLAY_RESULT = PANEL / "video-vcd2-verify-cache-replay-v251-complete.json"
REPLAY_RESULT_SHA = "d44ac65d3e2c92efa7e435194e475c07b544fc2566ac98ec5212b130795c2f83"
REPLAY_SOURCE = f"backend/scripts/{replay.MODULE.split('.')[-1]}.py"
SLOTS = (11, 12)
CONDITIONS = tuple(raw.case_name(slot, role) for slot in SLOTS for role in raw.ROLES)
CODE = (
    f"backend/scripts/{MODULE.split('.')[-1]}.py",
    "backend/tests/test_video_vcd2_remaining_webcams_v253.py",
    f"docs/{NAME}-method.md",
)


def frozen_roster(old):
    full = read(raw.direct_file(raw.TRIAL / "private-plan.json"))
    require(
        tuple(full) == raw.CONDITIONS and replay.archive_set_commitment(full) == old["plan_sha256"],
        "Exact historical v202 roster required",
    )
    selected = {condition: full[condition] for condition in CONDITIONS}
    require(
        all(
            row["slot"] in SLOTS
            and row["role"] in raw.ROLES
            and raw.case_name(row["slot"], row["role"]) == condition
            and row["status"] == "prepared"
            and 0 < row["bytes"] <= raw.MAX_APP_VIDEO_BYTES
            for condition, row in selected.items()
        ),
        "The six original prepared controls are required",
    )
    return selected


def case_inventory():
    require(sha(raw.PUBLIC) == replay.OLD_FREEZE_SHA, "Exact published v202 freeze required")
    old = read(raw.PUBLIC)
    plan = frozen_roster(old)
    for condition, row in plan.items():
        prior_paths = (*raw.archive_paths(condition), raw.authored(condition), raw.work_dir(condition))
        require(
            not any(path.exists() or path.is_symlink() for path in prior_paths),
            "An earlier attempt exists for a reserved control",
        )
        replay.same_input(row, old)
    return plan


def identity(plan):
    old, _, _, historical = replay.historical()
    require(plan == frozen_roster(old), "The successor must keep the six frozen inputs")
    result = read(REPLAY_RESULT)
    replay_source_sha = sha(ROOT / REPLAY_SOURCE)
    require(
        sha(REPLAY_RESULT) == REPLAY_RESULT_SHA
        and result["freeze_sha256"] == sha(replay.PUBLIC)
        and replay_source_sha == read(replay.PUBLIC)["code_sha256"][REPLAY_SOURCE]
        and result["replay_receipt_sha256"] == sha(raw.direct_file(replay.archive_paths()[1]))
        and result["same_sealed_input"] is result["affected_routes_recovered"] is True,
        "Exact successful cache-fix replay required",
    )
    runtime = replay.runtime_bindings(old, patched=True)
    runtime["app_manifest_sha256"] = runtime.pop("model_manifest_sha256")
    return {
        "schema": "chipmunk.video-vcd2-remaining-webcams-freeze/253",
        **historical,
        **runtime,
        "cache_fix_result_sha256": REPLAY_RESULT_SHA,
        "patched_fetchers_sha256": replay.PATCHED_FETCHERS,
        "planned_cases": len(plan),
        "planned_slots": len(SLOTS),
        "roles": list(raw.ROLES),
        "plan_sha256": replay.archive_set_commitment(plan),
        "code_sha256": {**{name: sha(ROOT / name) for name in CODE}, REPLAY_SOURCE: replay_source_sha},
        "all_face_swap_negatives": True,
        "serving_change_permitted": False,
    }


def preflight():
    return scoped(raw.preflight, {"case_inventory": case_inventory, "identity": identity})()


def freeze():
    require(not SEALED.exists(), "Preserve every earlier archive attempt")
    return scoped(
        raw.freeze,
        {
            "case_inventory": case_inventory,
            "identity": identity,
            "TRIAL": TRIAL,
            "WORK": WORK,
            "PUBLIC": PUBLIC,
        },
    )()


def verify_plan():
    return scoped(
        raw.verify_plan, {"identity": identity, "TRIAL": TRIAL, "PUBLIC": PUBLIC, "CONDITIONS": CONDITIONS}
    )()


def verify(condition):
    require(condition in CONDITIONS, "One of the six fixed controls is required")
    return scoped(raw.verify, {"verify_plan": verify_plan, "TRIAL": TRIAL})(condition)


def authored(condition):
    return TRIAL / condition / ARM / "authored"


def work_dir(condition):
    return WORK / condition / ARM


def audit_path(condition):
    return authored(condition) / "audit.json"


def archive_paths(condition):
    archive = SEALED / f"{condition}.tar"
    return archive, archive.with_suffix(".tar.receipt.json")


def active_case():
    return scoped(raw.active_case, {"TRIAL": TRIAL})()


def run(condition):
    _, _, bound = verify(condition)
    with active_case():
        require(
            not authored(condition).exists()
            and not any(WORK.glob(f"*/{ARM}"))
            and all(not authored(case).exists() or archive_paths(case)[1].is_file() for case in CONDITIONS),
            "Preserve attempts and seal and release the existing local case first",
        )
        free, needed = shutil.disk_usage(BASE).free, raw.disk_required(bound["bytes"])
        if free < needed:
            return {"status": "pending_disk_headroom", "free_bytes": free, "required_bytes": needed}
        stage = authored(condition)
        stage.parent.mkdir(exist_ok=True)

        def launch(argv, **kwargs):
            output = stage / "output"
            require(
                argv[3:] == ["scripts.video_appearance_native_v108", "authored", str(output)],
                "Exact guarded worker invocation required",
            )
            return subprocess.Popen([*argv[:3], MODULE, "_worker", condition, str(output)], **kwargs)

        scoped(
            raw.prior.original.runtime.run,
            {
                "TRIAL": stage.parent,
                "verify": lambda: verify(condition),
                "prerequisites": lambda *_: True,
                "__file__": __file__,
                "subprocess": SimpleNamespace(
                    Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
                ),
            },
        )("authored")
    return None


def worker(condition, output):
    return scoped(raw.worker, {"TRIAL": TRIAL, "WORK": WORK, "verify": verify})(condition, output)


def qualification(condition):
    return scoped(raw.qualification, {"TRIAL": TRIAL, "WORK": WORK, "__file__": __file__})(condition)


def audit_case(condition):
    raw.edge.enforce_read_only()
    return scoped(raw.audit_case, {"TRIAL": TRIAL, "verify": verify, "qualification": qualification})(
        condition
    )


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


def paired_flag_changes(by_slot):
    return raw.paired_flag_changes(
        {
            slot: [
                row
                if row.get("appearance_route_status") == "ok"
                and row.get("appearance_check_status") == "ok"
                and row.get("appearance_measured_windows", 0) > 0
                else {**row, "status": "comparison_unavailable"}
                for row in rows
            ]
            for slot, rows in by_slot.items()
        }
    )


def verified_seal(condition, sealed, row):
    stage = authored(condition)
    entries = {entry["path"]: entry for entry in sealed["entries"] if entry["kind"] == "file"}
    require(
        sealed["work_path"] == str(work_dir(condition))
        and all(
            entries[f"stage/{name}"]["sha256"] == sha(raw.direct_file(stage / name))
            for name in ("inputs.json", "freeze.json", f"{ARM}.yaml", "profile.json", "worker.log")
        )
        and read(stage / "inputs.json") == row
        and sha(stage / "freeze.json") == sha(PUBLIC)
        and sha(stage / f"{ARM}.yaml") == read(PUBLIC)["app_manifest_sha256"],
        "Sealed stage must bind the frozen inputs and runtime",
    )
    uploaded = replay.has_uploaded_bytes(entries.values(), row["bytes"], row["sha256"])
    require(
        read(audit_path(condition))["status"] != "scored" or uploaded, "Scored seal lacks the exact upload"
    )
    return uploaded, replay.collision_log((stage / "worker.log").read_text())


def readout():
    frozen, plan = verify_plan()
    checks = {}

    def checked_audit_path(condition):
        # The inherited readout verifies the full tar before it calls this hook.
        _, receipt = archive_paths(condition)
        checks[condition] = verified_seal(condition, read(raw.direct_file(receipt)), plan[condition])
        return audit_path(condition)

    aggregate = scoped(
        raw.readout,
        {
            "verify_plan": lambda: (frozen, plan),
            "archive_paths": archive_paths,
            "audit_path": checked_audit_path,
            "authored": authored,
            "SLOTS": SLOTS,
            "PUBLIC": PUBLIC,
            "paired_flag_changes": paired_flag_changes,
        },
    )()
    return {
        **aggregate,
        "schema": "chipmunk.video-vcd2-remaining-webcams-result/253",
        "historical_result_sha256": frozen["historical_result_sha256"],
        "cache_fix_result_sha256": frozen["cache_fix_result_sha256"],
        "verified_sealed_uploads": sum(uploaded for uploaded, _ in checks.values()),
        "sealed_cases_with_historical_cache_collision_signature": sum(
            collision for _, collision in checks.values()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "freeze", "readout"):
        commands.add_parser(name)
    for name in ("run", "review", "seal", "release", "_worker", "_audit"):
        operation = commands.add_parser(name)
        operation.add_argument("condition", choices=CONDITIONS)
        if name == "_worker":
            operation.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        return worker(args.condition, args.output)
    command = audit_case if args.command == "_audit" else globals()[args.command]
    result = command(args.condition) if hasattr(args, "condition") else command()
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
