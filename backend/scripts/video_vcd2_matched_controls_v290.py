"""Apply the exact v200 ordinary-control recipe to all eight v287 recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts import video_vcd2_controls_v200 as controls
from scripts import video_vcd2_delivery_v206 as delivery
from scripts import video_vcd2_unseen_controls_v287 as source
from scripts.video_appearance_corrected_v112 import scoped

BASE, ROLES = controls.BASE, controls.ROLES
NAME = "video-vcd2-matched-controls-v290"
TRIAL = BASE.ARCHIVE / NAME
PUBLIC = source.PANEL / f"{NAME}-freeze.json"
SOURCE_RESULT = source.PANEL / "video-vcd2-unseen-controls-complete-v287.json"
PREVIOUS = source.PANEL / "video-vcd2-controls-freeze-v200.json"
EXPECTED = {
    source.PUBLIC: "e7a7c400330dd22c1e9cd6d43d4ae91fbe5f3648657fb7c0c0877f103b44cc76",
    SOURCE_RESULT: "386e619b9e2bde56cc35919ff5ffa5bd16a6adc4955cfb9997e75749cac8a1df",
    PREVIOUS: "bba4426b8227f7ca1618613428e59c6c505bb8a247b2747b0cdfbbf99f9c24ab",
}
SLOTS = tuple(range(1, 9))
CODE = (
    Path(__file__),
    Path(delivery.__file__),
    BASE.ROOT / "backend/tests/test_video_vcd2_matched_controls_v290.py",
    BASE.ROOT / f"docs/{NAME}-method.md",
)
require, read, sha, write = controls.require, controls.read, controls.sha, controls.write


def source_bindings(cases, slots):
    require(
        [row["slot"] for row in cases] == [row["slot"] for row in slots] == list(SLOTS),
        "All eight ordered v287 slots required",
    )
    resolutions, bindings = Counter(), []
    for row, entry in zip(cases, slots, strict=True):
        selected = entry["selected"]
        require(
            entry["status"] == "qualified"
            and entry["attempts"]
            == [{"attempt": f"slot-{row['slot']:02d}-attempt-01", "status": "qualified"}]
            and selected["attempt"] == entry["attempts"][0]["attempt"],
            "Exact v287 qualification attempt required",
        )
        folder = source.TRIAL / selected["attempt"]
        acquired, qualification = (
            read(folder / filename) for filename in ("acquisition.json", "qualification.json")
        )
        streams = [item for item in qualification["header"]["streams"] if item.get("codec_type") == "video"]
        require(
            len(streams) == 1 and streams[0]["codec_name"] == "ffv1",
            "A qualified FFV1 metadata record is required",
        )
        recipe = controls.recipe(qualification, streams[0])
        require(f"{recipe.width}x{recipe.height}" == row["resolution"], "Qualified source geometry differs")
        resolutions[row["resolution"]] += 1
        bindings.append((acquired["bytes"], len(recipe.pts), selected["qualification_sha256"]))
    return resolutions, bindings


def plan():
    require(
        all(sha(path) == digest for path, digest in EXPECTED.items()),
        "Exact published v287 and v200 receipts required",
    )
    published, result, previous = map(read, (source.PUBLIC, SOURCE_RESULT, PREVIOUS))
    private = source.TRIAL / "freeze.json"
    frozen = read(private)
    cases = frozen["cases"]
    require(
        sha(private) == published["private_freeze_sha256"]
        and source.public_record(frozen, cases) == published
        and all(sha(BASE.ROOT / path) == digest for path, digest in frozen["code_sha256"].items()),
        "Exact v287 private source roster required",
    )
    inventory = scoped(controls.inventory, {"source": SimpleNamespace(TRIAL=source.TRIAL)})
    slots = inventory(cases, set())
    resolutions, bindings = source_bindings(cases, slots)
    require(
        result["status"] == "complete"
        and result["freeze_sha256"] == EXPECTED[source.PUBLIC]
        and result["planned_groups"] == len(slots) == len(SLOTS)
        and result["outcomes"] == {"qualified": len(SLOTS)}
        and result["qualified_by_resolution"] == dict(resolutions)
        and result["verified_bytes"] == sum(row[0] for row in bindings)
        and result["verified_frames"] == sum(row[1] for row in bindings)
        and result["qualification_bundle_sha256"]
        == hashlib.sha256(json.dumps(bindings).encode()).hexdigest(),
        "v287 verified qualification bundle differs",
    )
    recipe = controls.build_identity()
    require(
        all(previous.get(key) == value for key, value in recipe.items())
        and recipe["tool_sha256"] == published["decoder_sha256"]
        and recipe["relative_interval_s"] == frozen["relative_ordinary_interval_s"] == [3, 6],
        "Original v200 recipe and tools required",
    )
    identity = {
        **recipe,
        "schema": "chipmunk.video-vcd2-matched-controls/290",
        "source_freeze_sha256": EXPECTED[source.PUBLIC],
        "source_result_sha256": EXPECTED[SOURCE_RESULT],
        "source_private_freeze_sha256": published["private_freeze_sha256"],
        "source_qualification_bundle_sha256": result["qualification_bundle_sha256"],
        "v200_recipe_freeze_sha256": EXPECTED[PREVIOUS],
        "code_sha256": recipe["code_sha256"] | {str(path.relative_to(BASE.ROOT)): sha(path) for path in CODE},
        "source_groups": len(slots),
        "planned_controls": len(slots) * len(ROLES),
        "source_frames": result["verified_frames"],
        "storage_projection": controls.storage_projection(slots),
    }
    return {"identity": identity, "slots": slots}


def public_record(frozen):
    return {**frozen["identity"], "private_freeze_sha256": sha(TRIAL / "freeze.json")}


def admit(frozen):
    require(
        read(TRIAL / "freeze.json") == frozen and read(PUBLIC) == public_record(frozen),
        "Prospective v290 freeze required",
    )


def seal(frozen):
    counts, frames, observations = Counter(), Counter(), []
    for entry in frozen["slots"]:
        folder = TRIAL / f"slot-{entry['slot']:02d}-attempt-01"
        path = folder / "qualification.json"
        receipt = read(path)
        output_plan = delivery.receipt_plan(entry, receipt)
        clock = read(source.TRIAL / entry["selected"]["attempt"] / "qualification.json")["clock"]
        require(
            output_plan.origin_pts == clock["pts"][0]
            and list(output_plan.pts) == [pts - clock["pts"][0] for pts in clock["pts"]]
            and list(output_plan.interval) == clock["ordinary_frame_interval"]
            and str(output_plan.tb) == clock["time_base"],
            "Prepared controls differ from the qualified source clock",
        )
        for role, case in receipt["cases"].items():
            target = delivery.direct_file(folder / case["file"])
            require(sha(target) == case["sha256"], "Verified control bytes changed")
            counts[role] += 1
            frames[role] += case["verified_frames"]
        observations.append(sha(path))
    require(
        counts == {role: len(SLOTS) for role in ROLES}
        and all(frames[role] == frozen["identity"]["source_frames"] for role in ROLES),
        "All 24 matched controls required",
    )
    result = {
        "schema": "chipmunk.video-vcd2-matched-controls-result/290",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "source_groups": len(SLOTS),
        "controls": dict(counts),
        "verified_frames": dict(frames),
        "private_receipt_bundle_sha256": hashlib.sha256(json.dumps(observations).encode()).hexdigest(),
        "model_calls": 0,
    }
    write(TRIAL / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "prepare", "seal"))
    parser.add_argument("--slot", type=int, choices=SLOTS)
    args = parser.parse_args()
    require((args.command == "prepare") == (args.slot is not None), "Specify one slot only for preparation")
    frozen = plan()
    if args.command == "preflight":
        result = {
            key: frozen["identity"][key]
            for key in (
                "source_groups",
                "source_frames",
                "planned_controls",
                "relative_interval_s",
                "storage_projection",
                "model_calls",
            )
        }
    elif args.command == "freeze":
        require(not PUBLIC.exists(), "Preserve the published freeze")
        BASE.archive_ready()
        BASE.private_directory(TRIAL)
        write(TRIAL / "freeze.json", frozen)
        write(PUBLIC, public_record(frozen))
        result = {"status": "frozen", "freeze_sha256": sha(PUBLIC)}
    else:
        admit(frozen)
        if args.command == "prepare":
            BASE.archive_ready()
            scoped(controls.prepare, {"TRIAL": TRIAL, "source": SimpleNamespace(TRIAL=source.TRIAL)})(
                frozen, args.slot, 1
            )
            return
        result = seal(frozen)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
