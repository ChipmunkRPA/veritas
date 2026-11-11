"""Expand the exact v276 reference-edit construction to six unused source groups."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
from PIL import Image

from scripts import image_flux_photo_preparation_v276 as prior

flux = prior.flux
ROOT, DATA, PANELS = prior.ROOT, prior.DATA, prior.PANELS
NAME = "image-flux-photo-expansion-v285"
MODULE = "scripts.image_flux_photo_expansion_v285"
TRIAL = DATA / NAME
PUBLIC = PANELS / f"{NAME}-freeze.json"
require, sha, write = prior.require, prior.sha, prior.write
CASES = tuple(
    (f"{placement}-{number}", bounds, prompt)
    for number in range(1, 4)
    for placement, bounds, prompt in prior.CASES
)
BATCHES = tuple(tuple(CASES[start : start + 2]) for start in range(0, len(CASES), 2))
PARENT_RECEIPTS = {
    "image-flux-photo-preparation-freeze-v276.json": "56fca6203ac581d4a5b6a290fc588d97483147fb47523d0b1dc41200939e112d",
    "image-flux-photo-preparation-complete-v276.json": "2839c1745421e5f0c738c87a46aa7744976a445eb3afac4870f29585bc80eabd",
    "image-flux-detector-comparison-v280-freeze.json": "0729d5b3ed70eefbb7ae743a494e39e43670ec0b98e1dd0dcaa595e5c55d9ca0",
    "image-flux-detector-comparison-complete-v280.json": "23eeac4f9c85f5e6cac279caea9a0ea693c48044ca7802083f93ee1897266fa2",
}
CODE = (
    Path(__file__),
    ROOT / "backend/tests/test_image_flux_photo_expansion_v285.py",
    ROOT / f"docs/{NAME}-method.md",
)


def scoped(function, overrides):
    # The isolated model runtime has no psutil and cannot import the v112 adapter.
    globals_ = {**function.__globals__, **overrides}
    bound = FunctionType(
        function.__code__, globals_, function.__name__, function.__defaults__, function.__closure__
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


base_select = scoped(prior.select, {"CASES": CASES})


def choose(candidates, groups, prior_rows):
    excluded = {key for row in prior_rows for key in row["group"]}
    previous_hashes = {row["source_jpeg_sha256"] for row in prior_rows}
    matching_keys = {row["key"] for row in candidates if row["source_jpeg_sha256"] in previous_hashes}
    exposed = [group for group in groups if (excluded | matching_keys).intersection(group)]
    require(
        len(prior_rows) == len(previous_hashes) == len(exposed) == 2,
        "Both consumed pilot components must be found",
    )
    selected, eligible = base_select(candidates, [group for group in groups if group not in exposed])
    require(
        len(selected) == len({tuple(group) for _, group in selected}) == len(CASES),
        "Six independent source groups required",
    )
    return selected, eligible


def pilot_rows():
    receipts = {name: prior.read_exact(PANELS / name, digest) for name, digest in PARENT_RECEIPTS.items()}
    completed = receipts["image-flux-photo-preparation-complete-v276.json"]
    scored = receipts["image-flux-detector-comparison-complete-v280.json"]
    require(
        completed["status"] == scored["status"] == "complete"
        and completed["licensed_source_groups"] == 2
        and completed["generator_calls"] == 2
        and scored["unique_model_calls"] == 40
        and receipts["image-flux-detector-comparison-v280-freeze.json"]["source_groups"] == 2,
        "Complete two-source pilot and paired detector exposure required",
    )
    return prior.generation_inputs()


def audit():
    consumed = pilot_rows()
    selected, report = scoped(
        prior.audit, {"select": lambda candidates, groups: choose(candidates, groups, consumed)}
    )()
    return selected, {
        **report,
        "pilot_source_groups_excluded": len(consumed),
        "scope": {
            **report["scope"],
            "v276_v280": "Both completed pilot source groups and all their variants are excluded",
        },
    }


inputs = scoped(prior.inputs, {"CASES": CASES})
prepare = scoped(prior.prepare, {"audit": audit, "inputs": inputs})
verify_prepared = scoped(prior.verify_prepared, {"CASES": CASES})
base_plan = scoped(prior.freeze_plan, {"CASES": CASES, "verify_prepared": verify_prepared})


def freeze_plan(folder):
    plan = base_plan(folder)
    return {
        **plan,
        "schema": "chipmunk.image-flux-photo-expansion-freeze/285",
        "code_sha256": plan["code_sha256"] | {str(path.relative_to(ROOT)): sha(path) for path in CODE},
        "parent_receipts_sha256": PARENT_RECEIPTS,
        "planned_calls": len(CASES),
        "batches": [[slot for slot, _, _ in batch] for batch in BATCHES],
        "scope": "Six additional licensed crops for construction feasibility with matched ordinary controls. Source groups do not overlap the known v276/v280 pilot groups. No detector calls or fit.",
    }


freeze = scoped(
    prior.freeze,
    {
        "TRIAL": TRIAL,
        "PUBLIC": PUBLIC,
        "audit": audit,
        "inputs": inputs,
        "verify_prepared": verify_prepared,
        "freeze_plan": freeze_plan,
    },
)


def generation_inputs():
    return scoped(prior.generation_inputs, {"PUBLIC": PUBLIC, "freeze_plan": freeze_plan})(TRIAL)


def paths(number):
    require(type(number) is int and 1 <= number <= len(BATCHES), "One of three frozen batches required")
    batch_root = TRIAL / f"batch-{number:02d}"
    stage = batch_root / "authored"
    return batch_root, stage, stage / "output"


def batch_inputs(number):
    paths(number)
    planned = tuple(slot for slot, _, _ in BATCHES[number - 1])
    selected = [row for row in generation_inputs() if row["slot"] in planned]
    require(tuple(row["slot"] for row in selected) == planned, "Exact two frozen batch inputs required")
    return selected


def worker(number):
    _, stage, output = paths(number)
    scoped(
        prior.worker,
        {
            "TRIAL": TRIAL,
            "STAGE": stage,
            "OUTPUT": output,
            "PUBLIC": PUBLIC,
            "CASES": BATCHES[number - 1],
            "generation_inputs": lambda: batch_inputs(number),
        },
    )()


def run(number):
    from scripts import video_appearance_runtime_v108 as guard

    batch_root, stage, output = paths(number)
    batch_inputs(number)
    batch_root.mkdir(mode=0o700, exist_ok=True)

    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(output)],
            "Exact bounded v285 worker required",
        )
        kwargs["env"].update(
            PYTHONNOUSERSITE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            HF_HOME=str(stage / "hub-offline"),
        )
        return subprocess.Popen(
            [str(flux.ENV / "bin/python"), "-B", "-m", MODULE, "worker", str(number)], **kwargs
        )

    proxy = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(
        guard.run,
        {
            "TRIAL": batch_root,
            "verify": lambda: batch_inputs(number),
            "prerequisites": lambda *_: True,
            "subprocess": proxy,
            "__file__": __file__,
        },
    )("authored")


def read_pixels(path, expected, mode):
    require(path.resolve(strict=True) == path and sha(path) == expected, "Retained output PNG differs")
    with Image.open(path) as image:
        require(
            image.format == "PNG" and image.mode == mode and image.size == (prior.SIZE, prior.SIZE),
            "Exact source-local PNG geometry required",
        )
        return np.asarray(image)


def verify_case(before, row, folder):
    slot = before["slot"]
    require(
        row["slot"] == slot
        and row["status"] in {"qualified_geometry", "no_in_mask_change", "unsupported_output_geometry"},
        "Unexpected frozen case outcome",
    )
    if row["status"] == "unsupported_output_geometry":
        return {"placement": slot.rsplit("-", 1)[0], "status": row["status"]}
    prepared = TRIAL / slot
    source = read_pixels(prepared / "source.png", before["png_sha256"]["source"], "RGB")
    mask = read_pixels(prepared / "mask.png", before["png_sha256"]["mask"], "L")
    raw = read_pixels(folder / "raw.png", row["raw_png_sha256"], "RGB")
    differences = np.any(source != raw, axis=2)
    inside, outside = (int((differences & selected).sum()) for selected in (mask > 0, mask == 0))
    require(
        inside == row["raw_changes_inside"] and outside == row["raw_changes_outside"],
        "Raw RGB footprint differs",
    )
    if row["status"] == "no_in_mask_change":
        require(inside == 0, "The retained native edit contains unexpected changed pixels")
    else:
        composite, truth, proof = flux.compose(source, raw, mask)
        require(
            all(row[key] == value for key, value in proof.items())
            and np.array_equal(
                read_pixels(folder / "composite.png", row["composite_png_sha256"], "RGB"), composite
            )
            and np.array_equal(read_pixels(folder / "changed.png", row["changed_png_sha256"], "L"), truth),
            "Exact composition or actual change mask differs",
        )
    return {
        "placement": slot.rsplit("-", 1)[0],
        "status": row["status"],
        "raw_changes_inside": inside,
        "raw_changes_outside": outside,
        "composite_changed_pixels": row.get("changed_pixels", 0),
        "composite_outside_changes": row.get("outside_support_changes"),
    }


def verified_batch(number):
    _, stage, output = paths(number)
    selected = batch_inputs(number)
    profile_path, receipt = stage / "profile.json", output / "result.json"
    profile, result = (json.loads(path.read_bytes()) for path in (profile_path, receipt))
    require(
        profile["child_started"]
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["resource_limit_observations"]
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["worker_receipt_sha256"] == sha(receipt)
        and profile["resource_journal_sha256"] == sha(stage / "resources.jsonl")
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["attempt_journal_sha256"] == sha(output / "attempts.jsonl")
        and result["new_neural_calls"] == profile["new_neural_calls"] == len(selected)
        and result["network_attempts"] == result["detector_calls"] == 0,
        "Complete bounded batch receipt required",
    )
    rows = [
        verify_case(before, row, output / before["slot"])
        for before, row in zip(selected, result["cases"], strict=True)
    ]
    require(
        dict(Counter(row["status"] for row in rows)) == result["case_outcomes"],
        "Case outcomes differ from the retained batch",
    )
    return rows, {"result_sha256": sha(receipt), "profile_sha256": sha(profile_path)}


def coverage():
    generation_inputs()
    totals = {}
    for number in range(1, len(BATCHES) + 1):
        _, stage, _ = paths(number)
        if not stage.exists():
            totals[str(number)] = {"status": "unattempted", "verified": 0}
            continue
        try:
            rows, _ = verified_batch(number)
        except (OSError, ValueError, KeyError, TypeError):
            totals[str(number)] = {"status": "unverified", "verified": 0}
        else:
            totals[str(number)] = {
                "status": "verified",
                "verified": len(rows),
                "outcomes": dict(Counter(row["status"] for row in rows)),
            }
    return {"planned_sources": len(CASES), "batches": totals}


def seal():
    rows, receipts = [], {}
    for number in range(1, len(BATCHES) + 1):
        observations, receipts[str(number)] = verified_batch(number)
        rows.extend(observations)
    require(len(rows) == len(CASES), "All six frozen source outcomes required")
    result = {
        "schema": "chipmunk.image-flux-photo-expansion-result/285",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "source_groups": len(rows),
        "generator_calls": len(rows),
        "detector_calls": 0,
        "case_outcomes": dict(Counter(row["status"] for row in rows)),
        "cases": rows,
        "batch_receipts": receipts,
        "semantic_success_measured": False,
        "serving_admitted": False,
    }
    write(TRIAL / "summary.json", result)
    return result


def preflight():
    chosen, report = audit()
    return {
        **report,
        "cases": [
            {"slot": row["slot"], "control_changes": row["changed_pixels"]} for row, _ in inputs(chosen)
        ],
        "model_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "prepare", "freeze", "run", "worker", "coverage", "seal")
    )
    parser.add_argument("batch", nargs="?", type=int)
    args = parser.parse_args()
    if args.command in {"run", "worker"}:
        {"run": run, "worker": worker}[args.command](args.batch)
    else:
        require(args.batch is None, "Only a generation worker accepts a batch")
        report = {
            "preflight": preflight,
            "prepare": lambda: prepare(TRIAL),
            "freeze": freeze,
            "coverage": coverage,
            "seal": seal,
        }[args.command]()
        print(json.dumps(report, indent=2), flush=True)
