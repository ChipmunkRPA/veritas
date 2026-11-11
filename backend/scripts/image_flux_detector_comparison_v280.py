"""Compare frozen image localization readouts on the two qualified v276 edits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts import image_bg_real_backup_jpeg_v264 as v264
from scripts import image_flux_photo_preparation_v276 as photo
from scripts.video_appearance_corrected_v112 import scoped

original, previous, monitor = v264.original, v264.previous, v264.monitor
ROOT, DATA, PANELS = photo.ROOT, photo.DATA, photo.PANELS
NAME = "image-flux-detector-comparison-v280"
MODULE = "scripts.image_flux_detector_comparison_v280"
TRIAL, RAW_ROOT = DATA / NAME, DATA / f"{NAME}-maps"
PUBLIC = PANELS / f"{NAME}-freeze.json"
read, write, sha, require = v264.read, v264.write, v264.sha, v264.require
FORMATS = {0: "native_png", 75: "jpeg75"}
QUALITIES, ARMS, KINDS = tuple(FORMATS), v264.ARMS, v264.KINDS
PHOTOS, PHASES = 2, 1
RECEIPTS = {
    photo.PUBLIC: "56fca6203ac581d4a5b6a290fc588d97483147fb47523d0b1dc41200939e112d",
    photo.OUTPUT / "result.json": "11fcb8a91b7e259554f88bf339ba2d377c0a15cbc49ff0cd8d46476c867d3ef6",
    photo.STAGE / "profile.json": "484080c8c351aba1ecd48fac84e57a840d40e6f72e1c2239b9343f5f26aeaebe",
    v264.PUBLIC: "4db71eddb75033e8ac41d9716d2eb5cf7356a1d544a51b91de08657fd7ed44f9",
    PANELS
    / "image-bg-real-backup-jpeg-v264-complete.json": "438f754a667e90e5b42b977085a36725021cdb8180a3d6e9fca0a9206e267fc4",
}
CODE = (
    Path(__file__),
    ROOT / "backend/tests/test_image_flux_detector_comparison_v280.py",
    ROOT / f"docs/{NAME}-method.md",
)


def source_cases():
    require(
        all(sha(path) == digest for path, digest in RECEIPTS.items()),
        "Exact v276 source and v264 scoring receipts required",
    )
    selected = photo.generation_inputs()
    result, profile = read(photo.OUTPUT / "result.json"), read(photo.STAGE / "profile.json")
    require(
        result["status"] == "complete"
        and result["freeze_sha256"] == RECEIPTS[photo.PUBLIC]
        and result["attempt_journal_sha256"] == sha(photo.OUTPUT / "attempts.jsonl")
        and result["case_outcomes"] == {"qualified_geometry": PHOTOS}
        and result["new_neural_calls"] == PHOTOS
        and result["detector_calls"] == result["network_attempts"] == 0
        and profile["worker_receipt_sha256"] == RECEIPTS[photo.OUTPUT / "result.json"]
        and profile["resource_journal_sha256"] == sha(photo.STAGE / "resources.jsonl")
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0,
        "Verified v276 resource and geometry qualification required",
    )
    cases, index = [], {}
    for number, (before, after) in enumerate(zip(selected, result["cases"], strict=True)):
        slot = before["slot"]
        require(
            slot == after["slot"]
            and after["status"] == "qualified_geometry"
            and after["outside_support_changes"] == 0,
            "Every prepared source needs an exact completed edit",
        )
        cases.append({"source_id": slot, "component": f"photo-{number + 1}"})
        index[slot] = (before, after)
    require(
        len(cases) == PHOTOS and len({tuple(row["group"]) for row in selected}) == PHOTOS,
        "Two distinct source groups required",
    )
    return cases, index


def verified_png(path, expected, mode):
    require(path.resolve(strict=True) == path and sha(path) == expected, "Retained PNG bytes differ")
    with Image.open(path) as image:
        require(
            image.format == "PNG" and image.mode == mode and image.size == (photo.SIZE, photo.SIZE),
            "Expected exact native PNG geometry",
        )
        return image.copy()


def material(cases, index, position):
    slot = cases[position]["source_id"]
    before, after = index[slot]
    folder, output = photo.TRIAL / slot, photo.OUTPUT / slot
    prepared = {
        key: verified_png(
            folder / f"{key}.png", value, "L" if key == "mask" or key.endswith("_changed") else "RGB"
        )
        for key, value in before["png_sha256"].items()
    }
    images = {
        "ai": verified_png(output / "composite.png", after["composite_png_sha256"], "RGB"),
        "original": prepared["source"],
        **{key: prepared[key] for key in KINDS[2:]},
    }
    mask = np.asarray(prepared["mask"])
    require(
        np.isin(mask, (0, 255)).all() and int(np.count_nonzero(mask)) == after["support_pixels"],
        "Exact intended native mask required",
    )
    footprints = {key: original.changed_pixels(images["original"], images[key]) for key in KINDS}
    require(
        not footprints["original"].any()
        and all(
            masked.any() and not masked[mask == 0].any()
            for key, masked in footprints.items()
            if key != "original"
        ),
        "Native observed footprints differ",
    )
    exact = verified_png(output / "changed.png", after["changed_png_sha256"], "L")
    require(
        np.array_equal(np.asarray(exact), footprints["ai"].astype(np.uint8) * 255)
        and original.digest(footprints["ai"].tobytes()) == after["truth_mask_sha256"]
        and int(footprints["ai"].sum()) == after["changed_pixels"],
        "Qualified AI footprint must match exact native RGB differences",
    )
    for kind in KINDS[2:]:
        require(
            np.array_equal(np.asarray(prepared[f"{kind}_changed"]), footprints[kind].astype(np.uint8) * 255)
            and int(footprints[kind].sum()) == before["changed_pixels"][kind],
            "Ordinary source-relative footprint differs",
        )
    return images, footprints


def encode(image, quality):
    if quality == 0:
        return original.v28.encoded_view(image, "native")
    return original.jpeg_view(image, quality)


def source_entry(case, images, footprints):
    entry = {
        **case,
        "changed_pixels": {key: int(value.sum()) for key, value in footprints.items()},
        "inputs": {},
        "decoded_outside_native_footprint_changes": {},
    }
    for quality in QUALITIES:
        encoded = {key: encode(images[key], quality) for key in KINDS}
        entry["inputs"][str(quality)] = {key: original.view_binding(*pair) for key, pair in encoded.items()}
        source = encoded["original"][1]
        entry["decoded_outside_native_footprint_changes"][FORMATS[quality]] = {
            key: int((original.changed_pixels(source, pair[1]) & ~footprints[key]).sum())
            for key, pair in encoded.items()
        }
    return entry


def frozen_identity():
    _, policy, inherited = v264.public_contract()
    previous_freeze = read(v264.PUBLIC)
    chain = original.prior_chain()
    require(
        original.fixed_versions() == previous_freeze["versions"]
        and original.rotary_files() == previous_freeze["rotary_source_sha256"],
        "Unchanged v264 numerical runtime required",
    )
    keys = (
        "kinds",
        "arms",
        "methods",
        "sensitive_policy",
        "useful_iou",
        "useful_recall",
        "device",
        "dtype",
        "cpu_threads",
        "asset_sha256",
        "candidate_checkpoint_sha256",
        "versions",
        "rotary_source_sha256",
        "training_chain_sha256",
    )
    require(all(previous_freeze[key] == policy[key] for key in keys), "Frozen v264 scoring policies differ")
    return {
        "schema": "chipmunk.image-flux-detector-comparison-freeze/280",
        "source_groups": PHOTOS,
        "formats": list(FORMATS.values()),
        "unique_model_calls": PHOTOS * len(KINDS) * len(QUALITIES) * len(ARMS),
        "exact_first_input_repeats": len(QUALITIES) * len(ARMS),
        "same_native_exact_rgb_truth_both_formats": True,
        "preparation_model_calls": 0,
        "serving_admitted": False,
        "parent_receipts_sha256": {path.name: digest for path, digest in RECEIPTS.items()},
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in CODE}
        | inherited
        | {str(Path(v264.__file__).relative_to(ROOT)): sha(Path(v264.__file__))},
        **{key: previous_freeze[key] for key in keys},
    }, chain


def prepare(trial, raw_root):
    require(
        trial == TRIAL
        and raw_root == RAW_ROOT
        and not trial.exists()
        and raw_root.is_dir()
        and not any(raw_root.iterdir()),
        "Fresh private v280 trial and map directories required",
    )
    cases, index = source_cases()
    frozen, chain = frozen_identity()
    require(
        sha(Path(chain["final_checkpoint"])) == frozen["candidate_checkpoint_sha256"]
        and all(sha(ROOT / path) == digest for path, digest in frozen["asset_sha256"].items()),
        "Frozen model bytes differ",
    )
    trial.mkdir(mode=0o700)
    (trial / "masks").mkdir(mode=0o700)
    sources = []
    for n, case in enumerate(cases):
        images, footprints = material(cases, index, n)
        target = trial / "masks" / f"{n:03d}.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, **footprints)
        sources.append({**source_entry(case, images, footprints), "masks_sha256": sha(target)})
    write(trial / "inputs.json", {"raw_root": str(raw_root), "sources": sources})
    write(trial / "freeze.json", {**frozen, "private_input_bindings_sha256": sha(trial / "inputs.json")})


def private_lease(trial, expected, profile=None):
    require(trial == TRIAL and sha(trial / "freeze.json") == expected, "Exact prepared v280 trial required")
    frozen, _ = frozen_identity()
    require(
        read(trial / "freeze.json")
        == {**frozen, "private_input_bindings_sha256": sha(trial / "inputs.json")},
        "Frozen sources or scoring policy differ",
    )
    inputs = read(trial / "inputs.json")
    cases, index = source_cases()
    require(
        [{"source_id": row["source_id"], "component": row["component"]} for row in inputs["sources"]] == cases
        and Path(inputs["raw_root"]) == RAW_ROOT,
        "Prepared cases or map destination changed",
    )
    if profile is not None:
        assert_profile(profile, "prepare", ["prepare", str(trial), str(RAW_ROOT)])
    return inputs, cases, index


def lease(trial, expected, profile=None):
    require(sha(PUBLIC) == expected, "Prospective v280 freeze required before scoring")
    return private_lease(trial, expected, profile)


tasks = scoped(original.tasks, {"PHOTOS": PHOTOS, "PHASES": PHASES})
batch_views = scoped(original.batch_views, {"tasks": tasks, "material": material, "jpeg_view": encode})
MONITOR_IDENTITY = {"MODULE": MODULE, "__file__": __file__}
assert_profile = scoped(monitor.assert_profile, MONITOR_IDENTITY)


def launch(argv, **kwargs):
    require(argv[1:3] == ["-m", MODULE], "Exact v280 bounded worker required")
    return subprocess.Popen([*argv[:3], "_worker", *argv[3:]], **kwargs)


run = scoped(
    monitor.run,
    {
        **MONITOR_IDENTITY,
        "assert_profile": assert_profile,
        "subprocess": SimpleNamespace(
            Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
        ),
    },
)
phase_rows = scoped(v264.phase_rows, {"assert_profile": assert_profile})
producer_rows = scoped(original.producer_rows, {"phase_rows": phase_rows, "tasks": tasks})
score = scoped(original.score, {"lease": lease, "batch_views": batch_views, "tasks": tasks})
readout = scoped(
    original.readout,
    {"lease": lease, "producer_rows": producer_rows, "batch_views": batch_views, "tasks": tasks},
)
coverage = scoped(
    original.coverage,
    {
        "lease": lease,
        "producer_rows": producer_rows,
        "PHASES": PHASES,
        "PHOTOS": PHOTOS,
        "QUALITIES": QUALITIES,
    },
)


def summarize(rows, groups, quality):
    require(
        len(groups) == len(set(groups)) == PHOTOS
        and [(r["n"], r["kind"]) for r in rows] == tasks(0)
        and all(
            r["quality"] == quality
            and r["component"] == groups[r["n"]]
            and set(r["methods"]) == set(original.METHODS)
            for r in rows
        ),
        "Every paired view must be present",
    )
    methods = {}
    for name in original.METHODS:
        by_kind = {kind: [r["methods"][name]["ai"] for r in rows if r["kind"] == kind] for kind in KINDS}
        positives = by_kind["ai"]
        methods[name] = {
            "positive_groups": PHOTOS,
            "positive_groups_with_a_region": sum(m["predicted_positive_pixels"] > 0 for m in positives),
            "positive_groups_with_overlap": sum(m["intersection"] > 0 for m in positives),
            "useful_positive_groups": sum(m["iou"] >= 0.30 and m["recall"] >= 0.50 for m in positives),
            "positive_iou_by_group": [m["iou"] for m in positives],
            "positive_recall_by_group": [m["recall"] for m in positives],
            "controls": {
                kind: {
                    "groups": PHOTOS,
                    "flagged_groups": sum(m["predicted_positive_pixels"] > 0 for m in by_kind[kind]),
                    "flagged_area_fraction_by_group": [m["flagged_area_fraction"] for m in by_kind[kind]],
                }
                for kind in KINDS[1:]
            },
        }
    return {"format": FORMATS[quality], "source_groups": PHOTOS, "methods": methods}


def write_result(path, result):
    prepared = read(path.parent / "inputs.json")["sources"]
    write(
        path,
        {
            **{key: value for key, value in result.items() if key != "quality_results"},
            "schema": "chipmunk.image-flux-detector-comparison-result/280",
            "format_results": {FORMATS[int(key)]: value for key, value in result["quality_results"].items()},
            "transport_changes_outside_native_footprint": {
                fmt: {
                    kind: [row["decoded_outside_native_footprint_changes"][fmt][kind] for row in prepared]
                    for kind in KINDS
                }
                for fmt in FORMATS.values()
            },
            "scope": "Two source groups. Native observed RGB footprints shared by PNG and JPEG75. Descriptive transfer pilot without population intervals, fit or serving admission.",
        },
    )


seal = scoped(
    original.seal,
    {
        "lease": lease,
        "producer_rows": producer_rows,
        "phase_rows": phase_rows,
        "PHASES": PHASES,
        "tasks": tasks,
        "QUALITIES": QUALITIES,
        "summarize": summarize,
        "write": write_result,
    },
)


def preflight():
    cases, index = source_cases()
    frozen_identity()
    entries = [source_entry(case, *material(cases, index, n)) for n, case in enumerate(cases)]
    return {
        "source_groups": len(entries),
        "bound_views": len(entries) * len(KINDS) * len(QUALITIES),
        "native_changed_pixels": [row["changed_pixels"] for row in entries],
        "decoded_outside_native_footprint_changes": [
            row["decoded_outside_native_footprint_changes"] for row in entries
        ],
        "model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    worker = sys.argv[1:2] == ["_worker"]
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        ("prepare", "score", "readout", "seal")
        if worker
        else ("preflight", "prepare", "freeze", "score", "readout", "seal", "coverage")
    ):
        command = commands.add_parser(name)
        if worker:
            command.add_argument("trial", type=Path)
            command.add_argument(
                "raw_root" if name == "prepare" else "freeze_sha256", type=Path if name == "prepare" else str
            )
        if name == "score":
            command.add_argument("arm", choices=ARMS)
        if name in {"score", "readout"}:
            command.add_argument("quality", type=int, choices=QUALITIES)
            command.add_argument("phase", type=int, choices=range(PHASES))
    args = parser.parse_args(sys.argv[2:] if worker else None)
    if worker:
        require(args.trial == TRIAL, "Exact isolated v280 trial required")
        {"prepare": prepare, "score": score, "readout": readout, "seal": seal}[args.command](
            *list(vars(args).values())[1:]
        )
    elif args.command == "preflight":
        print(json.dumps(preflight(), indent=2))
    elif args.command == "prepare":
        RAW_ROOT.mkdir(mode=0o700, exist_ok=True)
        run("prepare", [str(TRIAL), str(RAW_ROOT)])
    elif args.command == "freeze":
        path = TRIAL / "freeze.json"
        private_lease(TRIAL, sha(path), original.prep_profile(TRIAL))
        with PUBLIC.open("xb") as stream:
            stream.write(path.read_bytes())
        print(json.dumps({"freeze_sha256": sha(PUBLIC)}))
    elif args.command == "coverage":
        coverage(TRIAL, sha(PUBLIC))
    else:
        run(args.command, [str(TRIAL), sha(PUBLIC), *(str(value) for value in list(vars(args).values())[1:])])


if __name__ == "__main__":
    main()
