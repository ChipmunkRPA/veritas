"""Prospective positive and mixed-edit transfer on the exact eight v298 sources."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import image_flux_photo_expansion_v285 as expansion
from scripts import image_flux_photo_preparation_v276 as pilot

ROOT, DATA, PANELS, flux = pilot.ROOT, pilot.DATA, pilot.PANELS, pilot.flux
TRIAL = DATA / "image-copy-positive-transfer-v300"
PUBLIC = TRIAL / "private-freeze.json"
PREPARED = TRIAL / "private-inputs.json"
V298 = DATA / "image-copy-transfer-v298"
MODULE = "scripts.image_copy_positive_transfer_v300"
SIZE = pilot.SIZE
GROUPS, BATCHES = 8, 4
FORMATS = ("native_png", "jpeg90")
REMOTE_BOX = (16, 16, 64, 64)
VARIANTS = (
    "original",
    "tone",
    "blur",
    "translated_copy",
    "reflected_copy",
    "remote_reflected_copy",
    "ai",
    "mixed",
)
CUE_REGIONS = (
    ("ai", "target"),
    ("mixed", "target"),
    ("mixed", "remote"),
    ("remote_reflected_copy", "remote"),
    ("original", "remote"),
)
RECEIPTS = {
    PANELS
    / "image-copy-transfer-v298-complete.json": "df4f7902a3cee6d906fc6ffaa72d0e3e585e6eb7e98be4f9daeff582745e7716",
    V298 / "private-freeze.json": "3e31ef323002875d6f3ef4261e4867e5926fb4a3bce4656a60ff2f8c7957d8a9",
    V298 / "private-result.json": "663fad4f6f4f59fbece46fcb8272e76ad963e4f0617d8d8b749859970c33a2ec",
    PANELS
    / "image-flux-photo-expansion-v285-freeze.json": "119e72d3d2672689560a3833f43237710fefbf616e3475352ddca4764c7f9827",
    PANELS
    / "image-flux-photo-expansion-complete-v285.json": "8882d6f2ad434ee29c77846003ef16291a5aa7169a63038574cb502120eacaf2",
    PANELS
    / "image-flux-product-comparison-v288-freeze.json": "74737eb4624484009b76577e9328289544f9714c1703323cffd60247882cf821",
    PANELS
    / "image-flux-product-comparison-complete-v288.json": "26feef3b938bca091a3e84d48c8f5d52fcd0921d425c7dcfc8c0a742ff2225ba",
}
CODE = (
    Path(__file__),
    Path(pilot.__file__),
    Path(expansion.__file__),
    Path(flux.__file__),
    ROOT / "backend/scripts/image_copy_positive_product_v300.py",
    ROOT / "backend/scripts/image_copy_transfer_v298.py",
    ROOT / "backend/scripts/image_copy_corroboration_v295.py",
    ROOT / "backend/scripts/video_appearance_runtime_v108.py",
    ROOT / "backend/tests/test_image_copy_positive_transfer_v300.py",
    ROOT / "docs/image-copy-positive-transfer-v300-method.md",
)
require, sha, write, read_exact = pilot.require, pilot.sha, pilot.write, pilot.read_exact
scoped = expansion.scoped


def bounds(box):
    x, y, width, height = box
    return x, y, x + width, y + height


def schedule(plan):
    prompts = {placement: (mask, prompt) for placement, mask, prompt in pilot.CASES}
    rows = []
    for n, source in enumerate(plan["rows"]):
        position = source["placement"]
        mask, prompt = prompts[position]
        require(
            n == source["slot"] and bounds(source["target_box_xywh"]) == mask, "Inherited v298 region differs"
        )
        rows.append(
            {
                "slot": f"{position}-{n + 1:02d}",
                "source_index": n,
                "placement": position,
                "jpeg_phase": source["jpeg_phase"],
                "mask_ltrb_exclusive": list(mask),
                "target_box_xywh": source["target_box_xywh"],
                "donor_box_xywh": source["donor_box_xywh"],
                "mixed_copy_box_xywh": list(REMOTE_BOX),
                "seed": pilot.SEED,
                "prompt": prompt,
            }
        )
    require(
        len(rows) == GROUPS
        and Counter((row["placement"], row["jpeg_phase"]) for row in rows)
        == {
            ("interior", "aligned"): 2,
            ("interior", "shifted"): 2,
            ("edge", "aligned"): 2,
            ("edge", "shifted"): 2,
        },
        "All eight exact source roles and balanced geometries required",
    )
    return rows


def previous_inputs():
    receipts = {path: read_exact(path, expected) for path, expected in RECEIPTS.items()}
    public = receipts[PANELS / "image-copy-transfer-v298-complete.json"]
    plan, result = (receipts[V298 / name] for name in ("private-freeze.json", "private-result.json"))
    require(
        public
        == {
            **result["aggregate"],
            "freeze_sha256": result["freeze_sha256"],
            "private_result_sha256": RECEIPTS[V298 / "private-result.json"],
        }
        and result["freeze_sha256"] == RECEIPTS[V298 / "private-freeze.json"]
        and public["status"] == "complete"
        and public["scored_views"] == public["planned_views"] == 80
        and public["generator_calls"] == public["detector_calls"] == 0
        and not result["input_errors"]
        and len({row["source"]["author"] for row in plan["rows"]}) == GROUPS,
        "The exact complete v298 ordinary-only panel is required",
    )
    producer = receipts[PANELS / "image-flux-photo-expansion-v285-freeze.json"]
    qualified = receipts[PANELS / "image-flux-photo-expansion-complete-v285.json"]
    require(
        (producer["model_revision"], producer["seed"], producer["steps"], producer["acquisition_sha256"])
        == (flux.REVISION, pilot.SEED, flux.STEPS, flux.ACQUISITION_SHA)
        and producer["guidance"] == 1.0
        and producer["scheduler"] == flux.SCHEDULER
        and not producer["use_kv_cache"]
        and qualified["case_outcomes"] == {"qualified_geometry": 6}
        and qualified["freeze_sha256"] == RECEIPTS[PANELS / "image-flux-photo-expansion-v285-freeze.json"],
        "Exact licensed reference-edit lineage required",
    )
    return plan, result["rows"]


def product_contract():
    path = PANELS / "image-flux-product-comparison-v288-freeze.json"
    old = read_exact(path, RECEIPTS[path])
    completed_path = PANELS / "image-flux-product-comparison-complete-v288.json"
    completed = read_exact(completed_path, RECEIPTS[completed_path])
    require(completed["freeze_sha256"] == RECEIPTS[path], "Exact v288 product receipt required")
    relevant = (
        "backend/app/",
        "backend/scripts/image_flux_product_comparison_v288.py",
        "backend/scripts/image_regional_inference_v28.py",
    )
    code = {name: value for name, value in old["code_sha256"].items() if name.startswith(relevant)}
    require(
        all(sha(ROOT / name) == value for name, value in code.items()), "Frozen v288 product paths changed"
    )
    return {
        "routes": {
            "native_png": old["current_routes"]["native_png"],
            "jpeg90": old["current_routes"]["jpeg75"],
        },
        "asset_sha256": {
            name: value
            for name, value in old["asset_sha256"].items()
            if name.startswith(("models/weights/splice/", "models/weights/fused/"))
        },
        "code_sha256": code,
        "configured_product_route": old["configured_product_route"],
        "versions": old["versions"],
        "useful_iou": old["useful_iou"],
        "useful_recall": old["useful_recall"],
    }


def metadata_contract():
    plan, _ = previous_inputs()
    return {
        "schema": "chipmunk.image-copy-positive-transfer-freeze/300",
        "prior_receipts_sha256": {str(path.relative_to(ROOT)): value for path, value in RECEIPTS.items()},
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in CODE},
        "model_revision": flux.REVISION,
        "asset_receipt_sha256": flux.ACQUISITION_SHA,
        "runtime_lock_sha256": flux.LOCK_SHA,
        "steps": flux.STEPS,
        "guidance": 1.0,
        "scheduler": flux.SCHEDULER,
        "use_kv_cache": False,
        "source_groups": GROUPS,
        "generation_batches": BATCHES,
        "generation_calls": GROUPS,
        "formats": list(FORMATS),
        "variants": list(VARIANTS),
        "cue_regions": [list(pair) for pair in CUE_REGIONS],
        "current_product": product_contract(),
        "product_unique_views": GROUPS * len(FORMATS) * len(VARIANTS),
        "cases": schedule(plan),
        "generation_fresh_for_v298_sources": True,
        "source_photo_panel_previously_observed_on_ordinary_controls": True,
    }


def preflight():
    import psutil

    from scripts import local_probe_limits_v24 as limits

    contract = metadata_contract()
    receipt = read_exact(flux.BUNDLE / "verified-assets.json", flux.ACQUISITION_SHA)
    require(
        receipt["revision"] == flux.REVISION
        and receipt["verified_bytes"] == 4_619_700_724
        and (flux.ENV / "bin/python").is_file()
        and sha(flux.LOCK) == flux.LOCK_SHA
        and sha(flux.WHEEL) == flux.WHEEL_SHA,
        "Exact qualified FLUX bundle and isolated runtime required",
    )
    available = psutil.virtual_memory().available
    free = shutil.disk_usage(DATA).free
    return {
        "source_groups": GROUPS,
        "generation_calls_planned": GROUPS,
        "model_free_cue_proposals_planned": 80,
        "product_views_planned": contract["product_unique_views"],
        "available_ram_bytes": available,
        "internal_disk_free_bytes": free,
        "external_disk_free_bytes": shutil.disk_usage(flux.ARCHIVE).free,
        "inherited_headroom_admitted": (
            available >= limits.START_RAM
            and free >= limits.START_DISK
            and flux.ARCHIVE.parent.parent.is_mount()
            and shutil.disk_usage(flux.ARCHIVE).free >= limits.START_DISK
        ),
        "product_asset_paths_present": all(
            (ROOT / name).is_file() for name in contract["current_product"]["asset_sha256"]
        ),
        "media_opened": False,
        "model_calls": 0,
    }


def verify_ordinary(source, row, expected):
    from scripts import image_copy_transfer_v298 as ordinary

    proposal, rendered = ordinary.controls(source, row["target_box_xywh"], row["donor_box_xywh"])
    expected = {(item["variant"], item["format"]): item for item in expected}
    require(
        set(expected) == {(kind, fmt) for kind in ordinary.VARIANTS for fmt in FORMATS},
        "Exact v298 coverage required",
    )
    for kind, rgb in rendered.items():
        truth = np.any(source != rgb, axis=2)
        for fmt in FORMATS:
            pixels, payload_hash = ordinary.encode_view(rgb, fmt)
            binding = expected[kind, fmt]
            require(
                payload_hash == binding["submitted_encoded_sha256"]
                and flux.digest(pixels) == binding["submitted_rgb_sha256"]
                and flux.digest(truth) == binding["native_change_mask_sha256"]
                and flux.digest(proposal) == binding["proposal_mask_sha256"],
                "An inherited ordinary control or source differs from v298",
            )
    return proposal


def prepare():
    from scripts import image_copy_transfer_v298 as ordinary

    contract = metadata_contract()
    plan, results = previous_inputs()
    require(not TRIAL.exists(), "Fresh private v300 trial directory required")
    prepared = []
    for item, original in zip(contract["cases"], plan["rows"], strict=True):
        source = ordinary.load_crop(original)
        selection = verify_ordinary(
            source, item, [row for row in results if row["slot"] == item["source_index"]]
        )
        prepared.append((item, source, selection.astype(np.uint8) * 255))
    TRIAL.mkdir(mode=0o700)
    rows = []
    for item, source, mask in prepared:
        folder = TRIAL / item["slot"]
        folder.mkdir(mode=0o700)
        inputs = {"source": source, "mask": mask}
        rows.append(
            {
                **item,
                "png_sha256": {
                    name: flux.save_png(folder / f"{name}.png", pixels) for name, pixels in inputs.items()
                },
                "rgb_or_mask_sha256": {name: flux.digest(pixels) for name, pixels in inputs.items()},
            }
        )
    write(PREPARED, {"metadata_contract": contract, "rows": rows})
    return {"prepared_source_groups": len(rows), "media_opened": True, "model_calls": 0}


def freeze_document():
    prepared = read_exact(PREPARED, sha(PREPARED))
    contract = metadata_contract()
    require(prepared["metadata_contract"] == contract, "Prepared code, source roles or method changed")
    _, parent_rows = previous_inputs()
    source_bindings = {
        item["slot"]: item
        for item in parent_rows
        if item["variant"] == "original" and item["format"] == "native_png"
    }
    for frozen, row in zip(contract["cases"], prepared["rows"], strict=True):
        require({key: row[key] for key in frozen} == frozen, "Prepared source case differs")
        source = prepared_source(row)
        selection = expansion.read_pixels(TRIAL / row["slot"] / "mask.png", row["png_sha256"]["mask"], "L")
        target = rectangle_mask(row["target_box_xywh"])
        require(
            flux.digest(source)
            == row["rgb_or_mask_sha256"]["source"]
            == source_bindings[row["source_index"]]["submitted_rgb_sha256"]
            and flux.digest(selection) == row["rgb_or_mask_sha256"]["mask"]
            and np.array_equal(selection, target.astype(np.uint8) * 255),
            "Prepared crop or mask no longer matches the sealed source",
        )
    return {**contract, "private_inputs_sha256": sha(PREPARED)}


def freeze():
    require(not PUBLIC.exists(), "Preserve the existing exact freeze")
    write(PUBLIC, freeze_document())
    return {"freeze_sha256": sha(PUBLIC), "source_groups": GROUPS, "model_calls": 0}


def frozen_rows(expected):
    require(re.fullmatch(r"[a-f0-9]{64}", expected), "Exact prospective digest required")
    frozen = read_exact(PUBLIC, expected)
    require(
        frozen == {**metadata_contract(), "private_inputs_sha256": frozen["private_inputs_sha256"]},
        "Frozen experiment changed",
    )
    prepared = read_exact(PREPARED, frozen["private_inputs_sha256"])
    require(
        all(
            all(row[key] == value for key, value in case.items())
            for case, row in zip(frozen["cases"], prepared["rows"], strict=True)
        ),
        "Frozen source case geometry changed",
    )
    return prepared["rows"]


def batch_rows(batch, expected):
    require(type(batch) is int and 1 <= batch <= BATCHES, "One of four frozen two-source stages required")
    return frozen_rows(expected)[2 * (batch - 1) : 2 * batch]


def stage_paths(kind, batch):
    root = TRIAL / f"{kind}-{batch:02d}"
    stage = root / "authored"
    return root, stage, stage / "output"


def guarded(kind, batch, expected, executable, action, prerequisite):
    from scripts import video_appearance_runtime_v108 as guard

    prerequisite()
    root, stage, output = stage_paths(kind, batch)
    root.mkdir(mode=0o700, exist_ok=True)

    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(output)],
            "Exact v108 guarded worker route required",
        )
        kwargs["env"].update(
            PYTHONNOUSERSITE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            HF_HOME=str(stage / "hub-offline"),
        )
        return subprocess.Popen([str(executable), "-B", "-m", MODULE, action, str(batch), expected], **kwargs)

    proxy = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(
        guard.run,
        {
            "TRIAL": root,
            "verify": prerequisite,
            "prerequisites": lambda *_: True,
            "subprocess": proxy,
            "__file__": __file__,
        },
    )("authored")


def generation(batch, expected):
    guarded(
        "generate", batch, expected, flux.ENV / "bin/python", "_generate", lambda: batch_rows(batch, expected)
    )


def generation_worker(batch, expected):
    rows = batch_rows(batch, expected)
    _, stage, output = stage_paths("generate", batch)
    cases = tuple((row["slot"], row["mask_ltrb_exclusive"], row["prompt"]) for row in rows)
    scoped(
        pilot.worker,
        {
            "TRIAL": TRIAL,
            "STAGE": stage,
            "OUTPUT": output,
            "PUBLIC": PUBLIC,
            "CASES": cases,
            "generation_inputs": lambda: rows,
        },
    )()


def generated(batch, expected):
    def paths(number):
        return stage_paths("generate", number)

    observations, receipts = scoped(
        expansion.verified_batch,
        {
            "paths": paths,
            "batch_inputs": lambda number: batch_rows(number, expected),
            "verify_case": scoped(expansion.verify_case, {"TRIAL": TRIAL}),
            "PUBLIC": PUBLIC,
            "__file__": __file__,
        },
    )(batch)
    output = paths(batch)[2]
    rows = read_exact(output / "result.json", receipts["result_sha256"])["cases"]
    return list(zip(batch_rows(batch, expected), rows, observations, strict=True)), output, receipts


def prepared_source(row):
    return expansion.read_pixels(TRIAL / row["slot"] / "source.png", row["png_sha256"]["source"], "RGB")


def rectangle_mask(box):
    left, top, right, bottom = bounds(box)
    require(0 <= left < right <= SIZE and 0 <= top < bottom <= SIZE, "In-frame rectangle required")
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def material(source, generated_image, row):
    from scripts import image_copy_transfer_v298 as ordinary

    target, controls = ordinary.controls(source, row["target_box_xywh"], row["donor_box_xywh"])
    remote = rectangle_mask(row["mixed_copy_box_xywh"])
    donor = rectangle_mask(row["donor_box_xywh"])
    require(
        not (target & remote).any()
        and not (remote & donor).any()
        and int(remote.sum()) == int(donor.sum()) == 64 * 64,
        "Mixed copy target and donor must be disjoint from the generated target",
    )
    left, top, right, bottom = bounds(row["donor_box_xywh"])
    reflected = source[top:bottom, left:right][:, ::-1]
    remote_control = source.copy()
    remote_control[remote] = reflected.reshape(-1, 3)
    controls["remote_reflected_copy"] = remote_control
    empty = np.zeros_like(target)
    variants = {name: (pixels, empty, np.any(source != pixels, axis=2)) for name, pixels in controls.items()}
    if generated_image is not None:
        require(
            generated_image.dtype == source.dtype and generated_image.shape == source.shape,
            "Expected native generated RGB composite",
        )
        ai_truth = np.any(source != generated_image, axis=2)
        require(
            ai_truth.any() and not (ai_truth & ~target).any(), "Generated edit must stay in its frozen region"
        )
        mixed = generated_image.copy()
        mixed[remote] = remote_control[remote]
        copy_truth = variants["remote_reflected_copy"][2]
        require(
            np.array_equal(np.any(source != mixed, axis=2), ai_truth | copy_truth)
            and np.array_equal(mixed[donor], source[donor]),
            "Mixed view must preserve both exact footprints and donor pixels",
        )
        variants.update(ai=(generated_image, ai_truth, empty), mixed=(mixed, ai_truth, copy_truth))
    return variants, {"target": target, "remote": remote}


def source_material(before, after, output):
    source = prepared_source(before)
    composite = None
    if after["status"] == "qualified_geometry":
        composite = expansion.read_pixels(
            output / before["slot"] / "composite.png", after["composite_png_sha256"], "RGB"
        )
    return material(source, composite, before)


def corroboration_rows(before, after, variants, masks):
    from scripts import image_copy_corroboration_v295 as cue
    from scripts import image_copy_transfer_v298 as ordinary

    encoded = {}
    for kind, region in CUE_REGIONS:
        for fmt in FORMATS:
            item = {
                "source_index": before["source_index"],
                "variant": kind,
                "region": region,
                "format": fmt,
                "generated_status": after["status"],
            }
            if kind not in variants:
                yield {**item, "status": "unassessed_generator_geometry"}
                continue
            pixels, generated_truth, ordinary_truth = variants[kind]
            if (kind, fmt) not in encoded:
                encoded[kind, fmt] = ordinary.encode_view(pixels, fmt)
            submitted, submitted_hash = encoded[kind, fmt]
            selection = masks[region]
            result = cue.corroborate(submitted, selection)
            require(
                result["proposal_mask_sha256"] == flux.digest(selection), "Existing region highlight changed"
            )
            copy_supported = result["status"] == "copy_consistent"
            copy_overlap = int((selection & ordinary_truth).sum())
            result_donor = result["donor"]
            yield {
                **item,
                **result,
                "submitted_sha256": submitted_hash,
                "generated_truth_sha256": flux.digest(generated_truth),
                "ordinary_truth_sha256": flux.digest(ordinary_truth),
                "supported_on_generated_pixels": int((selection & generated_truth).sum())
                if copy_supported
                else 0,
                "supported_on_ordinary_pixels": copy_overlap if copy_supported else 0,
                "exact_ordinary_donor": bool(
                    copy_overlap
                    and result_donor
                    and result_donor["transform"] == "horizontal_reflection"
                    and result_donor["box_xywh"] == before["donor_box_xywh"]
                ),
            }


def cue_stage(expected):
    import cv2

    cv2.setNumThreads(1)
    rows, batches, outcomes = [], {}, Counter()
    result_path = TRIAL / "private-cue-result.json"
    require(not result_path.exists(), "Preserve the existing cue result")
    for number in range(1, BATCHES + 1):
        observations, output, batches[str(number)] = generated(number, expected)
        for before, after, observation in observations:
            variants, masks = source_material(before, after, output)
            rows.extend(corroboration_rows(before, after, variants, masks))
            outcomes.update([observation["status"]])
    require(len(rows) == GROUPS * len(FORMATS) * len(CUE_REGIONS), "All expected cue views required")
    by_format = {}
    for fmt in FORMATS:
        results = {}
        for variant, region in CUE_REGIONS:
            selected = [
                row
                for row in rows
                if (row["format"], row["variant"], row["region"]) == (fmt, variant, region)
            ]
            require(len(selected) == GROUPS, "A planned case silently disappeared")
            results[f"{variant}_{region}"] = {
                "planned": GROUPS,
                "outcomes": dict(sorted(Counter(row["status"] for row in selected).items())),
                "copy_support_overlaps_generated": sum(
                    row.get("supported_on_generated_pixels", 0) > 0 for row in selected
                ),
                "copy_support_overlaps_ordinary": sum(
                    row.get("supported_on_ordinary_pixels", 0) > 0 for row in selected
                ),
                "exact_ordinary_donors": sum(row.get("exact_ordinary_donor", False) for row in selected),
            }
        by_format[fmt] = results
    aggregate = {
        "schema": "chipmunk.image-copy-positive-cue-aggregate/300",
        "freeze_sha256": expected,
        "source_groups": GROUPS,
        "planned_cue_proposals": len(rows),
        "generation_calls": GROUPS,
        "generation_outcomes": dict(sorted(outcomes.items())),
        "generated_batch_receipts": batches,
        "detector_calls": 0,
        "authorship_verdicts": 0,
        "by_format": by_format,
    }
    write(result_path, {"aggregate": aggregate, "rows": rows})
    return {**aggregate, "private_result_sha256": sha(result_path)}


def coverage(expected):
    frozen_rows(expected)
    batches = {}
    for n in range(1, BATCHES + 1):
        _, stage, output = stage_paths("generate", n)
        batches[str(n)] = {
            "generation": "result_present"
            if (output / "result.json").is_file()
            else "attempted"
            if stage.exists()
            else "unattempted",
            "product": {
                fmt: (stage_paths(f"product-{fmt}", n)[2] / "result.json").is_file() for fmt in FORMATS
            },
        }
    return {
        "source_groups": GROUPS,
        "batches": batches,
        "cue_result_present": (TRIAL / "private-cue-result.json").is_file(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "prepare",
            "freeze",
            "generate",
            "_generate",
            "cue",
            "coverage",
            "product-native_png",
            "product-jpeg90",
            "_product-native_png",
            "_product-jpeg90",
            "product-seal",
        ),
    )
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    command, values = args.command, args.arguments
    if command in {"preflight", "prepare", "freeze"}:
        require(not values, "This stage takes no arguments")
        result = {"preflight": preflight, "prepare": prepare, "freeze": freeze}[command]()
    elif command in {"cue", "coverage", "product-seal"}:
        require(len(values) == 1, "Frozen digest required")
        if command == "product-seal":
            from scripts.image_copy_positive_product_v300 import seal

            result = seal(values[0])
        else:
            result = {"cue": cue_stage, "coverage": coverage}[command](values[0])
    else:
        require(len(values) == 2, "Explicit batch and frozen digest required")
        batch, expected = int(values[0]), values[1]
        if command in {"generate", "_generate"}:
            {"generate": generation, "_generate": generation_worker}[command](batch, expected)
        else:
            from scripts.image_copy_positive_product_v300 import product, worker

            fmt = command.split("-", 1)[1]
            (worker if command.startswith("_") else product)(fmt, batch, expected)
        result = None
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
