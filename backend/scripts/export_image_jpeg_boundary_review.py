# SPDX-License-Identifier: Apache-2.0
"""Verify the held JPEG bytes and recount released regions through the shared protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from app.bench import image_jpeg_boundary as review
from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_locations
from app.detectors.image import jpeg_boundary_localizer as detector
from scripts import image_region_rank_inputs as files
from scripts.export_image_jpeg_band_review import portable_mask
from scripts.export_image_region_rank_review import pack_truth
from scripts.image_cross_quality_fit import load_mask
from scripts.image_jpeg_boundary_transfer import method, read_case


def replay_region(models, directory, sample, setting, recorded, shape):
    arm = setting["arm"]
    if recorded["score"] is None or recorded["score"] <= setting["point"]:
        return None
    with torch.inference_mode():
        model = models["plain" if arm == "plain" else "paired"]
        interior, edge = model(torch.from_numpy(read_case(directory, sample)[None])).numpy()[0]
    score, grid = detector.proposals(interior, edge if arm == "boundary" else None)
    if recorded["score"] != score:
        raise ValueError("The proposed JPEG score differs from its frozen replay")
    upsampled = grid.repeat(detector.CELL, axis=0).repeat(detector.CELL, axis=1)
    canvas = np.zeros(shape, dtype=bool)
    canvas[: upsampled.shape[0], : upsampled.shape[1]] = upsampled
    return canvas


def export(study, fit, receipt, output):
    lock = fit / "selection.json.gz"
    review.review(receipt, lock)
    held = review.read(receipt)
    if held["source_method"] != method():
        raise ValueError("Local code differs from the frozen comparison")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    models = {}
    for name, digest in held["model_pins"].items():
        path = fit / f"{name}.safetensors"
        if files.sha(path) != digest:
            raise ValueError("Local weights differ from the frozen comparison")
        model = detector.model()
        model.load_state_dict(load_file(path), strict=True)
        models[name] = model.eval()

    output.mkdir(mode=0o700)
    masks = output / "masks"
    masks.mkdir(mode=0o700)
    parent = {
        row["fingerprints"]["original"]["rgb_sha256"]: row
        for row in files.read(study / "intake/intake.json")["cases"]
    }
    panels = ("held", "held-ordinary")
    rosters = [study / panel / "manifest.json" for panel in panels]
    if [files.sha(path) for path in rosters] != held["source_manifests"]:
        raise ValueError("The JPEG inventories differ from the held receipt")
    measured = {row["file"]: row for row in held["rows"]}
    cases, index, truth, predictions = [], [], [], []
    expected = {}
    for panel, roster in zip(panels, rosters, strict=True):
        directory = roster.parent
        inventory = files.read(roster)
        groups = {row["tag"]: row for row in inventory["groups"]}
        for sample in inventory["images"]:
            area = measured.pop(sample["file"])
            if any(area[key] != sample[key] for key in ("role", "quality", "sha256", "mask_sha256")):
                raise ValueError("An assessed JPEG differs from the inventory")
            group = groups[sample["group"]]
            if (area["source_group"], area["family"], area["shape"]) != (
                group["rgb_sha256"],
                group["family"],
                group["shape"],
            ):
                raise ValueError("A held result differs from its source grouping")
            height, width = area["shape"]
            photo = parent[area["source_group"]]
            source = photo["source"]["original"]["sha256"]
            identifier = f"{sample['group']}-{Path(sample['file']).stem}"
            base = {"id": identifier, "input_sha256": area["sha256"]}
            label = int(area["role"] in review.POSITIVE)
            native = load_mask(directory, sample)
            if list(native.shape) != area["shape"] or int(native.any()) != label:
                raise ValueError("The truth raster differs from its held case label or geometry")
            target = int(native.sum())
            cases.append(
                base
                | {
                    "split": "confirmation",
                    "modality": "image",
                    "label": label,
                    "group": source,
                    "family": area["family"],
                    "condition": f"jpeg{area['quality']}/{area['role']}",
                    "source_sha256": source,
                    "lineage": [source],
                    "license": {
                        "id": "CC BY 2.0 photograph and CC BY 4.0 dataset subject mask",
                        "evaluation_permitted": True,
                        "redistribution_permitted": True,
                    },
                    "localization_truth": {
                        "unit": "pixels",
                        "source_units": height * width,
                        "target_units": target,
                    },
                }
            )
            index.append(
                {
                    "id": identifier,
                    "input": f"{panel}/{sample['file']}",
                    "source": f"intake/{photo['tag']}.png",
                }
            )
            location = (
                {
                    "mask": pack_truth(
                        directory / sample["mask"],
                        sample["mask_sha256"],
                        masks / f"{identifier}-truth.pbm",
                        [width, height],
                    )
                }
                if label
                else {"regions": []}
            )
            truth.append(base | {"extent": [width, height]} | location)
            setting = held["chosen"][str(area["quality"])]
            chosen = area["checks"][setting["arm"]]
            if chosen["truth"] != target:
                raise ValueError("The scored pixel count differs from the original truth")
            selection = replay_region(models, directory, sample, setting, chosen, native.shape)
            if selection is None:
                location = {"regions": []}
                expected[identifier] = 0, 0
            else:
                location = {"mask": portable_mask(selection, masks / f"{identifier}-prediction.pbm")}
                expected[identifier] = chosen["selected"], chosen["intersection"]
            predictions.append(
                base
                | {
                    "status": "ok",
                    "response": chosen["score"] or 0.0,
                    "localization": {"status": "ok"} | location,
                }
            )
    if measured:
        raise ValueError("The held receipt includes JPEGs absent from the inventories")

    manifest = {"schema": independent.SCHEMA, "cases": cases}
    audited = independent.audit_manifest(manifest)
    binding = {"manifest_sha256": audited["manifest_sha256"], "split": "confirmation"}
    file_index = {"schema": INDEX_SCHEMA, "manifest_sha256": binding["manifest_sha256"], "cases": index}
    truth = binding | {"schema": TRUTH_SCHEMA, "cases": truth}
    confidence = binding | {
        "system": "Paired JPEG region and boundary",
        "method_kind": "weighted",
        "implementation_sha256": independent.digest(method()),
        "weight_sha256": list(held["model_pins"].values()),
        "response_units": "Uncalibrated region response",
        "score_kind": "raw",
        "predictions": predictions,
    }
    inputs = verify_inputs(manifest, file_index, study)
    areas = count_locations(manifest, truth, confidence, mask_directory=masks)
    for row in areas["predictions"]:
        actual = row["localization"]
        if (actual["reported_units"], actual["overlap_units"]) != expected[row["id"]]:
            raise ValueError("The shared mask counter differs from the detector's recorded area")
    for name, document in (
        ("manifest", manifest),
        ("file-index", file_index),
        ("truth", truth),
        ("predictions", confidence),
        ("areas", areas),
    ):
        files.save(output / f"{name}.json", document)
    files.save(
        output / "summary.json",
        {
            "groups": audited["groups"],
            "inputs": inputs,
            "truth_rasters_verified": len(truth["cases"]),
            "reported_regions_replayed": sum(x > 0 for x, _ in expected.values()),
            "areas_sha256": independent.digest(areas),
            "held_receipt_sha256": files.sha(receipt),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("study", "fit", "receipt", "output"):
        parser.add_argument(argument, type=Path)
    export(**vars(parser.parse_args()))
