# SPDX-License-Identifier: Apache-2.0
"""Bind exact JPEG removals and native masks to the independent evaluation format."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_locations
from scripts import image_jpeg_band_transfer as trial
from scripts import image_region_rank_inputs as files
from scripts.export_image_region_rank_review import pack_truth


def portable_mask(mask, path):
    import numpy as np

    height, width = mask.shape
    data = f"P4\n{width} {height}\n".encode("ascii") + np.packbits(mask, axis=1).tobytes()
    path.write_bytes(data)
    return {"file": path.name, "sha256": hashlib.sha256(data).hexdigest()}


def selected_mask(module, media, measured, selected):
    import cv2
    import numpy as np
    import torch

    from app.detectors.image import jpeg_band_localizer as detector
    from scripts.image_cross_quality_fit import load_mask

    cv2.setNumThreads(1)
    arm = selected["arm"]
    bands = trial.read_coefficients(media, measured)
    with torch.inference_mode():
        choices = [head(torch.from_numpy(values[None])).numpy()[0, 0] for head, values in zip(module, bands, strict=True)]
    responses = dict(zip(detector.BANDS, choices, strict=True))
    responses["combined"] = np.mean(choices, axis=0)
    score, grid = detector.strongest(responses[arm])
    expected = measured["checks"][arm]
    ground_truth = load_mask(media, measured)
    if {"score": score, **trial.region(grid, ground_truth)} != expected:
        raise ValueError("Independent native replay differs from the measured score or selected pixels")
    if score is None or score <= selected["point"]:
        return None
    active = np.repeat(np.repeat(grid, detector.CELL, axis=0), detector.CELL, axis=1)
    canvas = np.zeros_like(ground_truth)
    canvas[:active.shape[0], :active.shape[1]] = active
    return canvas


def export(study, calibration, output, panels):
    import torch
    from safetensors.torch import load_file

    from app.detectors.image import jpeg_band_localizer

    lock = calibration / "freeze.json.gz"
    weights = calibration / "bands.safetensors"
    frozen = files.read(lock)
    selected = trial.validate(frozen)
    if files.sha(weights) != frozen["weights"] or frozen["code"] != trial.implementation():
        raise ValueError("The native feature or weight implementation differs from calibration")
    torch.set_num_threads(3)
    module = jpeg_band_localizer.model().eval()
    module.load_state_dict(load_file(weights), strict=True)

    output.mkdir(mode=0o700)
    masks = output / "masks"
    masks.mkdir(mode=0o700)
    cases, truth, predictions, index = [], [], [], []
    parent = {item["fingerprints"]["original"]["rgb_sha256"]: item for item in files.read(study / "intake/intake.json")["cases"]}
    prediction_counts = {}
    for profile, result_file in panels:
        result = files.read(result_file)
        trial.validate(result)
        media = study / profile
        delivery_manifest = media / "manifest.json"
        if result["manifest"] != files.sha(delivery_manifest) or result["frozen"] != files.sha(lock) or selected != result["selected"]:
            raise ValueError("Measured rows do not bind the selected operating points and source deliveries")
        deliveries = {row["file"]: row for row in files.read(delivery_manifest)["images"]}
        if deliveries.keys() != {row["file"] for row in result["rows"]} or len(deliveries) != len(result["rows"]):
            raise ValueError("A planned JPEG result is missing or duplicated")
        for i, measured in enumerate(result["rows"], 1):
            row = deliveries[measured["file"]] | measured
            source = parent[row["source_group"]]
            h, w = row["shape"]
            mask_fields = {key: row[key] for key in ("group", "role", "quality", "sha256", "mask_sha256", "file")}
            if any(deliveries[row["file"]][key] != value for key, value in mask_fields.items()):
                raise ValueError("Native receipt and original JPEG row disagree")
            case_id = f"{profile}-{Path(row['file']).parent.name}-{Path(row['file']).stem}"
            item = {"id": case_id, "input_sha256": row["sha256"]}
            photo = source["source"]["original"]["sha256"]
            cases.append(item | {"split": "confirmation", "modality": "image", "label": int(row["role"] in files.POSITIVE),
                "condition": f"{profile}/jpeg{row['quality']}/{row['role']}", "group": photo, "source_sha256": photo,
                "family": row["family"], "lineage": [photo], "license": {
                    "id": "CC BY 2.0 photographs and CC BY 4.0 source masks",
                    "evaluation_permitted": True, "redistribution_permitted": True},
                "localization_truth": {"unit": "pixels", "source_units": w*h, "target_units": row["changed_pixels"]}})
            location = {"mask": pack_truth(media / row["mask"], row["mask_sha256"], masks / f"{case_id}-truth.pbm", [w, h])
                        } if row["changed_pixels"] else {"regions": []}
            truth.append(item | {"extent": [w, h], **location})
            index.append({"id": case_id, "input": (media / row["file"]).relative_to(study).as_posix(),
                          "source": f"intake/{source['tag']}.png"})
            setting = selected[str(row["quality"])]
            witness = selected_mask(module, media, row, setting)
            predicted = {"mask": portable_mask(witness, masks / f"{case_id}-prediction.pbm")} if witness is not None else {"regions": []}
            predictions.append(item | {"status": "ok", "response": row["checks"][setting["arm"]]["score"] or 0.,
                                       "localization": {"status": "ok", **predicted}})
            check = row["checks"][setting["arm"]]
            prediction_counts[case_id] = (check["selected"], check["intersection"]) if witness is not None else (0, 0)
            if i % 96 == 0:
                print(f"Independently checked {i}/{len(deliveries)} exact {profile} deliveries", flush=True)

    manifest = {"schema": independent.SCHEMA, "purpose": "Locate bounded generated removals in JPEGs from licensed photographs.",
                "publisher": "https://huggingface.co/datasets/anush-123/bg-real", "cases": cases}
    declared = independent.audit_manifest(manifest)
    manifest_sha = declared["manifest_sha256"]
    binding = {"manifest_sha256": manifest_sha, "split": "confirmation"}
    truth_doc = binding | {"schema": TRUTH_SCHEMA, "cases": truth}
    index_doc = {"schema": INDEX_SCHEMA, "manifest_sha256": manifest_sha, "cases": index}
    response = binding | {"system": "JPEG local frequency comparison", "method_kind": "weighted", "score_kind": "raw",
        "implementation_sha256": independent.digest(frozen["code"]), "weight_sha256": [frozen["weights"]],
        "response_units": "Uncalibrated local Fourier-band readout", "predictions": predictions}
    files_verified = verify_inputs(manifest, index_doc, study)
    locations = count_locations(manifest, truth_doc, response, mask_directory=masks)
    declared_cases = {case["id"]: case for case in cases}
    totals = defaultdict(Counter)
    for row in locations["predictions"]:
        measured = row["localization"]
        reported, overlap = measured["reported_units"], measured["overlap_units"]
        if (reported, overlap) != prediction_counts[row["id"]]:
            raise ValueError("Independent mask counts differ from the native receipt")
        case = declared_cases[row["id"]]
        target = case["localization_truth"]["target_units"]
        useful = target > 0 and 10 * overlap >= 3 * (target + reported - overlap) and 2 * overlap >= target
        totals[case["condition"]].update(planned=1, edited=target > 0, useful=useful, flagged=reported > 0,
                                         ordinary_flags=target == 0 and reported > 0)
    for name, document in (("manifest", manifest), ("file-index", index_doc), ("truth", truth_doc), ("candidate", response),
                           ("areas", locations), ("input-verification", files_verified)):
        files.save(output / f"{name}.json", document)
    public = {"source_groups": declared["groups"], "input_verification": files_verified,
              "minimum_iou": .30, "minimum_recall": .50, "areas_sha256": independent.digest(locations),
              "outcomes": dict(sorted(totals.items()))}
    files.save(output / "summary.json.gz", public)


if __name__ == "__main__":
    command = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "calibration", "output"):
        command.add_argument(name, type=Path)
    command.add_argument("--panel", action="append", nargs=2, metavar=("DIRECTORY", "RESULT"), required=True)
    args = command.parse_args()
    export(args.study, args.calibration, args.output, [(scope, Path(path)) for scope, path in args.panel])
