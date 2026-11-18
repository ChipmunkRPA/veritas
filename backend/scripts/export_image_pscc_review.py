# SPDX-License-Identifier: Apache-2.0
"""Recount held localizer regions and export exact inputs for a separate detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copyfile

from app.bench import image_pscc_localization as detector
from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_locations
from scripts.audio_native_alignment_panel import digest, read, save
from scripts.export_image_region_rank_review import pack_truth
from scripts.image_cross_quality_fit import load_mask
from scripts.image_jpeg_texture_prediction_panel import _inputs
from scripts.image_pscc_localization_panel import review
from scripts.image_region_rank_inputs import POSITIVE


def _copy_verified(source, signature, destination):
    if not destination.exists():
        copyfile(source, destination)
    if digest(destination) != signature:
        raise ValueError(f"A portable input differs from the recorded bytes: {destination.name}")
    return destination


def export(banks, intake, frozen, outcome, output):
    review(outcome, frozen)
    result = read(outcome)
    if [digest(bank / "manifest.json") for bank in banks] != [source["manifest_sha256"] for source in result["inputs"]]:
        raise ValueError("The exported inputs must be the same banks scored in the held comparison")
    parents = {row["fingerprints"]["original"]["rgb_sha256"]: row for row in read(intake / "intake.json")["cases"]}
    remaining = {(row["file"], row["sha256"]): row for row in result["rows"]}
    output.mkdir(mode=0o700)
    masks = output / "masks"
    files = output / "source-files"
    for folder in (masks, files):
        folder.mkdir(mode=0o700)
    cases, index, truths = [], [], []
    records = {arm: [] for arm in detector.ARMS}
    expected = {arm: {} for arm in detector.ARMS}
    for bank in banks:
        for directory, item, group in _inputs(bank):
            name, sha = item["file"], item["sha256"]
            row = remaining.pop((name, sha))
            source = parents[group["rgb_sha256"]]
            parent = source["source"]["original"]["sha256"]
            if row["mask_sha256"] != item["mask_sha256"] or row["source_group"] != group["rgb_sha256"]:
                raise ValueError("A held row uses a different edit mask or parent")
            source_media = _copy_verified(intake / f"{source['tag']}.png", parent, files / f"{parent}.png")
            delivery = _copy_verified(directory / name, sha, files / f"{sha}.jpg")
            case = {"id": f"{group['tag']}-{Path(name).stem}", "input_sha256": sha}
            height, width = group["shape"]
            edited = item["role"] in POSITIVE
            cases.append(case | {
                "split": "confirmation", "modality": "image", "label": int(edited), "group": parent,
                "family": group["family"], "condition": f"jpeg{item['quality']}/{item['role']}",
                "source_sha256": parent, "lineage": [parent],
                "license": source["source"].get("license", {
                    "id": "CC BY 2.0 photograph and CC BY 4.0 dataset subject mask",
                    "evaluation_permitted": True, "redistribution_permitted": True,
                }),
                "localization_truth": {"unit": "pixels", "source_units": width * height,
                                       "target_units": item["changed_pixels"]},
            })
            index.append({"id": case["id"], "input": delivery.relative_to(output).as_posix(),
                          "source": source_media.relative_to(output).as_posix()})
            if edited:
                truth_pin = pack_truth(directory / item["mask"], item["mask_sha256"],
                                       masks / f"{case['id']}-truth.pbm", [width, height])
                truth_region = {"mask": truth_pin}
            else:
                load_mask(directory, item)
                truth_region = {"regions": []}
            truths.append(case | {"extent": [width, height]} | truth_region)
            for arm, region in row["checks"].items():
                if pin := region["mask"]:
                    _copy_verified(outcome.parent / "masks" / pin["file"], pin["sha256"], masks / pin["file"])
                area = {"mask": pin} if pin else {"regions": []}
                records[arm].append(case | {"status": "ok", "response": region["score"] or 0.,
                                            "localization": {"status": "ok"} | area})
                expected[arm][case["id"]] = region["selected"], region["intersection"]
    if remaining:
        raise ValueError("Some held cases could not be bound to the original files")
    manifest = {"schema": independent.SCHEMA, "cases": cases}
    audited = independent.audit_manifest(manifest)
    manifest_sha = audited["manifest_sha256"]
    binding = {"manifest_sha256": manifest_sha, "split": "confirmation"}
    file_index = {"schema": INDEX_SCHEMA, "manifest_sha256": manifest_sha, "cases": index}
    truth = binding | {"schema": TRUTH_SCHEMA, "cases": truths}
    inputs = verify_inputs(manifest, file_index, output)
    for filename, content in (("manifest", manifest), ("file-index", file_index), ("truth", truth)):
        save(output / f"{filename}.json", content)

    receipt = {"processed_inputs": inputs, "groups": audited["groups"], "held_result_sha256": digest(outcome),
               "publisher": detector.PUBLISHER, "readouts": {}}
    weights = [commitment for name, commitment in result["method"]["source_files"].items() if name.endswith(".pth")]
    for arm, predictions in records.items():
        response = binding | {"system": f"PSCC-Net {arm}", "method_kind": "weighted",
                              "implementation_sha256": result["method"]["inference_sha256"],
                              "weight_sha256": weights, "response_units": "Uncalibrated region response",
                              "score_kind": "raw", "predictions": predictions}
        areas = count_locations(manifest, truth, response, mask_directory=masks)
        for row in areas["predictions"]:
            location = row["localization"]
            if expected[arm][row["id"]] != (location["reported_units"], location["overlap_units"]):
                raise ValueError("The independent source-pixel recount disagrees with the measured response")
        save(output / f"predictions-{arm}.json", response)
        save(output / f"recounted-{arm}.json", areas)
        receipt["readouts"][arm] = {"checked_cases": len(predictions),
                                   "nonempty_masks": sum(count > 0 for count, _ in expected[arm].values()),
                                   "recount_sha256": independent.digest(areas)}
    save(output / "summary.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("intake", "frozen", "outcome", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--bank", action="append", required=True, type=Path, dest="banks")
    export(**vars(parser.parse_args()))
