"""Qualify a new small-insertion panel with matched processing controls."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import requests
import yaml
from PIL import Image

from scripts import image_bg_real_acquisition_v221 as inputs
from scripts import image_bg_real_roster_v218 as roster
from scripts import image_bg_real_transport_v225 as transport
from scripts import image_relative_patch_transfer as shared

GENERATOR = "stable-diffusion-v1-5/stable-diffusion-inpainting"
GENERATOR_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
AREA = ((0.02, 0.05), (0.05, 0.10), (0.10, 0.15))
PER_STRATUM = 4
PREVIOUS = ("image-bg-real-acquisition-v225", "image-bg-real-backup-intake-v262")


def exposure():
    keys, source_commitments = roster.exposure_keys()
    graph, encoded, rgb = inputs.exposure_data()
    graph = list(graph)
    for study in PREVIOUS:
        retained = roster.DATA / study / "qualified-private.json"
        source_commitments[study] = roster.sha(retained)
        for record in roster.read(retained)["qualified"]:
            keys.update(roster.source_keys(record["source"]))
            for image in record["measurement"]["fingerprints"].values():
                graph.append(image)
                encoded.add(image["encoded_sha256"])
                rgb.add(image["rgb_sha256"])
    return keys, (graph, encoded, rgb), source_commitments


def candidate_rows(metadata, keys):
    commitments = {}
    for name, (_, size, digest) in roster.METADATA.items():
        path = metadata / name
        roster.require(path.is_file() and path.stat().st_size == size and roster.sha(path) == digest,
                       f"Unexpected metadata bytes for {name}")
        commitments[name] = digest
    model = next(row for row in roster.table(metadata / "model_revisions.csv") if row["key"] == "sd15_inpaint")
    roster.require(model["repo"] == GENERATOR and model["revision"] == GENERATOR_REVISION
                   and model["license"] == "creativeml-openrail-m", "Pinned insertion generator changed")
    with (metadata / "splits_v2.yaml").open() as stream:
        splits = {key: set(value) for key, value in yaml.safe_load(stream).items()}
    rows = roster.table(metadata / "manifest_v2.csv")
    by_id = roster.indexed(rows, "sample_id")
    by_recipient = roster.indexed(roster.table(metadata / "recipients.csv"), "source_image_id")
    by_licence = roster.indexed((row for row in roster.table(metadata / "attribution.csv")
                                if row["role"] == "recipient"), "bgreal_id")
    candidates, rejected = [], Counter()
    for edited in rows:
        if (edited["family"] != "F5" or edited["generator"] != GENERATOR
                or edited["edit_kind"] != "object_insertion" or not roster.base_png(edited)
                or edited["split"] not in roster.TEST_SPLITS):
            continue
        source = edited["source_image_id"]
        try:
            original = roster.one(by_id, f"{source}_A0", "exact original required")
            processed = roster.one(by_id, f"{source}_C2", "exact processing control required")
            licence = roster.one(by_licence, source, "exact photo attribution required")
            recipient = roster.one(by_recipient, source, "exact recipient record required")
            case = roster.pair(edited, original, licence, recipient, splits)
            split = case["publisher_split"]
            roster.require(roster.base_png(processed) and processed["source_image_id"] == source
                           and processed["family"] == "C2" and processed["label"] == "0"
                           and float(processed["edit_area"]) == 0.0
                           and processed["generator"] == "sd15_autoencoder" and processed["split"] == split
                           and "F5" in processed["control_of"].split(",") and processed["sample_id"] in splits[split]
                           and tuple(case["size"]) == roster.dimensions(processed)
                           and roster.SHA.fullmatch(processed["sha256"]), "matched processing control differs")
            control_path = roster.named_png(processed["image_path"], PurePosixPath("v2/images") / source[3],
                                            f"{source}_C2.png")
            case["processed"] = {"path": control_path, "sha256": processed["sha256"]}
            candidates.append(case)
        except ValueError as exc:
            rejected[str(exc)] += 1
    ordered, excluded = roster.select(candidates, keys)
    selected, availability = [], {}
    for minimum, maximum in AREA:
        eligible = [case for case in ordered if minimum <= case["semantic_edit_area"] < maximum]
        tier = f"{minimum:g}-{maximum:g}"
        availability[tier] = len(eligible)
        for case in eligible[:PER_STRATUM]:
            selected.append(case | {"slot": len(selected), "edit_area_band": tier})
    return {"metadata_sha256": commitments, "eligible_by_area_band": availability, "metadata_rejected": dict(rejected),
            "components_excluded": dict(Counter(item["reason"] for item in excluded)), "cases": selected}


def acquire(plan, target, keys, known):
    assets = target / "assets"
    assets.mkdir(mode=0o700)
    qualified, rejected = [], []
    deadline = time.monotonic() + 300
    with (target / "attempts.jsonl").open("x") as log, requests.Session() as session:
        session.trust_env = False
        for case in plan["cases"]:
            slot = case["slot"]
            measurements, images = {}, {}
            row = {"slot": slot, "band": case["edit_area_band"], "files": measurements}
            try:
                for role in ("original", "processed", "edited", "semantic_mask"):
                    asset = case[role]
                    body, record = transport.download(session, asset["path"], asset["sha256"], deadline)
                    path = assets / f"{slot:02d}-{role}.png"
                    with path.open("xb") as stream:
                        stream.write(body)
                    measurements[role] = record
                    images[role] = inputs.png_pixels(body, case["size"])
                exact, support = inputs.edit_truth(images["original"], images["edited"], images["semantic_mask"],
                                                   case["semantic_edit_area"])
                fingerprints = {role: inputs.fingerprint((assets / f"{slot:02d}-{role}.png").read_bytes(), images[role])
                                for role in ("original", "processed", "edited")}
                matches = inputs.prior_matches(case, fingerprints, keys, known)
                inputs.check(not matches, ",".join(matches))
                for earlier in qualified:
                    inputs.check(not any(inputs.near(old, new) for old in earlier["fingerprints"].values()
                                         for new in fingerprints.values()), "within_panel_near_duplicate")
                truth = assets / f"{slot:02d}-truth.png"
                Image.fromarray(exact.astype(np.uint8) * 255).save(truth)
                processed_changes = int(np.any(images["original"] != images["processed"], axis=2).sum())
                row.update(status="qualified", exact_changed_pixels=int(exact.sum()), semantic_mask_pixels=support,
                           processing_changed_pixels=processed_changes, source_pixels=exact.size,
                           truth_sha256=roster.sha(truth), fingerprints=fingerprints)
                qualified.append(row)
            except (inputs.Rejected, requests.RequestException) as error:
                reason = str(error) if isinstance(error, inputs.Rejected) else type(error).__name__
                row.update(status="excluded", reason=reason)
                rejected.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(f"group {slot}: {row['status']}", flush=True)
    return {
        "schema": "chipmunk.image-bg-real-insertion-intake/1",
        "plan_sha256": roster.sha(target / "plan.json"),
        "acquisition_journal_sha256": roster.sha(target / "attempts.jsonl"),
        "planned": len(plan["cases"]), "qualified": len(qualified),
        "qualified_by_area_band": dict(Counter(row["band"] for row in qualified)),
        "asset_count": sum(len(row["files"]) for row in qualified + rejected),
        "excluded_reasons": dict(Counter(row["reason"] for row in rejected)),
        "exact_changed_pixels_outside_semantic_mask": 0 if qualified else None,
        "exact_ai_pixel_area_range": [min(row["exact_changed_pixels"] / row["source_pixels"] for row in qualified),
                                      max(row["exact_changed_pixels"] / row["source_pixels"] for row in qualified)] if qualified else None,
        "processing_changed_pixels_range": [min(row["processing_changed_pixels"] for row in qualified),
                                            max(row["processing_changed_pixels"] for row in qualified)] if qualified else None,
        "forwards": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "acquire"))
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    keys, known, exposure_commitments = exposure()
    plan = candidate_rows(args.metadata, keys) | {"exposure_sha256": exposure_commitments,
                                                 "script_sha256": roster.sha(Path(__file__)),
                                                 "dataset_revision": roster.REVISION,
                                                 "generator_revision": GENERATOR_REVISION}
    if args.action == "freeze":
        args.output.mkdir(mode=0o700)
        shared.write(args.output / "plan.json", plan)
        print(json.dumps({key: value for key, value in plan.items() if key != "cases"}))
        return
    roster.require(plan == roster.read(args.output / "plan.json"), "The exact private roster is required")
    shared.write(args.output / "result.json", acquire(plan, args.output, keys, known))


if __name__ == "__main__":
    main()
