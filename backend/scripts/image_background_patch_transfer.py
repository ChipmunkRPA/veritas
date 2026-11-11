"""Prepare licensed, source-bound small patches from two published inpainting editors."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import requests
import yaml
from PIL import Image, ImageFilter

from scripts import image_bg_real_acquisition_v221 as source
from scripts import image_bg_real_insertion_score as prior_score
from scripts import image_bg_real_roster_v218 as roster
from scripts import image_bg_real_transport_v225 as transport
from scripts import image_crop_remote_context as prior
from scripts import image_relative_patch_transfer as shared

SPLIT = "test_generator_ood"
GENERATORS = {
    "sdxl": ("G1", "C2_sdxl", "sdxl_autoencoder", "sdxl_inpaint",
             "115134f363124c53c7d878647567d04daf26e41e", "openrail++"),
    "kandinsky": ("G2", "C2_kandinsky", "kandinsky_autoencoder", "kandinsky_inpaint",
                   "db790ad5cbcabed886f069ef2710774657621702", "apache-2.0"),
}
GROUPS_PER_GENERATOR = 6
CASE_ROLES = ("original", "edited", "local_processing", "global_processing", "blur")


def match_exposure(previous, holdout):
    keys, known, commitments = prior.historical_context(previous)
    _, plan, rows = prior_score.checked_inputs(holdout)
    for case, row in zip(plan["cases"], rows, strict=True):
        keys.update(roster.source_keys(case))
        for view in row.get("fingerprints", {}).values():
            known[0].append(view)
            known[1].add(view["encoded_sha256"])
            known[2].add(view["rgb_sha256"])
    commitments["insertion_remote_holdout"] = roster.sha(holdout / "result.json")
    return keys, known, commitments


def bindings(metadata, keys):
    input_commitments = {}
    for name, (_, size, sha256) in roster.METADATA.items():
        path = metadata / name
        roster.require(path.stat().st_size == size and roster.sha(path) == sha256, f"Pinned {name} changed")
        input_commitments[name] = sha256
    with (metadata / "splits_v2.yaml").open() as stream:
        split = set(yaml.safe_load(stream)[SPLIT])
    rows = {row["sample_id"]: row for row in roster.table(metadata / "manifest_v2.csv") if row["sample_id"] in split}
    attribution = roster.indexed((r for r in roster.table(metadata / "attribution.csv") if r["role"] == "recipient"),
                                 "bgreal_id")
    models = {row["key"]: row for row in roster.table(metadata / "model_revisions.csv")}
    selected, counts = [], {}
    for name, (family, control, control_model, model_key, revision, licence) in GENERATORS.items():
        model = models[model_key]
        roster.require(model["revision"] == revision and model["license"] == licence, "Generator terms changed")
        candidates, rejected = [], Counter()
        for edit in rows.values():
            if edit["family"] != family or edit["generator"] != model["repo"] or not roster.base_png(edit):
                continue
            src = edit["source_image_id"]
            try:
                original = rows[f"{src}_A0"]
                processed = rows[f"{src}_{control}"]
                terms = roster.one(attribution, src, "Explicit photo licence required")
                owner, photo = roster.attribution(terms, src)
                size = roster.dimensions(edit)
                roster.require(roster.SOURCE.fullmatch(src) and min(size) >= 256, "Native geometry unavailable")
                roster.require(all(r["source_image_id"] == src and r["split"] == SPLIT and roster.base_png(r)
                                   and roster.dimensions(r) == size and roster.SHA.fullmatch(r["sha256"])
                                   for r in (original, processed, edit)), "Matched native images changed")
                roster.require(original["family"] == "A0" and original["label"] == "0"
                               and processed["family"] == "C2" and processed["label"] == "0"
                               and processed["generator"] == control_model and edit["label"] == "1",
                               "Source labels differ")
                images = PurePosixPath("v2/images") / src[3]
                folder = PurePosixPath("v2/masks/edit") / src[3]
                candidates.append({
                    "recipient": src, "flickr_owner": owner, "flickr_photo_id": photo, "photo_attribution": terms,
                    "publisher_split": SPLIT, "size": list(size), "generator": name,
                    "original": {"path": roster.named_png(original["image_path"], images, f"{src}_A0.png"),
                                 "sha256": original["sha256"]},
                    "edited": {"path": roster.named_png(edit["image_path"], images, f"{src}_{family}.png"),
                               "sha256": edit["sha256"]},
                    "processed": {"path": roster.named_png(processed["image_path"], images, f"{src}_{control}.png"),
                                  "sha256": processed["sha256"]},
                    "semantic": {"path": roster.named_png(edit["mask_edit_path"], folder, f"{src}_{family}_edit.png"),
                                 "sha256": None},
                })
            except (ValueError, KeyError) as error:
                rejected[type(error).__name__] += 1
        ordered, excluded = roster.select(candidates, keys)
        for case in ordered[:GROUPS_PER_GENERATOR]:
            selected.append(case | {"slot": len(selected)})
            keys.update(roster.source_keys(case))
        counts[name] = {"eligible": len(ordered), "selected": min(len(ordered), GROUPS_PER_GENERATOR),
                        "metadata_rejections": dict(rejected), "component_exclusions": len(excluded)}
    return {"schema": "chipmunk.background-patch-intake/1", "metadata_sha256": input_commitments,
            "dataset_revision": roster.REVISION, "source_counts": counts, "cases": selected}


def region(semantic, case_id):
    height, width = semantic.shape
    tile_width, tile_height = width // 4, height // 4
    boxes = []
    for y in range(7):
        top = y * (height - tile_height) // 6
        for x in range(7):
            left = x * (width - tile_width) // 6
            box = (left, top, left + tile_width, top + tile_height)
            if semantic[top:top + tile_height, left:left + tile_width].all():
                boxes.append(box)
    source.check(bool(boxes), "no_full_tile_inside_publisher_edit")
    return min(boxes, key=lambda box: shared.digest(f"{case_id}:{box}".encode()))


def compose(native, box):
    left, top, right, bottom = box
    target = np.s_[top:bottom, left:right]
    original = native["original"]
    edited = original.copy()
    edited[target] = native["edited"][target]
    processed = original.copy()
    processed[target] = native["processed"][target]
    blurred = original.copy()
    blurred[target] = np.asarray(Image.fromarray(original).filter(ImageFilter.GaussianBlur(1.25)))[target]
    truth = np.any(edited != original, axis=2)
    permitted = np.zeros(truth.shape, dtype=bool)
    permitted[target] = True
    source.check(not (truth & ~permitted).any() and 0.04 <= truth.mean() <= 0.0625, "invalid_small_patch_footprint")
    return dict(zip(CASE_ROLES, (original, edited, processed, native["processed"], blurred), strict=True)), truth


def acquire(plan, destination, keys, known):
    assets = destination / "assets"
    assets.mkdir(mode=0o700)
    accepted, failures = Counter(), Counter()
    deadline = time.monotonic() + 450
    with (destination / "attempts.jsonl").open("x") as journal, requests.Session() as session:
        session.trust_env = False
        for case in plan["cases"]:
            slot = case["slot"]
            row = {"slot": slot, "generator": case["generator"], "assets": {}}
            try:
                native = {}
                for role in ("original", "edited", "processed", "semantic"):
                    asset = case[role]
                    body, receipt = transport.download(session, asset["path"], asset["sha256"], deadline)
                    (assets / f"{slot:02d}-{role}.png").write_bytes(body)
                    row["assets"][role] = receipt
                    native[role] = source.png_pixels(body, case["size"])
                fingerprint = source.fingerprint((assets / f"{slot:02d}-original.png").read_bytes(), native["original"])
                source.check(not source.prior_matches(case, {"original": fingerprint}, keys, known),
                             "prior_photo_or_thumbnail_overlap")
                sem = native["semantic"]
                source.check(np.array_equal(sem[:, :, 0], sem[:, :, 1]) and np.array_equal(sem[:, :, 0], sem[:, :, 2])
                             and np.isin(sem, (0, 255)).all(), "nonbinary_semantic_mask")
                box = region(sem[:, :, 0] > 0, case["recipient"])
                inputs, truth = compose(native, box)
                known[0].append(fingerprint)
                known[1].add(fingerprint["encoded_sha256"])
                known[2].add(fingerprint["rgb_sha256"])
                row.update(status="qualified", box=box, fingerprint=fingerprint, source_pixels=truth.size,
                           changed_pixels=int(truth.sum()), truth_sha256=shared.digest(truth.tobytes()),
                           case_sha256={role: shared.digest(pixels.tobytes()) for role, pixels in inputs.items()})
                accepted[case["generator"]] += 1
            except (source.Rejected, requests.RequestException) as error:
                reason = str(error) if isinstance(error, source.Rejected) else type(error).__name__
                row.update(status="excluded", reason=reason)
                failures[reason] += 1
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            print(f"{slot}: {row['status']}", flush=True)
    shared.write(destination / "result.json", {"schema": plan["schema"], "planned": len(plan["cases"]),
                 "qualified": dict(accepted), "excluded": dict(failures), "detector_forwards": 0,
                 "plan_sha256": roster.sha(destination / "plan.json"),
                 "attempts_sha256": roster.sha(destination / "attempts.jsonl")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "acquire"))
    for key in ("metadata", "previous", "holdout", "destination"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    keys, known, exposure = match_exposure(args.previous, args.holdout)
    plan = bindings(args.metadata, keys.copy()) | {"exposure_sha256": exposure,
            "implementation_sha256": roster.sha(Path(__file__))}
    if args.stage == "freeze":
        args.destination.mkdir(mode=0o700)
        shared.write(args.destination / "plan.json", plan)
        print(plan["source_counts"])
    else:
        roster.require(plan == roster.read(args.destination / "plan.json"), "Private roster differs")
        acquire(plan, args.destination, keys, known)


if __name__ == "__main__":
    main()
