# SPDX-License-Identifier: Apache-2.0
"""Acquire and replay a fresh commercial-permission BG-REAL F5 JPEG panel."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import requests

from scripts import image_bg_real_acquisition_v221 as publisher
from scripts import image_jpeg_signed_transitions as study

SCHEMA = "chipmunk.image-jpeg-small-bg-real-holdout/1"
N_PER_FAMILY = 6
MODELS = {"lama": ("big-lama", "object_removal"),
          "sd15-inpaint": ("stable-diffusion-v1-5/stable-diffusion-inpainting", "object_insertion")}
PHOTO_LICENSE = "https://creativecommons.org/licenses/by/2.0/"
PHOTO_FIELDS = ("Author", "AuthorProfileURL", "License", "OriginalLandingURL")


def plan(args):
    manifest_path, attribution_path = (args.metadata / key for key in ("manifest_v2.csv", "attribution.csv"))
    fixed = study.read(args.roster / "summary.json")
    declared = fixed["metadata_commitments"]
    with manifest_path.open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    with attribution_path.open(newline="") as stream:
        attribution = {row["bgreal_id"]: row for row in csv.DictReader(stream) if row["role"] == "recipient"}
    metadata = {path.name: study.digest(path.read_bytes()) for path in (manifest_path, attribution_path)}
    for name, actual in metadata.items():
        study.require(actual == declared[name]["sha256"], "Publisher metadata bytes changed")
    occupied_authors, occupied_ids = set(), set()
    for row in study.read(args.roster / "private-roster.json").get("planned", ()):
        occupied_ids.add(str(row["flickr_photo_id"]))
    for path in args.exclude:
        old = study.read(path)
        for row in old["qualified"]:
            source = row["source"]
            credit = source["photo_attribution"]
            occupied_authors.add(credit["AuthorProfileURL"].casefold())
            occupied_authors.add(credit["Author"].casefold())
            occupied_ids.add(str(source["flickr_photo_id"]))
            occupied_ids.add(source.get("source_image_id", ""))
    private_cache = study.read(args.cache)
    for row in private_cache["sources"]:
        if row.get("source_id"):
            occupied_ids.add(str(row["source_id"]))
        if row.get("author"):
            occupied_authors.add(row["author"].casefold())
    originals = {row["source_image_id"]: row for row in manifest
                 if row["family"] == "A0" and row["robustness"] == "none" and row["encoding"] == "png"}
    choices = []
    for name, (model, kind) in MODELS.items():
        options = []
        for row in manifest:
            source = row["source_image_id"]
            credit, before = attribution.get(source), originals.get(source)
            if (row["family"] != "F5" or row["encoding"] != "png" or row["robustness"] != "none"
                or row["generator"] != model or row["edit_kind"] != kind
                or (name != "lama" and not row["split"].startswith("test"))
                or not .002 <= float(row["edit_area"]) <= .075 or not before or not credit
                or credit["License"] != PHOTO_LICENSE or row["split"] != before["split"]
                or (row["width"], row["height"]) != (before["width"], before["height"])
                or source in occupied_ids or not credit["Author"] or not credit["AuthorProfileURL"]):
                continue
            options.append((row, before, credit))
        used = 0
        for row, before, credit in sorted(options, key=lambda group: (
                not group[0]["split"].startswith("test"),
                study.digest(f"{SCHEMA}/{name}/{group[0]['source_image_id']}".encode()))):
            if name == "lama" and row["split"] == "val":
                continue
            photo_id = credit["OriginalLandingURL"].rstrip("/").split("/")[-1]
            author = {credit[key].casefold() for key in ("AuthorProfileURL", "Author")}
            if photo_id in occupied_ids or author & occupied_authors:
                continue
            occupied_ids.add(photo_id)
            occupied_authors |= author
            choices.append({"family": name, "publisher_split": row["split"], "source_id": row["source_image_id"],
                "shape": [int(row["width"]), int(row["height"])], "edited_area": float(row["edit_area"]),
                "source": {"path": before["image_path"], "sha256": before["sha256"]},
                "edited": {"path": row["image_path"], "sha256": row["sha256"]},
                "mask": {"path": row["mask_edit_path"], "sha256": None},
                "attribution": {key: credit[key] for key in PHOTO_FIELDS}})
            used += 1
            if used == N_PER_FAMILY:
                break
        study.require(used == N_PER_FAMILY, f"Insufficient distinct qualified {name} sources")
    return {"schema": SCHEMA, "publisher_revision": publisher.roster.REVISION,
            "metadata": metadata, "exclude_sha256": [study.digest(path.read_bytes()) for path in args.exclude],
            "cache_sha256": study.digest(args.cache.read_bytes()), "rows": choices}


def acquire(folder, scope):
    dest = folder / "objects"
    dest.mkdir(mode=0o700, exist_ok=True)
    # Current official delivery endpoints. The shared downloader still
    # requires the pinned repo revision and validates the expected content hash.
    publisher.CDN |= {"us.aws.cdn.hf.co", "us.gcp.cdn.hf.co"}
    deadline = time.monotonic() + 100
    def fetch(item):
        session = requests.Session()
        outputs = {}
        try:
            for role in ("source", "edited", "mask"):
                row = item[role]
                body, receipt = publisher.download(session, row["path"], row["sha256"], deadline)
                path = dest / receipt["sha256"]
                try:
                    with path.open("xb") as stream:
                        stream.write(body)
                except FileExistsError:
                    study.require(path.read_bytes() == body, "Archive object differs")
                outputs[role] = receipt
            return outputs
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = list(workers.map(fetch, scope["rows"]))
    study.write(folder / "acquired.json", {"plan_sha256": study.digest((folder / "plan.json").read_bytes()), "rows": rows})


def pixel_pairs(folder, scope):
    acquisition = study.read(folder / "acquired.json")
    study.require(acquisition["plan_sha256"] == study.digest((folder / "plan.json").read_bytes()), "Acquisition plan changed")
    for row, captured in zip(scope["rows"], acquisition["rows"], strict=True):
        decoded = {}
        for role, receipt in captured.items():
            body = (folder / "objects" / receipt["sha256"]).read_bytes()
            study.require(study.digest(body) == receipt["sha256"] and len(body) == receipt["bytes"]
                          and receipt["resolver_revision"] == scope["publisher_revision"]
                          and (role == "mask" or receipt["sha256"] == row[role]["sha256"]),
                          "Local pixels differ from the publisher asset")
            decoded[role] = publisher.png_pixels(body, row["shape"])
        exact, _ = publisher.edit_truth(decoded["source"], decoded["edited"], decoded["mask"], row["edited_area"])
        yield row["source"]["sha256"], row["family"], decoded["source"], decoded["edited"], exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "acquire", "freeze", "run", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("candidate_run", type=Path)
    data = Path("backend/data/independent-evaluation")
    parser.add_argument("--cache", type=Path, default=data / "image-fresh-source-cache-v39/private-groups.json")
    parser.add_argument("--roster", type=Path, default=data / "image-bg-real-roster-v218")
    parser.add_argument("--exclude", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "plan":
        args.run.mkdir(mode=0o700)
        value = plan(args)
        study.write(args.run / "plan.json", value)
        print("Planned", len(value["rows"]), flush=True)
        return
    scope = study.read(args.run / "plan.json")
    study.require(scope == plan(args), "Source selection changed")
    if args.stage == "acquire":
        acquire(args.run, scope)
        print("Acquired", len(scope["rows"]), flush=True)
        return
    measured = args.candidate_run / "calibration.json"
    cal = study.read(measured)
    body = study.read(args.candidate_run / "freeze.json")
    study.require(cal["freeze_sha256"] == study.digest((args.candidate_run / "freeze.json").read_bytes())
                  and body["code"][Path(study.__file__).name] == study.digest(Path(study.__file__).read_bytes()),
                  "Frozen primary candidate is unavailable")
    pairs = list(pixel_pairs(args.run, scope))
    def views():
        return study.views_from_pairs(pairs, "held")
    frozen = {"schema": SCHEMA, "code": study.digest(Path(__file__).read_bytes()),
              "plan_sha256": study.digest((args.run / "plan.json").read_bytes()),
              "calibration_sha256": study.digest(measured.read_bytes()),
              "inputs": [row for row, *_ in views()]}
    commitment = args.run / "freeze.json"
    if args.stage == "freeze":
        study.write(commitment, frozen)
        print("Frozen", len(frozen["inputs"]), study.digest(commitment.read_bytes()), flush=True)
        return
    study.require(study.read(commitment) == frozen, "Frozen fresh JPEG inputs differ")
    rows = study.measure_views(views(), "held", frozen["inputs"], cal["weights"], cal["points"])
    result = {"freeze_sha256": study.digest(commitment.read_bytes()), "rows": rows, "summary": study.summarize(rows)}
    output = args.run / "held.json"
    if args.stage == "review":
        study.require(study.read(output) == result, "Fresh JPEG replay differs")
    else:
        study.write(output, result)
    print(json.dumps(result["summary"]["outcomes"], indent=2), flush=True)


if __name__ == "__main__":
    main()
