# SPDX-License-Identifier: Apache-2.0
"""Reserve and verify a source-disjoint panel of small JPEG edit deliveries."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts import image_bg_real_insertion_transfer as intake
from scripts import image_region_rank_inputs as deliveries

SCHEMA = "chipmunk.image-cross-quality-sources/1"
FAMILIES = {("big-lama", "object_removal"): "lama", (intake.GENERATOR, "object_insertion"): "sd15"}
TARGET = 12
PUBLIC = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"


def known_sources(retained):
    keys, known, pins = intake.exposure()
    plan = deliveries.read(retained)
    keys.update(("original_sha256", row["source"]) for row in plan["groups"])
    known[2].update(row["rgb_sha256"] for row in plan["groups"])
    # Include later tests that were not in the historical input roster.
    prior = PUBLIC / "image-edge-context-new-sources-2026-11-14.json"
    for item in deliveries.read(prior)["sources"]:
        keys.add(("original_sha256", item["original"]["sha256"]))
        identity = item.get("flickr_photo_id")
        if identity:
            keys.add(("photo", str(identity)))
    return keys, known, pins | {"retained-deliveries": deliveries.sha(retained), "later-padding": deliveries.sha(prior)}


def plan(metadata, retained, output):
    import yaml

    roster = intake.roster
    keys, _, pins = known_sources(retained)
    names = roster.METADATA
    hashes = {name: roster.sha(metadata / name) for name in names}
    if any(hashes[name] != entry[2] for name, entry in names.items()):
        raise ValueError("Pinned picture metadata changed")
    rows = roster.table(metadata / "manifest_v2.csv")
    index = roster.indexed(rows, "sample_id")
    credits = roster.indexed((row for row in roster.table(metadata / "attribution.csv") if row["role"] == "recipient"), "bgreal_id")
    recipients = roster.indexed(roster.table(metadata / "recipients.csv"), "source_image_id")
    splits = {name: set(values) for name, values in yaml.safe_load((metadata / "splits_v2.yaml").read_bytes()).items()}
    models = {row["key"]: row for row in roster.table(metadata / "model_revisions.csv")}
    if models["lama"] != roster.MODEL or models["sd15_inpaint"]["revision"] != intake.GENERATOR_REVISION:
        raise ValueError("The published edit generator pins changed")
    candidates = defaultdict(list)
    for edited in rows:
        family = FAMILIES.get((edited["generator"], edited["edit_kind"]))
        if (not family or edited["family"] != "F5" or not roster.base_png(edited)
                or not .002 <= float(edited["edit_area"]) <= .08 or min(roster.dimensions(edited)) < 512):
            continue
        source = edited["source_image_id"]
        try:
            case = roster.pair(edited, roster.one(index, f"{source}_A0", "Need an original image"),
                               roster.one(credits, source, "Need one attributable photo"),
                               roster.one(recipients, source, "Need one source record"), splits)
        except ValueError:
            continue
        candidates[family].append(case | {"generator": family})
    ordered = []
    independent = set(keys)
    available = {}
    for family in FAMILIES.values():
        eligible, _ = roster.select(candidates[family], independent)
        eligible.sort(key=lambda row: (row["publisher_split"] not in roster.TEST_SPLITS,
                                       hashlib.sha256(f"{SCHEMA}/{row['recipient']}".encode()).hexdigest()))
        available[family] = len(eligible)
        choices = eligible[:TARGET * 2]
        for row in choices:
            independent.update(roster.source_keys(row))
        ordered.extend(choices)
    output.mkdir(mode=0o700)
    deliveries.save(output / "source-plan.json", {"schema": SCHEMA, "source_revision": roster.REVISION, "metadata": hashes,
                    "exposures": pins, "candidate_counts": available, "target_per_family": TARGET, "cases": ordered})
    print(json.dumps({"independent_eligible": available, "reserved": len(ordered)}, sort_keys=True), flush=True)


def acquire(study, retained):
    import cv2
    import requests

    roster, transport, inputs = intake.roster, intake.transport, intake.inputs
    manifest = deliveries.read(study / "source-plan.json")
    keys, known, pins = known_sources(retained)
    assert manifest["source_revision"] == roster.REVISION and manifest["exposures"] == pins
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unknown fresh source contract")
    root = study / "deliveries"
    root.mkdir(mode=0o700, exist_ok=True)
    if any(not name.name.startswith(".") for name in root.iterdir()):
        raise ValueError("Use a new location for a completed or partial acquisition")
    cv2.setNumThreads(1)
    deadline = time.monotonic() + 240

    def hydrate(case):
        try:
            with requests.Session() as session:
                session.trust_env = False
                blobs = {role: transport.download(session, case[role]["path"], case[role]["sha256"], deadline)[0]
                         for role in ("original", "edited", "semantic_mask")}
            images = {role: inputs.png_pixels(raw, case["size"]) for role, raw in blobs.items()}
            truth, _ = inputs.edit_truth(images["original"], images["edited"], images["semantic_mask"], case["semantic_edit_area"])
            fingerprint = {role: inputs.fingerprint(blobs[role], images[role]) for role in ("original", "edited")}
            conflicts = inputs.prior_matches(case, fingerprint, keys, known)
            return case, images, truth, fingerprint, ",".join(conflicts)
        except (OSError, RuntimeError, ValueError, requests.RequestException) as error:
            return case, None, None, None, f"{type(error).__name__}: {error}"

    groups, rows, receipts, exclusions, counts = [], [], [], Counter(), Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for case, images, truth, fingerprint, reason in pool.map(hydrate, manifest["cases"]):
            family = case["generator"]
            if counts[family] == TARGET:
                continue
            if not reason and not .001 <= truth.mean() <= .08:
                reason = "exact edit extent outside small-edit panel"
            if not reason and any(inputs.near(value, prior) for value in fingerprint.values() for item in receipts
                                  for prior in item["fingerprints"].values()):
                reason = "within-panel near duplicate"
            if reason:
                exclusions[reason] += 1
                continue
            counts[family] += 1
            tag = f"fresh-{len(groups):03d}"
            record = {"source": case, "fingerprints": fingerprint, "group": tag}
            group, rendered = render(root, record, images, truth)
            receipts.append(record)
            groups.append(group)
            rows.extend(rendered)
    deliveries.save(root / "manifest.json", {"schema": deliveries.SCHEMA, "groups": groups, "images": rows,
                    "source_manifests": {"fresh": deliveries.sha(study / "source-plan.json")}})
    deliveries.save(study / "qualification.json", {"accepted": receipts, "exclusions": dict(exclusions), "counts": dict(counts)})
    print(json.dumps({"qualified": counts, "excluded": sum(exclusions.values()), "deliveries": len(rows)}, sort_keys=True), flush=True)


def render(root, record, images, truth):
    import numpy as np
    from PIL import Image

    case, tag = record["source"], record["group"]
    folder = root / tag
    folder.mkdir(mode=0o700)
    group = {"tag": tag, "source": case["original"]["sha256"], "family": case["generator"], "split": "fresh",
             "shape": list(truth.shape), "first_changed_pixels": int(truth.sum()),
             "rgb_sha256": record["fingerprints"]["original"]["rgb_sha256"]}
    rows = []
    for role, pixels in deliveries.views(images["original"], images["edited"], truth).items():
        changed = np.any(images["original"] != pixels, axis=2) if role in deliveries.POSITIVE else np.zeros_like(truth)
        mask = folder / f"{role}-truth.png"
        Image.fromarray(changed.astype(np.uint8) * 255).save(mask, "PNG")
        for quality in deliveries.QUALITY:
            media = folder / f"{role}-{quality}.jpg"
            with io.BytesIO() as stream:
                Image.fromarray(pixels).save(stream, "JPEG", quality=quality, subsampling=2)
                body = stream.getvalue()
            media.write_bytes(body)
            rows.append({"group": tag, "role": role, "quality": quality, "file": media.relative_to(root).as_posix(),
                         "sha256": hashlib.sha256(body).hexdigest(), "mask": mask.relative_to(root).as_posix(),
                         "mask_sha256": deliveries.sha(mask), "changed_pixels": int(changed.sum())})
    return group, rows


def restore(receipt, root):
    import cv2
    import requests

    bundle = deliveries.read(receipt)
    assert bundle["schema"] == SCHEMA and bundle["source_revision"] == intake.roster.REVISION
    assert [item["group"] for item in bundle["accepted"]] == [f"fresh-{i:03d}" for i in range(len(bundle["accepted"]))]
    root.mkdir(mode=0o700)
    cv2.setNumThreads(1)

    def fetch(record):
        case = record["source"]
        with requests.Session() as session:
            session.trust_env = False
            blobs = {role: intake.transport.download(session, case[role]["path"], case[role]["sha256"], time.monotonic() + 35)[0]
                     for role in ("original", "edited", "semantic_mask")}
        images = {role: intake.inputs.png_pixels(raw, case["size"]) for role, raw in blobs.items()}
        assert {role: intake.inputs.fingerprint(blobs[role], images[role]) for role in ("original", "edited")} == record["fingerprints"]
        truth, _ = intake.inputs.edit_truth(images["original"], images["edited"], images["semantic_mask"], case["semantic_edit_area"])
        return render(root, record, images, truth)

    groups, rows = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for group, rendered in pool.map(fetch, bundle["accepted"]):
            groups.append(group)
            rows.extend(rendered)
    path = root / "manifest.json"
    deliveries.save(path, {"schema": deliveries.SCHEMA, "groups": groups, "images": rows,
                            "source_manifests": {"fresh": bundle["source_plan_sha256"]}})
    if deliveries.sha(path) != bundle["delivery_manifest_sha256"]:
        raise ValueError("Fresh JPEG delivery commitments differ from the tested files")
    print(json.dumps({"exact_deliveries": len(rows), "photos": len(groups)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="step", required=True)
    choose, fetch = (commands.add_parser(name) for name in ("plan", "acquire"))
    choose.add_argument("metadata", type=Path)
    for command in (choose, fetch):
        command.add_argument("retained", type=Path)
        command.add_argument("study", type=Path)
    replay = commands.add_parser("restore")
    replay.add_argument("receipt", type=Path)
    replay.add_argument("directory", type=Path)
    args = cli.parse_args()
    if args.step == "plan":
        plan(args.metadata, args.retained, args.study)
    elif args.step == "restore":
        restore(args.receipt, args.directory)
    else:
        acquire(args.study, args.retained)
