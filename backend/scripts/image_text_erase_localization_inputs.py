# SPDX-License-Identifier: Apache-2.0
"""Prepare a source-aligned transfer test from public, paired scene-text removals."""

from __future__ import annotations

import argparse
import hashlib
import io
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from scripts import image_region_rank_inputs as files

REVISION = "cf94f7eaffc654146cba02fe2f704cc1a6e4998d"
PUBLISHER = f"https://huggingface.co/datasets/QinmingZhou/TextEraseBench/resolve/{REVISION}"
INDEX_SHA = "2b83789cb3817a7e3cba5cc61e0cdd8f99091377bdc222ac473838a5ffcc9b94"
FIELDS = ("shot", "bg", "mask")
PLANNED = 24
LICENSE = {"id": "TextEraseBench CC BY 4.0 (publisher terms)", "evaluation_permitted": True,
           "redistribution_permitted": False}


def acquire(row, directory):
    expected = row["lfs"]["oid"]
    length = row["size"]
    if not 0 < length < 12 * 2**20 or not re.fullmatch(r"(shot|bg|mask)/[0-9]{5}\.png", row["path"]):
        raise ValueError("Unexpected publisher file size or path")
    with urlopen(Request(f"{PUBLISHER}/{row['path']}", headers={"User-Agent": "Chipmunk-Verify-dataset-evaluation"}),
                 timeout=25) as response:
        body = response.read(length + 1)
    if len(body) != length or hashlib.sha256(body).hexdigest() != expected:
        raise ValueError(f"A scene-text file differs from the publisher's commitment: {row['path']}")
    (directory / row["path"].replace("/", "-")).write_bytes(body)
    return body


def prepare_pair(blobs):
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    # Some published .png paths contain JPEG bytes.
    try:
        shot, background, mask = [Image.open(io.BytesIO(body), formats=["PNG", "JPEG"]) for body in blobs]
    except UnidentifiedImageError:
        return {"reason": "unsupported_encoding"}, None
    width, height = shot.size
    if (max(width, height, *background.size, *mask.size) > 4096 or min(width, height) < 256
            or shot.size != mask.size
            or abs(background.width * height / (background.height * width) - 1) > .012):
        return {"reason": "unsupported_dimensions", "sizes": [image.size for image in (shot, background, mask)]}, None
    scale = min(1., 768. / max(width, height))
    size = round(width * scale), round(height * scale)
    original, rendered = [np.array(image.convert("RGB").resize(size, Image.Resampling.LANCZOS))
                          for image in (shot, background)]
    selected = np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST)) >= 128
    support = cv2.dilate(selected.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    if not .003 <= support.mean() <= .12:
        return {"reason": "annotation_outside_small_edit_scope", "support_fraction": float(support.mean())}, None
    comparison = cv2.dilate(support.astype(np.uint8), np.ones((17, 17), dtype=np.uint8)) == 0
    clean, changed = [cv2.GaussianBlur(image.astype(np.float32), (0, 0), 2) for image in (original, rendered)]
    close = float(np.mean(np.abs(clean - changed).mean(axis=2)[comparison] <= 18))
    status = {"outside_target_close_fraction": close, "shape": list(support.shape),
              "annotation_fraction": float(selected.mean()), "support_fraction": float(support.mean())}
    if close < .88:
        return status | {"reason": "unresolved_background_alignment"}, None
    edited = np.where(support[..., None], rendered, original)
    truth = np.any(edited != original, axis=2)
    if truth.mean() < .003:
        return status | {"reason": "insufficient_visible_change"}, None
    return status | {"reason": None, "changed_pixels": int(truth.sum())}, (original, edited, truth)


def render(group, pair, directory):
    import cv2
    import numpy as np
    from PIL import Image

    original, edited, truth = pair
    output = directory / group
    output.mkdir(mode=0o700)
    row = []
    versions = files.views(original, edited, truth)
    versions["telea_fill"] = cv2.inpaint(original, truth.astype(np.uint8), 4, cv2.INPAINT_TELEA)
    for role, pixels in versions.items():
        annotation = (original != pixels).any(axis=-1) if role in files.POSITIVE else np.zeros_like(truth)
        mask = output / f"{role}-truth.png"
        Image.fromarray(annotation.astype(np.uint8) * 255).save(mask)
        for quality in files.QUALITY:
            delivery = output / f"{role}-{quality}.jpg"
            Image.fromarray(pixels).save(delivery, "JPEG", quality=quality, subsampling=2)
            row.append({"group": group, "role": role, "quality": quality, "file": delivery.relative_to(directory).as_posix(),
                        "sha256": files.sha(delivery), "mask": mask.relative_to(directory).as_posix(),
                        "mask_sha256": files.sha(mask), "changed_pixels": int(annotation.sum())})
    return row


def build(index, output, *, start=0, previous_intake=None):
    import cv2

    if files.sha(index) != INDEX_SHA:
        raise ValueError("Expected the pinned public dataset inventory")
    cv2.setNumThreads(1)
    inventory = {row["path"]: row for row in files.read(index) if row["type"] == "file"}
    stems = [Path(path).stem for path in inventory if re.fullmatch(r"shot/[0-9]{5}\.png", path)]
    ranked = sorted(stems, key=lambda stem: hashlib.sha256(f"scene-text-edit-transfer/1/{stem}".encode()).digest())
    if not 0 <= start <= len(ranked) - PLANNED:
        raise ValueError("The requested slice is outside the publisher roster")
    roster = ranked[start:start + PLANNED]
    prior = files.read(previous_intake) if previous_intake else None
    if prior and (prior["revision"] != REVISION or set(roster) & {row["case"] for row in prior["dispositions"]}):
        raise ValueError("The reserve overlaps the previous publisher inputs")
    output.mkdir(mode=0o700)
    intake, deliveries = output / "intake", output / "deliveries"
    intake.mkdir(mode=0o700)
    deliveries.mkdir(mode=0o700)
    plan = {stem: [inventory[f"{field}/{stem}.png"] for field in FIELDS] for stem in roster}
    reservation = {"revision": REVISION, "index_sha256": INDEX_SHA, "cases": plan}
    if start or previous_intake:
        reservation.update(start=start, previous_intake_sha256=files.sha(previous_intake) if previous_intake else None)
    files.save(output / "plan.json", reservation)
    groups, rows, checked, cases = [], [], [], []
    seen = {row["fingerprints"]["original"]["rgb_sha256"] for row in prior["cases"]} if prior else set()
    with ThreadPoolExecutor(max_workers=5) as pool:
        for stem in roster:
            assets = plan[stem]
            bodies = list(pool.map(acquire, assets, (intake for _ in assets)))
            status, pair = prepare_pair(bodies)
            checked.append({"case": stem, "assets": {item["path"]: item["lfs"]["oid"] for item in assets}, **status})
            if pair is None:
                continue
            original, _, truth = pair
            rgb_hash = hashlib.sha256(original.tobytes()).hexdigest()
            if rgb_hash in seen:
                checked[-1]["reason"] = "duplicate_normalized_source"
                continue
            seen.add(rgb_hash)
            tag = f"text-erase-{stem}"
            parent = assets[0]["lfs"]["oid"]
            (intake / f"{tag}.png").write_bytes(bodies[0])
            (intake / f"shot-{stem}.png").unlink()
            cases.append({"tag": tag, "fingerprints": {"original": {"rgb_sha256": rgb_hash}},
                          "source": {"original": {"sha256": parent}, "license": LICENSE}})
            groups.append({"tag": tag, "source": parent, "rgb_sha256": rgb_hash, "family": "provider_text_removal",
                           "split": "transfer", "shape": list(truth.shape), "first_changed_pixels": int(truth.sum())})
            rows.extend(render(tag, pair, deliveries))
    files.save(intake / "intake.json", {"cases": cases, "dispositions": checked, "revision": REVISION})
    files.save(deliveries / "manifest.json", {"schema": files.SCHEMA, "groups": groups, "images": rows,
                                            "source_manifests": {"publication": files.sha(output / "plan.json")}})
    print(f"Verified {len(checked)} paired releases. {len(groups)} qualified for a local JPEG comparison.", flush=True)


def compare_pairs(deliveries, output):
    """Replay the app's earlier-copy route on the exact single-upload test cases."""
    import cv2
    import numpy as np
    from PIL import Image

    from app.bench.image_jpeg_boundary import useful
    from app.evidence.image_reference import inspect
    from scripts.image_cross_quality_fit import load_mask

    cv2.setNumThreads(1)
    manifest = deliveries / "manifest.json"
    inputs = files.read(manifest)["images"]
    references = {(row["group"], row["quality"]): row for row in inputs if row["role"] == "ordinary"}
    options = dict.fromkeys((
        "allow_embedded_photo", "allow_exact_jpeg_identity", "allow_jpeg_detail_loss",
        "allow_jpeg_detail_loss_extra_qualities", "allow_oriented_crop", "allow_oriented_resize",
        "allow_perspective", "allow_png_opacity", "allow_proportional_resize", "allow_same_jpeg",
        "allow_scaled_screenshot", "allow_shifted_jpeg", "allow_structure_jpeg",
    ), True)

    def measure(item):
        source = references[item["group"], item["quality"]]
        earlier, actual = (deliveries / row["file"] for row in (source, item))
        truth = load_mask(deliveries, item)
        with TemporaryDirectory(prefix="chipmunk-scene-text-comparison-") as temporary:
            evidence = Path(temporary) / "evidence"
            result = inspect(earlier, actual, evidence_dir=evidence, **options)
            if result["reference"]["sha256"] != source["sha256"] or result["submitted"]["sha256"] != item["sha256"]:
                raise ValueError("The earlier-copy check received different files")
            selected = overlap = 0
            if pin := result.get("mask"):
                mask = evidence / pin["file"]
                if files.sha(mask) != pin["sha256"]:
                    raise ValueError("The measured regions differ from the exported mask")
                with Image.open(mask) as bitmap:
                    marks = np.asarray(bitmap) != 0
                if marks.shape != truth.shape:
                    raise ValueError("The export and changed pixels use different coordinates")
                selected, overlap = int(marks.sum()), int((marks & truth).sum())
        counts = {"selected": selected, "intersection": overlap, "truth": int(truth.sum())}
        return {key: item[key] for key in ("group", "role", "quality", "sha256", "mask_sha256")} | {
            "source_sha256": source["sha256"], "status": result["status"], "schema": result["schema"],
            **counts, "useful": useful(counts), "regions": len(result["regions"]),
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(measure, inputs))
    files.save(output, {"manifest_sha256": files.sha(manifest), "options": options, "rows": rows})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("index", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--start", type=int, default=0)
    prepare.add_argument("--previous-intake", type=Path)
    compare = commands.add_parser("compare-pairs")
    compare.add_argument("deliveries", type=Path)
    compare.add_argument("output", type=Path)
    args = vars(cli.parse_args())
    action = args.pop("action")
    (build if action == "prepare" else compare_pairs)(**args)
