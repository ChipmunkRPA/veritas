# SPDX-License-Identifier: Apache-2.0
"""Replay the JPEG grid check on separately encoded, source-bound color sampling variants."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.image_previous_grid_panel import (
    POSITIVES,
    PUBLIC,
    authored,
    decode,
    digest,
    measure_box,
    read,
    save,
    sources,
    useful,
)

SCHEMA = "chipmunk.image-previous-grid-color-sampling/1"
TYPES = ("sample_422", "sample_444", "progressive_422", "progressive_444")
PROFILES = {
    "first": (90, "image-previous-grid"),
    "reserve": (90, "image-previous-grid-reserve"),
    "insertion85": (85, "image-previous-grid-85-insertions"),
    "removal85": (85, "image-previous-grid-85-removals"),
}
CAMERA = {85: "image-previous-grid-85-cameras", 90: "image-previous-grid-camera-controls"}


def frozen(stem):
    return PUBLIC / f"{stem}-2026-11-17.json.gz"


def encode(pixels, quality, sampling=2, *, progressive=False, orientation=1):
    from PIL import Image

    options = {}
    if orientation != 1:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    with io.BytesIO() as output:
        Image.fromarray(pixels).save(output, "JPEG", quality=quality, subsampling=sampling,
                                      progressive=progressive, **options)
        return output.getvalue()


def camera_views(path, rows):
    import numpy as np
    from PIL import Image, ImageEnhance

    parent_sha = rows[0]["parent_sha256"]
    blob = (path / rows[0]["parent"]).read_bytes()
    if digest(blob) != parent_sha or any(row["parent_sha256"] != parent_sha for row in rows):
        raise ValueError("Camera parent bytes differ from their frozen receipt")
    x, y = rows[0]["origin"]
    pixels = decode(blob)
    window = pixels[y - 3:y + 581, x - 5:x + 771]
    crop = np.s_[3:579, 5:773]
    earlier = decode(encode(window, 55))[crop]
    return {"cropped_retained_jpeg": window[crop], "cropped_after_coarser_save": earlier,
            "global_contrast_after_crop": np.array(ImageEnhance.Contrast(Image.fromarray(earlier)).enhance(.88))}


def source_views(rows, manifest):
    matches = [item for item in manifest if item[1] == rows[0]["source"]]
    if len(matches) != 1:
        raise ValueError("The photopair must match its retained source index")
    target, parent_sha, _, child_sha = matches[0]
    original, edited = ((target / f"{role}.png").read_bytes() for role in ("original", "edited"))
    if (digest(original), digest(edited)) != (parent_sha, child_sha):
        raise ValueError("The photopair does not match its recorded bytes")
    _, _, parent, prior, views = authored(decode(original), decode(edited), parent_sha)
    if any(row["parent_delivery_sha256"] != (parent_sha if row["role"].startswith("direct_") else digest(parent))
           for row in rows):
        raise ValueError("An earlier parent does not reproduce its frozen bytes")
    return prior, views


def identity(record):
    measurement = record["measurement"]
    reading = {key: value for key, value in measurement.items() if key != "delivery_subsampling"} if measurement else None
    return record["status"], reading, record["region"]


def exports(pixels, truth, row, quality, folder, *, odd=False):
    import numpy as np
    from PIL import Image, ImageOps

    from app.detectors.image import jpeg_bitstream
    from app.detectors.image.jpeg_previous_grid import read_luma
    from app.evidence import image_previous_grid_binding as binding
    from app.evidence import image_previous_grid_source as source

    if odd:
        # Keep the origin and every preceding pixel. Both edges lose one sample.
        pixels, truth = pixels[:-1, :-1], truth[:-1, :-1]
    height, width = truth.shape
    current = encode(pixels, quality)
    source_pin = digest(current)
    if not odd and source_pin != row["input_sha256"]:
        raise ValueError(f"The 4:2:0 source encoding is not byte-identical for {row['file']}")
    record = source.record(source_pin, current)
    status, measurement, box = identity(record)
    historical = source.record(source_pin, current, schema=binding.QUALITY_SCHEMA)
    if (status, measurement, box) != identity(historical):
        raise ValueError("New and historical 4:2:0 measurements disagree")
    if status not in {"matched", "no_region", "no_prior_grid"}:
        raise ValueError(f"Unexpected 4:2:0 comparison state {status}")
    luma, table = read_luma(current)
    variants = {}
    orientation = 2 + int(row["input_sha256"][-2:], 16) % 7
    for order, name in enumerate(TYPES):
        progressive = order >= 2
        rotation = orientation if progressive else 1
        sampling = 1 - order % 2
        delivery = encode(pixels, quality, sampling, progressive=progressive, orientation=rotation)
        pin = digest(delivery)
        jpeg = jpeg_bitstream.read(delivery)
        label = "4:2:2" if sampling else "4:4:4"
        if ((jpeg.subsampling, jpeg.ijg_quality, jpeg.frame) != (label, quality, "progressive" if progressive else "baseline")
                or (jpeg.width, jpeg.height) != (width, height)):
            raise ValueError("An alternative JPEG does not meet the pinned encoding profile")
        alternate_luma, alternate_table = read_luma(delivery)
        if not np.array_equal(luma, alternate_luma) or not np.array_equal(table, alternate_table):
            raise ValueError(f"Stored brightness or table changed for {name} on {row['file']}")
        proposed = source.record(pin, delivery)
        observed_status, observed_reading, observed_box = identity(proposed)
        if observed_status != status or observed_reading != measurement:
            raise ValueError(f"An alternative JPEG changed its review on {row['file']}")
        if box is not None:
            mask = Image.new("L", (width, height))
            mask.paste(255, tuple(box))
            mask.getexif()[274] = rotation
            if observed_box != list(ImageOps.exif_transpose(mask).getbbox()):
                raise ValueError("The alternative region does not map to the same displayed pixels")
        elif observed_box is not None:
            raise ValueError("The alternative JPEG unexpectedly returned a region")
        if observed_reading and proposed["measurement"]["delivery_subsampling"] != label:
            raise ValueError("The result does not identify the measured color sampling")
        if source.record(pin, delivery, schema=binding.QUALITY_SCHEMA)["status"] != "outside_profile":
            raise ValueError("A historical report unexpectedly gained scope")
        if folder is not None:
            (folder / f"{name}.jpg").write_bytes(delivery)
        variants[name] = {"sha256": pin, "region": observed_box, "orientation": rotation}
    overlap = None if box is None else measure_box((box[0], box[1], box[2] - box[0], box[3] - box[1]), truth)
    return {"sha256": source_pin, "dimensions": [width, height], "quality": quality, "role": row["role"],
            "luma_sha256": digest(luma.astype(np.uint8).tobytes()), "status": status, "region": box, "overlap": overlap,
            "measurement_sha256": digest(json.dumps(measurement, sort_keys=True).encode()),
            "original_420_sha256": row["input_sha256"],
            "useful": useful(overlap), "variants": variants}


def benchmark(args):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    if args.assets:
        args.assets.mkdir()
    jobs = []
    for profile, (quality, stem) in PROFILES.items():
        path = getattr(args, profile)
        groups = defaultdict(list)
        for row in read(frozen(stem))["rows"]:
            if row["quality"] == quality:
                groups[row["source"]].append(row)
        manifest = tuple(sources(path))
        for key, rows in sorted(groups.items()):
            jobs.append((profile, quality, rows, path, manifest, key == min(groups)))
    for quality, stem in CAMERA.items():
        groups = defaultdict(list)
        for row in read(frozen(stem))["rows"]:
            groups[row["camera"]["group"]].append(row | {"input_sha256": row["reading"]["source_sha256"]})
        for key, rows in sorted(groups.items()):
            jobs.append((f"camera{quality}", quality, rows, args.camera, None, key == min(groups)))

    def measure(job):
        profile, quality, rows, path, manifest, odd = job
        if manifest is None:
            prior = None
            views = camera_views(path, rows)
        else:
            prior, views = source_views(rows, manifest)
        findings = []
        for row in rows:
            role = row["role"]
            pixels = views[role]
            truth = np.any(prior != pixels, axis=2) if role in POSITIVES else np.zeros(pixels.shape[:2], dtype=bool)
            if role in POSITIVES and int(truth.sum()) != row["pixels_changed"]:
                raise ValueError("The raster edit count differs from the original label")
            root = args.assets / profile / row["file"].removesuffix(".jpg") if args.assets else None
            if root is not None:
                root.mkdir(parents=True)
            findings.append({"profile": profile, "file": row["file"], "odd": False, **exports(pixels, truth, row, quality, root)})
            if odd:
                extra = root / "odd" if root else None
                if extra:
                    extra.mkdir()
                findings.append({"profile": profile, "file": row["file"], "odd": True,
                                 **exports(pixels, truth, row, quality, extra, odd=True)})
        return findings

    outputs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, batch in enumerate(pool.map(measure, jobs), 1):
            outputs.extend(batch)
            if number % 4 == 0 or number == len(jobs):
                print(f"Verified {number}/{len(jobs)} source and format groups", flush=True)
    receipts = {name: digest(frozen(stem).read_bytes()) for name, (_, stem) in PROFILES.items()}
    receipts.update({f"camera{quality}": digest(frozen(stem).read_bytes()) for quality, stem in CAMERA.items()})
    save(args.output, {"schema": SCHEMA, "parents": receipts, "rows": outputs, "summary": summary(outputs)})
    print(summary(outputs), flush=True)


def summary(rows):
    counts = defaultdict(Counter)
    for row in rows:
        scoped = counts[("odd" if row["odd"] else "standard") + "/" + row["profile"]]
        scoped["earlier_grid"] += row["status"] in {"matched", "no_region"}
        scoped["inputs"] += 1
        scoped["localized_edits"] += row["role"] in POSITIVES and row["useful"]
        scoped["missed_edits"] += row["role"] in POSITIVES and not row["useful"]
        scoped["misplaced_edits"] += row["role"] in POSITIVES and row["status"] == "matched" and not row["useful"]
        scoped["other_highlights"] += row["role"] not in POSITIVES and row["status"] == "matched"
        scoped["variants"] += len(row["variants"])
    return counts


def verify(path):
    from app.evidence.image_previous_grid_binding import orient_region

    data = read(path)
    if data["schema"] != SCHEMA or data["summary"] != summary(data["rows"]):
        raise ValueError("The receipt does not reproduce its recorded color sampling comparison counts")
    expected = {}
    profiles = PROFILES | {f"camera{quality}": (quality, stem) for quality, stem in CAMERA.items()}
    for profile, (quality, stem) in profiles.items():
        receipt = frozen(stem)
        if data["parents"].get(profile) != digest(receipt.read_bytes()):
            raise ValueError("A parent receipt does not match its public commitment")
        rows = [row for row in read(receipt)["rows"] if row.get("quality", quality) == quality]
        first = min(row["camera"]["group"] if profile.startswith("camera") else row["source"] for row in rows)
        for row in rows:
            camera = row.get("camera")
            group = camera["group"] if camera else row["source"]
            pin = row["reading"]["source_sha256"] if camera else row["input_sha256"]
            expected[profile, row["file"], False] = pin
            if group == first:
                expected[profile, row["file"], True] = pin
    actual = {(row["profile"], row["file"], row["odd"]): row["original_420_sha256"] for row in data["rows"]}
    if actual != expected or len(actual) != len(data["rows"]):
        raise ValueError("The receipt differs from the frozen input coverage")
    for row in data["rows"]:
        if set(row["variants"]) != set(TYPES) or row["useful"] != useful(row["overlap"]):
            raise ValueError("The receipt lacks its prescribed sampling variants or native overlap")
        for variant in row["variants"].values():
            region = orient_region(row["region"], variant["orientation"], *row["dimensions"]) if row["region"] else None
            if region != variant["region"]:
                raise ValueError("The JPEG variant uses different displayed coordinates")
    print(data["summary"])


def app(receipt, assets, output):
    from scripts.image_previous_grid_app import run

    releases = [
        ("q90-422", "reserve", "674faa841f45/sharp-90.jpg", "sample_422", True),
        ("q90-progressive-444", "reserve", "674faa841f45/sharp-90.jpg", "progressive_444", True),
        ("q85-444", "insertion85", "0e686e4c8bf6/feather-85.jpg", "sample_444", True),
        ("q85-progressive-422", "insertion85", "0e686e4c8bf6/feather-85.jpg", "progressive_422", True),
        ("ordinary-422", "insertion85", "0e686e4c8bf6/ordinary-85.jpg", "progressive_422", False),
    ]
    bank = {(row["profile"], row["file"]): row for row in read(receipt)["rows"] if not row["odd"]}
    cases = []
    for name, profile, case, variant, should_match in releases:
        row = bank[profile, case]
        outcome = row["variants"][variant]
        region = outcome["region"]
        if bool(region) != should_match or should_match and not row["useful"]:
            raise ValueError("The serving proof requires the verified modified and ordinary inputs")
        file = f"{profile}/{case.removesuffix('.jpg')}/{variant}.jpg"
        body = {"role": name, "quality": row["quality"], "file": file, "input_sha256": outcome["sha256"],
                "subsampling": "4:2:2" if variant.endswith("422") else "4:4:4"}
        cases.append((name, body, region, should_match, True))
    cases.append(("option-off", cases[0][1], None, False, False))
    run(assets, output, browser=True, case_rows=cases)
    tested = read(output / "receipt.json")["cases"]
    if [row["useful_added_region"] for row in tested] != [True] * 4 + [False] * 2:
        raise ValueError("The revised color sampling profiles did not surface all four verified review regions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    measure = modes.add_parser("measure")
    for name in (*PROFILES, "camera", "output"):
        measure.add_argument(name, type=Path)
    measure.add_argument("--assets", type=Path)
    review = modes.add_parser("verify")
    review.add_argument("receipt", type=Path)
    worker = modes.add_parser("app")
    for name in ("receipt", "assets", "output"):
        worker.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "measure":
        benchmark(args)
    elif args.command == "verify":
        verify(args.receipt)
    else:
        app(args.receipt, args.assets, args.output)
