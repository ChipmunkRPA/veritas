# SPDX-License-Identifier: Apache-2.0
"""Evaluate native brightness and color differences between matched JPEG copies."""

from __future__ import annotations

import argparse
import hashlib
import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.bench.image_jpeg_boundary import useful
from app.detectors.image import reference_native_jpeg
from scripts import image_region_rank_inputs as files
from scripts.image_cross_quality_fit import load_mask

SCHEMA = "chipmunk.paired-jpeg-native-difference-transfer/1"
QUALITIES = (75, 90)
POSITIVE = frozenset(("sharp", "feather"))


def counts(mask, truth):
    if mask is None:
        return {"selected": 0, "intersection": 0, "truth": int(truth.sum())}
    if mask.shape != truth.shape:
        raise ValueError("The native residual and staged edit use different coordinates")
    return {"selected": int(mask.sum()), "intersection": int((mask & truth).sum()), "truth": int(truth.sum())}


def checked_payload(deliveries, item):
    payload = (deliveries / item["file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise ValueError("The delivered JPEG does not match its manifest")
    return payload


def delivery_controls(deliveries, references):
    """Change only JPEG metadata or processing and bound the behavior of the framing gate."""
    controls = []
    seen = set()
    for item in references.values():
        if item["quality"] != 75 or item["sha256"] in seen:
            continue
        seen.add(item["sha256"])
        original = checked_payload(deliveries, item)
        with Image.open(io.BytesIO(original), formats=["JPEG"]) as earlier:
            pixels = np.asarray(earlier)

            def encode(rgb):
                with io.BytesIO() as output:
                    Image.fromarray(rgb).save(output, "JPEG", quality=75, subsampling=2)
                    return output.getvalue()

            variants = {"jpeg_comment": original[:2] + b"\xff\xfe\x00\x07check" + original[2:],
                        "decode_and_reencode": encode(pixels), "one_pixel_shift": encode(np.roll(pixels, 1, axis=1))}
            for label, payload in variants.items():
                with Image.open(io.BytesIO(payload), formats=["JPEG"]) as later:
                    status, result, _ = reference_native_jpeg.measure_pair(earlier, later, original, payload)
                controls.append({"source_sha256": item["sha256"], "submitted_sha256": hashlib.sha256(payload).hexdigest(),
                                 "variant": label, "status": status, "regions": len(result["regions"]) if result else 0})
    return controls


def summary(rows):
    results = {}
    for quality in QUALITIES:
        stats = Counter()
        for row in rows:
            if row["quality"] != quality:
                continue
            role = row["role"]
            edited = role in POSITIVE
            status = row["status"]
            baseline, native = row["baseline"], row["native"]
            # A rescue is eligible only when the existing app returned no region.
            rescue = native if baseline["selected"] == 0 else baseline
            stats.update(edits=edited, unchanged=role == "ordinary", retouches=not edited and role != "ordinary",
                         baseline_useful=edited and useful(baseline), native_useful=edited and useful(native),
                         rescue_useful=edited and useful(rescue), outside_profile=status == "different_delivery_profile",
                         unresolved=status == "native_alignment_unresolved", native_extra=edited and useful(native) and not useful(baseline),
                         native_lost=edited and useful(baseline) and not useful(native))
            if not edited:
                for label, record in (("baseline", baseline), ("native", native), ("rescue", rescue)):
                    stats[f"{label}_{role}_flags"] += record["selected"] > 0
        results[str(quality)] = dict(stats)
    return results


def evaluate(deliveries, baseline_path, output):
    """Replay on the same deliveries previously measured through the serving path."""
    cv2.setNumThreads(1)
    manifest_path = deliveries / "manifest.json"
    manifest = files.read(manifest_path)
    manifest_sha = files.sha(manifest_path)
    baseline = files.read(baseline_path)
    if baseline["manifest_sha256"] != manifest_sha:
        raise ValueError("The baseline measured different deliveries")
    images = manifest["images"]
    keys = ("group", "role", "quality", "sha256", "mask_sha256")

    def committed(row):
        return tuple(row[k] for k in keys)

    incumbents = {committed(row): row for row in baseline["rows"]}
    if len(incumbents) != len(images) or set(incumbents) != {committed(row) for row in images}:
        raise ValueError("The serving replay is incomplete")
    references = {(row["group"], row["quality"]): row for row in images if row["role"] == "ordinary"}

    def measure(item):
        source = references[item["group"], item["quality"]]
        original = checked_payload(deliveries, source)
        questioned = checked_payload(deliveries, item)
        truth = load_mask(deliveries, item)
        incumbent = incumbents[committed(item)]
        if incumbent["source_sha256"] != source["sha256"] or incumbent["truth"] != int(truth.sum()):
            raise ValueError("The serving replay used a different source or edited area")
        with Image.open(io.BytesIO(original), formats=["JPEG"]) as a, Image.open(io.BytesIO(questioned), formats=["JPEG"]) as b:
            status, result, mask = reference_native_jpeg.measure_pair(a, b, original, questioned, qualities=QUALITIES)
        return {k: item[k] for k in keys} | {
            "source_sha256": source["sha256"], "status": status,
            "baseline_status": incumbent["status"], "baseline_schema": incumbent["schema"],
            "baseline": {k: incumbent[k] for k in ("selected", "intersection", "truth")},
            "native": counts(mask, truth), "regions": result["regions"] if result is not None else [],
            "native_mask_sha256": hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() if mask is not None else None,
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(measure, images))
    report = summary(rows)
    controls = delivery_controls(deliveries, references)
    files.save(output, {"schema": SCHEMA, "manifest_sha256": manifest_sha, "baseline_sha256": files.sha(baseline_path),
                        "profile": {"qualities": QUALITIES, "exact_decoded_tile_floor": reference_native_jpeg.MIN_EXACT_SHARE,
                                    "residual_floor": reference_native_jpeg.RESIDUAL_FLOOR, "chroma_weight": reference_native_jpeg.COLOR_WEIGHT,
                                    "minimum_pixels": reference_native_jpeg.MIN_AREA, "closure_width": reference_native_jpeg.CLOSURE_WIDTH},
                        "packages": {name: version(name) for name in ("Pillow", "numpy", "opencv-python-headless")},
                        "summary": report, "rows": rows, "delivery_controls": controls})
    print(report, flush=True)
    print(dict(Counter((row["variant"], row["status"], row["regions"] > 0) for row in controls)), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("deliveries", type=Path)
    cli.add_argument("baseline_path", type=Path)
    cli.add_argument("output", type=Path)
    evaluate(**vars(cli.parse_args()))
