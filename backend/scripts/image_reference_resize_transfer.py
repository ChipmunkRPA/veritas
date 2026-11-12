# SPDX-License-Identifier: Apache-2.0
"""Evaluate earlier-copy inspection after a proportional, same-framing resize."""

from __future__ import annotations

import argparse
import io
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from PIL import __version__ as pillow_version

from app.detectors.image import reference_alignment, reference_compare
from scripts import image_reference_alignment_transfer as panel

SCHEMA = "chipmunk.image-reference-resize-transfer/1"
KERNELS = (Image.Resampling.BICUBIC, Image.Resampling.BILINEAR, Image.Resampling.LANCZOS)
SCALES = (.67, .82, 1.17)
MAX_ASPECT_DRIFT = .008
MIN_SCALE, MAX_SCALE = .50, 1.50
MIN_VOTES = 2
POSITIVE = ("sharp", "feathered")
ORDINARY = ("resized_original", "earlier_jpeg", "global_contrast")
HARD = ("cropped_and_resized", "rotated_and_resized", "unrelated")
ROLES = (*POSITIVE, *ORDINARY, *HARD)


def native_crop(region, original_size, submitted_size):
    """Map half-open submitted edges outward to integral original crop coordinates."""
    x, y, width, height = (region[key] for key in ("x", "y", "w", "h"))
    rw, rh = original_size
    sw, sh = submitted_size
    left, top = x * rw // sw, y * rh // sh
    right = ((x + width) * rw + sw - 1) // sw
    bottom = ((y + height) * rh + sh - 1) // sh
    return {"x": left, "y": top, "w": right - left, "h": bottom - top}


def compare(reference, submitted, carrier):
    """Project a corroborated same-framing reference onto the submitted pixel grid."""
    rh, rw = reference.shape[:2]
    sh, sw = submitted.shape[:2]
    sx, sy = sw / rw, sh / rh
    if ((rw, rh) == (sw, sh) or min(rw, rh, sw, sh) < reference_alignment.MIN_SIDE
            or min(sx, sy) < MIN_SCALE or max(sx, sy) > MAX_SCALE or abs(sx / sy - 1) > MAX_ASPECT_DRIFT):
        return {"status": "unsupported_resize_geometry", "regions": []}, None
    original = Image.fromarray(reference)
    candidates = [np.asarray(original.resize((sw, sh), kernel)) for kernel in KERNELS]
    support = reference_alignment.measure(candidates[0], submitted)
    if support is None or support["reference_minus_submitted"] != {"x": 0, "y": 0}:
        return {"status": "same_framing_unresolved", "regions": []}, None
    metadata = {"framing_check": support, "submitted_over_reference_xy": [[sw, rw], [sh, rh]]}
    votes = np.zeros((sh, sw), np.uint8)
    supported = []
    for kernel, projected in zip(KERNELS, candidates, strict=True):
        record, mask = reference_compare.compare_across_deliveries(projected, submitted, carrier)
        if record["status"] == "comparable" and mask is not None:
            votes += mask
            supported.append(kernel.name.lower())
    if len(supported) < MIN_VOTES:
        return {"status": "resize_delivery_unresolved", "regions": [], **metadata}, None
    regions, mask, omitted = reference_compare._regions(votes >= MIN_VOTES, max(48, round(votes.size * .0015)))
    for region in regions:
        region["reference_region"] = native_crop(region, (rw, rh), (sw, sh))
    return {"status": "comparable", "regions": regions, "additional_regions": omitted,
            "supported_resize_kernels": supported, **metadata}, mask


def cases(row, native, other):
    source = native["original"]
    earlier = Image.fromarray(source)
    edited = Image.fromarray(native["edited"])
    feathered = Image.fromarray(panel.blend.local(source, native["edited"], row["box"], feather=True))
    width, height = earlier.size
    scale = SCALES[row["slot"] % len(SCALES)]
    size = round(width * scale), round(height * scale)
    # The fixture cycles Pillow and OpenCV implementations before scoring.
    render = row["slot"] % 3
    def resized(image):
        if render == 0:
            method = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            return cv2.resize(np.asarray(image), size, interpolation=method)
        kernel = Image.Resampling.BICUBIC if render == 1 else Image.Resampling.LANCZOS
        return np.asarray(image.resize(size, kernel))

    unedited = resized(earlier)
    sharp = resized(edited)
    soft = resized(feathered)
    lossy = resized(Image.fromarray(reference_compare._prior_jpeg(source, 83)))
    contrast = resized(ImageEnhance.Contrast(earlier).enhance(1.12))
    crop = resized(earlier.crop((width // 11, height // 11, width - width // 10, height - height // 10)))
    rotated = resized(earlier.rotate(2.0, Image.Resampling.BICUBIC))
    different = resized(Image.fromarray(other))
    for role, image in zip(ROLES, (sharp, soft, unedited, lossy, contrast, crop, rotated, different), strict=True):
        # Exact effect of the authored edit on the delivered spatial grid before encoding.
        truth = np.any(image != unedited, axis=2) if role in POSITIVE else np.zeros(image.shape[:2], bool)
        yield role, image, truth, scale


def freeze(panels):
    plans = [json.loads((path / "plan.json").read_bytes()) for path in panels]
    for key in ("flickr_photo_id", "flickr_owner"):
        groups = [{row[key] for row in plan["cases"]} for plan in plans]
        if [len(group) for group in groups] != [12, 12] or len(set.union(*groups)) != 24:
            raise ValueError("Expected distinct 12-source and photographer panels")
    methods = (Path(__file__), Path(reference_alignment.__file__), Path(reference_compare.__file__),
               Path(panel.paired.__file__), Path(panel.paired.panel.__file__), Path(panel.blend.__file__),
               Path(panel.paired.product.previous.__file__))
    return {"schema": SCHEMA, "code_sha256": {p.name: panel.shared.file_digest(p) for p in methods},
            "intake_commitments": [{name: panel.shared.file_digest(path / name) for name in
                                    ("plan.json", "attempts.jsonl", "result.json")} for path in panels],
            "opencv": cv2.__version__, "pillow": pillow_version, "numpy": np.__version__, "roles": list(ROLES),
            "formats": list(panel.FORMATS), "scales": list(SCALES), "candidate_kernels": [k.name for k in KERNELS],
            "thresholds": {"aspect_drift": MAX_ASPECT_DRIFT, "minimum_scale": MIN_SCALE, "maximum_scale": MAX_SCALE,
                           "minimum_vote": MIN_VOTES, "iou": .30, "recall": .50},
            "advance": "Per panel and format: at least six useful feathered locations, at least eight assessed resized originals, at most four ordinary highlights (no more than one on resized originals), and zero comparable hard controls."}


def measure(intake):
    sources = list(panel.paired.inputs(intake))
    for index, (row, native, _) in enumerate(sources):
        reference = native["original"]
        neighbor = sources[(index + 1) % len(sources)][1]["original"]
        for role, pixels, truth, scale in cases(row, native, neighbor):
            for fmt in panel.FORMATS:
                body, image = panel.paired.encoded(pixels, fmt)
                submitted = np.asarray(image)
                with Image.open(io.BytesIO(body)) as carrier:
                    incumbent, incumbent_mask = reference_alignment.compare(reference, submitted, carrier)
                    result, mask = compare(reference, submitted, carrier)
                if mask is not None and int(mask.sum()) != sum(r["selected_pixels"] for r in result["regions"]):
                    raise ValueError("Selected mask totals disagree")
                yield {"slot": row["slot"], "role": role, "format": fmt, "scale": scale,
                       "source_sha256": panel.shared.digest(reference.tobytes()), "input_sha256": panel.shared.digest(body),
                       "truth_sha256": panel.shared.digest(truth.tobytes()), "status": result["status"],
                       "incumbent_status": incumbent["status"], "regions": result["regions"],
                       "mask_sha256": panel.shared.digest(mask.tobytes()) if mask is not None else None,
                       "candidate": panel.metrics.measure(mask if mask is not None else np.zeros_like(truth), truth),
                       "incumbent": panel.metrics.measure(incumbent_mask if incumbent_mask is not None else np.zeros_like(truth), truth)}


def rollup(rows):
    result = {}
    for fmt in panel.FORMATS:
        counts = Counter()
        for row in rows:
            if row["format"] != fmt:
                continue
            role = row["role"]
            counts[(role, "views")] += 1
            counts[(role, "comparable")] += row["status"] == "comparable"
            counts[(role, "incumbent_comparable")] += row["incumbent_status"] == "comparable"
            for arm in ("candidate", "incumbent"):
                result_row = row[arm]
                counts[(role, f"{arm}_selected")] += result_row["reported"] > 0
                counts[(role, f"{arm}_useful")] += bool(result_row["target"] and result_row["iou"] >= .30 and result_row["recall"] >= .50)
        result[fmt] = {role: {metric: value for (name, metric), value in counts.items() if name == role} for role in ROLES}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for field in ("earlier", "later", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    locked = freeze((args.earlier, args.later))
    freeze_path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.shared.write(freeze_path, locked)
        return
    if json.loads(freeze_path.read_bytes()) != locked:
        raise ValueError("Frozen inputs changed")
    cv2.setNumThreads(1)
    started = time.monotonic()
    reports, journals = {}, {}
    for name, intake in (("earlier", args.earlier), ("later", args.later)):
        rows = list(measure(intake))
        if len(rows) != 192 or len({(r["slot"], r["role"], r["format"]) for r in rows}) != 192:
            raise ValueError("Unexpected confirmation denominator")
        path = args.output / f"{name}.json"
        panel.shared.write(path, rows)
        reports[name] = rollup(rows)
        journals[name] = panel.shared.file_digest(path)
        print(name, "verified", len(rows), "views", flush=True)
    decisions = {}
    for fmt in panel.FORMATS:
        decisions[fmt] = all((r := report[fmt])["feathered"]["candidate_useful"] >= 6 and r["resized_original"]["comparable"] >= 8
                             and r["resized_original"]["candidate_selected"] <= 1
                             and sum(r[k]["candidate_selected"] for k in ORDINARY) <= 4
                             and not any(r[k]["comparable"] for k in HARD) for report in reports.values())
    panel.shared.write(args.output / "aggregate.json", {"schema": SCHEMA, "freeze_sha256": panel.shared.file_digest(freeze_path),
                       "journal_sha256": journals, "views": 384, "elapsed_s": round(time.monotonic() - started, 3),
                       "results": reports, "advances": decisions})


if __name__ == "__main__":
    main()
