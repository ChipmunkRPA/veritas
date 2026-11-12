# SPDX-License-Identifier: Apache-2.0
"""Evaluate exact reference matching after a native crop and a right-angle orientation."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.detectors.image import reference_compare, reference_png_exact_resize, reference_png_orientation
from app.evidence import image_reference
from scripts import image_reference_orientation_transfer as prior

SCHEMA = "chipmunk.image-reference-oriented-crop-transfer/1"
ROLES = ("unedited_crop", "local_edit", "identity_crop", "tone", "off_angle", "jpeg", "different_photo", "resized", "wide_edits", "ambiguous")
POSITIVE = frozenset(ROLES[:2])
PHOTO_GROUPS = 28
MAX_REFERENCE_SIDE = 1024
MIN_FRACTION, MAX_FRACTION = .55, .90
MAX_PROPOSALS = 32
SAMPLES = ((0, 0), (3, 11), (8, 4), (12, 17), (17, 7))
TRANSFORMS = tuple(reference_png_orientation.LABELS)


def packed(pixels):
    rgb = pixels.astype(np.uint32)
    return rgb[:, :, 0] << 16 | rgb[:, :, 1] << 8 | rgb[:, :, 2]


def offset_proposals(reference, submitted, tiles):
    height, width = submitted.shape
    ys, xs = reference.shape[0] - height + 1, reference.shape[1] - width + 1
    if min(ys, xs) < 1:
        return np.empty((0, 2), dtype=int)
    votes = np.zeros((ys, xs), dtype=np.uint8)
    for _, _, (y_bounds, x_bounds) in tiles:
        top, left = y_bounds.start, x_bounds.start
        agreed = np.ones((ys, xs), dtype=bool)
        for dy, dx in SAMPLES:
            y, x = top + dy, left + dx
            agreed &= reference[y:y + ys, x:x + xs] == submitted[y, x]
        votes += agreed
    return np.argwhere(votes >= reference_png_exact_resize.MIN_TILES)


def exact_crop(reference, submitted, carrier):
    rh, rw = reference.shape[:2]
    sh, sw = submitted.shape[:2]
    fraction = sh * sw / (rh * rw)
    if (carrier.format != "PNG" or carrier.mode != "RGB" or carrier.n_frames != 1 or max(rh, rw) > MAX_REFERENCE_SIDE
            or min(sh, sw) < 160 or not MIN_FRACTION <= fraction <= MAX_FRACTION):
        return None, None
    exact = reference_png_exact_resize
    tiles = exact._textured_tiles(submitted)
    if not exact._spread([(row, column) for row, column, _ in tiles]):
        return None, None
    candidate = None
    submitted_packed = packed(submitted)
    # Search the identity as well. It is a competing explanation, not an accepted orientation.
    for transform, oriented in reference_png_orientation._projections(reference):
        offsets = offset_proposals(packed(oriented), submitted_packed, tiles)
        if len(offsets) > MAX_PROPOSALS:
            return None, None
        for y, x in offsets:
            x, y = int(x), int(y)
            projected = oriented[y:y + sh, x:x + sw]
            support = exact._supported(projected, submitted, tiles)
            if support is None:
                continue
            if candidate is not None or transform is None:
                return None, None
            equal, witnesses = support
            candidate = transform, x, y, projected, equal, witnesses
    if candidate is None:
        return None, None
    transform, x, y, projected, equal, witnesses = candidate
    difference = ~equal
    regions, mask, omitted = reference_compare._regions(difference, max(48, round(difference.size * .0004)))
    oriented_crop = {"x": x, "y": y, "w": sw, "h": sh}
    native_crop = reference_png_orientation._native_bounds(oriented_crop, rw, rh, transform)
    for region in regions:
        oriented_region = {**region, "x": region["x"] + x, "y": region["y"] + y}
        region["reference_region"] = reference_png_orientation._native_bounds(oriented_region, rw, rh, transform)
    proof = {"transform": transform, "projected_crop": oriented_crop, "native_crop": native_crop,
             "matched_pixels": int(equal.sum()), "matched_tiles": [list(cell) for cell in witnesses],
             "projected_rgb_sha256": prior.sha(projected.tobytes()), "difference_mask_sha256": prior.sha(difference.tobytes())}
    return {"proof": proof, "regions": regions, "omitted_regions": omitted}, mask


def freeze(archive):
    original = prior.binding(archive)
    order = sorted(original["sources"], key=lambda item: prior.sha(f"exact-crop-orientation/{item['sha256']}".encode()))
    selected = [row | {"case": slot, "transform": TRANSFORMS[slot % len(TRANSFORMS)]}
                for slot, row in enumerate(order[:PHOTO_GROUPS])]
    donor = order[PHOTO_GROUPS]
    return {"schema": SCHEMA, "reference_groups": selected, "unrelated_donor": donor, "source_receipt": original["public_source_receipt"],
            "roles": list(ROLES), "reference_side_limit": MAX_REFERENCE_SIDE, "crop_area_fraction": [MIN_FRACTION, MAX_FRACTION],
            "max_proposals": MAX_PROPOSALS, "samples_in_native_tile": [list(pair) for pair in SAMPLES],
            "criteria": {"minimum_correct_unchanged": 15, "minimum_correct_edits": 15, "minimum_correct_per_orientation": 1,
                         "maximum_incorrect_proofs": 0, "minimum_improvement_over_existing": 10},
            "code_sha256": {path.name: prior.sha(path.read_bytes()) for path in (Path(__file__), Path(prior.__file__),
                            Path(reference_png_orientation.__file__), Path(reference_png_exact_resize.__file__))}}


def source(archive, row, size):
    with Image.open(archive / row["file"]) as raw, ImageOps.exif_transpose(raw) as oriented:
        return ImageOps.fit(oriented.convert("RGB"), size, Image.Resampling.LANCZOS)


def box_for(size, case):
    width, height = size
    crop_w = round(width * (.78 + (case % 3) * .04))
    crop_h = round(height * (.79 + (case % 4) * .03))
    x = 12 + (case * 41) % (width - crop_w - 20)
    y = 11 + (case * 37) % (height - crop_h - 20)
    return x, y, x + crop_w, y + crop_h


def deliveries(reference, donor, row):
    case, transform = row["case"], row["transform"]
    op = prior.reviewer.ORIENTATIONS[transform]
    with reference.transpose(op) as oriented:
        crop_box = box_for(oriented.size, case)
        crop = oriented.crop(crop_box)
        width, height = crop.size
        pixels = np.asarray(crop)
        y, x = np.ogrid[:height, :width]
        truth = ((x - width * (.32 + .11 * (case % 3))) / (width * .064)) ** 2 + ((y - height * .58) / (height * .072)) ** 2 < 1
        altered = pixels.copy()
        altered[truth] ^= np.array([21, 37, 57], dtype=np.uint8)
        wide = pixels.copy()
        wide[9:35, 11:40] ^= np.array([25, 53, 77], dtype=np.uint8)
        wide[-37:-12, -43:-10] ^= np.array([67, 19, 39], dtype=np.uint8)
        variants = (crop, Image.fromarray(altered), reference.crop(box_for(reference.size, case)),
                    crop.point([(value * 3 + 11) // 4 for value in range(256)] * 3),
                    crop.rotate(1.3, resample=Image.Resampling.BICUBIC), crop,
                    ImageOps.fit(donor, crop.size, Image.Resampling.LANCZOS),
                    crop.resize((width - 7, height - 6), Image.Resampling.BICUBIC), Image.fromarray(wide))
        for role, image in zip(ROLES[:-1], variants, strict=True):
            yield role, prior.encode(image, "JPEG" if role == "jpeg" else "PNG"), crop_box, truth if role == "local_edit" else None


def observe(case, transform, role, earlier_path, earlier, files, after, crop_box, truth):
    path = files / f"{case:02d}-{role}.{'jpg' if role == 'jpeg' else 'png'}"
    path.write_bytes(after)
    with Image.open(io.BytesIO(after)) as submitted:
        record, mask = exact_crop(np.asarray(earlier), np.asarray(submitted), submitted)
    old = None
    if role in POSITIVE:
        result = image_reference.inspect(earlier_path, path, allow_integer_alignment=True, allow_proportional_resize=True)
        old = {"status": result["status"], "region_count": len(result["regions"])}
    return {"case": case, "role": role, "transform": transform, "expected_projected_crop": list(crop_box),
            "earlier_sha256": prior.sha(earlier_path.read_bytes()), "submitted_sha256": prior.sha(after), "earlier_file": earlier_path.name,
            "submitted_file": path.name, "truth_sha256": prior.sha(truth.tobytes()) if truth is not None else None,
            "candidate": record, "candidate_mask_sha256": prior.sha(mask.tobytes()) if mask is not None else None, "existing": old}


def measure(args, lock):
    source_rows = lock["reference_groups"]
    donor = source(args.archive, lock["unrelated_donor"], (805, 655))
    rows = []
    files = args.run / "inputs"
    files.mkdir()
    for row in source_rows:
        case = row["case"]
        reference = source(args.archive, row, (840, 640) if case % 2 == 0 else (704, 800))
        before = prior.encode(reference, "PNG" if case % 4 < 2 else "JPEG")
        earlier_path = files / f"{case:02d}-earlier.{'png' if case % 4 < 2 else 'jpg'}"
        earlier_path.write_bytes(before)
        with Image.open(io.BytesIO(before)) as earlier:
            earlier.load()
            for role, after, crop_box, truth in deliveries(earlier, donor, row):
                rows.append(observe(case, row["transform"], role, earlier_path, earlier, files, after, crop_box, truth))
        left = np.asarray(reference)[:, :reference.width // 2]
        symmetric = Image.fromarray(np.concatenate((left, left[:, ::-1]), axis=1))
        symmetric_path = files / f"{case:02d}-symmetric-earlier.png"
        symmetric_path.write_bytes(prior.encode(symmetric, "PNG"))
        margin_x, margin_y = symmetric.width // 12, symmetric.height // 12
        centered = (margin_x, margin_y, symmetric.width - margin_x, symmetric.height - margin_y)
        later = symmetric.crop(centered)
        rows.append(observe(case, "mirror_horizontal", "ambiguous", symmetric_path, symmetric, files,
                            prior.encode(later, "PNG"), centered, None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = freeze(args.archive)
    path = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        prior.write(path, lock)
        print(prior.sha(path.read_bytes()), flush=True)
        return
    prior.require(json.loads(path.read_bytes()) == lock, "The oriented crop plan changed after freezing")
    rows = measure(args, lock)
    prior.write(args.run / "rows.json", rows)
    print(f"{len(rows)} held deliveries", flush=True)


if __name__ == "__main__":
    main()
