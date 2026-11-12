# SPDX-License-Identifier: Apache-2.0
"""Hold out rotated, cropped and resized native PNG pairs with exact change truth."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.evidence import image_reference
from scripts import image_background_patch_score as generated
from scripts import image_reference_orientation_transfer as prior
from scripts import image_reference_oriented_crop_transfer as crop
from scripts import image_reference_oriented_resize_candidate as candidate
from scripts import review_image_reference_oriented_resize_transfer as review

TRANSFORMS = tuple(prior.reviewer.ORIENTATIONS)
SIZES = ((688, 548), (720, 598), (644, 512))
SCALES = (.82, .91, 1.09, 1.14)
KERNELS = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR,
           "lanczos": Image.Resampling.LANCZOS, "opencv_area": None}
ROLES = {"photo": ("clean", "local", "fractional", "off_angle", "stretch", "jpeg", "tone", "unrelated", "identity"),
         "generated": ("clean", "generated", "fractional", "tone"), "ui": ("other_document",)}
RULES = {"minimum_clean": 16, "minimum_local": 10, "minimum_generated": 3, "minimum_improved": 18,
         "minimum_orientations": 7, "maximum_incorrect": 0}
sha = prior.sha


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha(path.read_bytes())


def freeze(args):
    recorded = prior.binding(args.commons)
    eligible = (row for row in recorded["sources"] if row["slot"] not in {19, 27})
    order = sorted(eligible, key=lambda row: sha(f"oriented-resize/1/{row['sha256']}".encode()))
    rows = list(generated.inputs(args.generated))
    prior.require(len(rows) == 7, "Need all seven qualified generated patch sources")
    kept = [{"slot": row["slot"], "generator": row["generator"], "case_sha256": row["case_sha256"],
             "truth_sha256": row["truth_sha256"], "assets_sha256": {name: value["sha256"] for name, value in row["assets"].items()}}
            for row, _, _ in rows]
    captures = read(args.ui / "captures.json")
    prior.require({(row["slot"], row["variant"]) for row in captures["rows"]} == {(i, j) for i in range(12) for j in range(2)}
                  and len(captures["rows"]) == 24 and len({row["png_sha256"] for row in captures["rows"]}) == 24,
                  "Expected 24 unique native Chromium inputs")
    for row in captures["rows"]:
        prior.require(digest(args.ui / row["name"]) == row["png_sha256"], "Document fixture image changed")
    sources = (candidate, candidate.exact, candidate.reference_crop_resize, candidate.orientation, prior, crop, generated, image_reference, review)
    code = (Path(__file__), Path(__file__).with_name("capture_image_reference_oriented_resize.cjs"),
            *(Path(module.__file__) for module in sources))
    return {"schema": review.SCHEMA, "photo_sources": order[:21], "unrelated_donor": order[21],
            "commons_receipt_sha256": recorded["public_source_receipt"], "generated_sources": kept,
            "generated_intake_sha256": {name: digest(args.generated / name) for name in ("plan.json", "result.json", "attempts.jsonl")},
            "chromium": captures, "roles": {name: list(values) for name, values in ROLES.items()}, "criteria": RULES,
            "transforms": list(TRANSFORMS), "kernels": list(KERNELS), "scales": list(SCALES),
            "candidate_requirements": {"min_exact_fraction": candidate.exact.MIN_GLOBAL_EQUAL,
                                        "min_tiles": candidate.exact.MIN_TILES, "min_spread": candidate.exact.MIN_AXIS,
                                        "max_difference_box_fraction": candidate.exact.MAX_MISMATCH_BOX,
                                        "max_proposals_per_orientation": candidate.exact.TOP_PROPOSALS},
            "code_sha256": {path.name: digest(path) for path in code}}


def ltrb(box):
    x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
    return x, y, x+w, y+h


def project(image, box, size, method):
    with image.crop(box) as cropped:
        if method == "opencv_area":
            return Image.fromarray(cv2.resize(np.asarray(cropped), size, interpolation=cv2.INTER_AREA))
        return cropped.resize(size, KERNELS[method])


def oriented_edit_bounds(truth, op):
    with Image.fromarray(truth).transpose(op) as mask:
        bounds = mask.getbbox()
    prior.require(bounds is not None, "Generated edit truth is empty")
    return bounds


def box_for(oriented, slot, mask=None):
    width, height = oriented.size
    w, h = round(width * (.79 + .035 * (slot % 4))), round(height * (.78 + .033 * (slot % 4)))
    if mask is None:
        x, y = 9 + (slot * 21) % (width-w-17), 8 + (slot * 17) % (height-h-15)
    else:
        x = max(0, min(width-w, (mask[0]+mask[2]-w)//2))
        y = max(0, min(height-h, (mask[1]+mask[3]-h)//2))
        prior.require(x <= mask[0] < mask[2] <= x+w and y <= mask[1] < mask[3] <= y+h, "The crop must contain the qualified edit")
    return x, y, x+w, y+h


def local_edit(original, slot):
    width, height = original.size
    x, y = round(width * (.27 + .15*(slot % 4))), round(height * (.37 + .07*(slot % 4)))
    w, h = round(width * .14), round(height * .17)
    with original.crop((x, y, x+w, y+h)) as part:
        replacement = ImageEnhance.Brightness(part.filter(ImageFilter.GaussianBlur(1.2))).enhance(1.21)
    result = original.copy()
    result.paste(replacement, (x, y))
    return result


def source_rows(args, lock):
    for slot, record in enumerate(lock["photo_sources"]):
        image = crop.source(args.commons, record, SIZES[slot % len(SIZES)])
        yield "photo", slot, prior.encode(image, "JPEG" if slot % 2 else "PNG"), None, None
    for slot, (_, native, truth) in enumerate(generated.inputs(args.generated)):
        yield "generated", slot, prior.encode(Image.fromarray(native["original"]), "PNG"), Image.fromarray(native["edited"]), truth


def positives(reference, transform, box, size, method, edit_source, slot):
    op = prior.reviewer.ORIENTATIONS[transform]
    with reference.transpose(op) as oriented:
        clear = project(oriented, box, size, method)
    if edit_source is None:
        edited = local_edit(clear, slot)
    else:
        with edit_source.transpose(op) as modified:
            edited = project(modified, box, size, method)
    return clear, edited


def deliveries(domain, slot, reference, edit_source, edit_mask, donor):
    transform = TRANSFORMS[slot % len(TRANSFORMS)]
    op = prior.reviewer.ORIENTATIONS[transform]
    method = tuple(KERNELS)[slot % len(KERNELS)]
    scale = SCALES[(slot // len(KERNELS) + slot) % len(SCALES)]
    with reference.transpose(op) as oriented:
        box = box_for(oriented, slot, oriented_edit_bounds(edit_mask, op) if edit_mask is not None else None)
        w, h = round((box[2]-box[0])*scale), round((box[3]-box[1])*scale)
    clear, edited = positives(reference, transform, box, (w, h), method, edit_source, slot)
    yield "clean", clear, transform, box, method, np.zeros((h, w), dtype=bool)
    truth = (np.asarray(clear) != np.asarray(edited)).any(axis=2)
    prior.require(bool(truth.any()), "Expected native changed-pixel truth")
    yield "local" if edit_source is None else "generated", edited, transform, box, method, truth
    fractional = clear.transform(clear.size, Image.Transform.AFFINE, (1, 0, .38, 0, 1, -.34), Image.Resampling.BICUBIC)
    yield "fractional", fractional, None, None, None, None
    yield "tone", ImageEnhance.Contrast(clear).enhance(.87), None, None, None, None
    if domain == "photo":
        yield "off_angle", clear.rotate(1.4, resample=Image.Resampling.BICUBIC), None, None, None, None
        yield "stretch", clear.resize((w-19, h+14), Image.Resampling.BICUBIC), None, None, None, None
        yield "jpeg", clear, None, None, None, None
        yield "unrelated", donor.resize(clear.size, Image.Resampling.LANCZOS), None, None, None, None
        yield "identity", project(reference, box_for(reference, slot), clear.size, method), None, None, None, None


def native_bounds(box, size, transform):
    op = prior.reviewer.ORIENTATIONS[transform]
    inverse = {Image.Transpose.ROTATE_90: Image.Transpose.ROTATE_270, Image.Transpose.ROTATE_270: Image.Transpose.ROTATE_90}.get(op, op)
    shape = size[::-1] if op in {Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270,
                                 Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE} else size
    with Image.new("1", shape) as mask:
        mask.paste(1, box)
        with mask.transpose(inverse) as native:
            return native.getbbox()


def observe(domain, slot, role, earlier, before, after, submitted_bytes, transform, box, method, truth, files):
    ref_path, sub_path = files / f"{domain}-{slot:02d}-earlier", files / f"{domain}-{slot:02d}-{role}"
    if not ref_path.exists():
        ref_path.write_bytes(earlier)
    sub_path.write_bytes(submitted_bytes)
    old = image_reference.inspect(ref_path, sub_path, allow_integer_alignment=True, allow_proportional_resize=True,
                                  allow_oriented_crop=True)
    with Image.open(io.BytesIO(submitted_bytes)) as carrier:
        proof, mask = candidate.compare(np.asarray(before), np.asarray(carrier.convert("RGB")), carrier)
    counts = None if truth is None else [int(truth.sum()), int(mask.sum()) if mask is not None else 0,
                                         int((truth & mask).sum()) if mask is not None else 0]
    expected = None if transform is None else {"transform": transform, "kernel": method, "projected_crop": list(box),
                                               "native_crop": list(native_bounds(box, before.size, transform)),
                                               "truth_sha256": sha(truth.tobytes())}
    return {"domain": domain, "slot": slot, "role": role, "earlier_sha256": sha(earlier), "submitted_sha256": sha(submitted_bytes),
            "earlier_size": list(before.size), "submitted_size": list(after.size), "expected": expected, "counts": counts,
            "selected_mask_sha256": sha(mask.tobytes()) if mask is not None else None,
            "current": {"status": old["status"], "schema": old["schema"], "regions": len(old["regions"])}, "proof": proof}


def run(args, lock):
    files = args.run / "inputs"
    files.mkdir()
    output = []
    donor = crop.source(args.commons, lock["unrelated_donor"], (688, 548))
    for domain, slot, earlier, edit, mask in source_rows(args, lock):
        with Image.open(io.BytesIO(earlier)) as encoded:
            before = encoded.convert("RGB")
        for role, after, transform, box, method, truth in deliveries(domain, slot, before, edit, mask, donor):
            data = prior.encode(after, "JPEG" if role == "jpeg" else "PNG")
            output.append(observe(domain, slot, role, earlier, before, after, data, transform, box, method, truth, files))
    for slot in range(12):
        earlier = (args.ui / f"{slot}-0.png").read_bytes()
        with (Image.open(io.BytesIO(earlier)) as ref, Image.open(args.ui / f"{slot}-1.png") as other,
              other.convert("RGB").transpose(prior.reviewer.ORIENTATIONS[TRANSFORMS[slot % len(TRANSFORMS)]]) as oriented):
            box = box_for(oriented, slot)
            size = round((box[2]-box[0])*.89), round((box[3]-box[1])*.89)
            method = tuple(KERNELS)[slot % len(KERNELS)]
            after = project(oriented, box, size, method)
            output.append(observe("ui", slot, "other_document", earlier, ref.convert("RGB"), after,
                                  prior.encode(after, "PNG"), None, None, None, None, files))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for name in ("commons", "generated", "ui", "run"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = freeze(args)
    target = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        prior.write(target, lock)
        print(digest(target), flush=True)
    else:
        prior.require(read(target) == lock, "The source-qualified geometry or code changed")
        rows = run(args, lock)
        prior.write(args.run / "result.json", {"schema": review.SCHEMA, "freeze": lock, "freeze_sha256": digest(target),
                    "rows": rows, "outcomes": review.outcomes(lock, rows)})
        print(json.dumps(review.outcomes(lock, rows), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
