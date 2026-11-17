"""Compare localized changes after a new crop, rotation, or perspective transform."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

POINT_GRID = (5., 7., 9., 11., 14., 18.)
QUALITY = (75, 90)
GEOMETRIES = ("rotated_crop", "perspective_crop")
EDITED = ("sharp", "feather")
ORDINARY = ("ordinary", "whole_tone", "whole_blur")
NO_MATCH = ("other_photo", "half_replaced")
ROLES = (*EDITED, *ORDINARY, "local_blur", *NO_MATCH)
METHOD = {
    "earlier": "Unchanged quality-90 JPEG from each pinned source group.",
    "submitted": "Earlier and edited JPEG90 pixels transformed with two crop mappings and saved at quality 75 or 90.",
    "controls": "Same mapping on the original, whole-photo tone or blur, a known local blur, another photo, and a replaced half.",
    "truth": "Nearest-neighbor projection of the exact change mask before the first JPEG save. Require 0.30 IoU and 0.50 recall.",
    "point": "Maximum useful calibration ranges by quality with at most one ordinary file alert and zero incompatible pair acceptances.",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as body:
        return hashlib.file_digest(body, "sha256").hexdigest()


def pack(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def save(path, value):
    with path.open("xb") as output:
        output.write(gzip.compress(pack(value), mtime=0))


def load(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def project(matrix, points):
    import numpy as np

    points = np.asarray(points, dtype=float)
    mapped = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    return mapped[:, :2] / mapped[:, 2:]


def geometry(size, kind, alternate):
    import cv2
    import numpy as np

    w, h = size
    ww, hh = round(w * .92), round(h * .92)
    corners = np.array([(0, 0), (ww - 1, 0), (ww - 1, hh - 1), (0, hh - 1)], dtype=np.float32)
    if kind == "rotated_crop":
        center = np.array([(w - 1) / 2, (h - 1) / 2])
        rotation = cv2.getRotationMatrix2D((0., 0.), -6. if alternate else 6., .88)[:, :2]
        source = (corners - corners.mean(axis=0)) @ rotation.T + center
    else:
        points = [(0.10, 0.045), (.92, .12), (.96, .93), (.045, .83)]
        if alternate:
            points = [(1 - x, y) for x, y in reversed(points)]
            points = [points[2], points[3], points[0], points[1]]
        source = np.array(points) * (w - 1, h - 1)
    return cv2.getPerspectiveTransform(corners, source.astype(np.float32)), (ww, hh)


def input_data(root, delivery):
    body = (root / delivery["file"]).read_bytes()
    if digest(body) != delivery["sha256"]:
        raise ValueError("Source JPEG differs from its recorded bytes")
    return body


def render(pixels, other, change, role, matrix, size):
    import cv2
    import numpy as np

    if role in NO_MATCH:
        part = cv2.resize(other, pixels.shape[1::-1], interpolation=cv2.INTER_AREA)
        if role == "other_photo":
            pixels = part
        else:
            pixels = pixels.copy()
            pixels[:, pixels.shape[1] // 2:] = part[:, pixels.shape[1] // 2:]
    image = cv2.warpPerspective(pixels, matrix, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    if role == "whole_tone":
        image = np.rint(np.clip(image * np.array((1.06, .97, 1.10)) + (-4, 6, -8), 0, 255)).astype(np.uint8)
    if role == "whole_blur":
        image = cv2.GaussianBlur(image, (0, 0), .7)
    truth = cv2.warpPerspective(change.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    return image, truth.astype(bool)


def pixels(data):
    import io

    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as src:
        return np.array(src.convert("RGB"))


def jpeg(image, quality):
    import io

    from PIL import Image

    handle = io.BytesIO()
    Image.fromarray(image).save(handle, format="JPEG", quality=quality, subsampling=2)
    return handle.getvalue()


def useful(intersection, selected, actual):
    return bool(actual and 2 * intersection >= actual and 10 * intersection >= 3 * (selected + actual - intersection))


def reading(result, selection, truth):
    if result is None or result["status"] != "comparable":
        return {"assessed": False, "regions": []}
    rows = []
    for region in result["regions"]:
        x, y, w, h = (region[key] for key in ("x", "y", "w", "h"))
        group = selection[y:y+h, x:x+w]
        rows.append({"box": [x, y, w, h], "selected": int(group.sum()), "intersection": int((group & truth[y:y+h, x:x+w]).sum())})
    return {"assessed": True, "regions": rows}


def incumbent(directory, earlier, uploaded, truth):
    import numpy as np
    from PIL import Image

    from app.evidence import image_reference

    with TemporaryDirectory(prefix="baseline-", dir=directory) as temp:
        folder = Path(temp)
        reference, submitted = folder / "ref.jpg", folder / "target.jpg"
        reference.write_bytes(earlier)
        submitted.write_bytes(uploaded)
        evidence = folder / "evidence"
        result = image_reference.inspect(reference, submitted, evidence_dir=evidence, allow_proportional_resize=True,
                                         allow_oriented_crop=True, allow_oriented_resize=True, allow_embedded_photo=True,
                                         allow_exact_jpeg_identity=True, allow_shifted_jpeg=True, allow_structure_jpeg=True,
                                         allow_same_jpeg=True, allow_jpeg_detail_loss=True, allow_jpeg_detail_loss_extra_qualities=True)
        mask = None
        if result.get("mask"):
            with Image.open(evidence / result["mask"]["file"]) as image:
                mask = np.array(image) > 0
        return reading(result, mask, truth)


def records(directory, sources, levels, examples=None):
    import numpy as np

    from app.detectors.image import reference_perspective as detector
    from scripts import image_cross_quality_fit as source_data

    rows = []
    keys = set()
    for root, split in sources:
        manifest = json.loads((root / "manifest.json").read_text())
        groups = list(source_data.images(manifest, (split,)))
        for index, (group, cases) in enumerate(groups):
            source = group["source"]
            if source in keys:
                raise ValueError("Repeated original across source rosters")
            keys.add(source)
            earlier = input_data(root, cases["ordinary", 90])
            origin = pixels(earlier)
            other = pixels(input_data(root, groups[(index + 1) % len(groups)][1]["ordinary", 90]))
            for motion in GEOMETRIES:
                matrix, size = geometry(origin.shape[1::-1], motion, index % 2)
                for role in ROLES:
                    original_role = role if role in (*EDITED, "local_blur") else "ordinary"
                    delivery = cases[original_role, 90]
                    image = origin if original_role == "ordinary" else pixels(input_data(root, delivery))
                    change = source_data.load_mask(root, delivery)
                    warped, truth = render(image, other, change, role, matrix, size)
                    target_pixels = int(truth.sum())
                    for quality in QUALITY:
                        uploaded = jpeg(warped, quality)
                        measured = detector.measure(origin, pixels(uploaded))
                        key_points = [(x * (size[0] - 1), y * (size[1] - 1)) for y in (.05, .5, .95) for x in (.05, .5, .95)]
                        error = None if measured is None else round(float(np.max(np.linalg.norm(
                            project(matrix, key_points) - project(measured.transform, key_points), axis=1))), 5)
                        base = None if levels is None else incumbent(directory, earlier, uploaded, truth)
                        points = POINT_GRID if levels is None else (levels[str(quality)],)
                        variants = {}
                        for point in points:
                            result, mask = detector.compare(measured, origin.shape[1::-1], point)
                            variants[str(point)] = reading(result, mask, truth)
                        row = {
                            "source": source, "family": group["family"], "role": role, "geometry": motion, "quality": quality,
                            "earlier_jpeg_sha256": digest(earlier), "submitted_jpeg_sha256": digest(uploaded),
                            "truth_pixels": target_pixels, "truth_sha256": digest(np.packbits(truth, bitorder="little").tobytes()),
                            "extent_error_px": error, "baseline": base, "checks": variants,
                        }
                        rows.append(row)
                        if (examples is not None and role in EDITED and group["family"] == "lama" and not base["assessed"]
                                and any(useful(r["intersection"], r["selected"], target_pixels) for r in next(iter(variants.values()))["regions"])):
                            label = f"{motion}-{role}-q{quality}"
                            target = examples / label
                            if not target.exists():
                                target.mkdir()
                                (target / "earlier.jpg").write_bytes(earlier)
                                (target / "submitted.jpg").write_bytes(uploaded)
                                (target / "truth.json").write_bytes(pack(row))
            print(f"{split} {group['tag']}: {len(rows)} deliveries complete", flush=True)
    return rows


def tally(rows, point, key="checks"):
    counts = Counter()
    for row in rows:
        readout = row[key][str(point)] if key == "checks" else row[key]
        if key == "checks" and row["baseline"] and row["baseline"]["assessed"]:
            readout = row["baseline"]
        if row["role"] in EDITED:
            counts["edits"] += 1
            counts["useful"] += any(useful(r["intersection"], r["selected"], row["truth_pixels"]) for r in readout["regions"])
            counts["edit_unassessed"] += not readout["assessed"]
            counts["edits_flagged"] += bool(readout["regions"])
        elif row["role"] in ORDINARY:
            counts["ordinary"] += 1
            counts["ordinary_alerts"] += bool(readout["regions"])
            counts["ordinary_unassessed"] += not readout["assessed"]
        elif row["role"] in NO_MATCH:
            counts["incompatible"] += 1
            counts["incompatible_accepts"] += readout["assessed"]
        else:
            counts["local_processing"] += 1
            counts["local_processing_flags"] += bool(readout["regions"])
    return dict(sorted(counts.items()))


def fit(rows):
    levels, calibration = {}, {}
    for quality in QUALITY:
        subset = [r for r in rows if r["quality"] == quality]
        counts = {str(point): tally(subset, point) for point in POINT_GRID}
        allowed = [p for p in POINT_GRID if counts[str(p)]["ordinary_alerts"] <= 1 and not counts[str(p)]["incompatible_accepts"]]
        if not allowed:
            raise ValueError(f"No qualifying calibration point at JPEG quality {quality}")
        winner = max(allowed, key=lambda p: (counts[str(p)]["useful"], -counts[str(p)]["ordinary_alerts"], p))
        levels[str(quality)], calibration[str(quality)] = winner, counts
    return levels, calibration


def summarize(rows, levels):
    totals = {}
    predicates = {"all": lambda row: True, **{family: (lambda row, feature=family: row["family"] == feature)
                                             for family in sorted({row["family"] for row in rows})}}
    for name, include in predicates.items():
        totals[name] = {}
        for quality in QUALITY:
            filtered = [r for r in rows if r["quality"] == quality and include(r)]
            point = levels[str(quality)]
            totals[name][str(quality)] = {"current": tally(filtered, point, "baseline"), "combined": tally(filtered, point)}
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "assess", "verify"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--source", action="append", nargs=2, metavar=("MEDIA", "SPLIT"))
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--examples", type=Path)
    args = parser.parse_args()
    if args.action == "verify":
        data = load(args.receipt)
        frozen = data["freeze"]
        assert fit(frozen["rows"]) == (frozen["points"], frozen["calibration"])
        known = {r["source"] for r in frozen["rows"]}
        assert not known.intersection(r["source"] for r in data["rows"])
        assert summarize(data["rows"], frozen["points"]) == data["totals"]
        print(json.dumps(data["totals"], indent=2))
        return
    import cv2

    cv2.setNumThreads(1)
    sources = [(Path(root), split) for root, split in args.source]
    pins = [{"sha256": file_sha(root / "manifest.json"), "split": split} for root, split in sources]
    if args.action == "fit":
        rows = records(args.directory, sources, None)
        points, calibration = fit(rows)
        save(args.receipt, {"method": METHOD, "manifests": pins, "points": points, "calibration": calibration, "rows": rows})
        print(json.dumps({"points": points, "calibration": calibration}, indent=2), flush=True)
    else:
        frozen = load(args.freeze)
        if args.examples:
            args.examples.mkdir()
        rows = records(args.directory, sources, frozen["points"], args.examples)
        assert not {r["source"] for r in frozen["rows"]}.intersection(r["source"] for r in rows)
        totals = summarize(rows, frozen["points"])
        save(args.receipt, {"freeze": frozen, "holdout_manifests": pins, "rows": rows, "totals": totals})
        print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
