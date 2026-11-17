"""Measure localized changes between HEIF originals and newly delivered JPEGs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import image_reference_perspective_panel as shared

POINTS = (14., 18., 22., 26.)
GEOMETRIES = ("offset_crop", "tilted_crop")
ROLES = ("sharp", "feather", "ordinary", "whole_tone", "whole_blur", "local_blur", "other_photo", "half_replaced")
SCHEMA = "chipmunk.reference-heif-panel/1"


def inputs(source, role):
    from scripts.image_cross_quality_fit import images

    manifest = json.loads((source / "manifest.json").read_text())
    groups = list(images(manifest, ("fresh",)))
    if len(groups) != 24:
        raise ValueError("The evaluation uses the committed 24-source comparison")
    return groups[:6] if role == "development" else groups[6:]


def geometry(size, motion, index):
    import cv2
    import numpy as np

    width, height = size
    framing = np.array(((.12, .07), (.94, .07), (.94, .92), (.12, .92)))
    if motion == "tilted_crop":
        framing = np.array(((.10, .055), (.90, .115), (.94, .89), (.055, .85)))
    if index % 2:
        framing = framing[[1, 0, 3, 2]]
        framing[:, 0] = 1 - framing[:, 0]
    crop = (.82 * width, .85 * height)
    scale = (680, 960, 1280)[index % 3] / max(crop)
    w, h = (round(value * scale) for value in crop)
    output = np.array(((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)), dtype=np.float32)
    fitted = cv2.getPerspectiveTransform(output, (framing * (width - 1, height - 1)).astype(np.float32))
    return fitted, (w, h)


def earlier_file(pixels, path):
    import cv2
    from PIL import Image
    from pillow_heif import from_pillow

    h, w = pixels.shape[:2]
    scale = 1760 / max(h, w)
    enlarged = cv2.resize(pixels, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    with Image.fromarray(enlarged) as image:
        from_pillow(image).save(path, quality=91)
    return enlarged.shape[1::-1]


def measurement(reference, jpeg, truth, native_size, known, points):
    import io

    import numpy as np
    from PIL import Image

    from app.detectors.image import reference_heif, reference_perspective

    with Image.open(io.BytesIO(jpeg)) as image:
        submitted = reference_heif.display_pixels(image)
    measured = reference_heif.measure(reference, submitted)
    geometry_error = None
    if measured is not None:
        w, h = submitted.shape[1::-1]
        probes = [(x * (w - 1), y * (h - 1)) for y in (.1, .5, .9) for x in (.1, .5, .9)]
        geometry_error = float(np.max(np.linalg.norm(
            shared.project(known, probes) - shared.project(measured.transform, probes), axis=1)))
    return {str(point): shared.reading(*reference_perspective.compare(measured, native_size, point), truth)
            for point in points}, geometry_error


def assess(source, output, *, role, freeze=None):
    import cv2
    import numpy as np
    from PIL import Image

    from app.detectors.image import reference_heif, reference_perspective
    from app.engine.media import refuse_truncated_images
    from scripts.image_cross_quality_fit import load_mask

    refuse_truncated_images()
    cv2.setNumThreads(1)
    groups = inputs(source, role)
    paths = (Path(reference_heif.__file__), Path(reference_perspective.__file__), Path(__file__))
    method = {path.name: shared.file_sha(path) for path in paths}
    if role == "held" and (freeze is None or freeze["method"] != method
                           or set(group["source"] for group, _ in groups) & set(freeze["source_ids"])):
        raise ValueError("Held sources must follow the frozen method and be separate from development")
    output.mkdir(mode=0o700)
    cases, evidence = [], output / "deliveries"
    evidence.mkdir()
    points = POINTS if freeze is None else (freeze["point"],)
    for index, (group, inputs_by_role) in enumerate(groups):
        directory = evidence / group["tag"]
        directory.mkdir()
        origin = shared.pixels(shared.input_data(source, inputs_by_role["ordinary", 90]))
        other = shared.pixels(shared.input_data(source, groups[(index + 1) % len(groups)][1]["ordinary", 90]))
        earlier = directory / "earlier.heic"
        native_size = earlier_file(origin, earlier)
        with Image.open(earlier) as native:
            assert reference_heif.supported_source(native)
            reference = reference_heif.display_pixels(native)
        earlier_sha = shared.file_sha(earlier)
        sx, sy = np.array(native_size) / origin.shape[:2][::-1]
        source_to_native = np.array(((sx, 0., (sx - 1) / 2), (0., sy, (sy - 1) / 2), (0., 0., 1.)))
        for motion in GEOMETRIES:
            matrix, size = geometry(origin.shape[1::-1], motion, index)
            for name in ROLES:
                body = inputs_by_role[name if name in ("sharp", "feather", "local_blur") else "ordinary", 90]
                pixels = shared.pixels(shared.input_data(source, body))
                before = load_mask(source, body)
                displayed, confirmed = shared.render(pixels, other, before, name, matrix, size)
                for quality in (75, 85 if index % 2 else 90):
                    jpeg = shared.jpeg(displayed, quality)
                    path = directory / f"{motion}-{name}-{quality}.jpg"
                    path.write_bytes(jpeg)
                    result, error = measurement(reference, jpeg, confirmed, native_size, source_to_native @ matrix, points)
                    cases.append({"source": group["source"], "file": path.relative_to(output).as_posix(),
                                  "jpeg_sha256": shared.digest(jpeg), "earlier_sha256": earlier_sha,
                                  "geometry": motion, "role": name, "quality": quality,
                                  "truth_pixels": int(confirmed.sum()), "observations": result,
                                  "max_native_alignment_error_px": None if error is None else round(error, 5)})
        print(role, index + 1, "/", len(groups), flush=True)
    value = {"schema": SCHEMA, "role": role, "method": method, "rows": cases,
             "source_ids": sorted(group["source"] for group, _ in groups),
             "input_manifest_sha256": shared.file_sha(source / "manifest.json")}
    if freeze is None:
        counts = {str(point): summarize(cases, point) for point in points}
        eligible = [point for point in points if counts[str(point)]["ordinary_flags"] <= 1
                    and counts[str(point)]["incompatible_accepted"] == 0]
        if not eligible:
            raise ValueError("No development point meets the ordinary and unmatched checks")
        point = max(eligible, key=lambda value: (counts[str(value)]["useful"], -counts[str(value)]["ordinary_flags"], value))
        value.update(point=point, counts=counts)
    else:
        value.update(point=freeze["point"], counts=summarize(cases, freeze["point"]))
    shared.save(output / "receipt.json.gz", value)
    return value


def summarize(cases, point):
    counts = Counter({key: 0 for key in ("edited", "ordinary", "incompatible", "local_blur", "useful",
                                         "ordinary_flags", "incompatible_accepted", "local_blur_flags", "assessed")})
    for row in cases:
        observed = row["observations"][str(point)]
        name = row["role"]
        flagged = bool(observed["regions"])
        counts["assessed"] += observed["assessed"]
        if name in shared.EDITED:
            counts["edited"] += 1
            counts["useful"] += any(shared.useful(item["intersection"], item["selected"], row["truth_pixels"])
                                    for item in observed["regions"])
        elif name in shared.NO_MATCH:
            counts["incompatible"] += 1
            counts["incompatible_accepted"] += observed["assessed"]
        elif name == "local_blur":
            counts["local_blur"] += 1
            counts["local_blur_flags"] += flagged
        else:
            counts["ordinary"] += 1
            counts["ordinary_flags"] += flagged
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = shared.load(args.source)
        assert result["schema"] == SCHEMA
        counts = (summarize(result["rows"], result["point"]) if result["role"] == "held"
                  else {str(point): summarize(result["rows"], point) for point in POINTS})
        assert counts == result["counts"]
        print(counts)
    else:
        freeze = shared.load(args.freeze) if args.freeze else None
        result = assess(args.source, args.output, role="held" if freeze else "development", freeze=freeze)
        print(result["point"], result["counts"])


if __name__ == "__main__":
    main()
