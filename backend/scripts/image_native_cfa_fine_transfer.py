# SPDX-License-Identifier: Apache-2.0
"""Measure small alternate interpolation regions in native PNGs."""

import argparse
import hashlib
import importlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
SCHEMA = "chipmunk.image-native-fine-interpolation/1"
ROSTERS = {"development": ("image-native-cfa-cc0-roster-2026-11-13.json", 6),
           "held": ("image-native-cfa-cc0-png-roster-2026-11-13.json", 10),
           "transfer": ("image-native-cfa-dimensions-cc0-roster-2026-11-13.json", 10)}
SIZES = (32, 48, 64)
SHIFTS = ((1, 0), (0, 1), (1, 1))
POSITIVE = tuple(f"{size}_{style}" for size in SIZES for style in ("sharp", "feather"))
CONTROL = ("untouched", "even_copy", "local_tone", "local_blur", "local_fractional", "global_shift", "jpeg_saved_as_png")
DESIGN = {"scope": "2048 by 1536 bare opaque 16-bit PNG, developed on the native sensor grid.",
          "method": "Learn the normal phase in two spatial folds on 32 px cells. If the coarse rule has no region, search "
                    "16 px cells at half-cell spacing. Require one alternate phase with closure >= .62 and at least .25 "
                    "above the reference in at least two-thirds of the 2x2, 2x3, 3x2 or 3x3 cells.",
          "calibration": "Second-highest fine response maximum across six development ordinary photo groups, floored at .25.",
          "useful": "Minimum changed-pixel recall .50 and intersection over union .30.",
          "outcomes": "A camera interpolation change is a local editing cue with zero AI-origin confidence."}


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def source_plan():
    groups = {}
    used, models = set(), set()
    for split, (name, count) in ROSTERS.items():
        rows = read(PANELS / name)[:count]
        require(len(rows) == count, "A source role is incomplete")
        for row in rows:
            model = row["model"].casefold()
            require(row["license"] == "CC0-1.0" and row["sha256"] not in used and model not in models,
                    "Photo ancestry, rights or camera roles overlap")
            used.add(row["sha256"])
            models.add(model)
        groups[split] = rows
    return groups


def cases(photo, ordinal, split):
    import cv2
    import numpy as np

    bank = tuple(ROSTERS).index(split)
    height, width = photo.shape[:2]
    x = 416 + 8 * ((17 * ordinal + 5 + 3 * bank) % 137)
    y = 320 + 8 * ((13 * ordinal + 7 + 5 * bank) % 101)
    donor_x, donor_y = 160 + 16 * (ordinal % 5), 128 + 16 * (ordinal % 4)
    dx, dy = SHIFTS[(ordinal + bank) % len(SHIFTS)]
    empty = np.zeros(photo.shape[:2], bool)

    def patch(donor, size, feather=False):
        pixels = photo.copy()
        region = pixels[y:y+size, x:x+size]
        if feather:
            yy, xx = np.indices((size, size))
            rim = np.minimum.reduce((yy+1, xx+1, size-yy, size-xx))
            alpha = ((1 - np.cos(np.pi * np.minimum(1., rim / 6))) / 2)[..., None]
            donor = np.rint(region * (1-alpha) + donor * alpha).astype(np.uint16)
        changed = empty.copy()
        changed[y:y+size, x:x+size] = np.any(region != donor, axis=2)
        region[:] = donor
        return pixels, changed

    yield "untouched", photo, empty
    base = photo[y:y+48, x:x+48]
    yield "even_copy", patch(photo[donor_y:donor_y+48, donor_x:donor_x+48], 48)[0], empty
    yield "local_tone", patch(np.rint(base * .85 + 1300).astype(np.uint16), 48)[0], empty
    yield "local_blur", patch(cv2.GaussianBlur(base, (5, 5), 1.1), 48)[0], empty
    affine = np.array([[1, 0, .5], [0, 1, .375]], np.float32)
    yield "local_fractional", patch(cv2.warpAffine(base, affine, (48, 48), borderMode=cv2.BORDER_REFLECT_101), 48)[0], empty
    rows, cols = np.minimum(np.arange(height)+1, height-1), np.minimum(np.arange(width)+1, width-1)
    yield "global_shift", photo[np.ix_(rows, cols)], empty
    ok, jpeg = cv2.imencode(".jpg", np.ascontiguousarray((photo[..., ::-1] >> 8).astype(np.uint8)), [cv2.IMWRITE_JPEG_QUALITY, 90])
    require(ok, "Could not create the JPEG delivery control")
    yield "jpeg_saved_as_png", cv2.imdecode(jpeg, cv2.IMREAD_COLOR)[..., ::-1].copy(), empty
    for size in SIZES:
        donor = photo[donor_y+dy:donor_y+dy+size, donor_x+dx:donor_x+dx+size]
        for name in ("sharp", "feather"):
            modified, changed = patch(donor, size, name == "feather")
            require(changed.any(), "The authored local shift changed no pixels")
            yield f"{size}_{name}", modified, changed


def native_inputs(options):
    import cv2
    from scripts import image_native_cfa_closure_transfer as baseline

    cv2.setNumThreads(1)
    wheel = baseline.raw_source.verified_wheel(options.cache / baseline.raw_source.WHEEL)
    with tempfile.TemporaryDirectory(prefix="chipmunk-cfa-fine-decoder-") as directory:
        baseline.raw_source.unpack_decoder(wheel, directory)
        sys.path.insert(0, directory)
        try:
            decoder = importlib.import_module("rawpy")
            require(Path(decoder.__file__).resolve().is_relative_to(Path(directory).resolve()), "Expected the pinned RAW decoder")
            for index, row in enumerate(source_plan()[options.phase]):
                path = options.archive / f"sample-{row['id']}{Path(row['url']).suffix.lower()}"
                require(file_hash(path) == row["sha256"], "The camera-source checksum differs")
                photo, _ = baseline.raw_render(decoder, path)
                yield row, cases(photo, index, options.phase)
        finally:
            sys.path.pop(0)


def readout(values, truth):
    import numpy as np
    from app.detectors.image import native_cfa, native_cfa_fine
    from scripts.image_native_cfa_png_transfer import png_body

    body = png_body(values)
    decoded = native_cfa.decoded_rgb(body)
    row = {"png_sha256": digest(body), "truth_sha256": digest(truth.tobytes()), "changed": int(truth.sum()),
           "phase": None, "coarse": None, "fine": None}
    if decoded is None:
        return row | {"status": "unsupported_png"}
    require(np.array_equal(values, decoded), "PNG changed native samples")
    coarse = native_cfa.measure(decoded)
    if coarse.phase is None:
        return row | {"status": coarse.status}

    def region(box):
        x, y, x1, y1 = box
        return {"region": list(box), "intersection": int(truth[y:y1, x:x1].sum())}

    # The new check is a fallback. A selected coarse box retains priority.
    fine = native_cfa_fine.propose(decoded, coarse.phase) if coarse.region is None else None
    return row | {"status": "assessed", "phase": coarse.phase,
                  "coarse": region(coarse.region) if coarse.region else None,
                  "fine": fine | region(fine["region"]) if fine is not None else None}


def threshold(rows):
    maxima = {}
    for row in rows:
        if row["role"] in CONTROL:
            score = row["fine"]["response"] if row["fine"] else 0.
            maxima[row["id"]] = max(score, maxima.get(row["id"], 0.))
    require(len(maxima) == ROSTERS["development"][1], "An ordinary calibration source is missing")
    return max(.25, sorted(maxima.values())[-2])


def review(record, development=None):
    split, rows = record["split"], record["rows"]
    require(record["schema"] == SCHEMA and record["design"] == DESIGN, "Unexpected fixed protocol")
    require([(row["id"], row["role"]) for row in rows] == [(source["id"], role) for source in source_plan()[split]
            for role in (*CONTROL, *POSITIVE)], "A frozen camera or delivery is missing")
    if split == "development":
        point = threshold(rows)
    else:
        require(development is not None and development["split"] == "development", "The development receipt is required")
        review(development)
        point = threshold(development["rows"])
    require(point == record["point"], "The genuine calibration point differs")
    counts = {role: Counter() for role in (*CONTROL, *POSITIVE)}
    for row in rows:
        case = row["role"]
        require((row["changed"] > 0) == (case in POSITIVE) and row["status"] in {"assessed", "no_repeatable_stencil", "unsupported_png"},
                "The recorded edit truth or coverage differs")
        require((type(row["phase"]) is int and row["phase"] in range(4)) if row["status"] == "assessed"
                else (row["phase"], row["coarse"], row["fine"]) == (None, None, None), "The reading differs from the recorded coverage")
        for name in ("coarse", "fine"):
            if reading := row[name]:
                a, b, c, d = reading["region"]
                require(0 <= a < c <= 2048 and 0 <= b < d <= 1536 and row["phase"] in range(4)
                        and 0 <= reading["intersection"] <= min(row["changed"], (c-a) * (d-b)), "A region escaped its source")
                if name == "fine":
                    require(a % 8 == b % 8 == 0 and c-a in {32, 48} and d-b in {32, 48} and row["coarse"] is None
                            and reading["alternate_phase"] in range(4) and reading["alternate_phase"] != row["phase"]
                            and reading["compared_cells"] == (c-a)*(d-b)//256
                            and math.isfinite(reading["response"]) and 0 < reading["response"] <= 2.25
                            and 2*reading["compared_cells"] <= 3*reading["matching_cells"] <= 3*reading["compared_cells"],
                            "Fine phase cells differ from the native geometry")
        coarse, fine = row["coarse"], row["fine"]
        expanded = coarse or (fine if fine and fine["response"] > point else None)

        def useful(region):
            if not region or not row["changed"]:
                return False
            x, y, x1, y1 = region["region"]
            common = region["intersection"]
            return 2*common >= row["changed"] and 10*common >= 3*((x1-x)*(y1-y) + row["changed"] - common)

        counts[case].update(files=1, assessed=int(row["status"] == "assessed"), coarse_flags=int(coarse is not None),
                            coarse_useful=int(useful(coarse)), expanded_flags=int(expanded is not None), expanded_useful=int(useful(expanded)))
    result = {key: dict(value) for key, value in counts.items()}
    require("results" not in record or record["results"] == result, "Stored results do not reproduce")
    return result


def run(options):
    rows = []
    for source, deliveries in native_inputs(options):
        for role, values, truth in deliveries:
            rows.append({"id": source["id"], "role": role, **readout(values, truth)})
        print(options.phase, "camera", source["id"], flush=True)
    development = read(options.calibration) if options.calibration else None
    record = {"schema": SCHEMA, "design": DESIGN, "split": options.phase, "rows": rows,
              "point": threshold(rows if development is None else development["rows"])}
    record["results"] = review(record, development)
    options.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*ROSTERS, "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(review(read(args.output), read(args.calibration) if args.calibration else None), sort_keys=True, indent=2))
    else:
        if args.cache is None or args.archive is None or (args.phase != "development" and args.calibration is None):
            parser.error("Native scoring requires --cache and --archive, and the held roles require --calibration")
        run(args)
