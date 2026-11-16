# SPDX-License-Identifier: Apache-2.0
"""Rebuild and evaluate bounded right-angle matching on larger JPEG uploads."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from scripts.image_exact_turned_copy_transfer import digest, overlap, useful

SCHEMA = "chipmunk.image-jpeg-native-rotation-transfer/1"
PANELS = Path("docs/evaluation-panels")
ROSTERS = {
    "development": PANELS / "image-native-cfa-cc0-png-roster-2026-11-13.json",
    "held": PANELS / "image-native-cfa-dimensions-cc0-roster-2026-11-13.json",
}
COMMONS = PANELS / "image-affine-native-scale-2026-11-11.json"
EDITS = ("quarter", "half", "feather")
CONTROLS = ("ordinary", "blur", "straight", "mirror")
GEOMETRIES = {"medium": (1920, 1280), "large": (3072, 2048)}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, document):
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def qualify(archive, phase):
    roster = read(ROSTERS[phase])
    if phase == "development":
        roster = roster[:4]
    assert len(roster) == (4 if phase == "development" else 10)
    assert len({item["model"] for item in roster}) == len(roster)
    for item in roster:
        path = archive / f"sample-{item['id']}{Path(item['url']).suffix.lower()}"
        assert item["license"] == "CC0-1.0" and sha(path) == item["sha256"]
        yield item, path


def native_rgb(decoder, path):
    import numpy as np

    with decoder.imread(str(path)) as raw:
        height, width = raw.sizes.height, raw.sizes.width
        rgb = raw.postprocess(use_camera_wb=False, use_auto_wb=True, output_bps=8, user_flip=0)
    assert rgb.dtype == np.uint8 and rgb.shape == (height, width, 3)
    return np.rot90(rgb) if height > width else rgb


def select(rgb, w, h, group):
    import numpy as np

    if w > rgb.shape[1] or h > rgb.shape[0]:
        raise ValueError("CC0 original lacks the required native dimensions")
    x, y = (rgb.shape[1] - w) // 2, (rgb.shape[0] - h) // 2
    native = rgb[y : y + h, x : x + w]
    return np.ascontiguousarray(np.rot90(native) if group % 3 == 1 else native)


def placements(image, group, sw, sh):
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dx, dy = w // 7 + 37 * group % 193, h // 6 + 21 * group % 141

    def texture_at(x, y):
        area = gray[y : y + sh, x : x + sw]
        return float(np.mean(abs(np.diff(area, axis=0))) + np.mean(abs(np.diff(area, axis=1)))), float(
            area.std()
        )

    pool = []
    for y in range(h // 2 + 32, h - sh - 30, 123):
        for x in range(w // 2 + 32, w - sw - 30, 127):
            texture, contrast = texture_at(x, y)
            if texture >= 5 and contrast >= 10:
                pool.append((x, y))
    if not pool:
        candidates = (
            (x, y)
            for y in range(30, h - sh - 30, 123)
            for x in range(30, w - sw - 30, 127)
            if max(abs(x - dx), abs(y - dy)) >= max(sw, sh) + 32
        )
        pool = [max(candidates, key=lambda pos: texture_at(*pos))]
    sx, sy = min(pool, key=lambda position: digest(f"{SCHEMA}/{group}/{sw}/{sh}/{position}".encode()))
    return sx, sy, dx, dy


def treatments(image, group):
    import cv2
    import numpy as np

    yield "ordinary", image, None, None
    for name, width, height, turns, blend in (
        ("quarter", 64, 80, 1, 0),
        ("half", 80, 88, 2, 0),
        ("feather", 80, 96, 3, 7),
        ("straight", 64, 80, 0, 0),
        ("mirror", 64, 80, 0, 0),
        ("blur", 64, 80, 0, 0),
    ):
        sw, sh = (height, width) if turns % 2 else (width, height)
        sx, sy, x, y = placements(image, group, sw, sh)
        output = image.copy()
        target = output[y : y + height, x : x + width]
        copied = image[sy : sy + sh, sx : sx + sw]
        if name == "mirror":
            copied = copied[:, ::-1]
        elif name == "blur":
            copied = cv2.GaussianBlur(target, (0, 0), 1.25)
        copied = np.rot90(copied, turns)
        if blend:
            yy, xx = np.indices(target.shape[:2])
            alpha = np.minimum(1, np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy]) / blend)[
                ..., None
            ]
            copied = np.rint(target * (1 - alpha) + copied * alpha).astype(np.uint8)
        difference = np.any(target != copied, axis=2)
        assert difference.any()
        target[:] = copied
        truth = [[x, y, width, height], [sx, sy, sw, sh]] if name in EDITS else None
        yield name, output, truth, digest(difference.tobytes()) if truth else None


def freeze(raw_archive, wheel, commons, destination, phase):
    import importlib

    import cv2
    import numpy as np
    from PIL import Image

    from scripts import image_rawnind_phase_v239 as decoder_package
    from scripts.image_jpeg_turned_copy_transfer import encode_jpeg
    from scripts.image_offgrid_patch_transfer import hard_controls

    cv2.setNumThreads(1)
    destination.mkdir(mode=0o700)
    rows = []

    def add(group, raster, shape, quality, role, truth, mask):
        encoded = encode_jpeg(raster, quality)
        name = f"{group}-{shape}-{role}.jpg"
        (destination / name).write_bytes(encoded)
        rows.append(
            {
                "file": name,
                "group": group,
                "geometry": shape,
                "size": list(raster.shape[1::-1]),
                "quality": quality,
                "role": role,
                "truth": truth,
                "changed_crop_sha256": mask,
                "sha256": digest(encoded),
            }
        )

    sources = []
    with tempfile.TemporaryDirectory(prefix="chipmunk-jpeg-raw-decoder-") as directory:
        decoder_package.unpack_decoder(decoder_package.verified_wheel(wheel), directory)
        sys.path.insert(0, directory)
        try:
            decoder = importlib.import_module("rawpy")
            assert Path(decoder.__file__).resolve().is_relative_to(Path(directory).resolve())
            for group, (item, path) in enumerate(qualify(raw_archive, phase)):
                picture = native_rgb(decoder, path)
                group_id = str(item["id"])
                sources.append(item | {"group": group_id, "native_size": list(picture.shape[1::-1])})
                for shape, size in GEOMETRIES.items():
                    raw = select(picture, *size, group)
                    for role, raster, truth, mask in treatments(raw, group):
                        add(group_id, raster, shape, (90, 95)[group % 2], role, truth, mask)
                print(f"Froze CC0 camera group {group + 1}/{4 if phase == 'development' else 10}", flush=True)
        finally:
            sys.path.pop(0)
    repeats = []
    if phase == "held":
        for index, row in enumerate(hard_controls(commons)):
            with Image.open(row["path"]) as image:
                original = np.asarray(image.convert("RGB"))
            if min(original.shape[:2]) < 1024 or max(original.shape[:2]) > 3072:
                continue
            group = f"repeat-{row['slot']}"
            repeats.append(
                {key: row[key] for key in ("author", "page", "licence_url", "jpeg_sha256")} | {"group": group}
            )
            add(group, original, "repeat", (90, 95)[index % 2], "repetition", None, None)
    frozen = {
        "schema": SCHEMA,
        "phase": phase,
        "sources": sources,
        "repeated_structures": repeats,
        "rows": rows,
        "camera_roster_sha256": sha(ROSTERS[phase]),
        "commons_sha256": sha(COMMONS),
        "decoder_sha256": sha(wheel),
        "detector_sha256": sha(Path("backend/app/detectors/image/jpeg_native_turned_copy.py")),
    }
    write(destination / "freeze.json", frozen)


def baseline(rgb):
    import cv2
    import numpy as np

    from app.detectors.image import copy_affine, copy_affine_large, copy_move
    from app.evidence.image_copy_affine_binding import LARGE_VERSION, supported

    if not supported(rgb.shape[1], rgb.shape[0], version=LARGE_VERSION):
        return {"status": "unsupported_geometry", "pairs": []}
    try:
        proposals = copy_affine_large.candidates(rgb)
    except copy_move.CandidateBudgetExceeded:
        return {"status": "unassessed_match_budget", "pairs": []}
    pairs = []
    if proposals:
        gray = copy_affine.luma(rgb)
        for matrix, bounds in proposals:
            if (masks := copy_affine.dense_pair(gray, matrix, bounds)) is not None:
                pairs.append([list(cv2.boundingRect(mask.astype(np.uint8))) for mask in masks])
    return {"status": "supported_pair" if pairs else "no_pair", "pairs": pairs}


def measure(directory, row):
    import cv2
    import numpy as np
    from PIL import Image

    from app.detectors.image import jpeg_native_turned_copy, jpeg_turned_copy

    cv2.setNumThreads(1)
    data = (directory / row["file"]).read_bytes()
    assert digest(data) == row["sha256"]
    with Image.open(io.BytesIO(data), formats=["JPEG"]) as picture:
        assert list(picture.size) == row["size"]
        pixels = np.asarray(picture)
    existing = baseline(pixels)
    status, items = jpeg_native_turned_copy.analyse(pixels)
    for item in items:
        assert jpeg_turned_copy.witness(pixels, item["pair"], item["turns"]) == item["measurement"]
    return {"existing": existing, "native": {"status": status, "pairs": items}}


def tally(rows, observed):
    result = {}
    for strata in ("all", "medium", "large", "repeat", "q90", "q95"):
        counts, gained = Counter(), set()
        for row, observation in zip(rows, observed, strict=True):
            if strata not in ("all", row["geometry"], f"q{row['quality']}"):
                continue
            expected = row["truth"]
            old, new = (observation[arm] for arm in ("existing", "native"))
            flag = bool(new["pairs"])
            old_hit = bool(expected and useful(old["pairs"], expected))
            has_hit = bool(expected and useful([p["pair"] for p in new["pairs"]], expected))
            counts.update(
                edits=bool(expected),
                ordinary=not expected,
                existing_useful=old_hit,
                native_useful=has_hit,
                any_useful=old_hit or has_hit,
                gained=has_hit and not old_hit,
                native_misplaced=flag and bool(expected) and not has_hit,
                ordinary_flags=flag and not expected,
                unassessed=new["status"] not in {"no_pair", "supported_pair"},
            )
            if has_hit and not old_hit:
                gained.add(row["group"])
        result[strata] = dict(counts, gained_sources=len(gained))
    return result


def review(record):
    plan, observed = record["freeze"], record["observed"]
    phase = plan["phase"]
    assert plan["schema"] == SCHEMA and plan["camera_roster_sha256"] == sha(ROSTERS[phase])
    assert plan["commons_sha256"] == sha(COMMONS) and plan["sources"]
    assert len(plan["rows"]) == len(observed) and len({r["sha256"] for r in plan["rows"]}) == len(observed)
    expected = {
        (source["group"], shape, role)
        for source in plan["sources"]
        for shape in GEOMETRIES
        for role in (*EDITS, *CONTROLS)
    }
    expected.update((row["group"], "repeat", "repetition") for row in plan["repeated_structures"])
    assert {(r["group"], r["geometry"], r["role"]) for r in plan["rows"]} == expected
    for row, observation in zip(plan["rows"], observed, strict=True):
        native = observation["native"]
        assert bool(row["truth"]) == bool(row["changed_crop_sha256"]) == (row["role"] in EDITS)
        assert native["status"] in {"no_pair", "supported_pair", "ambiguous", "unassessed_match_budget"}
        assert len(native["pairs"]) == (native["status"] == "supported_pair")
        for item in native["pairs"]:
            width, height = row["size"]
            a, b = item["pair"]
            assert not overlap(a, b) and item["turns"] in (1, 2)
            assert a[2:] == (b[2:][::-1] if item["turns"] == 1 else b[2:])
            assert all(
                0 <= x <= width - w and 0 <= y <= height - h and 32 <= min(w, h) <= max(w, h) <= 118
                for x, y, w, h in (a, b)
            )
            witness = item["measurement"]
            mean = witness["interior_mean"]
            assert 0 <= mean <= 3.2 and 0 <= witness["interior_p90"] <= 6
            assert (
                sum(v is not None and v + 0.00005 >= max(7, 2.5 * mean) for v in witness["perimeter_means"])
                >= 3
            )
    result = tally(plan["rows"], observed)
    assert result == record.get("results", result)
    return result


def capture(directory, output, workers):
    from itertools import repeat

    plan = read(directory / "freeze.json")
    assert plan["detector_sha256"] == sha(Path("backend/app/detectors/image/jpeg_native_turned_copy.py"))
    readings = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for measured in pool.map(measure, repeat(directory), plan["rows"]):
            readings.append(measured)
            if len(readings) % 14 == 0:
                print(f"Measured {len(readings)}/{len(plan['rows'])} committed JPEGs", flush=True)
    record = {"freeze": plan, "observed": readings}
    record["results"] = review(record)
    write(output, record)
    print(json.dumps(record["results"], sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("freeze")
    prepare.add_argument("phase", choices=ROSTERS)
    for name in ("raw_archive", "wheel", "commons", "output"):
        prepare.add_argument(name, type=Path)
    measured = modes.add_parser("capture")
    measured.add_argument("inputs", type=Path)
    measured.add_argument("output", type=Path)
    measured.add_argument("--workers", type=int, default=3)
    verify = modes.add_parser("review")
    verify.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.raw_archive, args.wheel, args.commons, args.output, args.phase)
    elif args.mode == "capture":
        capture(args.inputs, args.output, args.workers)
    else:
        print(json.dumps(review(read(args.receipt)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
