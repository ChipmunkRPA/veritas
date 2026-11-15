# SPDX-License-Identifier: Apache-2.0
"""Compare compact camera interpolation cores on selected and offset edits."""

import argparse
import gzip
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from app.detectors.image import native_cfa_core as core
from scripts import image_native_cfa_fine_transfer as fine

SCHEMA = "chipmunk.native-cfa-compact-core/1"
SPLITS = tuple(fine.ROSTERS)
LEVELS = ("tight", "wider")
RETAINED = ("32_feather", "32_sharp", "48_feather", "64_feather", *fine.CONTROL)
NEW = {
    "offset_feather6": True, "offset_feather9": True,
    "offset_even_feather": False, "offset_fractional_feather": False,
    "whole_resize": False, "adaptive_development": False,
}
DESIGN = {
    "resolution": "32 by 32 native pixels with four eight-pixel core cells, scanned at a four-pixel step.",
    "comparison": "Core uses at least three cells supporting the same alternate phase. Surrounded uses at least two and "
                  "requires six of eight exterior samples recovering the native phase, with at least one on each side.",
    "calibration": "Fit individual thresholds to the maximum and second-highest ordinary-camera maxima across the "
                   "development role. Apply strict excess. Preserve earlier selected regions on all inputs.",
    "transfer": "Six and nine pixel cosine blends on differently placed non-grid crops, with matched ordinary even copies "
                "and fractional translations. Additional controls resize the whole image or develop the RAW adaptively.",
    "useful": "The crop covers half the exact changed pixels and reaches 0.30 intersection over union.",
    "confidence_weight": 0,
}
read, require = fine.read, fine.require


def prior_path(split):
    return fine.PANELS / f"image-native-cfa-fine-{split}-2026-11-15.json"


def extent_path(split):
    return fine.PANELS / f"image-native-cfa-extent-{split}-2026-11-15.json"


def retained_inputs(split):
    old = read(prior_path(split))
    extend = read(extent_path(split))
    fine.review(old, read(prior_path("development")) if split != "development" else None)
    changes = {(row["id"], row["role"]): row["extent"] for row in extend["measurements"]["prior"]}
    return {(row["id"], row["role"]): row | {"existing": row["coarse"] or changes[row["id"], row["role"]]
                                           or active_fine(row)} for row in old["rows"]}


def active_fine(record):
    # Use the same fixed fine point as the released development option.
    value = record["fine"]
    return value if value and value["response"] > .25 else None


def new_cases(photo, id_value, index, split):
    import cv2
    import numpy as np

    seed = bytes.fromhex(fine.digest(f"{SCHEMA}/{split}/{id_value}".encode()))
    x = 416 + int.from_bytes(seed[:2], "little") % 1035
    y = 320 + int.from_bytes(seed[2:4], "little") % 779
    # Deliberate non-grid starts exercise the nearest four-pixel search instead
    # of exploiting the aligned grid in the existing camera study.
    x += not x % 4
    y += not y % 4
    source_x, source_y = 80 + 2 * seed[4], 96 + 2 * seed[5]
    source_x += x % 2
    source_y += y % 2
    dx, dy = fine.SHIFTS[(index + tuple(fine.ROSTERS).index(split)) % len(fine.SHIFTS)]
    focused = photo[y:y+32, x:x+32]
    yy, xx = np.indices(focused.shape[:2])
    margin = np.minimum.reduce((yy+1, xx+1, 32-yy, 32-xx))
    empty = np.zeros(photo.shape[:2], bool)

    def patch(name, donor, width, altered):
        result = photo.copy()
        alpha = ((1 - np.cos(np.pi * np.minimum(1., margin / width))) / 2)[..., None]
        blend = np.rint(focused * (1 - alpha) + donor * alpha).astype(np.uint16)
        truth = empty.copy()
        if altered:
            truth[y:y+32, x:x+32] = np.any(blend != focused, axis=-1)
        result[y:y+32, x:x+32] = blend
        return name, result, truth

    different = photo[source_y+dy:source_y+dy+32, source_x+dx:source_x+dx+32]
    same = photo[source_y:source_y+32, source_x:source_x+32]
    yield patch("offset_feather6", different, 6, True)
    yield patch("offset_feather9", different, 9, True)
    yield patch("offset_even_feather", same, 6, False)
    fractional = cv2.warpAffine(focused, np.array([[1, 0, .5], [0, 1, .375]], np.float32), (32, 32),
                                borderMode=cv2.BORDER_REFLECT_101)
    yield patch("offset_fractional_feather", fractional, 6, False)
    height, width = photo.shape[:2]
    downsampled = cv2.resize(photo, (int(.93 * width), int(.93 * height)), interpolation=cv2.INTER_AREA)
    yield "whole_resize", cv2.resize(downsampled, (width, height), interpolation=cv2.INTER_CUBIC), empty


def scan(im, truth, held):
    from app.detectors.image import native_cfa_fine
    from scripts.image_native_cfa_png_transfer import png_body

    if held is None:
        original = fine.readout(im, truth)
        seed = active_fine(original)
        original["existing"] = (original["coarse"]
                                or (native_cfa_fine.refine_extent(im, original["phase"], seed["region"]) if seed else None)
                                or seed)
    else:
        original = held
        require(fine.digest(png_body(im)) == held["png_sha256"] and fine.digest(truth.tobytes()) == held["truth_sha256"],
                "An established original or truth mask differs")
    phase = original["phase"]
    existing = original["existing"]
    proposed = core.propose(im, phase) if phase is not None and existing is None else dict.fromkeys(core.ARMS)

    def bind(item):
        if item is None:
            return None
        a, b, c, d = item["region"]
        return item | {"intersection": int(truth[b:d, a:c].sum())}

    witnesses = {key: original[key] for key in ("phase", "status", "changed", "png_sha256", "truth_sha256")}
    return witnesses | {"existing": bind(existing), "proposed": {arm: bind(value) for arm, value in proposed.items()}}


def measure(split, cache, archive):
    import numpy as np

    from scripts.image_native_cfa_closure_transfer import raw_render

    baseline = retained_inputs(split)
    rows = []
    args = SimpleNamespace(phase=split, cache=cache, archive=archive)
    for photo, index, rgb in fine.native_photos(args):
        import rawpy

        source_id = photo["id"]
        path = archive / f"sample-{source_id}{Path(photo['url']).suffix.lower()}"
        for role, im, mask in fine.cases(rgb, index, split):
            if role in RETAINED:
                rows.append({"id": source_id, "role": role} | scan(im, mask, baseline[source_id, role]))
        for role, im, mask in new_cases(rgb, source_id, index, split):
            rows.append({"id": source_id, "role": role} | scan(im, mask, None))
        adaptive, _ = raw_render(rawpy, path, adaptive=True)
        rows.append({"id": source_id, "role": "adaptive_development"} | scan(adaptive, np.zeros(rgb.shape[:2], bool), None))
        print(f"{split} camera {index + 1}/{fine.ROSTERS[split][1]} measured", flush=True)
    return rows


def points(rows):
    ordinary = defaultdict(lambda: dict.fromkeys(core.ARMS, 0.))
    for row in rows:
        if row["changed"]:
            continue
        group = ordinary[row["id"]]
        for arm, item in row["proposed"].items():
            group[arm] = max(group[arm], item["response"] if item else 0.)
    require(len(ordinary) == fine.ROSTERS["development"][1], "Development source groups differ")
    limits = {}
    for arm in core.ARMS:
        ranked = sorted((row[arm] for row in ordinary.values()), reverse=True)
        limits[arm] = dict(zip(LEVELS, ranked[:2], strict=True))
    return limits


def validate(split, rows):
    baseline = retained_inputs(split)
    expected = [((source["id"], role)) for source in fine.source_plan()[split]
                for role in (*(k for k in (*fine.CONTROL, *fine.POSITIVE) if k in RETAINED), *NEW)]
    require([(row["id"], row["role"]) for row in rows] == expected, "A frozen source or delivery is missing")
    for row in rows:
        if old := baseline.get((row["id"], row["role"])):
            require(all(row[key] == old[key] for key in ("phase", "status", "changed", "png_sha256", "truth_sha256"))
                    and bool(row["existing"]) == bool(old["existing"])
                    and (row["existing"] is None or row["existing"]["region"] == old["existing"]["region"]),
                    "An original source commitment or current selection differs")
        else:
            require(NEW[row["role"]] == (row["changed"] > 0), "New source labels differ")
        require(set(row["proposed"]) == set(core.ARMS), "A comparison is missing")
        for arm, reading in row["proposed"].items():
            if reading is None:
                continue
            require(row["existing"] is None and row["status"] == "assessed"
                    and reading["alternate_phase"] in range(4) and reading["alternate_phase"] != row["phase"],
                    "An unsupported phase or existing selection entered the new check")
            a, b, c, d = reading["region"]
            require(c-a == d-b == core.BOX and 8 <= a < c <= 2040 and 8 <= b < d <= 1528 and a % core.STEP == b % core.STEP == 0
                    and 0 <= reading["intersection"] <= min(core.BOX**2, row["changed"]), "A crop is outside its source")
            internal, external = reading["core_cells"], reading["outer_cells"]
            require(len(internal) == 4 and len(external) == 8
                    and all(pair is not None for pair in internal)
                    and all(pair is None or (len(pair) == 2 and all(math.isfinite(x) and 0 <= x <= 1 for x in pair))
                            for pair in internal + external), "Native cell readings differ")
            matches, strength, sides, edges = core.cell_support(internal, external)
            valid = matches >= 3 if arm == "core" else matches >= 2 and sides >= 6 and edges == 4
            score = strength if arm == "core" else strength * sides / 8
            require(valid and score > 0 and reading["response"] == round(score, 8), "The proposed response does not reproduce")


def useful(item, changed):
    if not item or not changed:
        return False
    a, b, c, d = item["region"]
    intersection = item["intersection"]
    return 2 * intersection >= changed and 10 * intersection >= 3 * (changed + (c-a)*(d-b) - intersection)


def summarize(rows, cuts):
    counts = defaultdict(Counter)
    for row in rows:
        before = row["existing"]
        base_useful = useful(before, row["changed"])
        base_flag = before is not None
        counts[row["role"]]["files"] += 1
        counts[row["role"]]["assessed"] += row["status"] == "assessed"
        counts[row["role"]]["existing_flags"] += base_flag
        counts[row["role"]]["existing_useful"] += base_useful
        for arm, by_level in cuts.items():
            reading = row["proposed"][arm]
            for label, cutoff in by_level.items():
                now = reading if reading and reading["response"] > cutoff else before
                good = useful(now, row["changed"])
                prefix = f"{arm}/{label}"
                counts[row["role"]][f"{prefix}/added_flags"] += now is not None and not base_flag
                counts[row["role"]][f"{prefix}/added_useful"] += good and not base_useful
    return {key: dict(count) for key, count in counts.items()}


def review(record):
    require(record["schema"] == SCHEMA and record["design"] == DESIGN and set(record["readings"]) == set(SPLITS)
            and record["sources"] == {path.name: fine.file_hash(path) for split in SPLITS
                                      for path in (prior_path(split), extent_path(split))}, "The frozen source study differs")
    for split, rows in record["readings"].items():
        validate(split, rows)
    cuts = points(record["readings"]["development"])
    results = {split: summarize(rows, cuts) for split, rows in record["readings"].items()}
    require(cuts == record["points"] and ("results" not in record or results == record["results"]), "The native crop outcome differs")
    return results


def save(path, data):
    raw = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0))


def run(output, cache, archive):
    readings = {}
    with tempfile.TemporaryDirectory(prefix="chipmunk-core-roles-") as directory:
        for split in SPLITS:
            source = Path(directory) / f"{split}.json.gz"
            subprocess.run([sys.executable, "-B", "-m", "scripts.image_native_cfa_core_transfer", "capture",
                            split, str(source), str(cache), str(archive)], check=True)
            with gzip.open(source, "rt", encoding="utf-8") as stream:
                readings[split] = json.load(stream)
            if split == "development":
                cuts = points(readings[split])
                output.with_suffix(".points.json").write_text(json.dumps(cuts, indent=2) + "\n")
    receipt = {"schema": SCHEMA, "design": DESIGN, "points": cuts, "readings": readings,
               "sources": {path.name: fine.file_hash(path) for split in SPLITS for path in (prior_path(split), extent_path(split))}}
    receipt["results"] = review(receipt)
    save(output, receipt)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    phases = cli.add_subparsers(dest="phase", required=True)
    native = phases.add_parser("run")
    for argument in ("output", "cache", "archive"):
        native.add_argument(argument, type=Path)
    capture = phases.add_parser("capture")
    capture.add_argument("split", choices=SPLITS)
    for argument in ("output", "cache", "archive"):
        capture.add_argument(argument, type=Path)
    portable = phases.add_parser("review")
    portable.add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.phase == "run":
        run(args.output, args.cache, args.archive)
    elif args.phase == "capture":
        save(args.output, measure(args.split, args.cache, args.archive))
    else:
        with gzip.open(args.receipt, "rt", encoding="utf-8") as stream:
            print(json.dumps(review(json.load(stream)), indent=2, sort_keys=True))
