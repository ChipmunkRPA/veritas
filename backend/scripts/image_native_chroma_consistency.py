# SPDX-License-Identifier: Apache-2.0
"""Compare native JPEG color and brightness around small standalone edits."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.bench.image_jpeg_boundary import POSITIVE, QUALITIES, incumbent_flag, incumbent_hit, useful
from scripts import image_local_spectral_peaks as common
from scripts import image_region_rank_inputs as files

SCHEMA = "chipmunk.native-jpeg-chroma-consistency/1"
ARMS = ("missing_color_detail", "extra_color_detail", "weak_color_brightness_agreement")
FREQUENCIES = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2))
SHAPES = ((2, 3), (3, 2), (3, 3), (3, 5), (5, 3), (4, 8), (8, 4), (5, 5), (7, 7), (6, 10), (10, 6))
ORDINARY_GROUPS = 1
CELL = 16


def features(data):
    import cv2
    import jpeglib
    import numpy as np
    from PIL import Image
    from scipy.fft import dctn, idctn

    from app.detectors.image import jpeg_bitstream

    structure = jpeg_bitstream.read(data)
    if (structure is None or structure.frame != "baseline" or structure.precision != 8
            or [(part.id, part.h, part.v) for part in structure.components] != [(1, 2, 2), (2, 1, 1), (3, 1, 1)]
            or any(structure.table_bits.get(part.table_id) != 8 for part in structure.components)
            or not 256 <= min(structure.width, structure.height) <= max(structure.width, structure.height) <= 768):
        return None
    with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
        if image.getexif().get(274, 1) != 1:
            return None
    height, width = structure.height // CELL, structure.width // CELL
    with NamedTemporaryFile(suffix=".jpg") as encoded:
        encoded.write(data)
        encoded.flush()
        jpeg = jpeglib.read_dct(encoded.name)
        tables = np.asarray(jpeg.qt, dtype=np.float32)[np.asarray(jpeg.quant_tbl_no, dtype=np.intp)]
        y = np.asarray(jpeg.Y[:2*height, :2*width], dtype=np.float32) * tables[0]
        planes = [np.asarray(array[:height, :width], dtype=np.float32) * table
                  for array, table in zip((jpeg.Cb, jpeg.Cr), tables[1:], strict=True)]
    spatial = idctn(y, axes=(-2, -1), norm="ortho").transpose(0, 2, 1, 3).reshape(height*CELL, width*CELL)
    pooled = spatial.reshape(height*8, 2, width*8, 2).mean(axis=(1, 3))
    aligned_y = dctn(pooled.reshape(height, 8, width, 8).transpose(0, 2, 1, 3), axes=(-2, -1), norm="ortho")
    rows, cols = zip(*FREQUENCIES, strict=True)
    luma, cb, cr = (plane[..., rows, cols] for plane in (aligned_y, *planes))

    def local(values):
        return cv2.blur(values.mean(axis=2), (3, 3), borderType=cv2.BORDER_REFLECT_101)

    luma_power, cb_power, cr_power = (local(array ** 2) for array in (luma, cb, cr))
    brightness = local(aligned_y[..., :1, :1].reshape(height, width, 1)) / (8 * 28)
    saturation = local(np.stack([plane[..., 0, 0]**2 for plane in planes], axis=2)) ** .5 / (8 * 22)
    activity = np.log1p(luma_power ** .5 / 5.)
    color = np.log1p((cb_power + cr_power) ** .5 / 4.)
    coherence = np.maximum(np.abs(local(luma*cb)) / np.sqrt((luma_power + 25.)*(cb_power + 16.)),
                           np.abs(local(luma*cr)) / np.sqrt((luma_power + 25.)*(cr_power + 16.)))
    return np.stack((brightness, activity, saturation), axis=-1), np.stack((color, coherence), axis=-1)


def fields(data):
    import numpy as np
    from scipy.spatial import cKDTree

    reading = features(data)
    if reading is None:
        return None
    geometry, response = reading
    height, width, _ = geometry.shape
    context = geometry.reshape(-1, 3)
    measurements = response.reshape(-1, 2)
    points = np.indices((height, width)).reshape(2, -1).T
    distances, neighbors = cKDTree(context).query(context, k=min(144, len(context)))
    result = np.full((height*width, 3), np.nan, dtype=np.float64)
    for index, ranks in enumerate(neighbors):
        valid = (np.abs(points[ranks] - points[index]).max(axis=1) >= 4) & (distances[index] <= 1.2)
        take = ranks[valid][:24]
        if len(take) < 12:
            continue
        samples = measurements[take]
        centre = np.median(samples, axis=0)
        deviation = np.maximum([.20, .065], 1.4826*np.median(np.abs(samples - centre), axis=0))
        color, alignment = (measurements[index] - centre) / deviation
        result[index] = -color, color, -alignment
    return result.reshape(height, width, 3)


def permitted_shapes(height, width):
    return ((h, w) for h, w in SHAPES if h <= height and w <= width and h*w <= .16*height*width)


def proposals(data):
    import numpy as np

    reading = fields(data)
    if reading is None:
        return dict.fromkeys(ARMS)
    height, width, _ = reading.shape
    best = dict.fromkeys(ARMS)
    for h, w in permitted_shapes(height, width):
        tiles = np.lib.stride_tricks.sliding_window_view(reading, (h, w), axis=(0, 1))
        valid = np.isfinite(tiles).sum(axis=(-2, -1)) >= .85*h*w
        windows = np.where(np.isfinite(tiles), tiles, -10.)
        scores = np.where(valid, np.quantile(windows, .40, axis=(-2, -1)), -np.inf)
        for arm_index, arm in enumerate(ARMS):
            y, x = np.unravel_index(np.argmax(scores[..., arm_index]), scores.shape[:2])
            score = scores[y, x, arm_index]
            if not np.isfinite(score) or best[arm] and score <= best[arm]["score"]:
                continue
            best[arm] = {"box": [int(x)*CELL, int(y)*CELL, w*CELL, h*CELL], "score": float(score)}
    return best


def measure_truth(choices, truth):
    import cv2
    import numpy as np

    height, width = (n // CELL for n in truth.shape)
    total = int(truth.sum())
    for proposed in choices.values():
        if proposed is not None:
            x, y, w, h = proposed["box"]
            proposed.update(intersection=int(truth[y:y+h, x:x+w].sum()), selected=w*h, truth=total)
    if not total:
        return False
    masses = cv2.integral(truth.astype(np.uint8))[::CELL, ::CELL]
    for h, w in permitted_shapes(height, width):
        overlaps = masses[h:, w:] - masses[h:, :-w] - masses[:-h, w:] + masses[:-h, :-w]
        if np.any((overlaps*2 >= total) & (10*overlaps >= 3*(h*w*CELL**2+total-overlaps))):
            return True
    return False


def measure(bank, baseline):
    import cv2

    from app.detectors.image import jpeg_dq, regional
    from app.detectors.image.adapters.regional import repeat_components
    from app.detectors.image.regional_support import readout
    from scripts import image_region_rank_reference
    from scripts.image_cross_quality_fit import load_mask

    cv2.setNumThreads(1)
    manifest, groups, _ = common.plan(bank / "manifest.json")
    with baseline.open() as stream:
        recorded = json.loads(next(stream))
        responses = {row["file"]: row for row in map(json.loads, stream)}
    expected = {"manifest_sha256": files.sha(bank / "manifest.json"),
                "code_sha256": files.sha(Path(image_region_rank_reference.__file__)),
                "regional_sha256": regional.CHECKPOINT_SHA256, "jpeg_sha256": files.sha(Path(jpeg_dq.__file__)),
                "regional_rule_sha256": files.sha(Path(readout.__code__.co_filename)), "repeat_jpeg90": True,
                "repeat_rule_sha256": files.sha(Path(repeat_components.__code__.co_filename))}
    if recorded != expected or set(responses) != {row["file"] for row in manifest["images"]}:
        raise ValueError("Existing checks did not run on every exact JPEG under their current code")
    rows = []
    for index, entry in enumerate(manifest["images"], 1):
        data = (bank / entry["file"]).read_bytes()
        truth = load_mask(bank, entry)
        reference = responses[entry["file"]]
        if (hashlib.sha256(data).hexdigest() != entry["sha256"] or reference["sha256"] != entry["sha256"]
                or list(truth.shape) != groups[entry["group"]]["shape"]):
            raise ValueError("An evaluated JPEG or original mask commitment changed")
        chosen = proposals(data)
        possible = measure_truth(chosen, truth)
        rows.append({**entry, "shape": list(truth.shape), "source": groups[entry["group"]]["source"],
                     "current": {field: reference[field] for field in ("jpeg", "regional")},
                     "arms": chosen, "geometry_can_fit": possible})
        if index % 28 == 0 or index == len(manifest["images"]):
            print(f"Measured chroma consistency on {index}/{len(manifest['images'])} files.", flush=True)
    return rows


def summary(rows, arm, point):
    counts = Counter()
    added_edits, extra = set(), set()
    for row in rows:
        positive = row["role"] in POSITIVE
        proposal = row["arms"][arm]
        located = positive and incumbent_hit(row)
        flag = not positive and incumbent_flag(row)
        selected = proposal is not None and proposal["score"] > point
        best_hits = positive and proposal is not None and useful(proposal)
        counts.update(edits=positive, controls=not positive, existing_locations=located, existing_control_flags=flag,
                      oracle_geometry=positive and row["geometry_can_fit"], assessed=proposal is not None,
                      useful_before_cutoff=best_hits, added_locations=not located and selected and best_hits,
                      misplaced=positive and selected and not best_hits,
                      added_flags=not positive and selected and not flag,
                      added_unchanged_flags=row["role"] == "ordinary" and selected and not flag)
        if not positive and selected and not flag:
            extra.add(row["source"])
        if not located and selected and best_hits:
            added_edits.add(row["source"])
    return dict(counts) | {"added_edit_groups": len(added_edits), "added_control_groups": len(extra)}


def operating_points(rows):
    result = {}
    for quality in QUALITIES:
        partition = [row for row in rows if row["quality"] == quality]
        arms, choices = {}, []
        for arm in ARMS:
            maximum = {}
            for row in partition:
                value = row["arms"][arm]
                if row["role"] not in POSITIVE and value is not None and not incumbent_flag(row):
                    source = row["source"]
                    maximum[source] = max(value["score"], maximum.get(source, float("-inf")))
            if len(maximum) <= ORDINARY_GROUPS:
                raise ValueError("Too few measured ordinary calibration sources")
            point = sorted(maximum.values(), reverse=True)[ORDINARY_GROUPS]
            metrics = summary(partition, arm, point)
            arms[arm] = {"point": point, "counts": metrics}
            choices.append((metrics["added_locations"], -metrics["added_flags"], -metrics["misplaced"], arm))
        result[str(quality)] = {"arms": arms, "chosen": max(choices)[-1]}
    return result


def recount(rows, points):
    return {str(quality): {arm: summary([r for r in rows if r["quality"] == quality], arm, frozen["point"])
                           for arm, frozen in points[str(quality)]["arms"].items()}
            for quality in QUALITIES}


def freeze(args):
    _, _, calibration = common.plan(args.calibration / "manifest.json")
    held = [common.plan(path / "manifest.json")[2] for path in (args.fresh, *args.transfer)]
    for bank in held:
        if any(set(values) & set(bank["identities"][key]) for key, values in calibration["identities"].items()):
            raise ValueError("A held source overlaps calibration")
    rows = measure(args.calibration, args.baseline)
    files.save(args.output, {"schema": SCHEMA, "calibration": calibration, "held": held,
                             "code_sha256": files.sha(Path(__file__)), "rows": rows, "points": operating_points(rows),
                             "baseline_sha256": files.sha(args.baseline)})


def held(args):
    lock = files.read(args.freeze)
    _, _, identity = common.plan(args.bank / "manifest.json")
    if (lock["schema"] != SCHEMA or lock["code_sha256"] != files.sha(Path(__file__)) or identity not in lock["held"]
            or lock["points"] != operating_points(lock["rows"])):
        raise ValueError("Source, implementation or operating points changed after calibration")
    rows = measure(args.bank, args.baseline)
    report = recount(rows, lock["points"])
    files.save(args.output, {"schema": SCHEMA, "freeze_sha256": files.sha(args.freeze), "rows": rows, "summary": report,
                             "baseline_sha256": files.sha(args.baseline)})
    print(report, flush=True)


def review(args):
    lock, report = map(files.read, (args.freeze, args.result))
    if (lock["schema"] != SCHEMA or report["schema"] != SCHEMA or report["freeze_sha256"] != files.sha(args.freeze)
            or lock["points"] != operating_points(lock["rows"]) or report["summary"] != recount(report["rows"], lock["points"])):
        raise ValueError("The reported cutoffs or counts disagree with the retained ledger")
    print({key: {arm: report["summary"][key][arm] for arm in (point["chosen"],)} for key, point in lock["points"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, arguments in (("freeze", ("calibration", "baseline", "fresh", "output")),
                            ("held", ("freeze", "bank", "baseline", "output")), ("review", ("freeze", "result"))):
        command = subcommands.add_parser(name)
        for field in arguments:
            command.add_argument(field, type=Path)
        if name == "freeze":
            command.add_argument("--transfer", type=Path, action="append", default=[])
    args = parser.parse_args()
    {"freeze": freeze, "held": held, "review": review}[args.command](args)
