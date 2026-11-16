"""Compare small geometric copy proposals on source-disjoint JPEGs."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.image-small-affine-jpeg/1"
EDIT_SPECS = {
    "sharp44": (44, 0, (-16., 8., 20., -27.), (.80, 1.20, 1., 1.13)),
    "sharp64": (64, 0, (21., -22., 12., 0.), (1.12, .90, 1.10, .82)),
    "feather80": (80, 7, (-31., 17., 0., -13.), (1.05, 1.18, 1.22, .76)),
}
CONTROLS = ("original", "local_blur", "local_tone", "local_resize", "global_resave")
QUALITIES = (75, 90)
METHODS = ("current", "with_luma_search", "with_color_and_detail")
MIN_DETAIL = .93
MAX_COLOR_ERROR = 8.5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if data.startswith(b"\x1f\x8b") else data)


def write(path, record):
    data = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def jpeg(pixels, quality):
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, "JPEG", quality=quality, subsampling=2)
    return stream.getvalue()


def views(source, slot):
    import cv2
    import numpy as np

    height, width = source.shape[:2]
    grid_y, grid_x = np.ogrid[:height, :width]
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    target_x, target_y = 3 * width // 4 + slot % 11, height // 3 + slot % 31

    def texture(position, half):
        x, y = position
        patch = gray[y-half:y+half, x-half:x+half]
        return float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean())

    for name, (size, feather, angles, scales) in EDIT_SPECS.items():
        half = size // 2
        donors = [(x, y) for x in range(half + 16, width // 2 - 16, 61)
                  for y in range(half + 16, height - half - 16, 71)]

        donors.sort(key=lambda position: texture(position, half))
        donor_x, donor_y = donors[int(len(donors) * (.55, .72, .87)[slot % 3])]
        family = slot % 4
        matrix = cv2.getRotationMatrix2D((donor_x, donor_y), angles[family], scales[family])
        matrix[:, 2] += [target_x - donor_x, target_y - donor_y]
        rim = size / 2 - np.maximum(abs(grid_x-donor_x), abs(grid_y-donor_y))
        alpha = ((rim > 0).astype(np.float32) if not feather else np.clip(rim / feather, 0, 1).astype(np.float32))
        weight = cv2.warpAffine(alpha, matrix, (width, height))[..., None]
        moved = cv2.warpAffine(source, matrix, (width, height), flags=cv2.INTER_CUBIC)
        edited = np.rint(source * (1-weight) + moved * weight).astype(np.uint8)
        receiver = np.any(edited != source, axis=2)
        donor = (alpha > 0) & cv2.warpAffine(
            receiver.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        ).astype(bool)
        require(not np.any(donor & receiver) and min(int(donor.sum()), int(receiver.sum())) > 500,
                "Invalid edit ground truth")
        yield name, edited, (donor, receiver)

    yield "original", source, None
    area = slice(target_y-35, target_y+35), slice(target_x-35, target_x+35)
    patch = source[area]
    blurred = cv2.GaussianBlur(patch, (0, 0), 1.4)
    toned = np.clip(np.rint(patch * .87 + 23), 0, 255).astype(np.uint8)
    resized = cv2.resize(cv2.resize(patch, (53, 53), interpolation=cv2.INTER_AREA), (70, 70),
                         interpolation=cv2.INTER_CUBIC)
    for name, local in zip(CONTROLS[1:4], (blurred, toned, resized), strict=True):
        ordinary = source.copy()
        ordinary[area] = local
        yield name, ordinary, None
    from PIL import Image

    with Image.open(io.BytesIO(jpeg(source, 82))) as prior:
        yield "global_resave", np.array(prior), None


def prepare(archive, wheel, roster_path, repetitions, output):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    from PIL import __version__ as pillow_version

    from scripts import image_jpeg_native_rotation_transfer as native
    from scripts import image_rawnind_phase_v239 as decoder_package

    cv2.setNumThreads(1)
    roster = read(roster_path)
    require(len(roster) == len({item["sha256"] for item in roster}) == len({item["id"] for item in roster}) == 18,
            "The confirmation roster needs eighteen distinct originals")
    output.mkdir(mode=0o700)
    rows = []

    def stage(group, role, raster, truth):
        stem = f"{group}-{role}"
        if truth is not None:
            np.savez_compressed(output / f"{stem}.npz", donor=truth[0], receiver=truth[1])
        for quality in QUALITIES:
            data = jpeg(raster, quality)
            name = f"{stem}-{quality}.jpg"
            (output / name).write_bytes(data)
            rows.append({"group": group, "role": role, "quality": quality, "file": name, "sha256": sha(data),
                         "size": list(raster.shape[1::-1]), "truth": [sha(mask.tobytes()) for mask in truth] if truth else None,
                         "truth_boxes": [list(cv2.boundingRect(mask.astype(np.uint8))) for mask in truth] if truth else None})

    with tempfile.TemporaryDirectory(prefix="chipmunk-small-jpeg-raw-") as directory:
        decoder_package.unpack_decoder(decoder_package.verified_wheel(wheel), directory)
        sys.path.insert(0, directory)
        try:
            import rawpy

            require(Path(rawpy.__file__).resolve().is_relative_to(Path(directory).resolve()), "The pinned decoder is required")
            for slot, item in enumerate(roster):
                path = archive / item["file"]
                require(item["license"] == "CC0-1.0" and native.sha(path) == item["sha256"], "Unqualified photo")
                with rawpy.imread(str(path)) as raw:
                    pixels = raw.postprocess(use_camera_wb=False, use_auto_wb=True, output_bps=8, user_flip=0)
                require(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[-1] == 3,
                        "The RAW file did not develop as RGB")
                original = Image.fromarray(pixels)
                if original.height > original.width:
                    original = original.transpose(Image.Transpose.ROTATE_90)
                if slot % 3 == 1:
                    original = original.transpose(Image.Transpose.ROTATE_90)
                edge = (768, 1024)[slot % 2]
                size = tuple(round(edge * side / max(original.size)) for side in original.size)
                require(min(size) >= 384, "Camera image outside the supported range")
                photo = np.asarray(original.resize(size, Image.Resampling.LANCZOS))
                for role, raster, truth in views(photo, slot):
                    stage(item["id"], role, raster, truth)
                print(f"Prepared camera group {slot+1}/{len(roster)}", file=sys.stderr, flush=True)
        finally:
            sys.path.pop(0)
    repeated = []
    repeat_freeze = read(repetitions / "freeze.json")
    credit = {item["group"]: item for item in repeat_freeze["repeated_structures"]}
    for row in repeat_freeze["rows"]:
        if row["role"] != "repetition":
            continue
        data = (repetitions / row["file"]).read_bytes()
        require(row["sha256"] == sha(data) and row["group"] in credit, "The repetition control differs")
        with Image.open(io.BytesIO(data)) as encoded:
            photograph = ImageOps.exif_transpose(encoded).convert("RGB")
        photograph.thumbnail((960, 960), Image.Resampling.LANCZOS)
        require(min(photograph.size) >= 384, "Repeated structure outside the supported range")
        stage(row["group"], "repetition", np.asarray(photograph), None)
        repeated.append(credit[row["group"]] | {"input_sha256": row["sha256"]})
    require(len(repeated) == len(credit) == 9, "Missing repetition controls")
    contract = {"detail_minimum": MIN_DETAIL, "color_error_maximum": MAX_COLOR_ERROR,
                "minimum_iou": .30, "minimum_recall": .50,
                "runtime": {"opencv": cv2.__version__, "pillow": pillow_version, "numpy": np.__version__}}
    write(output / "freeze.json", {"schema": SCHEMA, "contract": contract, "models": [item["model"] for item in roster],
                                   "roster_sha256": native.sha(roster_path), "wheel_sha256": native.sha(wheel),
                                   "repetition_credits": repeated, "rows": rows})


def mask_record(masks, truth):
    import numpy as np

    counts = [int(mask.sum()) for mask in masks]
    cross = [[int(np.count_nonzero(mask & known)) for known in truth] for mask in masks] if truth else None
    return {"pixels": counts, "sha256": [sha(mask.tobytes()) for mask in masks], "intersections": cross}


def observe(root, row):
    import numpy as np
    from PIL import Image

    data = (root / row["file"]).read_bytes()
    require(sha(data) == row["sha256"], "JPEG bytes differ from the freeze")
    with Image.open(io.BytesIO(data), formats=("JPEG",)) as encoded:
        require(encoded.mode == "RGB" and list(encoded.size) == row["size"], "Wrong JPEG raster")
        pixels = np.asarray(encoded)
    truth = None
    if row["truth"] is not None:
        with np.load(root / (row["file"].rsplit("-", 1)[0] + ".npz")) as labels:
            truth = tuple(labels[key] for key in ("donor", "receiver"))
        require([sha(mask.tobytes()) for mask in truth] == row["truth"], "Truth pixels differ")
    return {"sha256": row["sha256"], **measure(pixels, truth)}


def measure(pixels, truth):
    import cv2
    import numpy as np

    from app.detectors.image import copy_affine, copy_affine_small, copy_move

    cv2.setNumThreads(1)
    gray = copy_affine.luma(pixels)
    current, proposed = [], []
    current_assessable = fallback_assessable = True
    try:
        for item in copy_affine.candidates(pixels):
            x, y, w, h = item.box_a
            rect = math.floor(x), math.floor(y), min(pixels.shape[1], math.ceil(x+w)), min(pixels.shape[0], math.ceil(y+h))
            if (masks := copy_affine.dense_pair(gray, np.asarray(item.affine.matrix), rect)) is not None:
                current.append(mask_record(masks, truth))
    except copy_move.CandidateBudgetExceeded:
        current_assessable = fallback_assessable = False
    if current_assessable and not current:
        try:
            for matrix, bounds, masks in copy_affine_small.matches_in_image(pixels):
                detail = copy_affine_small.alignment_detail(pixels, matrix, masks[0])
                proposed.append(mask_record(masks, truth) | {"detail": list(detail) if detail else None,
                                                            "matrix": matrix.tolist(), "seed": bounds})
                if detail and detail[0] <= MAX_COLOR_ERROR and detail[1] >= MIN_DETAIL:
                    break
        except copy_move.CandidateBudgetExceeded:
            fallback_assessable = False
    return {"truth_pixels": [int(mask.sum()) for mask in truth] if truth else None,
            "current_assessable": current_assessable, "fallback_assessable": fallback_assessable,
            "current": current, "proposed": proposed}


def useful(pair, truth):
    if not truth:
        return False
    for indices in ((0, 1), (1, 0)):
        intersections = [pair["intersections"][i][j] for i, j in enumerate(indices)]
        if all(2 * hit >= truth[j] and 10 * hit >= 3 * (truth[j] + pair["pixels"][i] - hit)
               for i, (j, hit) in enumerate(zip(indices, intersections, strict=True))):
            return True
    return False


def review(record):
    plan = record["freeze"]
    require(record["schema"] == plan["schema"] == SCHEMA and len(plan["rows"]) == len(record["observed"]) == 306,
            "Wrong comparison roster")
    require(plan["contract"]["detail_minimum"] == MIN_DETAIL and plan["contract"]["color_error_maximum"] == MAX_COLOR_ERROR,
            "The comparison points changed")
    return tally(plan["rows"], record["observed"])


def tally(rows, observations):
    distinct = len({row["sha256"] for row in rows})
    stats, groups = Counter(), {}
    for row, result in zip(rows, observations, strict=True):
        require(row["sha256"] == result["sha256"], "Comparison rows do not match the freeze")
        current, proposed = result["current"], result["proposed"]
        selected = next((item for item in proposed if item["detail"] is not None and item["detail"][0] <= MAX_COLOR_ERROR
                         and item["detail"][1] >= MIN_DETAIL), None)
        for arm, pairs in zip(METHODS, (current, current or proposed[:1], current or ([selected] if selected else [])), strict=True):
            truth = result["truth_pixels"]
            for pair in pairs:
                if truth:
                    require(all(0 <= val <= min(area, truth[j])
                                for area, values in zip(pair["pixels"], pair["intersections"], strict=True)
                                for j, val in enumerate(values)), "Incorrect overlap")
            hit = any(useful(pair, truth) for pair in pairs)
            key = f"{row['quality']}/{arm}/{row['role']}"
            stats[key + "/files"] += 1
            stats[key + "/useful"] += hit
            stats[key + "/flagged"] += bool(pairs)
            stats[key + "/misplaced"] += bool(truth and pairs and not hit)
            assessed = result["current_assessable"] if arm == "current" else result["fallback_assessable"]
            stats[key + "/unassessed"] += not assessed and not pairs
            if pairs and not truth:
                groups.setdefault(f"{row['quality']}/{arm}", set()).add(row["group"])
    return {"counts": dict(sorted(stats.items())), "distinct_deliveries": distinct,
            "ordinary_groups": {k: sorted(v) for k, v in sorted(groups.items())}}


def capture(prepared, receipt, workers):
    from app.detectors.image import copy_affine, copy_affine_small, copy_move

    require(copy_affine_small.MAX_COLOR_ERROR == MAX_COLOR_ERROR and copy_affine_small.MIN_DETAIL_CORRELATION == MIN_DETAIL,
            "Serving and experiment comparison points disagree")
    plan = read(prepared / "freeze.json")
    rows = plan["rows"]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        measured = []
        for index, item in enumerate(pool.map(observe, [prepared] * len(rows), rows, chunksize=2), 1):
            measured.append(item)
            if index % 24 == 0:
                print(f"Measured {index}/{len(rows)} JPEGs", file=sys.stderr, flush=True)
    bindings = (Path(__file__), *(Path(module.__file__) for module in (copy_affine, copy_affine_small, copy_move)))
    output = {"schema": SCHEMA, "freeze": plan, "observed": measured,
              "implementation_sha256": {path.name: sha(path.read_bytes()) for path in bindings}}
    output["summary"] = review(output)
    write(receipt, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command, names in (("prepare", ("archive", "wheel", "roster", "repetitions", "output")),
                           ("capture", ("prepared", "receipt")), ("review", ("receipt",))):
        sub = subcommands.add_parser(command)
        for name in names:
            sub.add_argument(name, type=Path)
        if command == "capture":
            sub.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.archive, args.wheel, args.roster, args.repetitions, args.output)
    elif args.command == "capture":
        capture(args.prepared, args.receipt, args.workers)
    else:
        result = read(args.receipt)
        require(result["summary"] == review(result), "Portable audit differs from published comparison")
        print(json.dumps(result["summary"], sort_keys=True))
