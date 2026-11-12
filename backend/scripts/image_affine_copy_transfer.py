"""Test dense pixel support for rotated and resized same-image copy pairs."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.detectors.image import copy_move
from scripts import image_background_patch_score as paired
from scripts import image_bg_real_insertion_score as metrics

SCHEMA = "chipmunk.image-affine-copy-transfer/1"
FORMATS = ("png", "jpeg75")
POSITIVE = ("sharp_affine", "feathered_affine")
CONTROLS = ("original", "contrast", "blur", "whole_resize", "local_tone")
MIN_CORRELATION = .94
MIN_DETAIL = 5.0
MIN_AREA = 300
MAX_PAIRS = 3
MAX_CANDIDATES = 1500
STUDY = Path(__file__)
sha, digest, write = paired.shared.file_digest, paired.shared.digest, paired.shared.write


def limits(box, width, height, margin=0):
    x, y, w, h = box
    return (max(0, math.floor(x) - margin), max(0, math.floor(y) - margin),
            min(width, math.ceil(x + w) + margin), min(height, math.ceil(y + h) + margin))


def rectangle(shape, box):
    result = np.zeros(shape, bool)
    left, top, right, bottom = limits(box, shape[1], shape[0])
    result[top:bottom, left:right] = True
    return result


def smooth(value):
    return cv2.boxFilter(value, cv2.CV_32F, (11, 11), normalize=True)


def support(a, b, valid):
    """Require similar textured pixel neighborhoods after alignment."""
    ma, mb = smooth(a), smooth(b)
    va, vb = np.maximum(0, smooth(a * a) - ma * ma), np.maximum(0, smooth(b * b) - mb * mb)
    covariance = smooth(a * b) - ma * mb
    detail = np.minimum(va, vb) >= MIN_DETAIL**2
    agreement = covariance >= MIN_CORRELATION * np.sqrt(va * vb)
    difference = smooth(np.abs(a - b)) <= 3 + .25 * np.sqrt(np.maximum(va, vb))
    selected = (valid & detail & agreement & difference).astype(np.uint8)
    return cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool) & valid


def dense_pair(gray, pair):
    """Refine an existing affine proposal without hypothesizing a copy direction."""
    height, width = gray.shape
    matrix = np.array(pair.affine.matrix, np.float64)
    x0, y0, x1, y1 = limits(pair.box_a, width, height, 36)
    size = (x1 - x0, y1 - y0)
    if min(size) < 16:
        return None
    local = matrix.copy()
    local[:, 2] += local[:, :2] @ np.array([x0, y0])
    aligned = cv2.warpAffine(gray, local, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    yy, xx = np.indices(aligned.shape)
    bx = local[0, 0] * xx + local[0, 1] * yy + local[0, 2]
    by = local[1, 0] * xx + local[1, 1] * yy + local[1, 2]
    a = gray[y0:y1, x0:x1]
    valid = (bx >= 6) & (bx < width - 6) & (by >= 6) & (by < height - 6)
    valid &= (bx - (xx + x0)) ** 2 + (by - (yy + y0)) ** 2 >= 48**2
    selected = support(a, aligned, valid)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), connectivity=8)
    left, top, right, bottom = limits(pair.box_a, width, height)
    seeds = Counter(labels[top-y0:bottom-y0, left-x0:right-x0].ravel())
    candidates = [label for label, seed in seeds.items()
                  if label and seed >= 100 and stats[label, cv2.CC_STAT_AREA] >= MIN_AREA]
    if not candidates:
        return None
    label = max(candidates, key=lambda value: (seeds[value], stats[value, cv2.CC_STAT_AREA], -value))
    first = np.zeros(gray.shape, np.uint8)
    first[y0:y1, x0:x1] = labels == label
    second = cv2.warpAffine(first, matrix, (width, height), flags=cv2.INTER_NEAREST)
    if int(second.sum()) < MIN_AREA or np.any(first & second):
        return None
    return first.astype(bool), second.astype(bool)


def fixtures(source, slot):
    """Generate exact source-pixel truth for fixed affine and ordinary deliveries."""
    height, width = source.shape[:2]
    side = min(height, width) // 5
    if side < 80:
        raise ValueError("Source is too small for the frozen patch geometry")
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    donors = [(x, y) for x in (width // 5, width // 3) for y in (height // 4, height // 2, 3 * height // 4)]
    radius = side // 2
    corners = cv2.cornerMinEigenVal(gray, 5)
    dx, dy = max(donors, key=lambda point: float(corners[point[1]-radius:point[1]+radius,
                                                        point[0]-radius:point[0]+radius].sum()))
    tx, ty = 3 * width // 4, height - dy
    angle, scale = (14., .83) if slot % 2 == 0 else (-17., 1.17)
    matrix = cv2.getRotationMatrix2D((dx, dy), angle, scale)
    matrix[:, 2] += [tx - dx, ty - dy]
    yy, xx = np.ogrid[:height, :width]
    radius_field = np.sqrt(((xx - dx) / radius) ** 2 + ((yy - dy) / (.82 * radius)) ** 2)
    donor = radius_field < 1
    patch = cv2.warpAffine(source, matrix, (width, height), flags=cv2.INTER_CUBIC)
    for role, blend in zip(POSITIVE, (donor.astype(np.float32), np.clip((1 - radius_field) / .24, 0, 1).astype(np.float32)), strict=True):
        alpha = cv2.warpAffine(blend, matrix, (width, height), flags=cv2.INTER_LINEAR)
        edited = np.rint(patch * alpha[:, :, None] + source * (1 - alpha[:, :, None])).astype(np.uint8)
        receiver = np.any(edited != source, axis=2)
        source_support = donor & cv2.warpAffine(receiver.astype(np.uint8), matrix, (width, height),
                                                 flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype(bool)
        if np.any(source_support & receiver) or min(int(source_support.sum()), int(receiver.sum())) < MIN_AREA:
            raise ValueError("Affine fixture has invalid source and receiver support")
        yield role, edited, (source_support, receiver)
    image = Image.fromarray(source)
    whole_resize = image.resize((round(width * .83), round(height * .83)), Image.Resampling.BICUBIC)
    whole_resize = whole_resize.resize((width, height), Image.Resampling.BICUBIC)
    tone = np.asarray(ImageEnhance.Contrast(image).enhance(1.15))
    local_tone = source.copy()
    local_tone[donor] = tone[donor]
    for role, control in zip(CONTROLS, (source, tone, np.asarray(image.filter(ImageFilter.GaussianBlur(.9))),
                                       np.asarray(whole_resize), local_tone), strict=True):
        yield role, control, (np.zeros_like(donor), np.zeros_like(donor))


def pair_overlap(masks, truth):
    ordered = [[metrics.measure(a, b) for a, b in zip(masks, order, strict=True)] for order in (truth, truth[::-1])]
    return max(ordered, key=lambda pair: min(row["iou"] or 0 for row in pair))


def useful(overlap):
    return all((row["iou"] or 0) >= .30 and (row["recall"] or 0) >= .50 for row in overlap)


def evaluate(pixels, truth):
    try:
        detected = copy_move.analyse(pixels, max_candidate_pairs=MAX_CANDIDATES)
    except copy_move.CandidateBudgetExceeded:
        return {"assessed": False, "baseline": [], "dense": []}
    gray = cv2.GaussianBlur(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), (0, 0), .85).astype(np.float32)
    baseline, dense = [], []
    for index, pair in enumerate(detected.distinct):
        masks = tuple(rectangle(gray.shape, box) for box in (pair.box_a, pair.box_b))
        baseline.append(pair_overlap(masks, truth))
        if index < MAX_PAIRS and (refined := dense_pair(gray, pair)) is not None:
            dense.append({"proposal": index, "overlap": pair_overlap(refined, truth),
                          "mask_sha256": [digest(mask.tobytes()) for mask in refined]})
    return {"assessed": True, "baseline": baseline, "dense": dense}


def freeze(intakes):
    plans = [json.loads((path / "plan.json").read_bytes()) for path in intakes]
    for field in ("flickr_photo_id", "flickr_owner"):
        sets = [{row[field] for row in plan["cases"]} for plan in plans]
        if [len(values) for values in sets] != [12, 12] or len(set.union(*sets)) != 24:
            raise ValueError("Confirmation panels must have disjoint photos and attributed photographers")
    files = (STUDY, Path(copy_move.__file__), Path(paired.__file__), Path(metrics.__file__))
    return {"schema": SCHEMA, "code_sha256": {path.name: sha(path) for path in files},
            "source_commitments": [{name: sha(folder / name) for name in ("plan.json", "attempts.jsonl", "result.json")}
                                   for folder in intakes], "opencv": cv2.__version__, "numpy": np.__version__,
            "formats": FORMATS, "positive": POSITIVE, "controls": CONTROLS,
            "localization": {"minimum_iou_each_area": .30, "minimum_recall_each_area": .50},
            "primary": "The dense-only pair must add at least three useful feathered localizations in at least three source groups per delivery, lose at most two useful baseline pairs across both positive roles, and add zero control flags."}


def compare(intake):
    for row, native, _ in paired.inputs(intake):
        source = native["original"]
        for role, pixels, truth in fixtures(source, row["slot"]):
            for delivery in FORMATS:
                body, image = paired.encoded(pixels, delivery)
                yield {"slot": row["slot"], "role": role, "delivery": delivery, "encoded_sha256": digest(body),
                       "truth_sha256": [digest(mask.tobytes()) for mask in truth], **evaluate(np.asarray(image), truth)}


def summarize(panel_rows):
    aggregate = {}
    decisions = {}
    for fmt in FORMATS:
        counts = {role: Counter() for role in (*POSITIVE, *CONTROLS)}
        added_groups = set()
        for panel, rows in panel_rows.items():
            for row in rows:
                if row["delivery"] != fmt:
                    continue
                role = row["role"]
                bucket = counts[role]
                bucket["views"] += 1
                bucket["assessed"] += row["assessed"]
                incumbent = any(useful(overlap) for overlap in row["baseline"])
                new = any(useful(pair["overlap"]) for pair in row["dense"])
                bucket["incumbent_flagged"] += bool(row["baseline"])
                bucket["dense_flagged"] += bool(row["dense"])
                bucket["incumbent_useful"] += incumbent
                bucket["dense_useful"] += new
                bucket["added_useful"] += new and not incumbent
                bucket["lost_useful"] += incumbent and not new
                bucket["added_flag"] += bool(row["dense"]) and not row["baseline"]
                if role == "feathered_affine" and new and not incumbent:
                    added_groups.add((panel, row["slot"]))
        aggregate[fmt] = counts
        decisions[fmt] = (len(added_groups) >= 3 and sum(counts[role]["lost_useful"] for role in POSITIVE) <= 2
                          and all(counts[role]["added_flag"] == 0 for role in CONTROLS))
    return aggregate, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("earlier", type=Path)
    parser.add_argument("later", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    folders = {"earlier": args.earlier, "later": args.later}
    expected = freeze(folders.values())
    freeze_path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir()
        write(freeze_path, expected)
        print(sha(freeze_path), flush=True)
        return
    if json.loads(freeze_path.read_bytes()) != json.loads(json.dumps(expected)):
        raise ValueError("Code or intake differs from the frozen comparison")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    started = time.monotonic()
    panel_rows = {}
    journals = {}
    for name, folder in folders.items():
        journal = args.output / f"{name}.jsonl"
        rows = []
        with journal.open("x") as stream:
            for row in compare(folder):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                rows.append(row)
                if len(rows) % 42 == 0:
                    print(f"{name}: {len(rows)} views measured", flush=True)
        if len(rows) != 168:
            raise ValueError("A confirmation panel is incomplete")
        journals[name] = sha(journal)
        panel_rows[name] = rows
    results, advances = summarize(panel_rows)
    write(args.output / "aggregate.json", {"schema": SCHEMA, "freeze_sha256": sha(freeze_path), "journal_sha256": journals,
                                            "views": 336, "elapsed_s": time.monotonic() - started,
                                            "results": results, "advances": advances})
    print(sha(args.output / "aggregate.json"), flush=True)


if __name__ == "__main__":
    main()
