"""Test a bounded tone correction for proposed same-image geometric copy pairs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from PIL import __version__ as pillow_version

from app.detectors.image import copy_affine, copy_move

SCHEMA = "chipmunk.image-copy-tone-transfer/1"
FORMATS = ("png", "jpeg75")
POSITIVE = ("sharp", "feathered", "sharp_tone", "feathered_tone")
CONTROLS = ("published", "global_tone", "local_tone", "resize")
MAX_SIDE = 800


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def retained_rows(archive, first_slot):
    plan = archive / "plan.json"
    journal = archive / "attempts.jsonl"
    result = json.loads((archive / "result.json").read_bytes())
    if (result["plan_sha256"] != digest(plan.read_bytes())
            or result["acquisition_journal_sha256"] != digest(journal.read_bytes())):
        raise ValueError("Retained qualification receipts changed")
    cases = json.loads(plan.read_bytes())["cases"]
    rows = []
    for case, line in zip(cases, journal.read_text().splitlines(), strict=True):
        qualified = json.loads(line)
        if qualified["status"] != "qualified" or case["slot"] != qualified["slot"]:
            raise ValueError("Retained original is outside the qualified roster")
        name = f"{case['slot']:02d}-original.png"
        expected = qualified["files"]["original"]["sha256"]
        if case["original"]["sha256"] != expected or digest((archive / "assets" / name).read_bytes()) != expected:
            raise ValueError("Retained original does not bind the qualified source")
        attribution = case["photo_attribution"]
        rows.append({"slot": first_slot + len(rows), "cohort": "retained", "name": name, "sha256": expected,
                     "author": attribution["Author"], "license": attribution["License"],
                     "page": attribution["OriginalLandingURL"], "source_size": case["size"]})
    if len(rows) != 8:
        raise ValueError("Expected eight retained photo groups")
    return rows


def freeze(sources, archive):
    roster = json.loads((sources / "roster.json").read_bytes())
    if len(roster) != 8 or len({item["author"] for item in roster}) != len(roster):
        raise ValueError("Expected the eight distinct credited photographers")
    rows = []
    for slot, photo in enumerate(roster):
        source = sources / f"{slot}.jpg"
        item = {"slot": slot, **photo, "cohort": "repeated_structure",
                "page": "https://commons.wikimedia.org/wiki/File:" + quote(photo["title"].replace(" ", "_"))}
        if source.exists():
            media_url = (sources / f"{slot}.url").read_text().strip()
            body = source.read_bytes()
            with Image.open(io.BytesIO(body)) as decoded:
                if decoded.format != "JPEG":
                    raise ValueError("Source bytes must decode as JPEG")
                native = ImageOps.exif_transpose(decoded)
                scale = min(1., MAX_SIDE / max(native.size))
                if round(min(native.size) * scale) < 384:
                    raise ValueError("Source cannot be reduced to supported geometry")
                item.update({"url": media_url, "sha256": digest(body), "source_size": native.size, "name": source.name})
        rows.append(item)
    if sum("sha256" in item for item in rows) < 4:
        raise ValueError("At least four planned photos must be acquired before freeze")
    rows.extend(retained_rows(archive, len(roster)))
    admitted = [row for row in rows if "sha256" in row]
    if len({row["author"] for row in admitted}) != len(admitted) or len({row["sha256"] for row in admitted}) != len(admitted):
        raise ValueError("Photo and attributed author must stay disjoint")
    paths = [Path(__file__), Path(copy_affine.__file__), Path(copy_move.__file__)]
    return {"schema": SCHEMA, "sources": rows, "code_sha256": {path.name: digest(path.read_bytes()) for path in paths},
            "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "pillow": pillow_version},
            "formats": FORMATS, "positive": POSITIVE, "controls": CONTROLS,
            "rule": "Per delivery add useful adjusted-copy locations on at least three source photos, preserve every incumbent "
                    "pair and useful location, and add at most one ordinary flag and no additional published-photo flag."}


def fixtures(source, slot):
    """Generate copies with two exact support masks and ordinary deliveries."""
    height, width = source.shape[:2]
    radius = min(height, width) // 8
    candidates = [(x, y) for x in (width // 5, width // 3)
                  for y in (height // 4, height // 2, 3 * height // 4)]
    corners = cv2.cornerMinEigenVal(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), 5)
    dx, dy = max(candidates, key=lambda p: float(corners[p[1]-radius:p[1]+radius,
                                                        p[0]-radius:p[0]+radius].sum()))
    tx, ty = 3 * width // 4, height - dy
    angle, scale, gain, bias = (14., .83, .9, 27.) if slot % 2 == 0 else (-17., 1.17, 1.1, -18.)
    matrix = cv2.getRotationMatrix2D((dx, dy), angle, scale)
    matrix[:, 2] += [tx - dx, ty - dy]
    yy, xx = np.ogrid[:height, :width]
    distance = np.hypot((xx - dx) / radius, (yy - dy) / (.82 * radius))
    donor = distance < 1
    patch = cv2.warpAffine(source, matrix, (width, height), flags=cv2.INTER_CUBIC).astype(np.float32)
    adjusted = np.clip(patch * gain + bias, 0, 255)
    for role, blend in zip(POSITIVE, (donor, np.clip((1 - distance) / .24, 0, 1)) * 2, strict=True):
        alpha = cv2.warpAffine(blend.astype(np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR)
        copied = adjusted if role.endswith("_tone") else patch
        edited = np.rint(copied * alpha[:, :, None] + source * (1 - alpha[:, :, None])).astype(np.uint8)
        receiver = np.any(edited != source, axis=2)
        support = donor & cv2.warpAffine(receiver.astype(np.uint8), matrix, (width, height),
                                         flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype(bool)
        if np.any(support & receiver) or min(int(support.sum()), int(receiver.sum())) < 300:
            raise ValueError("Copy has no distinct measured areas")
        yield role, edited, (support, receiver)
    image = Image.fromarray(source)
    global_tone = np.asarray(ImageEnhance.Contrast(image).enhance(1.15))
    local_tone = source.copy()
    local_tone[donor] = np.rint(np.clip(source[donor] * gain + bias, 0, 255)).astype(np.uint8)
    resized = image.resize((round(width * .83), round(height * .83)), Image.Resampling.BICUBIC)
    empty = np.zeros_like(donor)
    for role, control in zip(CONTROLS, (source, global_tone, local_tone,
                                      np.asarray(resized.resize((width, height), Image.Resampling.BICUBIC))), strict=True):
        yield role, control, (empty, empty)


def box_bounds(box, width, height):
    x, y, w, h = box
    return max(0, math.floor(x)), max(0, math.floor(y)), min(width, math.ceil(x+w)), min(height, math.ceil(y+h))


def mean(value):
    return cv2.boxFilter(value, cv2.CV_32F, (11, 11), normalize=True)


def tone_pair(gray, matrix, seed):
    """Fit one bounded luma transform on the seed and require consistent local agreement."""
    height, width = gray.shape
    left, top, right, bottom = seed
    x0, y0, x1, y1 = max(0, left - 36), max(0, top - 36), min(width, right + 36), min(height, bottom + 36)
    size = (x1 - x0, y1 - y0)
    if min(size) < 16:
        return None
    local = matrix.copy()
    local[:, 2] += local[:, :2] @ np.array([x0, y0])
    a = gray[y0:y1, x0:x1]
    b = cv2.warpAffine(gray, local, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    yy, xx = np.indices(a.shape)
    bx = local[0, 0] * xx + local[0, 1] * yy + local[0, 2]
    by = local[1, 0] * xx + local[1, 1] * yy + local[1, 2]
    valid = ((bx >= 6) & (bx < width - 6) & (by >= 6) & (by < height - 6)
             & ((bx - (xx + x0)) ** 2 + (by - (yy + y0)) ** 2 >= 48**2))
    ma, mb = mean(a), mean(b)
    va, vb = np.maximum(0, mean(a*a) - ma*ma), np.maximum(0, mean(b*b) - mb*mb)
    cov = mean(a*b) - ma*mb
    strong = valid & (np.minimum(va, vb) >= 25) & (cov >= .96 * np.sqrt(va*vb))
    seed_slices = np.s_[top-y0:bottom-y0, left-x0:right-x0]
    fit = np.zeros_like(strong)
    fit[seed_slices] = strong[seed_slices]
    if int(fit.sum()) < 100:
        return None
    local_gain = np.sqrt(vb[fit] / va[fit])
    gain = float(np.median(local_gain))
    bias = float(np.median(mb[fit] - gain * ma[fit]))
    if not (.75 <= gain <= 1.35 and abs(bias) <= 38):
        return None
    close = mean(np.abs(gain*a + bias - b)) <= 3 + .20 * np.sqrt(np.maximum(va*gain*gain, vb))
    selected = cv2.morphologyEx((strong & close).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats((selected.astype(bool) & valid).astype(np.uint8))
    seeds = Counter(labels[seed_slices].ravel())
    supported = [label for label, count in seeds.items()
                 if label and count >= 100 and stats[label, cv2.CC_STAT_AREA] >= 300]
    if not supported:
        return None
    label = max(supported, key=lambda value: (seeds[value], stats[value, cv2.CC_STAT_AREA], -value))
    first = np.zeros(gray.shape, np.uint8)
    first[y0:y1, x0:x1] = labels == label
    second = cv2.warpAffine(first, matrix, (width, height), flags=cv2.INTER_NEAREST)
    if int(second.sum()) < 300 or np.any(first & second):
        return None
    return (first.astype(bool), second.astype(bool)), (gain, bias)


def pixel_counts(masks, truth):
    def measure(predicted, actual):
        return [int(predicted.sum()), int(actual.sum()), int((predicted & actual).sum())]

    def min_iou(counts):
        return min(overlap / (predicted + target - overlap) for predicted, target, overlap in counts)

    orders = [[measure(mask, target) for mask, target in zip(masks, order, strict=True)]
              for order in (truth, truth[::-1])]
    return max(orders, key=min_iou)


def useful(counts):
    return all(target and 10*overlap >= 3*(predicted + target - overlap) and 2*overlap >= target
               for predicted, target, overlap in counts)


def pair_readout(masks, truth, index, correction=None):
    return {"proposal": index, "counts": pixel_counts(masks, truth),
            "mask_sha256": [digest(mask.tobytes()) for mask in masks], "correction": correction}


def score(pixels, truth):
    try:
        proposals = copy_affine.candidates(pixels)
    except copy_move.CandidateBudgetExceeded:
        return {"status": "unassessed_match_budget", "proposals": 0, "incumbent": [], "added": []}
    gray = copy_affine.luma(pixels) if proposals else None
    incumbent, added = [], []
    for index, proposal in enumerate(proposals):
        matrix = np.asarray(proposal.affine.matrix, dtype=np.float64)
        seed = box_bounds(proposal.box_a, pixels.shape[1], pixels.shape[0])
        if (masks := copy_affine.dense_pair(gray, matrix, seed)) is not None:
            incumbent.append(pair_readout(masks, truth, index))
        elif (normalized := tone_pair(gray, matrix, seed)) is not None:
            masks, correction = normalized
            added.append(pair_readout(masks, truth, index, correction))
    return {"status": "assessed", "proposals": len(proposals), "incumbent": incumbent, "added": added}


def summarize(rows):
    results = {fmt: {role: Counter() for role in (*POSITIVE, *CONTROLS)} for fmt in FORMATS}
    gains = {fmt: set() for fmt in FORMATS}
    for row in rows:
        fmt, role = row["format"], row["role"]
        old = any(useful(pair["counts"]) for pair in row["incumbent"])
        new = any(useful(pair["counts"]) for pair in row["added"])
        counts = results[fmt][role]
        counts.update(views=1, assessed=row["status"] == "assessed", proposals=bool(row["proposals"]),
                      incumbent_flag=bool(row["incumbent"]), added_pair=bool(row["added"]),
                      additional_flag=bool(row["added"]) and not row["incumbent"],
                      incumbent_useful=old, combined_useful=old or new)
        if role.endswith("_tone") and role in POSITIVE and new and not old:
            gains[fmt].add(row["slot"])
    advances = {}
    for fmt, roles in results.items():
        advances[fmt] = (len(gains[fmt]) >= 3
                         and sum(roles[role]["added_pair"] for role in CONTROLS) <= 1
                         and not roles["published"]["added_pair"])
    return results, advances, {fmt: len(slots) for fmt, slots in gains.items()}


def compare(sources, archive, protocol):
    for item in protocol["sources"]:
        if "sha256" not in item:
            continue
        location = sources if item["cohort"] == "repeated_structure" else archive / "assets"
        with Image.open(location / item["name"]) as original:
            source = ImageOps.exif_transpose(original).convert("RGB")
            source.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        for role, pixels, truth in fixtures(np.asarray(source), item["slot"]):
            for fmt in FORMATS:
                buffer = io.BytesIO()
                if fmt == "png":
                    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=6)
                else:
                    Image.fromarray(pixels).save(buffer, format="JPEG", quality=75, subsampling=2)
                body = buffer.getvalue()
                with Image.open(io.BytesIO(body)) as image:
                    decoded = np.asarray(image.convert("RGB"))
                yield {"slot": item["slot"], "cohort": item["cohort"], "role": role, "format": fmt, "input_sha256": digest(body),
                       "truth_sha256": [digest(mask.tobytes()) for mask in truth],
                       "truth_pixels": [int(mask.sum()) for mask in truth], **score(decoded, truth)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("sources", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    protocol = freeze(args.sources, args.archive)
    plan = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir()
        write(plan, protocol)
        print(digest(plan.read_bytes()), flush=True)
        return
    if json.loads(plan.read_bytes()) != json.loads(json.dumps(protocol)):
        raise ValueError("Source assets or code changed after freeze")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    started = time.monotonic()
    rows = []
    journal = args.output / "journal.jsonl"
    with journal.open("x", encoding="utf-8") as stream:
        for row in compare(args.sources, args.archive, protocol):
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            rows.append(row)
            if len(rows) % (len(FORMATS) * (len(POSITIVE) + len(CONTROLS))) == 0:
                print(f"{len(rows)} views measured", flush=True)
    results, advances, gained_groups = summarize(rows)
    write(args.output / "aggregate.json", {"schema": SCHEMA, "freeze_sha256": digest(plan.read_bytes()),
                                           "journal_sha256": digest(journal.read_bytes()), "views": len(rows),
                                           "elapsed_s": time.monotonic() - started, "results": results,
                                           "advances": advances, "gained_groups": gained_groups})


if __name__ == "__main__":
    main()
