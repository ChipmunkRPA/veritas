# SPDX-License-Identifier: Apache-2.0
"""Qualify local HDR gain-map disagreement on independently authored JPEG deliveries."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
from PIL import Image

from app.provenance.exiftool import _pillow_thumbnail

SCHEMA = "chipmunk.image-hdr-gainmap-transfer/1"
CONTAINER = "http://ns.google.com/photos/1.0/container/"
ITEM = CONTAINER + "item/"
HDR = "http://ns.adobe.com/hdr-gain-map/1.0/"
ROLES = ("untouched_camera", "clean", "jpeg75", "tone", "resize", "sharpen", "highlight_edit", "arbitrary_edit", "no_gainmap")
POS = {"highlight_edit", "arbitrary_edit"}
ORDINARY = set(ROLES) - POS - {"no_gainmap"}
METHODS = ("orphan_edge", "dark_relative_to_gain")
CELL = 8
SIDE = 384
SIDES = (5, 7, 9, 11)
CRITERIA = {"held_groups": 6, "useful_highlight": 3, "useful_arbitrary": 2, "maximum_ordinary_groups": 1}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def directory(length):
    return f'''<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description xmlns:hdrgm="{HDR}" hdrgm:Version="1.0" xmlns:Container="{CONTAINER}" xmlns:Item="{ITEM}">
<Container:Directory><rdf:Seq><rdf:li rdf:parseType="Resource"><Container:Item Item:Mime="image/jpeg" Item:Semantic="Primary"/>
</rdf:li><rdf:li rdf:parseType="Resource"><Container:Item Item:Mime="image/jpeg" Item:Semantic="GainMap" Item:Length="{length}"/>
</rdf:li></rdf:Seq></Container:Directory></rdf:Description></rdf:RDF></x:xmpmeta>'''.encode()


def _xml(value):
    if not value or len(value) > 65_536 or b"<!" in value:
        return None
    try:
        return ElementTree.fromstring(value)
    except ElementTree.ParseError:
        return None


def split(data):
    """Read a declared grayscale gain map whose JPEG bytes end at the file boundary."""
    with Image.open(io.BytesIO(data)) as base:
        xml = _xml(base.info.get("xmp"))
        if (base.format != "JPEG" or max(base.size) > 8192 or min(base.size) < SIDE or xml is None
                or base.getexif().get(274, 1) != 1):
            return None
        directory_nodes = xml.findall(f".//{{{CONTAINER}}}Directory")
        if len(directory_nodes) != 1:
            return None
        nodes = directory_nodes[0].findall(f".//{{{CONTAINER}}}Item")
        if [node.get(f"{{{ITEM}}}Semantic") for node in nodes] != ["Primary", "GainMap"]:
            return None
        if any(node.get(f"{{{ITEM}}}Mime") != "image/jpeg" or node.get(f"{{{ITEM}}}Padding", "0") != "0" for node in nodes):
            return None
        length = nodes[1].get(f"{{{ITEM}}}Length", "")
        if not length.isdecimal() or not 4 <= int(length) <= min(2_000_000, len(data) - 100):
            return None
        cut = len(data) - int(length)
        if data[cut-2:cut+2] != b"\xff\xd9\xff\xd8" or not data.endswith(b"\xff\xd9"):
            return None
        payload = data[cut:]
        with Image.open(io.BytesIO(payload)) as thumb:
            metadata = _xml(thumb.info.get("xmp"))
            if (thumb.format != "JPEG" or thumb.mode != "L" or max(thumb.size) > 2048 or min(thumb.size) < 96
                    or abs((base.width / base.height) / (thumb.width / thumb.height) - 1.) > .01 or metadata is None
                    or not any(node.get(f"{{{HDR}}}Version") == "1.0" for node in metadata.iter())):
                return None
            gain = thumb.copy()
        base_pixels = base.convert("RGB")
        return base_pixels, gain, payload


def encode(image, gain, quality):
    with io.BytesIO() as output:
        params = {"xmp": directory(len(gain))} if gain else {}
        image.save(output, "JPEG", quality=quality, subsampling=2, **params)
        return output.getvalue() + gain


def working(image, edge):
    factor = edge / max(image.size)
    return image.resize(tuple(round(n * factor) for n in image.size), Image.Resampling.LANCZOS)


def _windows(h, w, cell=1):
    for height in SIDES:
        for width in SIDES:
            if max(width, height) > 1.5 * min(width, height):
                continue
            xs, ys = np.meshgrid(np.arange(2, w - 1 - width), np.arange(2, h - 1 - height))
            yield cell*xs, cell*ys, cell*(xs + width), cell*(ys + height)


def _integral(data):
    return np.pad(data.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _sum(data, left, top, right, bottom):
    return data[bottom, right] - data[top, right] - data[bottom, left] + data[top, left]


def _box_from_gain(gain, alternative):
    grid = np.asarray(gain.resize((48, round(gain.height * 48 / gain.width)), Image.Resampling.BILINEAR), dtype=np.float32)
    h, w = grid.shape
    side = 9
    xs, ys = np.meshgrid(np.arange(4, w - side - 4), np.arange(4, h - side - 4))
    if alternative is None:
        interest = _sum(_integral(np.maximum(grid - np.median(grid), 0)), xs, ys, xs + side, ys + side)
        index = np.unravel_index(interest.argmax(), interest.shape)
    else:
        index = (int(alternative[0:4], 16) % xs.shape[0], int(alternative[4:8], 16) % xs.shape[1])
    x, y = int(xs[index]), int(ys[index])
    return [x / w, y / h, (x + side) / w, (y + side) / h]


def _authored(base, rect, slot):
    width, height = base.size
    left, top, right, bottom = [round(value * length) for value, length in zip(rect, (width, height, width, height), strict=True)]
    source = np.asarray(base).copy()
    target = source[top:bottom, left:right]
    h, w = target.shape[:2]
    y, x = np.ogrid[:h, :w]
    ramp = np.minimum(np.minimum(x, w-1-x), np.minimum(y, h-1-y))
    alpha = np.clip(ramp / max(1, min(h, w) * .14), 0, 1)
    # A locally replaced dark patch removes the base's highlight structure while retaining its prior gain map.
    smooth = cv2.GaussianBlur(target, (0, 0), 8).astype(np.float32)
    dark = smooth * (.48 + .03 * (slot % 4))
    candidate = np.rint(target * (1 - alpha[:, :, None]) + dark * alpha[:, :, None]).astype(np.uint8)
    truth = np.zeros(source.shape[:2], bool)
    truth[top:bottom, left:right] = (candidate != target).any(2)
    source[top:bottom, left:right] = candidate
    return Image.fromarray(source), truth


def sources(archive):
    manifest = json.loads((archive / "acquisition.json").read_bytes())
    if manifest["license"] != "https://creativecommons.org/licenses/by/4.0/" or len(manifest["files"]) != 10:
        raise ValueError("The approved original photo set differs")
    selected = sorted(manifest["files"], key=lambda row: sha(f"hdr-local/1/{row['sha256']}".encode()))
    for slot, row in enumerate(selected):
        data = (archive / row["name"]).read_bytes()
        if sha(data) != row["sha256"] or len(data) != row["bytes"]:
            raise ValueError("The publisher's original file differs")
        parsed = split(data)
        if parsed is None:
            raise ValueError("The publisher's gain map could not be verified")
        yield slot, row, data, parsed


def cases(archive):
    for slot, row, data, (base, gain, payload) in sources(archive):
        base = working(base, SIDE * 2)
        w, h = base.size
        changed = [_authored(base, _box_from_gain(gain, key), slot) for key in (None, row["sha256"])]
        pixels = np.asarray(base)
        tone = Image.fromarray(np.rint(pixels.astype(np.float32)*.90 + 10).astype(np.uint8))
        reduced = base.resize((round(w*.77), round(h*.77)), Image.Resampling.LANCZOS).resize(base.size, Image.Resampling.BICUBIC)
        blurred = cv2.GaussianBlur(pixels, (0, 0), .9)
        sharpen = Image.fromarray(np.clip(np.rint(1.20*pixels - .20*blurred), 0, 255).astype(np.uint8))
        yield slot, ROLES[0], data, None
        variants = [(base, None, 90), (base, None, 75), (tone, None, 90), (reduced, None, 90), (sharpen, None, 90),
                    (*changed[0], 90), (*changed[1], 90), (base, None, 90)]
        for role, (image, truth, quality) in zip(ROLES[1:], variants, strict=True):
            body = encode(image, payload if role != "no_gainmap" else b"", quality)
            with Image.open(io.BytesIO(body)) as original:
                if _pillow_thumbnail(original, original.getexif()) is not None:
                    raise ValueError("A no-preview variant unexpectedly retained an IFD1 thumbnail")
            yield slot, role, body, truth


def _rank(base, gain):
    def gradient(image):
        smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .8)
        x, y = (cv2.Sobel(smooth, cv2.CV_32F, int(axis == 0), int(axis == 1), ksize=3) / 8 for axis in (0, 1))
        return x, y, np.hypot(x, y)

    bx, by, bm = gradient(base)
    gx, gy, gm = gradient(gain)
    main = bm > max(4., float(np.percentile(bm, 76)))
    gain_edge = gm > max(5., float(np.percentile(gm, 86)))
    common = np.zeros(main.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            a, b, length, valid = (np.roll(v, (dy, dx), (0, 1)) for v in (bx, by, bm, main))
            common |= valid & (np.abs(a * gx + b * gy) > .75 * length * gm)
    edge = np.where(gain_edge, (~common).astype(np.float32), 0.)

    # For each gain range, estimate the usual SDR level from the dominant source pixels.
    # The separate score looks only for unusually dark replacement inside high-gain areas.
    groups = np.clip(gain // 16, 0, 15).astype(np.int64)
    dark, high = np.zeros_like(edge), np.zeros_like(main)
    for i in range(16):
        included = groups == i
        if i >= max(2, int(np.percentile(groups, 72))) and included.sum() >= 64:
            center = np.median(base[included])
            spread = np.median(np.abs(base[included] - center)) * 1.4826
            dark[included] = np.clip((center - base[included]) / max(14, spread), 0, 3)
            high |= included
    return ((edge, gain_edge), (dark, high))


def measure(data):
    split_pair = split(data)
    if split_pair is None:
        return None
    image, gain, _ = split_pair
    nw, nh = image.size
    base = working(image, SIDE).convert("L")
    w, h = base.size
    field = _rank(np.asarray(base, dtype=np.float32), np.asarray(gain.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32))
    ranked = {}
    for method, (scores, valid) in zip(METHODS, field, strict=True):
        values, count = _integral(scores), _integral(valid.astype(np.int64))
        best = (0., None)
        for x, y, right, bottom in _windows(h // CELL, w // CELL, CELL):
            total = _sum(count, x, y, right, bottom)
            core = _sum(values, x, y, right, bottom) / np.maximum(total, 1)
            outer = _sum(count, x-CELL, y-CELL, right+CELL, bottom+CELL) - total
            ring = (_sum(values, x-CELL, y-CELL, right+CELL, bottom+CELL) - core*total) / np.maximum(outer, 1)
            use = (total >= 35) & (outer >= 24)
            score = np.where(use, (core - ring) * np.minimum(1, np.sqrt(total / 80)), 0)
            k = np.unravel_index(score.argmax(), score.shape)
            if score[k] > best[0]:
                box = [round(q[k] * native / size) for q, native, size in zip((x, y, right, bottom), (nw, nh, nw, nh), (w, h, w, h), strict=True)]
                best = float(score[k]), box
        ranked[method] = {"score": best[0], "box": best[1]}
    return ranked


def binding(archive):
    locked = [{"slot": slot, "role": role, "sha256": sha(body), "truth": sha(mask.tobytes()) if mask is not None else None}
              for slot, role, body, mask in cases(archive)]
    return {"schema": SCHEMA, "criteria": CRITERIA, "code_sha256": sha(Path(__file__).read_bytes()),
            "intake_sha256": sha((archive / "acquisition.json").read_bytes()), "confidence_weight": 0, "cases": locked,
            "calibration_group_slots": list(range(4)), "held_group_slots": list(range(4, 10))}


def useful(box, truth):
    if truth is None or box is None:
        return False
    x, y, right, bottom = box
    hit = int(truth[y:bottom, x:right].sum())
    total = int(truth.sum())
    return 2*hit >= total and 10*hit >= 3*((right-x)*(bottom-y)+total-hit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "calibrate", "held"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    frozen = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(frozen, binding(args.archive))
        print(sha(frozen.read_bytes()), flush=True)
        return
    assert binding(args.archive) == json.loads(frozen.read_bytes())
    result = []
    for slot, role, body, mask in cases(args.archive):
        if (slot < 4) != (args.stage == "calibrate") or (args.stage == "calibrate" and role not in ORDINARY):
            continue
        scores = measure(body)
        if scores is not None:
            for arm in scores.values():
                arm["useful"] = useful(arm["box"], mask)
        result.append({"slot": slot, "role": role, "scores": scores})
    if args.stage == "calibrate":
        points = {method: max(row["scores"][method]["score"] for row in result) for method in METHODS}
        write(args.run / "points.json", {"freeze": sha(frozen.read_bytes()), "points": points, "calibration": result})
        print(points, flush=True)
        return
    operating = json.loads((args.run / "points.json").read_bytes())
    assert operating["freeze"] == sha(frozen.read_bytes())
    reports = {}
    for method, point in operating["points"].items():
        positive = {role: sum(row["scores"][method]["useful"] and row["scores"][method]["score"] > point
                              for row in result if row["role"] == role) for role in POS}
        ordinary = [row for row in result if row["role"] in ORDINARY and row["scores"][method]["score"] > point]
        groups = len({row["slot"] for row in ordinary})
        passed = (positive["highlight_edit"] >= CRITERIA["useful_highlight"] and positive["arbitrary_edit"] >= CRITERIA["useful_arbitrary"]
                  and groups <= CRITERIA["maximum_ordinary_groups"])
        reports[method] = {"useful": positive, "ordinary_views_flagged": len(ordinary), "ordinary_groups_flagged": groups,
                           "useful_before_point": {role: sum(row["scores"][method]["useful"] for row in result if row["role"] == role) for role in POS},
                           "point": point, "passed": passed}
    write(args.run / "held.json", {"held": result, "summary": reports})
    print(reports, flush=True)


if __name__ == "__main__":
    main()
