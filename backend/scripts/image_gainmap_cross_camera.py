# SPDX-License-Identifier: Apache-2.0
"""Transfer fixed gain-map localization points to original files from new cameras."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "docs/evaluation-panels/image-gainmap-localization-2026-11-12.json"
SCHEMA = "chipmunk.image-gainmap-cross-camera/1"
METHODS = ("orphan_edge", "dark_relative_to_gain")
ORDINARY = ("original", "jpeg90", "jpeg75", "tone", "resize", "sharpen", "aligned_crop", "aligned_rotation")
EDITED = ("highlight_edit", "arbitrary_edit")
ROLES = (*ORDINARY, *EDITED, "no_gainmap")
CRITERIA = {"min_held_author_groups": 6, "min_useful_edits": 6, "min_new_carrier_makes": 2,
            "max_flagged_ordinary_authors": 1, "max_mislocalized_edits": 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def configure():
    import cv2

    cv2.setNumThreads(1)


def central_crop(size):
    width, height = size
    return round(width / 8), round(height / 8), round(7 * width / 8), round(7 * height / 8)


def cases(archive, sources):
    import cv2
    import numpy as np
    from PIL import Image

    from scripts import image_gainmap_iso as expanded
    from scripts import image_gainmap_localization_transfer as prior

    for slot, row in enumerate(sources):
        raw = (archive / (row["key"] + ".jpg")).read_bytes()
        require(len(raw) == row["bytes"] and sha(raw) == row["local_sha256"]
                and hashlib.sha1(raw).hexdigest() == row["publisher_sha1"], "Publisher original changed")
        prior_pair = prior.split(raw)
        pair = expanded.split(raw)
        require((prior_pair is not None) == row["current_parser_admits"], "Historical reader differs from source intake")
        if pair is None:
            yield row["key"], "original", None, None, None, None
            continue
        original, gain, encoded_map = pair
        require(list(original.size) == row["dimensions"], "Wrong original image geometry")
        if prior_pair is not None:
            require(all(a.tobytes() == b.tobytes() for a, b in zip(prior_pair[:2], pair[:2], strict=True))
                    and encoded_map == prior_pair[2], "Expanded reader changed the supported XMP input")
        else:
            require(list(gain.size) == row["mpf_layout"]["gain_dimensions"]
                    and sha(encoded_map) == row["mpf_layout"]["gain_aux_sha256"], "Unexpected indexed auxiliary image")
        yield row["key"], "original", original, gain, None, sha(encoded_map)

        base = prior.working(original, prior.SIDE * 2)
        del original, pair, prior_pair, raw
        width, height = base.size
        pixels = np.asarray(base)
        tone = Image.fromarray(np.rint(pixels.astype(np.float32) * .90 + 10).astype(np.uint8))
        small = base.resize((round(width * .77), round(height * .77)), Image.Resampling.LANCZOS)
        resized = small.resize(base.size, Image.Resampling.BICUBIC)
        softened = cv2.GaussianBlur(pixels, (0, 0), .9)
        sharp = Image.fromarray(np.clip(1.2 * pixels - .2 * softened, 0, 255).astype(np.uint8))
        variants = [(base, gain, None, 90), (base, gain, None, 75), (tone, gain, None, 90),
                    (resized, gain, None, 90), (sharp, gain, None, 90),
                    (base.crop(central_crop(base.size)), gain.crop(central_crop(gain.size)), None, 90),
                    (base.transpose(Image.Transpose.ROTATE_90), gain.transpose(Image.Transpose.ROTATE_90), None, 90)]
        for target in (None, row["local_sha256"]):
            image, mask = prior._authored(base, prior._box_from_gain(gain, target), slot)
            variants.append((image, gain, mask, 90))
        variants.append((base, None, None, 90))
        for role, (image, paired_map, mask, quality) in zip(ROLES[1:], variants, strict=True):
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=quality, subsampling=2)
            with Image.open(buffer) as delivered:
                yield row["key"], role, delivered.convert("RGB"), paired_map, mask, None


def geometry(image):
    return {"dimensions": list(image.size), "sha256": sha(image.tobytes())} if image else None


def inputs(cases_iterator):
    for key, role, image, gain, mask, auxiliary in cases_iterator:
        yield {"key": key, "role": role, "base": geometry(image), "gain": geometry(gain), "auxiliary_sha256": auxiliary,
               "truth_pixels": int(mask.sum()) if mask is not None else 0,
               "truth_sha256": sha(mask.tobytes()) if mask is not None else None}


def script_hashes():
    paths = (Path(__file__), Path(__file__).with_name("image_gainmap_iso.py"),
             Path(__file__).with_name("image_gainmap_localization_transfer.py"))
    return {path.name: sha(path.read_bytes()) for path in paths}


def freeze(archive, source_path):
    sources = read(source_path)["records"]
    require(len(sources) == len({row["key"] for row in sources}) == 10, "Expected ten distinct source files")
    prior = read(PRIOR)
    return {"schema": SCHEMA, "criteria": CRITERIA, "points": {key: prior["held"][key]["point"].hex() for key in METHODS},
            "source_roster": sources, "prior_sha256": sha(PRIOR.read_bytes()),
            "scripts": script_hashes(), "views": list(inputs(cases(archive, sources)))}


def measure(image, gain, mask):
    import numpy as np
    from PIL import Image

    from scripts import image_gainmap_localization_transfer as prior

    if gain is None:
        return None
    original = prior.working(image, prior.SIDE).convert("L")
    width, height = original.size
    fields = prior._rank(np.asarray(original, dtype=np.float32),
                         np.asarray(gain.resize(original.size, Image.Resampling.BILINEAR), dtype=np.float32))
    outcomes = {}
    cell = prior.CELL
    for method, (scores, valid) in zip(METHODS, fields, strict=True):
        values, count = prior._integral(scores), prior._integral(valid.astype(np.int64))
        highest, box = 0., None
        for x, y, right, bottom in prior._windows(height // cell, width // cell, cell):
            core_count = prior._sum(count, x, y, right, bottom)
            core = prior._sum(values, x, y, right, bottom) / np.maximum(core_count, 1)
            outer = (x-cell, y-cell, right+cell, bottom+cell)
            ring_count = prior._sum(count, *outer) - core_count
            ring = (prior._sum(values, *outer) - core * core_count) / np.maximum(ring_count, 1)
            selected = np.where((core_count >= 35) & (ring_count >= 24),
                                (core-ring) * np.minimum(1, np.sqrt(core_count / 80)), 0)
            index = np.unravel_index(selected.argmax(), selected.shape)
            if selected[index] > highest:
                highest = float(selected[index])
                box = [round(axis[index] * size / working) for axis, size, working in
                       zip((x, y, right, bottom), image.size * 2, original.size * 2, strict=True)]
        hit = int(mask[box[1]:box[3], box[0]:box[2]].sum()) if box is not None and mask is not None else 0
        outcomes[method] = {"score": highest.hex(), "box": box, "hit": hit}
    return outcomes


def review(locked, measured):
    require(locked["schema"] == SCHEMA and locked["criteria"] == CRITERIA
            and locked["prior_sha256"] == sha(PRIOR.read_bytes()) and locked["scripts"] == script_hashes(),
            "Frozen study definition changed")
    expected_points = {key: read(PRIOR)["held"][key]["point"].hex() for key in METHODS}
    require(locked["points"] == expected_points, "Original points were changed")
    roster = {row["key"]: row for row in locked["source_roster"]}
    require(len(roster) == len(locked["source_roster"]) == 10, "Original source count changed")
    view_ids = [(view["key"], view["role"]) for view in locked["views"]]
    originals = {view["key"]: view for view in locked["views"] if view["role"] == "original"}
    require(set(originals) == set(roster) and view_ids == [(key, role) for key in roster
            for role in (ROLES if originals[key]["gain"] else ("original",))], "Original or derived view missing")
    results = {arm: {"ordinary_authors": set(), "new_carrier_makes": set(), "counts": Counter()} for arm in (*METHODS, "union")}
    for view, observed in zip(locked["views"], measured, strict=True):
        require((observed["key"], observed["role"]) == (view["key"], view["role"]), "View roster changed")
        row, role, scores = roster[view["key"]], view["role"], observed["scores"]
        require((scores is None) == (view["gain"] is None), "Assessed coverage changed")
        require(role != "no_gainmap" or scores is None, "A view without a gain map cannot carry a score")
        if scores is None:
            continue
        width, height = view["base"]["dimensions"]
        truth = view["truth_pixels"]
        require(set(scores) == set(METHODS) and bool(truth) == (role in EDITED), "Measurement roster changed")
        states = {}
        for method, response in scores.items():
            box, hit = response["box"], response["hit"]
            score = float.fromhex(response["score"])
            require(0 <= score <= 3 and type(hit) is int and 0 <= hit <= truth, "Invalid measurement")
            if box is not None:
                x, y, right, bottom = box
                area = (right - x) * (bottom - y)
                require(0 <= x < right <= width and 0 <= y < bottom <= height and hit <= area, "Invalid source box")
            else:
                require(hit == 0 and score == 0, "Missing response box")
                area = 0
            states[method] = score > float.fromhex(locked["points"][method]), bool(truth and 2*hit >= truth and 10*hit >= 3*(truth+area-hit))
        for arm, result in results.items():
            considered = list(states.values()) if arm == "union" else [states[arm]]
            flagged = any(flag for flag, _ in considered)
            useful = any(flag and match for flag, match in considered)
            misplaced = any(flag and not match for flag, match in considered)
            result["counts"]["assessed_" + role] += 1
            if role in EDITED:
                result["counts"]["useful_" + role] += useful
                result["counts"]["mislocalized_edits"] += misplaced
                if useful and not row["current_parser_admits"]:
                    result["new_carrier_makes"].add(row["make"])
            elif flagged:
                result["counts"]["flagged_ordinary"] += 1
                result["ordinary_authors"].add(row["author"])
    authors = len({roster[view["key"]]["author"] for view in locked["views"] if view["gain"] and view["role"] == "original"})
    summaries = {}
    for arm, result in results.items():
        counts = dict(result["counts"])
        makes = sorted(result["new_carrier_makes"])
        ordinary = len(result["ordinary_authors"])
        passed = (authors >= CRITERIA["min_held_author_groups"] and len(makes) >= CRITERIA["min_new_carrier_makes"]
                  and sum(counts.get("useful_"+role, 0) for role in EDITED) >= CRITERIA["min_useful_edits"]
                  and ordinary <= CRITERIA["max_flagged_ordinary_authors"]
                  and counts.get("mislocalized_edits", 0) <= CRITERIA["max_mislocalized_edits"])
        summaries[arm] = {"counts": counts, "new_carrier_makes": makes, "ordinary_authors_flagged": ordinary, "passed": passed}
    return {"assessed_authors": authors, "arms": summaries}


def observed(archive, locked):
    return [{"key": key, "role": role, "scores": measure(image, gain, mask)}
            for key, role, image, gain, mask, _ in cases(archive, locked["source_roster"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sources", type=Path)
    args = parser.parse_args()
    if args.phase != "review" or args.archive:
        configure()
    if args.phase == "freeze":
        if not (args.archive and args.sources):
            parser.error("freeze needs --archive and --sources")
        write(args.path, freeze(args.archive, args.sources))
        print(sha(args.path.read_bytes()))
    elif args.phase == "score":
        if not (args.archive and args.sources):
            parser.error("score needs --archive and --sources")
        locked = read(args.path)
        require(locked == freeze(args.archive, args.sources), "Frozen source views changed")
        records = observed(args.archive, locked)
        summary = review(locked, records)
        write(args.path.with_name("receipt.json"), {"freeze": locked, "measured": records, "summary": summary})
        print(json.dumps(summary, indent=2))
    else:
        receipt = read(args.path)
        locked = receipt["freeze"]
        summary = review(locked, receipt["measured"])
        require(summary == receipt["summary"], "Reported result differs")
        if args.archive:
            require(args.sources and locked == freeze(args.archive, args.sources), "Frozen source views changed")
            require(observed(args.archive, locked) == receipt["measured"], "Native response differs")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
