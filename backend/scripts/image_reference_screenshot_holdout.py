# SPDX-License-Identifier: Apache-2.0
"""Freeze and replay native screenshot comparison with exact two-source evidence."""

from __future__ import annotations

import argparse
import io
import tempfile
import time
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_png_screenshot as candidate
from app.evidence import image_reference
from scripts import image_reference_embedded_screenshot_transfer as earlier

SCHEMA = "chipmunk.image-reference-screenshot-holdout/1"
SOURCE_COUNTS = {"LaMa": 12, "sdxl": 3, "kandinsky": 3}
ROLES = ("clean", "edit", "local_tone", "global_tone", "resized", "unrelated", "duplicated", "jpeg", "chrome")
MATCHED = {"clean", "edit", "local_tone"}
GATE = {"minimum_lama_edit_locations": 9, "minimum_other_edit_locations": 5,
        "minimum_lama_clean_matches": 10, "minimum_other_clean_matches": 5,
        "maximum_wrong_locations": 0, "maximum_clean_highlights": 0, "maximum_incompatible_matches": 0}
ARMS = ("serving", "previous_inset", "boundary_witness")
require, read, write, sha, file_sha = earlier.require, earlier.read, earlier.write, earlier.sha, earlier.file_sha
base = earlier.prior.BASE


def sources(args):
    for source in earlier.prior.old.primary_roster(args)[1]:
        yield "LaMa", source["source"]
    candidates = read(args.alternate / "plan.json")["cases"]
    for family in ("sdxl", "kandinsky"):
        found = sorted((row for row in candidates if row["generator"] == family), key=lambda row: row["slot"])[3:]
        require(len(found) == SOURCE_COUNTS[family], "Alternate photo roster differs")
        yield from ((family, source) for source in found)


def photos(args):
    cache, lama = earlier.prior.old.primary_roster(args)
    alternates = {case["slot"]: (case, images) for case, images, _ in base.REGIONAL.paired.inputs(args.alternate)}
    primary = {row["source"]["flickr_photo_id"]: row for row in lama}
    for family, source in sources(args):
        if family == "LaMa":
            files = primary[source["flickr_photo_id"]]["measurement"]["files"]
            frames = []
            for role in ("original", "edited"):
                path = cache / files[role]["file"]
                require(file_sha(path) == files[role]["sha256"], "Qualified LaMa input changed")
                frames.append(base.pixels(path))
            original, edited = frames
            edited = base.feather(original, edited, np.any(original != edited, axis=2))
        else:
            case, images = alternates[source["slot"]]
            variants = base.REGIONAL.native_variants(images, case["box"])
            original, edited = variants["original"], variants["feathered_edit"]
        yield family, source, original, edited


def plan(args):
    layouts = []
    for slot, (_, _, original, edited) in enumerate(photos(args)):
        truth = np.any(original != edited, axis=2)
        left, top, right, bottom = base.source_box(truth)
        height, width = truth.shape
        w, h = min(width, 448 + (slot % 4) * 10), min(height, 394 + (slot % 3) * 19)
        a = max(0, min(width - w, (left + right - w) // 2))
        b = max(0, min(height - h, (top + bottom - h) // 2))
        require(truth[b:b+h, a:a+w].sum() >= .80 * truth.sum(), "The screenshot omits too much known edit area")
        layouts.append({"slot": slot, "layout": slot % 4, "width": 816, "height": 680,
                        "x": 67 + (slot * 43) % 185, "y": 76 + (slot * 31) % 100,
                        "w": w, "h": h, "source_box": [a, b, a + w, b + h]})
    return layouts


def prepare(args):
    args.run.mkdir(mode=0o700)
    browser = args.run / "browser"
    browser.mkdir()
    (browser / "photo-fixtures").mkdir()
    layouts = plan(args)
    write(browser / "layout-plan.json", layouts)
    for item, (_, _, original, edited) in zip(layouts, photos(args), strict=True):
        a, b, right, bottom = item["source_box"]
        for role, rgb in (("photo", original), ("edit", edited)):
            (browser / "photo-fixtures" / f"{item['slot']}-{role}.png").write_bytes(base.png(rgb[b:bottom, a:right]))


def cases(args):
    browser = args.run / "browser"
    layouts, capture_record = (read(browser / name) for name in ("layout-plan.json", "captures.json"))
    require(layouts == plan(args), "Capture layout changed")
    captures = {(row["slot"], row["role"]): row for row in capture_record["rows"]}
    require(len(captures) == len(capture_record["rows"]) == len(layouts) * 3, "Capture roster differs")
    roster = list(photos(args))
    for item, (family, source, original, edited) in zip(layouts, roster, strict=True):
        slot, x, y, w, h = (item[key] for key in ("slot", "x", "y", "w", "h"))
        a, b, right, bottom = item["source_box"]
        plain, blend = (pixels[b:bottom, a:right] for pixels in (original, edited))
        for role, capture in (("clean", "photo"), ("edit", "edit"), ("chrome", "template")):
            path = browser / "captures" / f"{slot}-{capture}.png"
            body, rgb = path.read_bytes(), base.pixels(path)
            require(sha(body) == captures[slot, capture]["png_sha256"], "Chromium capture bytes changed")
            truth = np.zeros(rgb.shape[:2], dtype=bool)
            if role in MATCHED:
                expected = plain if role == "clean" else blend
                require(np.array_equal(expected, rgb[y:y+h, x:x+w]), "Chromium changed native photo pixels")
                truth[y:y+h, x:x+w] = np.any(expected != plain, axis=2)
            yield (family, slot, role), source, original, body, rgb, truth, (earlier.bounds(a, b, w, h), earlier.bounds(x, y, w, h))

        tone = plain.copy()
        tone[h//4:h//4 + h//5, w//3:w//3 + w//6] = (tone[h//4:h//4 + h//5, w//3:w//3 + w//6] * .83 + 5).astype(np.uint8)
        resized = np.asarray(Image.fromarray(plain).resize((round(w * .87), round(h * .87)), Image.Resampling.BICUBIC))
        donor = roster[(slot + 1) % len(roster)][2][:h, :w]
        controls = (("local_tone", tone), ("global_tone", (plain * .84 + 6).astype(np.uint8)),
                    ("resized", resized), ("unrelated", donor), ("duplicated", plain), ("jpeg", plain))
        for role, patch in controls:
            body, rgb, (px, py) = earlier.canvas(patch, slot, role)
            truth = np.zeros(rgb.shape[:2], dtype=bool)
            if role == "local_tone":
                truth[py:py+h, px:px+w] = np.any(plain != tone, axis=2)
            yield (family, slot, role), source, original, body, rgb, truth, (earlier.bounds(a, b, w, h), earlier.bounds(px, py, w, h))


def frozen(args):
    roster = [earlier.identity(case) for case in cases(args)]
    selected = [row for row in roster if row["key"][-1] == "clean"]
    require(len(selected) == sum(SOURCE_COUNTS.values()), "Wrong source count")
    expected = {(family, slot, role) for family, slot, _ in (row["key"] for row in selected) for role in ROLES}
    require(len(expected) == len(roster) and expected == {tuple(row["key"]) for row in roster}, "Role roster differs")
    exposed = [row for panel in (args.development, args.confirmation) for row in read(panel / "freeze.json")["cases"]]
    for field in ("source_group", "contributor", "reference_rgb"):
        identities = {row[field] for row in selected}
        require(len(identities) == len(selected) and not identities & {row[field] for row in exposed},
                f"Held photo or contributor overlaps exposed cases: {field}")
    bound = (Path(__file__), Path(candidate.__file__), Path(candidate.reference_compare.__file__),
             Path(earlier.candidate.__file__), Path(earlier.__file__), Path(earlier.prior.old.__file__),
             Path(image_reference.__file__), args.alternate / "plan.json", args.alternate / "attempts.jsonl",
             args.primary / "qualified-private.json", args.backup / "qualified-private.json",
             args.development / "freeze.json", args.confirmation / "freeze.json",
             Path(__file__).with_name("capture_image_crop_multiscale_transfer.cjs"),
             args.run / "browser" / "layout-plan.json", args.run / "browser" / "captures.json")
    return {"schema": SCHEMA, "gate": GATE, "useful": {"iou": .30, "recall": .50},
            "source_counts": SOURCE_COUNTS, "arms": list(ARMS), "cases": roster, "confidence_weight": 0,
            "bound_inputs": {f"{i:02}-{path.name}": file_sha(path) for i, path in enumerate(bound)}}


def arm(result, mask, target, expected):
    proof = result and result.get("embedded_native_photo")
    selected = int(mask.sum()) if mask is not None else 0
    overlap = int((mask & target).sum()) if mask is not None else 0
    target_count = int(target.sum())
    geometry = bool(proof and (proof["reference_bounds"], proof["submitted_bounds"]) == expected)
    return {"accepted": bool(result and result["status"] == "comparable"), "exact_geometry": geometry,
            "useful": bool(geometry and target_count and 2 * overlap >= target_count
                           and 10 * overlap >= 3 * (target_count + selected - overlap)),
            "mask_sha256": sha(mask.tobytes()) if mask is not None else None, "selected": selected,
            "overlap": overlap, "regions": result["regions"] if result is not None else [],
            "proof": proof}


def measure(args):
    rows = []
    with tempfile.TemporaryDirectory(dir=args.run) as scratch:
        before, after = (Path(scratch) / name for name in ("earlier.png", "submitted"))
        for key, _, original, body, rgb, truth, expected in cases(args):
            before.write_bytes(base.png(original))
            after.write_bytes(body)
            current = image_reference.inspect(before, after, allow_proportional_resize=True,
                                              allow_oriented_crop=True, allow_oriented_resize=True)
            with Image.open(io.BytesIO(body)) as carrier:
                previous, previous_mask = earlier.candidate.compare(original, rgb, carrier)
                start = time.monotonic()
                result, mask = candidate.compare(original, rgb, carrier)
                runtime = time.monotonic() - start
            rows.append({"key": list(key), "target": int(truth.sum()), "candidate_seconds": round(runtime, 6),
                         "serving": {"accepted": current["status"] == "comparable", "regions": current["regions"]},
                         "previous_inset": arm(previous, previous_mask, truth, expected),
                         "boundary_witness": arm(result, mask, truth, expected)})
    return rows


def report(rows):
    by_arm = {}
    for label in ARMS:
        strata = {}
        for group in SOURCE_COUNTS:
            stratum = [row for row in rows if row["key"][0] == group]
            results = {role: [row[label] for row in stratum if row["key"][-1] == role] for role in ROLES}
            strata[group] = {"edits_located": sum(row.get("useful", False) for row in results["edit"]),
                             "clean_matches": sum(row.get("exact_geometry", False) for row in results["clean"]),
                             "local_tone_locations": sum(row.get("useful", False) for row in results["local_tone"]),
                             "clean_highlights": sum(bool(row["regions"]) for row in results["clean"]),
                             "wrong_locations": sum(row["accepted"] and not row.get("exact_geometry", False)
                                                    for role in MATCHED for row in results[role]),
                             "incompatible_acceptances": sum(row["accepted"] for role in set(ROLES) - MATCHED for row in results[role]),
                             "edit_pairs_comparable": sum(row["accepted"] for row in results["edit"])}
        by_arm[label] = strata
    results = by_arm["boundary_witness"]
    lama = results["LaMa"]
    others = [results[family] for family in ("sdxl", "kandinsky")]
    passed = (lama["edits_located"] >= GATE["minimum_lama_edit_locations"]
              and sum(row["edits_located"] for row in others) >= GATE["minimum_other_edit_locations"]
              and lama["clean_matches"] >= GATE["minimum_lama_clean_matches"]
              and sum(row["clean_matches"] for row in others) >= GATE["minimum_other_clean_matches"]
              and all(sum(row[field] for row in results.values()) <= GATE[f"maximum_{field}"]
                      for field in ("wrong_locations", "clean_highlights"))
              and sum(row["incompatible_acceptances"] for row in results.values()) <= GATE["maximum_incompatible_matches"])
    times = [row["candidate_seconds"] for row in rows]
    return {"by_arm": by_arm, "passed": passed, "views": len(rows),
            "median_seconds": median(times), "maximum_seconds": max(times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "run", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--alternate", type=Path, required=True)
    paths = {"primary": "image-bg-real-acquisition-v225", "backup": "image-bg-real-backup-intake-v262",
             "legacy": "image-residual-patch-held-2026-11-11",
             "development": "image-reference-embedded-screenshot-2026-11-12",
             "confirmation": "image-reference-embedded-screenshot-confirmation-2026-11-12"}
    for name, default in paths.items():
        parser.add_argument(f"--{name}", type=Path, default=earlier.ROOT / default)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "prepare":
        prepare(args)
        return
    commitment = args.run / "freeze.json"
    if args.stage == "freeze":
        require(not commitment.exists(), "This holdout is already frozen")
        write(commitment, frozen(args))
        print(file_sha(commitment), flush=True)
        return
    held = args.run / "held.json"
    require(frozen(args) == read(commitment), "Holdout inputs or analysis method differ from freeze")
    rows = measure(args)
    if args.stage == "review":
        prior = read(held)
        for row, earlier_row in zip(rows, prior, strict=True):
            del row["candidate_seconds"], earlier_row["candidate_seconds"]
        require(rows == prior, "Held placements, masks or comparator decisions differ")
        print(f"Reproduced {len(rows)} frozen screenshot comparisons", flush=True)
        return
    require(not held.exists(), "Held outputs already exist")
    write(held, rows)
    result = {"schema": SCHEMA, "freeze_sha256": file_sha(commitment), "held_sha256": file_sha(held),
              "report": report(rows), "confidence_weight": 0}
    write(args.run / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
