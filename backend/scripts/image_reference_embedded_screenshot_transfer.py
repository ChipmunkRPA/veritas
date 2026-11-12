# SPDX-License-Identifier: Apache-2.0
"""Measure exact screenshot containment on known local edits and controls."""

from __future__ import annotations

import argparse
import io
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.evidence import image_reference
from scripts import image_native_tile_agreement_transfer as prior
from scripts import image_reference_embedded_screenshot as candidate

SCHEMA = "chipmunk.image-reference-embedded-screenshot-transfer/1"
ROOT = Path("backend/data/independent-evaluation")
GATE = {"browser_correct_edits": 7, "alternate_correct_edits": 4, "browser_exact_clean": 7,
        "alternate_exact_clean": 4, "incorrect_placements": 0, "incompatible_acceptances": 0}
SOURCE_EXPECTED = {"browser": 12, "alternate": 6}
ALTERNATE_ROLES = ("clean", "edit", "local_tone", "global_tone", "resized", "unrelated", "duplicated", "jpeg", "chrome")
BROWSER_ROLES = ("clean", "edit", "chrome")
COMPARABLE = {"clean", "edit", "local_tone"}
require, read, write, sha, file_sha = prior.require, prior.read, prior.write, prior.sha, prior.file_sha


def bounds(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


def earlier_photos(args):
    if args.confirmation:
        roster_path = args.backup / "qualified-private.json"
        require(file_sha(roster_path) == read(args.backup / "summary.json")["qualified_private_sha256"], "Backup intake changed")
        backup = read(roster_path)
        cache, records = Path(backup["cache"]), backup["qualified"]
    else:
        cache, used = prior.old.primary_roster(args)
        used = {row["source"]["flickr_photo_id"] for row in used}
        records = read(args.primary / "qualified-private.json")["qualified"]
        records = [row for row in records if row["source"]["flickr_photo_id"] not in used]
    require(len(records) == SOURCE_EXPECTED["browser"], "Wrong roster for the retained Chromium capture")
    for row in records:
        images = []
        for role in ("original", "edited"):
            asset = row["measurement"]["files"][role]
            path = cache / asset["file"]
            require(file_sha(path) == asset["sha256"], "Qualified earlier image changed")
            images.append(prior.BASE.pixels(path))
        original, edited = images
        known = (original != edited).any(axis=2)
        yield row["source"], original, prior.BASE.feather(original, edited, known)


def prepare_browser(args):
    args.run.mkdir(mode=0o700)
    args.browser.mkdir(mode=0o700)
    assets = args.browser / "photo-fixtures"
    assets.mkdir()
    plan = []
    for slot, (_, earlier, edited) in enumerate(earlier_photos(args)):
        truth = (earlier != edited).any(axis=2)
        left, top, right, bottom = prior.BASE.source_box(truth)
        height, width = truth.shape
        w, h = min(width, 472 + (slot % 4) * 6), min(height, 420 + (slot % 3) * 17)
        a, b = max(0, min(width-w, (left+right-w)//2)), max(0, min(height-h, (top+bottom-h)//2))
        x, y = 60 + (slot * 47) % 161, 70 + (slot * 29) % 87
        crop = np.s_[b:b+h, a:a+w]
        require(truth[crop].sum() >= .8 * truth.sum(), "The confirmation screenshot dropped too much of the edit")
        plan.append({"slot": slot, "layout": (slot + 1) % 4, "width": 768, "height": 640,
                     "x": x, "y": y, "w": w, "h": h, "source_box": [a, b, a + w, b + h]})
        for role, pixels in (("photo", earlier), ("edit", edited)):
            (assets / f"{slot}-{role}.png").write_bytes(prior.BASE.png(pixels[crop]))
    write(args.browser / "layout-plan.json", plan)


def retained_browser(args):
    plan, captures = (read(args.browser / name) for name in ("layout-plan.json", "captures.json"))
    rows = {(row["slot"], row["role"]): row for row in captures["rows"]}
    require(len(rows) == len(plan) * 3, "Wrong Chromium capture roster")
    for item, (source, original, edited) in zip(plan, earlier_photos(args), strict=True):
        x, y, w, h = (item[key] for key in ("x", "y", "w", "h"))
        a, b, right, bottom = item["source_box"]
        prior_rect, screen_rect = bounds(a, b, w, h), bounds(x, y, w, h)
        earlier = original[b:bottom, a:right]
        for role, capture in zip(BROWSER_ROLES, ("photo", "edit", "template"), strict=True):
            path = args.browser / "captures" / f"{item['slot']}-{capture}.png"
            body, rgb = path.read_bytes(), prior.BASE.pixels(path)
            require(sha(body) == rows[item["slot"], capture]["png_sha256"], "Chromium capture bytes differ")
            if role in COMPARABLE:
                expected = (edited if role == "edit" else original)[b:bottom, a:right]
                require(np.array_equal(expected, rgb[y:y+h, x:x+w]), "Chromium did not preserve native photo pixels")
            truth = np.zeros(rgb.shape[:2], dtype=bool)
            if role == "edit":
                truth[y:y+h, x:x+w] = (earlier != expected).any(axis=2)
            yield ("browser", item["slot"], role), source, original, body, rgb, truth, (prior_rect, screen_rect)


def canvas(patch, slot, role):
    width, height = (1190, 660) if role == "duplicated" else (970, 640)
    palettes = (("#e7edf4", "#314765", "#fcfdfd"), ("#252329", "#563d60", "#e5ebea"),
                ("#edede8", "#8d6d3d", "#f8f4e8"))
    bg, accent, panel = palettes[slot % len(palettes)]
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 56), fill=accent)
    draw.text((24, 20), "Local review  /  Screenshot fixture", fill=panel)
    draw.rectangle((width - 210, 92, width - 28, 236), fill=panel)
    for i in range(5):
        draw.text((width - 196, 110 + i * 21), f"Archive entry {slot + 1}/{i + 2}", fill=accent)
    x, y = 88 + (slot * 47) % 157, 92 + (slot * 31) % 83
    if role == "duplicated":
        x = 63
        image.paste(Image.fromarray(patch), (width - patch.shape[1] - 61, y))
    if role != "chrome":
        image.paste(Image.fromarray(patch), (x, y))
    with io.BytesIO() as out:
        image.save(out, format="JPEG" if role == "jpeg" else "PNG", quality=85)
        body = out.getvalue()
    with Image.open(io.BytesIO(body)) as decoded:
        return body, np.asarray(decoded), (x, y)


def alternate_screens(args):
    archive = args.alternate if args.confirmation else args.legacy
    options = list(prior.BASE.REGIONAL.paired.inputs(archive))
    sources = {row["slot"]: row for row in read(archive / "plan.json")["cases"]}
    split = {family: sorted((row for row in options if row[0]["generator"] == family), key=lambda row: row[0]["slot"])[:3]
             for family in ("sdxl", "kandinsky")}
    roster = split["sdxl"] + split["kandinsky"]
    require(len(roster) == SOURCE_EXPECTED["alternate"], "Expected three source groups per alternate editor")
    for slot, (source, images, _) in enumerate(roster):
        variants = prior.BASE.REGIONAL.native_variants(images, source["box"])
        original, edit = (variants[key] for key in ("original", "feathered_edit"))
        known = (original != edit).any(axis=2)
        left, top, right, bottom = prior.BASE.source_box(known)
        height, width = known.shape
        w, h = min(width, 415 + slot * 5), min(height, 357 + slot * 9)
        a, b = max(0, min(width - w, (left + right - w) // 2)), max(0, min(height - h, (top + bottom - h) // 2))
        crop = np.s_[b:b+h, a:a+w]
        require(known[crop].sum() == known.sum(), "Stress screenshot cropped out the known edit")
        plain, blend = original[crop], edit[crop]
        tone = plain.copy()
        tone[known[crop]] = tone[known[crop]] * .82
        resized = np.asarray(Image.fromarray(plain).resize((round(w * .88), round(h * .88)), Image.Resampling.BICUBIC))
        donor = roster[(slot + 1) % len(roster)][1]["original"][:h, :w]
        for role, patch in zip(ALTERNATE_ROLES, (plain, blend, tone, (plain * .85 + 8).astype(np.uint8), resized,
                                                donor, plain, plain, plain), strict=True):
            body, rgb, (x, y) = canvas(patch, slot, role)
            truth = np.zeros(rgb.shape[:2], dtype=bool)
            if role in {"edit", "local_tone"}:
                truth[y:y+h, x:x+w] = (plain != patch).any(axis=2)
            yield ("alternate", slot, role), sources[source["slot"]], original, body, rgb, truth, (bounds(a, b, w, h), bounds(x, y, w, h))


def cases(args):
    yield from retained_browser(args)
    yield from alternate_screens(args)


def identity(case):
    key, source, original, body, rgb, truth, expected = case
    return {"key": list(key), "source_group": sha(str(source["flickr_photo_id"]).encode()),
            "contributor": sha(str(source["flickr_owner"]).strip().casefold().encode()),
            "family": source.get("generator", "LaMa"), "reference_rgb": sha(original.tobytes()),
            "screen_delivery": sha(body), "screen_rgb": sha(rgb.tobytes()), "target": int(truth.sum()),
            "truth": sha(truth.tobytes()), "geometry": list(expected) if key[-1] in COMPARABLE else None}


def freeze(args):
    roster = [identity(case) for case in cases(args)]
    groups = {tuple(row["key"][:2]): row for row in roster if row["key"][-1] == "clean"}
    require(len(groups) == sum(SOURCE_EXPECTED.values()), "Source group count differs")
    for field in ("source_group", "contributor", "reference_rgb"):
        require(len({row[field] for row in groups.values()}) == len(groups), "Photographs or credited contributors overlap")
    if args.confirmation:
        exposed = read(args.development / "freeze.json")["cases"]
        for field in ("source_group", "contributor", "reference_rgb"):
            require(not {row[field] for row in groups.values()} & {row[field] for row in exposed},
                    "Confirmation and development sources overlap")
    require({tuple(row["key"]) for row in roster} == {(panel, i, role) for panel, count in SOURCE_EXPECTED.items()
            for i in range(count) for role in (BROWSER_ROLES if panel == "browser" else ALTERNATE_ROLES)}, "Incomplete panel")
    publisher = args.backup if args.confirmation else args.primary
    alternate = args.alternate if args.confirmation else args.legacy
    bound = (Path(__file__), Path(candidate.__file__), Path(candidate.exact.__file__), Path(candidate.reference_compare.__file__),
             Path(image_reference.__file__), publisher / "qualified-private.json", alternate / "attempts.jsonl", alternate / "plan.json",
             args.browser / "layout-plan.json", args.browser / "captures.json", Path(prior.__file__).with_name("capture_image_crop_multiscale_transfer.cjs"))
    return {"schema": SCHEMA, "gate": GATE, "cases": roster, "confidence_weight": 0,
            "minimum_interior": {"corroborated": candidate.CORROBORATED_INTERIOR_AGREEMENT, "strict": candidate.STRICT_INTERIOR_AGREEMENT},
            "development_commitment": file_sha(args.development / "freeze.json") if args.confirmation else None,
            "proof_scope": "single native-scale rectangular photo inset in upright RGB PNG, unique integer placement",
            "code_and_sources": {path.name: file_sha(path) for path in bound}}


def compare_case(case, directory):
    key, _, original, body, rgb, truth, expected = case
    started = time.monotonic()
    with Image.open(io.BytesIO(body)) as carrier:
        result, mask = candidate.compare(original, rgb, carrier)
        elapsed = time.monotonic() - started
        strict, strict_mask = candidate.compare(original, rgb, carrier, minimum_interior=candidate.STRICT_INTERIOR_AGREEMENT)
    before, after = directory / "earlier.png", directory / "submitted.img"
    before.write_bytes(prior.BASE.png(original))
    after.write_bytes(body)
    baseline = image_reference.inspect(before, after, allow_integer_alignment=True, allow_proportional_resize=True,
                                       allow_oriented_crop=True, allow_oriented_resize=True)
    proof = None if result is None else result["embedded_native_photo"]
    overlap = 0 if mask is None else int((mask & truth).sum())
    selected = 0 if mask is None else int(mask.sum())
    target = int(truth.sum())
    useful = bool(target and 2 * overlap >= target and 10 * overlap >= 3 * (target + selected - overlap))
    strict_proof = None if strict is None else strict["embedded_native_photo"]
    strict_selected = 0 if strict_mask is None else int(strict_mask.sum())
    strict_overlap = 0 if strict_mask is None else int((strict_mask & truth).sum())
    strict_useful = bool(target and 2 * strict_overlap >= target and 10 * strict_overlap >= 3 * (target + strict_selected - strict_overlap))
    return {"key": list(key), "candidate_s": round(elapsed, 6), "proof": proof, "regions": [] if result is None else result["regions"],
            "exact_geometry": bool(proof and (proof["reference_bounds"], proof["submitted_bounds"]) == expected),
            "mask_sha256": None if mask is None else sha(mask.tobytes()), "selected": selected, "overlap": overlap,
            "strict_proof": strict_proof, "strict_selected": strict_selected, "strict_overlap": strict_overlap,
            "strict_exact_geometry": bool(strict_proof and (strict_proof["reference_bounds"], strict_proof["submitted_bounds"]) == expected),
            "strict_useful": strict_useful,
            "target": target, "useful": useful, "current_status": baseline["status"], "current_regions": len(baseline["regions"])}


def summarize(rows):
    tallies = defaultdict(lambda: {"views": 0, "proofs": 0, "exact_locations": 0, "useful": 0,
                                  "current_comparable": 0, "current_with_regions": 0, "with_regions": 0,
                                  "strict_proofs": 0, "strict_useful": 0})
    for row in rows:
        panel, _, role = row["key"]
        tally = tallies[f"{panel}/{role}"]
        tally["views"] += 1
        tally["proofs"] += row["proof"] is not None
        tally["exact_locations"] += row["exact_geometry"]
        tally["useful"] += row["exact_geometry"] and row["useful"]
        tally["current_comparable"] += row["current_status"] == "comparable"
        tally["current_with_regions"] += bool(row["current_regions"])
        tally["with_regions"] += bool(row["selected"])
        if "strict_useful" in row:
            tally["strict_proofs"] += row["strict_proof"] is not None
            tally["strict_useful"] += row["strict_exact_geometry"] and row["strict_useful"]
    incorrect = sum(row["proof"] is not None and not row["exact_geometry"] for row in rows if row["key"][-1] in COMPARABLE)
    incompatible = sum(row["proof"] is not None for row in rows if row["key"][-1] not in COMPARABLE)
    runtime = sorted(row["candidate_s"] for row in rows)
    clean = all(tallies[f"{panel}/clean"]["with_regions"] == 0 for panel in SOURCE_EXPECTED)
    passed = (incorrect <= GATE["incorrect_placements"] and incompatible <= GATE["incompatible_acceptances"] and clean and
              all(tallies[f"{panel}/edit"]["useful"] >= GATE[f"{panel}_correct_edits"] and
                  tallies[f"{panel}/clean"]["exact_locations"] >= GATE[f"{panel}_exact_clean"] for panel in SOURCE_EXPECTED))
    return {"by_role": dict(tallies), "incorrect_placements": incorrect, "incompatible_acceptances": incompatible,
            "clean_highlights": sum(tallies[f"{panel}/clean"]["with_regions"] for panel in SOURCE_EXPECTED),
            "candidate_median_seconds": runtime[len(runtime) // 2], "candidate_max_seconds": runtime[-1], "passed": passed}


def parser():
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("stage", choices=("prepare-browser", "freeze", "run"))
    cmd.add_argument("run", type=Path)
    cmd.add_argument("--confirmation", action="store_true")
    cmd.add_argument("--alternate", type=Path)
    cmd.add_argument("--development", type=Path, default=ROOT / "image-reference-embedded-screenshot-2026-11-12")
    for field, location in {"primary": "image-bg-real-acquisition-v225", "backup": "image-bg-real-backup-intake-v262",
                            "legacy": "image-residual-patch-held-2026-11-11", "browser": "image-native-tile-agreement-2026-11-12"}.items():
        cmd.add_argument(f"--{field}", type=Path, default=ROOT / location)
    return cmd


def main():
    args = parser().parse_args()
    cv2.setNumThreads(1)
    frozen = args.run / "freeze.json"
    if args.stage == "prepare-browser":
        require(args.confirmation, "The original Chromium receipts are already retained")
        prepare_browser(args)
        return
    require(not args.confirmation or args.alternate is not None, "Provide the separate alternate editor archive")
    if args.stage == "freeze":
        if not args.confirmation:
            args.run.mkdir(mode=0o700)
        write(frozen, freeze(args))
        print(file_sha(frozen), flush=True)
        return
    committed = read(frozen)
    require(freeze(args) == committed, "Exact panel or method differs from freeze")
    with tempfile.TemporaryDirectory(dir=args.run) as tmp:
        rows = [compare_case(case, Path(tmp)) for case in cases(args)]
    held = args.run / "held.json"
    write(held, rows)
    report = summarize(rows)
    write(args.run / "result.json", {"schema": SCHEMA, "freeze_sha256": file_sha(frozen), "held_sha256": file_sha(held),
                                     "report": report, "confidence_weight": 0})
    print(report, flush=True)


if __name__ == "__main__":
    main()
