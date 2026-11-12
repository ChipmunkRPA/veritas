"""Challenge the tone-corrected copy check on ordinary photos with sparse proposals."""

from __future__ import annotations

import argparse
import io
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.detectors.image import copy_affine, copy_move
from scripts import image_copy_tone_transfer as transfer

SCHEMA = "chipmunk.image-copy-tone-ordinary-stress/1"
LICENSES = {"https://creativecommons.org/licenses/by/2.0/", "https://creativecommons.org/licenses/by-sa/2.0/"}
DELIVERIES = ("rgb", "jpeg75")


def freeze(cache):
    raw = cache.read_bytes()
    source_cache = json.loads(raw)
    indexed = {row["key"]: row for row in source_cache["sources"]}
    roster = []
    for component in source_cache["candidate_only_unlinked_components"]:
        candidates = [indexed[key] for key in component]
        if not all(row.get("license_url") in LICENSES and "source_jpeg_sha256" in row for row in candidates):
            raise ValueError("A qualified component lacks source and reuse terms")
        row = min(candidates, key=lambda value: value["source_jpeg_sha256"])
        width, height = row["size"]
        if max(width, height) < 513 or round(min(width, height) * min(1., 1024 / max(width, height))) < 384:
            continue
        roster.append({"path": row["source_jpeg_file"], "sha256": row["source_jpeg_sha256"]})
    paths = (Path(__file__), Path(transfer.__file__), Path(copy_affine.__file__), Path(copy_move.__file__))
    return {"schema": SCHEMA, "cache_sha256": transfer.digest(raw), "roster": roster, "deliveries": DELIVERIES,
            "code_sha256": {path.name: transfer.digest(path.read_bytes()) for path in paths},
            "selection": "Smallest JPEG digest per previously qualified source component with supported dimensions. "
                         "Scan both fixed deliveries with sparse proposals only before running dense comparisons."}


def open_source(item):
    raw = Path(item["path"]).read_bytes()
    if transfer.digest(raw) != item["sha256"]:
        raise ValueError("Source bytes changed")
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        if min(image.size) < 384 or max(image.size) < 513:
            raise ValueError("Decoded source geometry differs from the eligibility roster")
        return image


def decode(image, fmt):
    if fmt == "rgb":
        return np.asarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75, subsampling=2)
    with Image.open(buffer) as jpeg:
        return np.asarray(jpeg.convert("RGB"))


def screen(plan, output):
    counts = {fmt: Counter() for fmt in DELIVERIES}
    shortlisted = []
    for slot, item in enumerate(plan["roster"]):
        image = open_source(item)
        for fmt in DELIVERIES:
            pixels = decode(image, fmt)
            count = counts[fmt]
            count["views"] += 1
            try:
                result = copy_move.analyse(pixels, max_candidate_pairs=copy_affine.MAX_CANDIDATES)
            except copy_move.CandidateBudgetExceeded:
                count["unassessed_match_budget"] += 1
                continue
            count["assessed"] += 1
            count["repetitive_image"] += result.repetitive_image
            count["ordinary_pair_proposals"] += bool(result.distinct)
            if result.distinct:
                shortlisted.append({"slot": slot, "format": fmt, "input_sha256": transfer.digest(pixels.tobytes()),
                                    "proposals": len(result.distinct[:copy_affine.MAX_PAIRS])})
        if (slot + 1) % 25 == 0:
            print(f"{slot + 1} ordinary source groups screened", flush=True)
    transfer.write(output / "screen.json", {"schema": SCHEMA, "freeze_sha256": sha(output / "freeze.json"),
                                            "counts": counts, "shortlist": shortlisted})


def sha(path):
    return transfer.digest(path.read_bytes())


def compare(plan, output):
    path = output / "screen.json"
    selected = json.loads(path.read_bytes())
    if selected["freeze_sha256"] != sha(output / "freeze.json"):
        raise ValueError("Sparse screen does not bind the frozen procedure")
    counts = {fmt: Counter() for fmt in DELIVERIES}
    records = []
    for row in selected["shortlist"]:
        item = plan["roster"][row["slot"]]
        fmt = row["format"]
        pixels = decode(open_source(item), fmt)
        if transfer.digest(pixels.tobytes()) != row["input_sha256"]:
            raise ValueError("Decoded shortlisted bytes differ from the frozen sparse screen")
        empty = np.zeros(pixels.shape[:2], bool)
        result = transfer.score(pixels, (empty, empty))
        if result["status"] != "assessed" or result["proposals"] != row["proposals"]:
            raise ValueError("The shortlisted sparse proposals did not reproduce")
        counts[fmt].update(shortlisted=1, incumbent_flag=bool(result["incumbent"]),
                           additional_pair=bool(result["added"]),
                           additional_flag=bool(result["added"]) and not result["incumbent"])
        records.append({**row, **result})
    transfer.write(output / "comparison.json", {"schema": SCHEMA, "screen_sha256": sha(path),
                                                "counts": counts, "records": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "screen", "compare"))
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = freeze(args.cache)
    file = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir()
        transfer.write(file, plan)
        print(f"{len(plan['roster'])} ordinary groups frozen under {sha(file)}", flush=True)
        return
    if json.loads(file.read_bytes()) != json.loads(json.dumps(plan)):
        raise ValueError("Source roster or code changed after freeze")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    started = time.monotonic()
    (screen if args.stage == "screen" else compare)(plan, args.output)
    print(f"{args.stage} completed in {time.monotonic() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
