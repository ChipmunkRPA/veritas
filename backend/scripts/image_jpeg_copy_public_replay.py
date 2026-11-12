# SPDX-License-Identifier: Apache-2.0
"""Reproduce the shipped JPEG copy checks using a fixed public-photo roster."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, features

from app.detectors.image import copy_move, copy_similarity, jpeg_copy_blocks, jpeg_dq, jpeg_offgrid_copy
from app.evidence import image_jpeg_copy_blocks_source, image_jpeg_offgrid_source, image_packets_v7
from app.evidence.image_copy_similarity_binding import (
    JPEG_BLOCK_BASIS,
    JPEG_OFFGRID_VERSION,
    OFFGRID_BASIS,
    ROLES,
)
from app.reports import verify_image_source
from app.reports.public_verify import SCHEMA as PUBLIC_PACKAGE_SCHEMA
from scripts import image_affine_native_scale as publisher
from scripts import image_jpeg_copy_blocks_transfer as original
from scripts import image_offgrid_patch_transfer as offgrid

SCHEMA = "chipmunk.public-jpeg-copy-replay/1"
ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "docs/evaluation-panels/image-affine-native-scale-2026-11-11.json"
SLOTS = (0, 2, 4, 5, 6, 8, 10, 11)
QUALITIES = (75, 90)
EDITED = ("aligned_64", "offgrid_40x48", "offgrid_64")
require, digest = original.require, original.digest


def _json(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _save(path: Path, value) -> None:
    with path.open("xb") as output:
        output.write(_json(value))


def roster() -> list[dict]:
    receipt = json.loads(ROSTER.read_bytes())
    published = receipt["source_roster"]["sources"]
    intake = {row["slot"]: row for row in receipt["source_intake"]["attempts"]}
    rows = []
    for slot in SLOTS:
        source, delivery = published[slot], intake[slot]
        url = source[f"{delivery['delivery']}_url"]
        endpoint = urlsplit(url)
        require(source["slot"] == slot and publisher.LICENSE.fullmatch(source["licence_url"])
                and bool(source["author"]) and len(source["original_sha1"]) == 40
                and delivery["status"] == "qualified" and delivery["delivery"] in {"original", "thumb"}
                and endpoint.scheme == "https" and endpoint.hostname in {"upload.wikimedia.org", "thumb.wikimedia.org"},
                "Public source is outside the pinned commercial-permissive photo roster")
        row = {key: source[key] for key in ("slot", "category", "page", "title", "author", "licence", "licence_url",
                                            "original_sha1", "source_dimensions")}
        rows.append({**row, "delivery": delivery["delivery"], "url": url, "source_sha256": delivery["sha256"],
                     "file": f"photo-{slot:02d}.jpg"})
    require(len({row["author"].casefold() for row in rows}) == len(SLOTS), "Require distinct credited contributors")
    return rows


def _source(path: Path, source: dict) -> tuple[bytes, Image.Image]:
    with path.open("rb") as stream:
        data = stream.read(publisher.MAX_DOWNLOAD + 1)
    require(0 < len(data) <= publisher.MAX_DOWNLOAD and digest(data) == source["source_sha256"],
            "The Wikimedia delivery differs from the published SHA-256")
    if source["delivery"] == "original":
        require(hashlib.sha1(data).hexdigest() == source["original_sha1"], "The publisher original SHA-1 differs")
    with Image.open(io.BytesIO(data), formats=["JPEG"]) as raw:
        require(raw.width * raw.height <= 45_000_000, "The source exceeded the photo decode bound")
        if source["delivery"] == "original":
            require(list(raw.size) == source["source_dimensions"], "The publisher original dimensions differ")
        image = ImageOps.contain(ImageOps.exif_transpose(raw).convert("RGB"), (768, 768), Image.Resampling.LANCZOS)
    require(min(image.size) >= 480 and max(image.size) == 768, "The photo cannot make the fixed JPEG input size")
    return data, image


def acquire(folder: Path) -> None:
    folder.mkdir(mode=0o700, exist_ok=True)
    for source in roster():
        path = folder / source["file"]
        if not path.exists():
            data = publisher._fetch(source["url"])
            require(digest(data) == source["source_sha256"], "Downloaded source digest differs")
            with path.open("xb") as output:
                output.write(data)
        data, _ = _source(path, source)
        print(f"Verified public {source['delivery']} {source['slot']:02d}: {digest(data)}", flush=True)


def variants(image: Image.Image, index: int):
    blurred = image.copy()
    box = (72, 81, 142, 157)
    blurred.paste(image.filter(ImageFilter.GaussianBlur(1.6)).crop(box), box[:2])
    yield "ordinary", image, None, None
    yield "contrast", ImageEnhance.Contrast(image).enhance(1.12), None, None
    yield "blur", blurred, None, None
    yield "aligned_64", *original.edit(image, index, (64, 64), False)
    yield "offgrid_40x48", *offgrid.authored(image, index + 29, (40, 48))
    yield "offgrid_64", *offgrid.authored(image, index + 29, (64, 64))


def views(image: Image.Image, index: int, slot: int):
    for treatment, pixels, truth, changed in variants(image, index):
        for quality in QUALITIES:
            body = original.encode(pixels, quality)
            yield {"slot": slot, "quality": quality, "treatment": treatment, "input_sha256": digest(body),
                   "truth": [list(box) for box in truth] if truth else None,
                   "changed_mask_sha256": digest(changed.tobytes()) if changed is not None else None}, body, changed


def freeze(folder: Path) -> dict:
    sources, planned = [], []
    for index, source in enumerate(roster()):
        _, image = _source(folder / source["file"], source)
        sources.append({**source, "working_rgb_sha256": digest(image.tobytes())})
        planned.extend(row for row, _, _ in views(image, index, source["slot"]))
    modules = (copy_move, copy_similarity, jpeg_copy_blocks, jpeg_dq, jpeg_offgrid_copy,
               image_jpeg_copy_blocks_source, image_jpeg_offgrid_source, verify_image_source, original, offgrid)
    pins = {Path(module.__file__).name: digest(Path(module.__file__).read_bytes()) for module in modules}
    pins.update({fn.__name__: digest(inspect.getsource(fn).encode()) for fn in (
        image_packets_v7.selected_translation_pairs, image_packets_v7.translation_pairs)})
    return {"schema": SCHEMA, "code": pins | {"runner": digest(Path(__file__).read_bytes())}, "sources": sources,
            "roster_receipt_sha256": digest(ROSTER.read_bytes()), "views": planned,
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": version("pillow"),
                        "jpeglib": version("jpeglib"), "jpeg": features.version_codec("jpg")},
            "useful_iou_each": .5, "changed_pixel_recall": .5, "confidence_weight": 0}


def _verify_archive(evidence: dict, assets: dict[str, bytes], source: Path, sha: str) -> str:
    zip_bytes = io.BytesIO()
    with ZipFile(zip_bytes, "w") as archive:
        archive.writestr("evidence.json", _json(evidence))
        for key, value in assets.items():
            archive.writestr(f"assets/{key}.png", value)
    bundle = {"schema": PUBLIC_PACKAGE_SCHEMA, "analysis": {"kind": "image"}, "source_commitment": {"sha256": sha}}
    with ZipFile(zip_bytes) as archive:
        return verify_image_source.check(archive, bundle, source)


def _native_proof(image, body, pairs, basis, measured, source, tamper):
    evidence = {"version": JPEG_OFFGRID_VERSION, "packets": [], "copy_similarity": {"pairs": []},
                "copy_translation": {"basis": basis, "jpeg_offgrid_review": measured, "pairs": []},
                "source": {"coordinate_space": "exif-oriented-original", "width": image.width,
                           "height": image.height, "exif_orientation": 1}}
    assets = {}
    for index, boxes in enumerate(pairs, 1):
        pair = {"id": f"T{index:02d}", "regions": {}, "artifacts": {}}
        for role, (x, y, w, h) in zip(ROLES, boxes, strict=True):
            crop = image.crop((x, y, x + w, y + h))
            with io.BytesIO() as buffer:
                crop.save(buffer, "PNG")
                payload = buffer.getvalue()
            pin = digest(payload)
            assets[pin] = payload
            pair["regions"][role] = dict(x=x, y=y, w=w, h=h)
            pair["artifacts"][role] = {"sha256": pin, "rgb_sha256": digest(crop.tobytes())}
        evidence["copy_translation"]["pairs"].append(pair)
    source.write_bytes(body)
    detail = _verify_archive(evidence, assets, source, digest(body))
    require(f"{2 * len(pairs)} unannotated crops matched" in detail, "Portable native crop verification did not run")
    if basis == JPEG_BLOCK_BASIS:
        require("matching encoded JPEG brightness blocks" in detail, "Exact block verification did not run")
    if basis == OFFGRID_BASIS:
        require("Recomputed the recorded color differences" in detail, "Exact RGB/perimeter verification did not run")
    if basis not in tamper:
        moved = copy.deepcopy(evidence)
        region = moved["copy_translation"]["pairs"][0]["regions"]["area_a"]
        region["x"] += 1 if region["x"] + region["w"] < image.width else -1
        _reject(moved, assets, source, digest(body), "pixels at its recorded source coordinates")
        if basis == OFFGRID_BASIS:
            altered = copy.deepcopy(evidence)
            altered["copy_translation"]["jpeg_offgrid_review"]["measurement"]["interior_mean"] += .01
            _reject(altered, assets, source, digest(body), "differs from the original crop")
        tamper.add(basis)
    return evidence["copy_translation"]["pairs"]


def _reject(evidence, assets, source, sha, expected):
    try:
        _verify_archive(evidence, assets, source, sha)
    except ValueError as error:
        require(expected in str(error), "Native verification rejected the wrong condition")
        return
    raise ValueError("Native verification accepted altered evidence")


def score(image, row, body, changed, source_path, tamper):
    rgb = np.asarray(image)
    stage, pairs, basis = image_packets_v7.selected_translation_pairs(rgb)
    baseline = pairs
    status, block_status, measured = stage, None, None
    if stage == "no_pair":
        block_status, added = jpeg_copy_blocks.analyse(body)
        if added:
            status, pairs, basis = "supported_pair", added, JPEG_BLOCK_BASIS
        elif block_status == "no_pair":
            measured, added = image_jpeg_offgrid_source.review(body, rgb)
            if added:
                status, pairs, basis = "supported_pair", added, OFFGRID_BASIS
    truth = row["truth"]
    witness = _native_proof(image, body, pairs, basis, measured, source_path, tamper) if pairs else []
    return {**row, "baseline_useful": original.match(baseline, truth, changed),
            "serving_useful": original.match(pairs, truth, changed), "status": status,
            "translation_status": stage, "jpeg_block_status": block_status, "offgrid": measured,
            "basis": basis, "verified_pairs": witness}


def measure(folder: Path, frozen: dict) -> dict:
    rows, tamper = [], set()
    committed = iter(frozen["views"])
    with TemporaryDirectory(dir=folder, prefix="native-check-") as scratch:
        source_path = Path(scratch) / "source.jpg"
        for index, source in enumerate(frozen["sources"]):
            _, working = _source(folder / source["file"], source)
            for row, body, changed in views(working, index, source["slot"]):
                require(row == next(committed), "Prepared input differs from the pre-score freeze")
                with Image.open(io.BytesIO(body), formats=["JPEG"]) as decoded:
                    rows.append(score(decoded.convert("RGB"), row, body, changed, source_path, tamper))
    require(next(committed, None) is None, "Unmeasured frozen input remains")
    summary = {}
    for row in rows:
        key = f"{row['quality']}/{row['treatment']}"
        bucket = summary.setdefault(key, Counter())
        bucket.update(views=1, baseline_useful=row["baseline_useful"], serving_useful=row["serving_useful"],
                      flagged=bool(row["verified_pairs"]), verified_crop_assets=2 * len(row["verified_pairs"]))
        if row["basis"]:
            bucket[row["basis"]] += 1
    ordinary = [row for row in rows if row["treatment"] not in EDITED]
    edited = [row for row in rows if row["treatment"] in EDITED]
    totals = {"ordinary": len(ordinary), "ordinary_flagged": sum(bool(row["verified_pairs"]) for row in ordinary),
              "edited": len(edited), "baseline_useful": sum(row["baseline_useful"] for row in edited),
              "serving_useful": sum(row["serving_useful"] for row in edited),
              "verified_crop_assets": sum(2 * len(row["verified_pairs"]) for row in rows)}
    return {"schema": SCHEMA, "rows": rows, "summary": summary, "totals": totals, "rows_sha256": digest(_json(rows)),
            "tamper_rejected_by_basis": sorted(basis for basis in tamper if basis)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "freeze", "run", "review"))
    parser.add_argument("folder", type=Path, help="Scratch folder for licensed source files and replay receipts")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.action == "acquire":
        acquire(args.folder)
        return
    plan = args.folder / "freeze.json"
    if args.action == "freeze":
        _save(plan, freeze(args.folder))
        print(f"Frozen {digest(plan.read_bytes())}")
        return
    frozen = json.loads(plan.read_bytes())
    require(_json(frozen) == _json(freeze(args.folder)), "Source, implementation, truth, or runtime changed after the freeze")
    measured = measure(args.folder, frozen)
    result = args.folder / "result.json"
    if args.action == "review":
        require(_json(measured) == result.read_bytes(), "Replay changed the source measurements or crop evidence")
    else:
        _save(result, measured)
    print(_json({key: value for key, value in measured.items() if key != "rows"}).decode(), end="")


if __name__ == "__main__":
    main()
