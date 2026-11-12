"""Freeze and replay a source-resolution geometric copy transfer."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from html import unescape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL import __version__ as pillow_version

from app.detectors.image import copy_affine, copy_affine_large, copy_move
from scripts import image_copy_tone_transfer as base

SCHEMA = "chipmunk.image-affine-native-scale/1"
CATEGORIES = ("architecture", "abandoned", "cityscapes", "building-exteriors", "lakes", "forests")
FORMATS = ("png", "jpeg75")
TREATMENTS = ("sharp", "feathered", *base.CONTROLS)
ARMS = ("source", "source_smoothed")
MAX_DOWNLOAD = 16 * 1024 * 1024
AUTHOR_TAGS = re.compile(r"<[^>]+>")
LICENSE = re.compile(r"https?://creativecommons\.org/licenses/(by|by-sa)/(2\.0|3\.0|4\.0)/?")


def sha(path: Path) -> str:
    return base.digest(path.read_bytes())


def plan(metadata: Path) -> dict:
    pools = {}
    for category in CATEGORIES:
        data = json.loads((metadata / f"{category}.json").read_bytes())
        candidates = []
        for page in data["query"]["pages"]:
            if not page.get("imageinfo"):
                continue
            item = page["imageinfo"][0]
            values = {name: value["value"] for name, value in item["extmetadata"].items()}
            rights = values.get("LicenseUrl", "").strip()
            author = " ".join(unescape(AUTHOR_TAGS.sub(" ", values.get("Artist", ""))).split())
            width, height = item["width"], item["height"]
            if (not LICENSE.fullmatch(rights) or not author or len(author) > 240 or item["mime"] != "image/jpeg"
                    or not 1400 <= max(width, height) <= 9000 or min(width, height) / max(width, height) < .47
                    or item["size"] > MAX_DOWNLOAD):
                continue
            candidates.append({"pageid": page["pageid"], "category": category, "title": page["title"],
                               "author": author, "licence": values["LicenseShortName"], "licence_url": rights,
                               "page": item["descriptionurl"], "original_url": item["url"],
                               "original_sha1": item["sha1"], "thumb_url": item["thumburl"],
                               "source_dimensions": [width, height]})
        pools[category] = sorted(candidates, key=lambda item: base.digest(("affine-scale-20261111/" + item["original_sha1"]).encode()))
    roster, authors, sources = [], set(), set()
    for _ in range(7):
        for category in CATEGORIES:
            while pools[category]:
                item = pools[category].pop(0)
                author = item["author"].casefold()
                if author in authors or item["original_sha1"] in sources:
                    continue
                roster.append({"slot": len(roster), **item})
                authors.add(author)
                sources.add(item["original_sha1"])
                break
    return {"schema": SCHEMA, "metadata_sha256": {name: sha(metadata / f"{name}.json") for name in CATEGORIES},
            "selection": "Up to seven distinct credited authors per Commons category in salted source order",
            "sources": roster}


def _fetch(url: str) -> bytes:
    host = urlsplit(url)
    hosts = {"upload.wikimedia.org", "thumb.wikimedia.org"}
    if host.scheme != "https" or host.hostname not in hosts:
        raise ValueError("Expected a public Wikimedia image URL")
    request = Request(url, headers={"User-Agent": "ChipmunkVerifyEvaluation/1.0 (local reproducibility study)"})
    with urlopen(request, timeout=25) as response:
        if urlsplit(response.url).hostname not in hosts:
            raise ValueError("Unexpected image host")
        content = response.read(MAX_DOWNLOAD + 1)
    if not 0 < len(content) <= MAX_DOWNLOAD:
        raise ValueError("Image exceeded the download budget")
    return content


def decode(raw: bytes) -> Image.Image:
    with Image.open(io.BytesIO(raw)) as source:
        if source.format != "JPEG" or max(source.size) > 9000 or source.width * source.height > 45_000_000:
            raise ValueError("Unexpected high-resolution JPEG geometry")
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
    if min(image.size) < 768 or max(image.size) <= 1024:
        raise ValueError("Source is outside the large-image comparison band")
    return image


def acquire(roster: dict, folder: Path):
    roster_sha256 = sha(folder / "plan.json")
    first = folder / "acquisition-first.json"
    prior = json.loads(first.read_bytes()) if first.exists() else None
    if prior is not None and prior["roster_sha256"] != roster_sha256:
        raise ValueError("The first acquisition journal belongs to a different roster")
    attempts = []
    rate_limited = set()
    for item in roster["sources"]:
        old = prior["attempts"][item["slot"]] if prior else None
        if old and old["status"] == "qualified":
            attempts.append(old)
            continue
        row = {"slot": item["slot"], "attempts": list(old["attempts"]) if old else []}
        # The first intake recognized only the original server, so the exact
        # publisher-provided thumbnail URLs have not been requested there.
        deliveries = ("thumb",) if old else ("thumb", "original")
        for delivery in deliveries:
            url = item[f"{delivery}_url"]
            host = urlsplit(url).hostname
            if host in rate_limited:
                row["attempts"].append({"delivery": delivery, "error": "rate_limited_host", "code": None})
                continue
            try:
                raw = _fetch(url)
                if delivery == "original" and hashlib.sha1(raw).hexdigest() != item["original_sha1"]:
                    raise ValueError("Publisher original digest changed")
                image = decode(raw)
            except (OSError, ValueError, URLError) as error:
                if isinstance(error, HTTPError) and error.code == 429:
                    rate_limited.add(host)
                row["attempts"].append({"delivery": delivery, "error": type(error).__name__,
                                        "code": getattr(error, "code", None)})
                continue
            name = f"source-{item['slot']:02d}.jpg"
            (folder / name).write_bytes(raw)
            row.update(status="qualified", name=name, delivery=delivery, sha256=base.digest(raw),
                       rgb_sha256=base.digest(image.tobytes()), size=list(image.size))
            break
        else:
            row["status"] = "unavailable"
        attempts.append(row)
        print(f"{row['slot']:02d} {row['status']}", flush=True)
    base.write(folder / "acquisition.json", {"roster_sha256": roster_sha256, "attempts": attempts})


def protocol(folder: Path):
    roster = json.loads((folder / "plan.json").read_bytes())
    acquired = json.loads((folder / "acquisition.json").read_bytes())
    if acquired["roster_sha256"] != sha(folder / "plan.json"):
        raise ValueError("Acquired sources differ from the fixed roster")
    sources = [{**a, **{key: item[key] for key in ("category", "author", "licence", "licence_url", "page")}}
               for a, item in zip(acquired["attempts"], roster["sources"], strict=True)
               if a["slot"] == item["slot"] and a["status"] == "qualified"]
    if len(sources) < 16 or len({source["sha256"] for source in sources}) != len(sources):
        raise ValueError("Need at least 16 distinct qualified high-resolution photo files")
    paths = (Path(__file__), Path(base.__file__), Path(copy_affine_large.__file__),
             Path(copy_affine.__file__), Path(copy_move.__file__))
    return {"schema": SCHEMA, "roster_sha256": sha(folder / "plan.json"),
            "acquisition_sha256": sha(folder / "acquisition.json"), "sources": sources,
            "formats": list(FORMATS), "treatments": list(TREATMENTS), "arms": list(ARMS),
            "code": {path.name: sha(path) for path in paths},
            "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "pillow": pillow_version},
            "rule": "Each delivery needs useful pairs on at least six source groups, including at least two feathered groups. "
                    "Allow flags from at most two ordinary source groups per delivery, at most one from unchanged evaluation photos. "
                    "Among passing arms prefer the one with more useful views, then fewer ordinary flags, then source."}


def score(rgb, truth):
    try:
        candidates = copy_affine_large.candidates(rgb)
    except copy_move.CandidateBudgetExceeded:
        return "unassessed_match_budget", 0, {}
    results = {arm: [] for arm in ARMS}
    if not candidates:
        return "assessed", 0, results
    for arm, gray in (("source", copy_affine.luma(rgb)), ("source_smoothed", copy_affine_large.luma(rgb))):
        for matrix, seed in candidates:
            if (masks := copy_affine.dense_pair(gray, matrix, seed)) is None:
                continue
            results[arm].append({"overlap": base.pixel_counts(masks, truth),
                                 "mask_sha256": [base.digest(mask.tobytes()) for mask in masks]})
    return "assessed", len(candidates), results


def useful(row):
    return all(10 * overlap >= 3 * (predicted + target - overlap) and 2 * overlap >= target
               for predicted, target, overlap in row["overlap"])


def summarize(rows):
    output = {}
    for arm in ARMS:
        by_format = {}
        for fmt in FORMATS:
            views = [row for row in rows if row["format"] == fmt]
            counts = {}
            matched = set()
            ordinary = set()
            for treatment in TREATMENTS:
                subset = [row for row in views if row["treatment"] == treatment]
                hits = [row for row in subset if any(useful(pair) for pair in row["results"].get(arm, []))]
                if treatment in ("sharp", "feathered"):
                    matched.update(row["slot"] for row in hits)
                else:
                    ordinary.update(row["slot"] for row in subset if row["results"].get(arm))
                counts[treatment] = {"views": len(subset),
                                     "unassessed": sum(row["status"] != "assessed" for row in subset),
                                     "proposals": sum(row["proposals"] > 0 for row in subset),
                                     "flagged": sum(bool(row["results"].get(arm)) for row in subset),
                                     "useful": len(hits)}
            by_format[fmt] = {"treatments": counts, "useful_groups": len(matched), "ordinary_groups": len(ordinary),
                              "passes": len(matched) >= 6 and counts["feathered"]["useful"] >= 2
                              and counts["published"]["flagged"] <= 1 and len(ordinary) <= 2}
        output[arm] = by_format
    passed = [arm for arm in ARMS if all(output[arm][fmt]["passes"] for fmt in FORMATS)]
    def ranking(arm):
        counts = [output[arm][fmt]["treatments"] for fmt in FORMATS]
        return (-sum(row[key]["useful"] for row in counts for key in ("sharp", "feathered")),
                sum(row[key]["flagged"] for row in counts for key in base.CONTROLS), ARMS.index(arm))
    return output, min(passed, key=ranking) if passed else None


def run(folder: Path, frozen: dict):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    rows = []
    started = time.monotonic()
    for source in frozen["sources"]:
        raw = (folder / source["name"]).read_bytes()
        image = decode(raw)
        if base.digest(raw) != source["sha256"] or base.digest(image.tobytes()) != source["rgb_sha256"]:
            raise ValueError("Decoded image no longer matches its source commitments")
        for treatment, pixels, truth in base.fixtures(np.asarray(image), source["slot"]):
            if treatment not in TREATMENTS:
                continue
            for fmt in FORMATS:
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer, format="PNG" if fmt == "png" else "JPEG",
                                             **({} if fmt == "png" else {"quality": 75, "subsampling": 2}))
                with Image.open(buffer) as decoded:
                    values = np.asarray(decoded.convert("RGB"))
                status, count, outcomes = score(values, truth)
                rows.append({"slot": source["slot"], "format": fmt, "treatment": treatment,
                             "input_sha256": base.digest(values.tobytes()),
                             "truth_sha256": [base.digest(mask.tobytes()) for mask in truth],
                             "status": status, "proposals": count, "results": outcomes})
        print(f"Finished source {source['slot']:02d}", flush=True)
    base.write(folder / "measurements.json", rows)
    results, selected = summarize(rows)
    base.write(folder / "aggregate.json", {"schema": SCHEMA, "freeze_sha256": sha(folder / "freeze.json"),
                                           "measurements_sha256": sha(folder / "measurements.json"),
                                           "sources": len(frozen["sources"]), "views": len(rows), "results": results,
                                           "selected": selected, "seconds": round(time.monotonic() - started, 3)})
    print(json.dumps({"views": len(rows), "selected": selected}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "acquire", "freeze", "run"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder
    if args.phase == "plan":
        base.write(folder / "plan.json", plan(folder / "intake"))
    elif args.phase == "acquire":
        acquire(json.loads((folder / "plan.json").read_bytes()), folder)
    elif args.phase == "freeze":
        base.write(folder / "freeze.json", protocol(folder))
    else:
        frozen = protocol(folder)
        if frozen != json.loads((folder / "freeze.json").read_bytes()):
            raise ValueError("Source or method changed after freeze")
        run(folder, frozen)


if __name__ == "__main__":
    main()
