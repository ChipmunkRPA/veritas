# SPDX-License-Identifier: Apache-2.0
"""Acquire a bounded, rights-checked set of camera photographs with original thumbnails."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
from pathlib import Path
from urllib import error, parse, request

from PIL import Image

from app.provenance.exiftool import _pillow_thumbnail

API = "https://commons.wikimedia.org/w/api.php"
CAMERAS = ("Canon EOS 450D", "Nikon D90", "Sony DSLR-A200", "Canon PowerShot SX230 HS", "Panasonic Lumix DMC-FZ200")
MAX_BYTES = 6_000_000
MAX_PIXELS = 24_000_000
PER_CAMERA = 3
MAX_ATTEMPTS = 10
AGENT = "ChipmunkVerifyEvaluation/1.0 (local forensic research)"
LICENSES = {f"https://creativecommons.org/licenses/{kind}/{version}" for kind in ("by", "by-sa")
            for version in ("2.0", "2.5", "3.0", "4.0")}


def write(path, value):
    with path.open("x") as output:
        json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")


def get(url, limit):
    with request.urlopen(request.Request(url, headers={"User-Agent": AGENT}), timeout=10) as response:
        host = parse.urlsplit(response.geturl()).hostname
        if host not in {"commons.wikimedia.org", "upload.wikimedia.org"}:
            raise ValueError("Unexpected Wikimedia redirect")
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Public source exceeds the frozen byte limit")
    return body


def plain(markup):
    return " ".join(html.unescape(re.sub(r"<[^>]*>", "", markup)).split())


def plan(camera):
    params = {"action": "query", "generator": "categorymembers", "gcmtitle": f"Category:Taken with {camera}",
              "gcmnamespace": 6, "gcmtype": "file", "gcmlimit": 60, "prop": "imageinfo",
              "iiprop": "url|timestamp|size|sha1|extmetadata", "format": "json"}
    body = get(API + "?" + parse.urlencode(params), 2_500_000)
    rows = []
    for item in json.loads(body).get("query", {}).get("pages", {}).values():
        info = item["imageinfo"][0]
        metadata = info["extmetadata"]
        author = plain(metadata.get("Artist", {}).get("value", ""))
        license_url = metadata.get("LicenseUrl", {}).get("value", "").rstrip("/")
        url = info["url"].split("?", 1)[0]
        if (not author or license_url not in LICENSES or not 100_000 <= info["size"] <= MAX_BYTES
                or not 1_000_000 <= info["width"] * info["height"] <= MAX_PIXELS or info["width"] < info["height"]
                or parse.urlsplit(url).hostname != "upload.wikimedia.org" or not url.casefold().endswith((".jpg", ".jpeg"))):
            continue
        rows.append({"category": camera, "author": author, "title": item["title"], "license": license_url,
                     "page": info["descriptionurl"], "url": url, "source_sha1": info["sha1"], "size": info["size"],
                     "native_size": [info["width"], info["height"]], "timestamp": info["timestamp"]})
    rows.sort(key=lambda row: hashlib.sha256(("thumbnail-native/1/" + row["title"]).encode()).hexdigest())
    return hashlib.sha256(body).hexdigest(), rows


def acquire(root):
    root.mkdir(mode=0o700)
    raw = root / "objects"
    raw.mkdir()
    plans = {camera: plan(camera) for camera in CAMERAS}
    write(root / "plans.json", plans)
    authors, verified, attempts = set(), [], []
    limited = False
    for camera, (_, rows) in plans.items():
        accepted = requested = 0
        for row in rows:
            author_key = row["author"].casefold()
            if author_key in authors:
                continue
            if accepted >= PER_CAMERA or requested >= MAX_ATTEMPTS:
                break
            requested += 1
            status = "download_failed"
            try:
                body = get(row["url"], MAX_BYTES)
                if len(body) != row["size"] or hashlib.sha1(body).hexdigest() != row["source_sha1"]:
                    raise ValueError("Original object digest differs from Commons")
                status = "no_supported_thumbnail"
                with Image.open(io.BytesIO(body)) as original:
                    exif = original.getexif()
                    preview = _pillow_thumbnail(original, exif)
                    if original.format == "JPEG" and preview and exif.get(274, 1) == 1 and exif.get(272):
                        with Image.open(io.BytesIO(preview)) as embedded:
                            if min(embedded.size) >= 64 and max(embedded.size) <= 512:
                                digest = hashlib.sha256(body).hexdigest()
                                with (raw / f"{digest}.jpg").open("xb") as out:
                                    out.write(body)
                                verified.append(row | {"sha256": digest, "camera_model": str(exif.get(272)),
                                                       "thumbnail_size": list(embedded.size),
                                                       "thumbnail_sha256": hashlib.sha256(preview).hexdigest()})
                                authors.add(author_key)
                                accepted += 1
                                status = "qualified"
            except error.HTTPError as exc:
                status = f"http_{exc.code}"
                limited = exc.code == 429
            except (ValueError, OSError, KeyError):
                pass
            attempts.append({"title": row["title"], "status": status})
            if limited:
                break
        if limited:
            break
    write(root / "qualified.json", {"rows": verified, "attempts": attempts})
    print({camera: sum(row["category"] == camera for row in verified) for camera in CAMERAS}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    acquire(parser.parse_args().directory)
