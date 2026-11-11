"""Bounded RawNIND transport and a model-free Bayer-phase feasibility replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import platform
import re
import stat
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import yaml
from PIL import Image

from scripts.image_bayer_phase_v237 import analyze_bayer_phase

HOST = "https://dataverse.uclouvain.be"
DOI = "doi:10.14428/DVN/DEQCIM"
CATALOG_URL = HOST + "/api/datasets/:persistentId/versions/1.0?persistentId=doi%3A10.14428%2FDVN%2FDEQCIM"
CATALOG_SHA = "8183a5969d4631e84d643887f5ad851ea2e71bd4c3c3abf40d260791f88969e8"
CATALOG_FILES = 2845
SIDECAR_ID = 30027
SIDECAR_SHA = "f9af6290fb94e4d64941cf789ea30e97a02e586f225a75e60722764d4a254fbc"
ROSTER_SHA = "9df5cd01077ed209d5138798f1b2d6b06711b3fabe06698346eae4f04afbcfa8"
RAW_BYTES = 111_284_720
LICENSE_URI = "https://creativecommons.org/licenses/by-sa/4.0/"
WHEEL = "rawpy-0.28.0-cp312-cp312-macosx_11_0_arm64.whl"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/ba/4e/"
    "20001556bb07f74517b0c105b7e3f5bb383a361edfcf3ea102d73066e40b/" + WHEEL
)
WHEEL_SHA = "60e4c82d2dbce6ae20f95d0894bcacf355d32b783cd017c374c800dfc4a0d9b3"
WHEEL_BYTES = 2_060_964
DIST_INFO = "rawpy-0.28.0.dist-info"
EXTENSIONS = ("arw", "crw")
MEMBER = re.compile(r"Bayer_(.+?)(_GT)?_ISO([0-9]+)_sha1=([0-9a-f]{40})\.(arw|crw)")
VIEWS = ("native", "jpeg90", "jpeg75", "crop_odd", "resize_075", "sharpen")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(body):
    return hashlib.sha256(body).hexdigest()


class DirectOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_exact(url, size):
    request = urllib.request.Request(url, headers={"User-Agent": "ChipmunkVerifyResearch/1.0"})
    with urllib.request.build_opener(DirectOnly()).open(request, timeout=15) as response:
        require(response.status == 200 and response.url == url, "Expected a direct publisher response")
        body = response.read(size + 1)
    require(len(body) == size, "Publisher body differs from its bound size")
    return body


def source_member(item, files, scene, extension, clean):
    name = item["filename"].rsplit("/", 1)[-1]
    match = MEMBER.fullmatch(name)
    if match is None or (match[1], match[5], bool(match[2])) != (scene, extension, clean):
        return None
    row = files.get(name)
    if row is None:
        return None
    f = row["dataFile"]
    require(
        row["restricted"] is False
        and row["version"] == 1
        and item["sha1"] == match[4]
        and f["checksum"]["type"] == "MD5"
        and f["checksum"]["value"] == f["md5"]
        and re.fullmatch(r"[0-9a-f]{32}", f["md5"])
        and type(f["id"]) is int
        and type(f["filesize"]) is int
        and 0 < f["filesize"] < 50_000_000,
        "An original RAW does not match its publisher provenance",
    )
    return {
        "id": f["id"],
        "file": name,
        "iso": int(match[3]),
        "md5": f["md5"],
        "sha1": match[4],
        "bytes": f["filesize"],
    }


def select_roster(catalog, sidecar):
    data = catalog["data"]
    require(
        data["datasetPersistentId"] == DOI
        and data["versionState"] == "RELEASED"
        and (data["versionNumber"], data["versionMinorNumber"], data["id"]) == (1, 0, 177)
        and data["license"]["rightsIdentifier"] == "CC-BY-SA-4.0"
        and data["license"]["uri"] == LICENSE_URI,
        "Only the published RawNIND V1.0 release is admitted",
    )
    files = {row["dataFile"]["filename"]: row for row in data["files"]}
    require(len(files) == len(data["files"]) == CATALOG_FILES, "Exact published file catalog required")
    selected = []
    for extension in EXTENSIONS:
        groups = {}
        for scene, record in sidecar["Bayer"].items():
            if record["test_reserve"] or record["unknown_sensor"]:
                continue
            clean = [source_member(item, files, scene, extension, True) for item in record["clean_images"]]
            noisy = [source_member(item, files, scene, extension, False) for item in record["noisy_images"]]
            if any(clean) and any(noisy):
                groups[scene] = min(
                    (item for item in clean if item), key=lambda item: (item["iso"], item["id"])
                )
        require(len(groups) >= 2, "Two author-bound scenes per capture format required")
        scenes = sorted(
            groups,
            key=lambda s: hashlib.sha256(f"chipmunk-rawnind-v237-sanity/{extension}/{s}".encode()).digest(),
        )
        selected.extend(groups[s] for s in scenes[:2])
    require(
        len({row["id"] for row in selected}) == len({row["sha1"] for row in selected}) == 4,
        "Four distinct captures required",
    )
    return selected


def owner_directory(root):
    info = root.stat()
    require(
        root.is_absolute()
        and root.resolve() == root
        and root.is_dir()
        and info.st_uid == os.getuid()
        and info.st_mode & 0o077 == 0,
        "Direct owner-only study directory required",
    )


def read_private(path, limit):
    info = path.lstat()
    require(
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.getuid()
        and info.st_mode & 0o077 == 0
        and 0 < info.st_size <= limit,
        "Bounded owner-only study file required",
    )
    with path.open("rb") as handle:
        body = handle.read(limit + 1)
    require(len(body) == info.st_size, "Study file changed during read")
    return body


def write_private(path, body):
    with tempfile.NamedTemporaryFile(dir=path.parent) as handle:
        handle.write(body)
        handle.flush()
        os.link(handle.name, path)


def freeze(root):
    require(
        root.is_absolute() and root.parent.resolve() == root.parent and not root.exists(),
        "Fresh direct study path required",
    )
    catalog = fetch_exact(CATALOG_URL, 2_302_430)
    require(sha256(catalog) == CATALOG_SHA, "Published catalog bytes differ")
    body = fetch_exact(f"{HOST}/api/access/datafile/{SIDECAR_ID}", 456_897)
    require(sha256(body) == SIDECAR_SHA, "Published camera sidecar bytes differ")
    roster = compact(select_roster(json.loads(catalog), yaml.safe_load(body)))
    require(sha256(roster) == ROSTER_SHA, "The four selected RAW commitments differ")
    root.mkdir(mode=0o700)
    owner_directory(root)
    write_private(root / "roster.json", roster)
    return {"status": "frozen", "planned_captures": 4, "raw_bytes": RAW_BYTES, "roster_sha256": ROSTER_SHA}


def frozen_roster(root):
    owner_directory(root)
    body = read_private(root / "roster.json", 4_096)
    require(sha256(body) == ROSTER_SHA, "Frozen roster differs")
    rows = json.loads(body)
    require(len(rows) == 4 and sum(row["bytes"] for row in rows) == RAW_BYTES, "Unexpected RAW roster size")
    return rows


def raw_path(root, ordinal, row):
    extension = MEMBER.fullmatch(row["file"])[5]
    return root / f"capture-{ordinal:02}.{extension}"


def verify_raw(body, row):
    require(
        len(body) == row["bytes"]
        and hashlib.md5(body, usedforsecurity=False).hexdigest() == row["md5"]
        and hashlib.sha1(body, usedforsecurity=False).hexdigest() == row["sha1"],
        "RAW source fails its publisher size or checksum",
    )


def verified_wheel(path):
    body = read_private(path, WHEEL_BYTES)
    require(len(body) == WHEEL_BYTES and sha256(body) == WHEEL_SHA, "Pinned official rawpy wheel required")
    return body


def acquire(root):
    rows = frozen_roster(root)
    for ordinal, row in enumerate(rows):
        path = raw_path(root, ordinal, row)
        if path.exists():
            verify_raw(read_private(path, row["bytes"]), row)
        else:
            body = fetch_exact(f"{HOST}/api/access/datafile/{row['id']}", row["bytes"])
            verify_raw(body, row)
            write_private(path, body)
    path = root / WHEEL
    if path.exists():
        verified_wheel(path)
    else:
        body = fetch_exact(WHEEL_URL, WHEEL_BYTES)
        require(sha256(body) == WHEEL_SHA, "Official wheel checksum differs")
        write_private(path, body)
    return {"status": "acquired", "captures": 4, "raw_bytes": RAW_BYTES, "wheel_bytes": WHEEL_BYTES}


def unpack_decoder(body, destination):
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = archive.infolist()
        require(
            len(members) == len({member.filename for member in members})
            and sum(member.file_size for member in members) <= 25_000_000,
            "Unexpected wheel members or expanded size",
        )
        for member in members:
            name = PurePosixPath(member.filename)
            require(
                not name.is_absolute()
                and ".." not in name.parts
                and "\\" not in member.filename
                and not stat.S_ISLNK(member.external_attr >> 16),
                "Unsafe wheel member",
            )
        metadata = BytesParser().parsebytes(archive.read(f"{DIST_INFO}/METADATA"))
        wheel_info = BytesParser().parsebytes(archive.read(f"{DIST_INFO}/WHEEL"))
        require(
            metadata["Name"] == "rawpy"
            and metadata["Version"] == "0.28.0"
            and metadata["License-Expression"] == "MIT"
            and wheel_info["Tag"] == "cp312-cp312-macosx_11_0_arm64",
            "Pinned wheel platform and license metadata required",
        )
        require(
            {"LICENSE", "LICENSE.LibRaw"} <= set(metadata.get_all("License-File", [])),
            "Rawpy and LibRaw license notices required",
        )
        for name in ("LICENSE", "LICENSE.LibRaw"):
            require(bool(archive.read(f"{DIST_INFO}/licenses/{name}").strip()), "Empty wheel license notice")
        archive.extractall(destination)


def decode(rawpy, path):
    with rawpy.imread(str(path)) as raw:
        pattern = raw.raw_pattern
        require(
            pattern is not None
            and pattern.shape == (2, 2)
            and sorted(raw.color_desc[index] for index in pattern.flat) == sorted(b"RGGB"),
            "Original 2×2 Bayer capture required",
        )
        native_size = (raw.sizes.height, raw.sizes.width, 3)
        pixels = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            half_size=False,
            four_color_rgb=False,
            dcb_iterations=0,
            dcb_enhance=False,
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
            noise_thr=None,
            median_filter_passes=0,
            use_camera_wb=True,
            use_auto_wb=False,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8,
            user_flip=0,
            no_auto_bright=True,
            no_auto_scale=False,
            bright=1.0,
            highlight_mode=rawpy.HighlightMode.Clip,
            gamma=(2.222, 4.5),
            chromatic_aberration=(1.0, 1.0),
        )
    require(
        pixels.dtype == np.uint8 and pixels.shape == native_size,
        "Native RAW rendering differs from stored geometry",
    )
    return pixels


def ordinary_views(pixels):
    yield "native", pixels
    for quality in (90, 75):
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(
            encoded, format="JPEG", quality=quality, subsampling=2, optimize=False, progressive=False
        )
        with Image.open(encoded) as image:
            yield f"jpeg{quality}", np.asarray(image.convert("RGB"))
    yield "crop_odd", pixels[1:-3, 3:-2]
    height, width = pixels.shape[:2]
    yield "resize_075", cv2.resize(pixels, (width * 3 // 4, height * 3 // 4), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(pixels, (3, 3), 0)
    yield "sharpen", cv2.addWeighted(pixels, 1.3, blurred, -0.3, 0)


def measure(pixels):
    result = analyze_bayer_phase(pixels)
    scores = result.scores[np.isfinite(result.scores)] if result.scores is not None else None
    return {
        "source_size": list(result.source_size),
        "rgb_sha256": sha256(np.ascontiguousarray(pixels)),
        "status": result.status,
        "available_tiles": list(result.available_tiles),
        "held_out_support": list(result.held_out_support) if result.held_out_support is not None else None,
        "score_quantiles_05_50_95_100": np.quantile(scores, [0.05, 0.50, 0.95, 1.0]).tolist()
        if scores is not None
        else None,
    }


def summarize(rows):
    result = {}
    for view in VIEWS:
        items = [row for row in rows if row["view"] == view]
        require(
            len(items) == 4 and {row["capture"] for row in items} == set(range(4)),
            "Incomplete matched replay",
        )
        scores = [row["score_quantiles_05_50_95_100"] for row in items if row["status"] == "scored"]
        supports = [row["held_out_support"] for row in items if row["held_out_support"] is not None]
        result[view] = {
            "status_counts": dict(Counter(row["status"] for row in items)),
            "available_tiles": sum(sum(row["available_tiles"]) for row in items),
            "mean_held_out_support": float(np.mean(supports)) if supports else None,
            "capture_median_of_score_quantiles": np.median(scores, axis=0).tolist() if scores else None,
        }
    return {
        "status": "complete",
        "captures": 4,
        "measured_views": len(rows),
        "views": result,
        "new_neural_calls": 0,
    }


def replay(root):
    require(
        sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info[:2] == (3, 12),
        "Pinned wheel requires CPython 3.12 on macOS ARM64",
    )
    rows = frozen_roster(root)
    for ordinal, row in enumerate(rows):
        verify_raw(read_private(raw_path(root, ordinal, row), row["bytes"]), row)
    body = verified_wheel(root / WHEEL)
    require(
        "rawpy" not in sys.modules and not (root / "replay-private.json").exists(),
        "Fresh isolated decoder replay required",
    )
    cv2.setNumThreads(1)
    measured = []
    with tempfile.TemporaryDirectory(dir=root, prefix="decoder-") as directory:
        unpack_decoder(body, directory)
        sys.path.insert(0, directory)
        try:
            rawpy = importlib.import_module("rawpy")
            require(
                Path(rawpy.__file__).resolve().is_relative_to(Path(directory)),
                "Isolated wheel module required",
            )
            for ordinal, row in enumerate(rows):
                pixels = decode(rawpy, raw_path(root, ordinal, row))
                for name, view in ordinary_views(pixels):
                    measured.append({"capture": ordinal, "view": name, **measure(view)})
                del pixels, view
            version = list(rawpy.libraw_version)
        finally:
            sys.path.pop(0)
    output = {
        "roster_sha256": ROSTER_SHA,
        "wheel_sha256": WHEEL_SHA,
        "libraw_version": version,
        "rows": measured,
    }
    summary = summarize(measured)
    write_private(root / "replay-private.json", compact(output))
    return {**summary, "roster_sha256": ROSTER_SHA, "wheel_sha256": WHEEL_SHA, "libraw_version": version}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "acquire", "replay"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    stage = {"freeze": freeze, "acquire": acquire, "replay": replay}[args.stage]
    print(json.dumps(stage(args.root), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
