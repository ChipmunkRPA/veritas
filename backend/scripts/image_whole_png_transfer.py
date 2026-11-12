# SPDX-License-Identifier: Apache-2.0
"""Freeze and measure source-disjoint native PNG coverage with the inherited image point."""

from __future__ import annotations

import argparse
import importlib
import io
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageOps

from scripts import image_rawnind_phase_v239 as raw
from scripts import image_whole_fresh_sources as earlier
from scripts import run_image_whole_fresh as runner

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
DOCS = ROOT / "docs/evaluation-panels"
JPEG = DOCS / "image-whole-fresh-sources-2026-11-12.json"
OLD = ROOT / earlier.OLDER
POINT = ROOT / earlier.POINT
WEIGHTS = ROOT / earlier.WEIGHTS
HUB = "https://huggingface.co"
SALT = "chipmunk-whole-image-native-png-2026-11-12/1/"
SCHEMA = "chipmunk.image-whole-png-transfer/1"
COUNTS = {"flux-schnell": 5, "sdxl": 5, "gemini": 6, "raw-camera": 4, "camera-jpeg-export": 8, "browser": 8}
GATE = {"minimum_total": 36, "minimum_generated_flags": 12, "minimum_per_generator": 3,
        "maximum_control_groups_flagged": 1, "maximum_raw_camera_flags": 0,
        "minimum_new_generated": 3, "minimum_new_generator_families": 2, "maximum_new_control_groups": 1}
SOURCES = {
    "T4": {"repository": "MrAlexGov/t4-imagegen-llm-judges", "revision": "9136726108ac95b1aabdf200c0969f6b3b39c1d1",
           "readme_sha256": "5b511d4fa261519101c6864e1d8e0353420682288ac63f1f3c16389e9e8e82e2",
           "generation_sha256": "a2cefc295ea19d8e688aa04799bc9ac0063e21d19a553c1676a0fa3abfff8bc0"},
    "CompoSET": {"repository": "CompoSET/CompoSET", "revision": "d428bf4f23f50ababb205185f179737cd9638db2",
                 "readme_sha256": "ab4d1035699f5aa1c291752931f22a8f09271c962c2d5fc3a8c41dc8d55db0d9"},
}
LICENSES = {"flux-schnell": "https://www.apache.org/licenses/LICENSE-2.0", "sdxl": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md",
            "gemini": "https://creativecommons.org/licenses/by/4.0/"}
ALLOW = ("huggingface.co", "hf.co", "xethub.hf.co")
require, read, write, digest, sha = earlier.require, earlier.read, earlier.write, earlier.digest, runner.sha


def rank(value):
    return digest((SALT + value).encode())


def url(source, filename):
    return f"{HUB}/datasets/{source['repository']}/resolve/{source['revision']}/{quote(filename, safe='/')}"


def png_identity(data):
    require(data[:16] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" and data[24:26] == b"\x08\x02", "Expected a true-color 8-bit PNG")
    with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
        require(image.mode == "RGB" and image.n_frames == 1 and "transparency" not in image.info
                and min(image.size) >= 768 and max(image.size) <= 4920 and image.getexif().get(274, 1) == 1,
                "Image is outside the reviewed PNG size or color scope")
        rgb_sha, perceptual = earlier.rgb_identity(image)
        return list(image.size), rgb_sha, f"{perceptual:016x}"


def png(image):
    with io.BytesIO() as output:
        image.save(output, "PNG", icc_profile=None)
        return output.getvalue()


def generated_plan():
    groups, rows = set(), []
    for source in SOURCES.values():
        card = earlier.get(url(source, "README.md"), 125_000, ALLOW)
        require(digest(card) == source["readme_sha256"], "Publisher PNG license or metadata changed")
        if source.get("generation_sha256"):
            raw_manifest = earlier.get(url(source, "generators.json"), 125_000, ALLOW)
            require(digest(raw_manifest) == source["generation_sha256"], "Exact per-generator terms changed")
    prompts = ("01_portrait", "02_text_sign", "03_hands", "04_composition", "05_landscape", "06_illustration",
               "07_product", "08_anime", "09_food", "10_infographic")
    for family in ("flux-schnell", "sdxl", "gemini"):
        pool = prompts if family != "gemini" else [f"s{i:03d}_base" for i in range(1, 81)]
        publisher = SOURCES["T4" if family != "gemini" else "CompoSET"]
        chosen = sorted(set(pool) - groups, key=rank)[:COUNTS[family]]
        require(len(chosen) == COUNTS[family], "Incomplete disjoint prompt roster")
        groups.update(chosen)
        for group in chosen:
            source_file = f"images/{family}/{group}.png" if family != "gemini" else f"images/{group}.png"
            rows.append({"id": f"{family}-{group}", "family": family, "label": "synthetic", "group": group,
                         "origin": "publisher-native-png", "source_url": url(publisher, source_file),
                         "license": LICENSES[family], "license_source": url(publisher, "README.md")})
    return rows


def acquire(directory):
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / "objects").mkdir(exist_ok=True)
    require(not (directory / "freeze.json").exists(), "Refuse to acquire over a sealed panel")
    plan = {"sources": SOURCES, "rows": generated_plan()}
    plan_path = directory / "generated-plan.json"
    if plan_path.exists():
        require(read(plan_path) == plan, "Resume must use the exact planned source roster")
    else:
        write(plan_path, plan)
    for row in plan["rows"]:
        destination = directory / "objects" / f"{row['id']}.png"
        data = destination.read_bytes() if destination.exists() else earlier.get(row["source_url"], 8_000_000, ALLOW)
        png_identity(data)
        if not destination.exists():
            with destination.open("xb") as stream:
                stream.write(data)
        print(row["id"], len(data), flush=True)


def raw_controls(directory):
    source = DATA / "image-rawnind-phase-v239-attempt02"
    original = read(source / "replay-private.json")
    roster = raw.frozen_roster(source)
    expected = {row["capture"]: row for row in original["rows"] if row["view"] == "native"}
    publisher = f"https://doi.org/{raw.DOI.removeprefix('doi:')}"
    with tempfile.TemporaryDirectory(prefix="chipmunk-png-raw-control-") as private:
        raw.unpack_decoder(raw.verified_wheel(source / raw.WHEEL), Path(private))
        sys.path.insert(0, private)
        try:
            decoder = importlib.import_module("rawpy")
            require(Path(decoder.__file__).resolve().is_relative_to(Path(private).resolve()), "Pinned RAW decode required")
            for slot, row in enumerate(roster):
                source_file = raw.raw_path(source, slot, row)
                raw.verify_raw(source_file.read_bytes(), row)
                pixels = raw.decode(decoder, source_file)
                previous = expected[slot]
                require(digest(pixels.tobytes()) == previous["rgb_sha256"], "Pinned camera development pixels changed")
                image = Image.fromarray(pixels)
                width, height = image.size
                w, h = (1152, 1152) if slot % 2 == 0 else (1408, 768)
                x, y = (width - w) // 2, (height - h) // 2
                require(min(x, y) >= 0, "Native camera crop lacks source extent")
                data = png(image.crop((x, y, x + w, y + h)))
                yield {"id": f"raw-{slot:02d}", "family": "raw-camera", "label": "ordinary", "group": f"raw-{raw.MEMBER.fullmatch(row['file'])[1]}",
                       "origin": "linear-demosaiced-native-camera-crop", "source_url": publisher,
                       "publisher_file": row["file"], "publisher_sha1": row["sha1"], "license": raw.LICENSE_URI,
                       "license_source": publisher, "native_crop": [x, y, w, h]}, data
        finally:
            sys.path.remove(private)


def jpeg_exports(prior_sources):
    archive = ROOT / earlier.ARCHIVE
    bound = earlier.ordinary_plan(archive)["sources"]
    previous = {row["sha256"] for row in prior_sources if row["label"] == "ordinary"}
    choices = []
    for row in sorted(bound, key=lambda row: rank(row["title"])):
        if row["archive_sha256"] in previous or len(choices) >= COUNTS["camera-jpeg-export"]:
            continue
        ordinal = len(choices)
        original = archive / row["file"]
        require(sha(original) == row["archive_sha256"], "Camera archive bytes changed")
        with Image.open(original, formats=["JPEG"]) as image:
            color = ImageOps.exif_transpose(image).convert("RGB")
            if min(color.size) < 1024 or max(color.size) > 4920:
                continue
            x, y, w, h = 0, 0, *color.size
            if ordinal % 2:
                w, h = 1408, 768
                x, y = (color.width - w) // 2, (color.height - h) // 2
                if min(x, y) < 0:
                    continue
                color = color.crop((x, y, x + w, y + h))
            data = png(color)
        choices.append(({"id": f"export-{ordinal:02d}", "family": "camera-jpeg-export", "label": "ordinary",
                         "group": row["group"], "origin": "lossless-png-export-of-publisher-jpeg", "author": row["author"],
                         "source_url": row["page"], "publisher_sha256": row["archive_sha256"], "license": row["license"],
                         "license_source": row["page"], "native_crop": [x, y, w, h]}, data))
    require(len(choices) == COUNTS["camera-jpeg-export"], "Incomplete separate ordinary photo roster")
    return choices


def browser_controls(directory):
    browser = read(directory / "screenshots.json")
    require(len(browser["paths"]) == 8 and len({item["group"] for item in browser["paths"]}) == 4, "Incomplete authored browser panel")
    for entry in browser["paths"]:
        name = Path(entry["file"])
        require(name.name == entry["file"] and name.suffix == ".png", "Invalid screenshot path")
        yield {"id": f"browser-{name.stem}", "family": "browser", "label": "ordinary", "group": f"browser-{entry['group']}",
               "origin": "native-browser-screenshot", "browser_version": browser["version"], "license": "project-authored"}, (directory / name).read_bytes()


def freeze(directory):
    plan = read(directory / "generated-plan.json")
    previous = read(JPEG)["sources"]
    older = read(OLD)["record"]
    encoded = {row["sha256"] for row in previous + older["inventory"]}
    exact = {row["rgb_sha256"] for row in previous + older["inventory"]}
    near = {int(row.get("difference_hash") or row["dhash64"], 16) for row in previous + older["inventory"]}
    cases = [(row, (directory / "objects" / f"{row['id']}.png").read_bytes()) for row in plan["rows"]]
    cases.extend(raw_controls(directory))
    cases.extend(jpeg_exports(previous))
    cases.extend(browser_controls(directory))
    require(Counter(row["family"] for row, _ in cases) == COUNTS and len({row["id"] for row, _ in cases}) == 36
            and sum(len(data) for _, data in cases) <= 100_000_000, "Frozen PNG roster or size budget changed")
    items = []
    for row, data in cases:
        size, rgb, diff = png_identity(data)
        payload = digest(data)
        require(payload not in encoded and rgb not in exact, "Earlier or duplicate encoded or decoded image")
        if row["family"] != "browser":
            require(all((int(diff, 16) ^ prior).bit_count() > 4 for prior in near), f"Earlier or duplicate photo or generated scene: {row['id']}")
            near.add(int(diff, 16))
        encoded.add(payload)
        exact.add(rgb)
        rel = Path("objects") / f"{row['id']}.png"
        if row["label"] == "ordinary":
            with (directory / rel).open("xb") as output:
                output.write(data)
        items.append({**row, "path": str(rel), "sha256": payload, "rgb_sha256": rgb, "dhash64": diff, "size": size, "bytes": len(data),
                      "format": "png"})
    point = read(POINT)["gate"]["raw_mean_logit"]
    write(directory / "freeze.json", {"schema": SCHEMA, "criteria": GATE, "sources": plan["sources"], "items": items,
          "operating_point": {"raw_mean_logit": point}, "operating_point_sha256": sha(POINT), "weight_receipt_sha256": sha(WEIGHTS),
          "old_panel_sha256": sha(OLD), "jpeg_panel_sha256": sha(JPEG), "experiment_sha256": sha(Path(__file__)),
          "screenshot_script_sha256": sha(Path(__file__).with_name("capture_image_whole_png_controls.cjs"))})
    print(f"Frozen {len(items)} PNG files. SHA-256 {sha(directory / 'freeze.json')}", flush=True)


def prepared(directory):
    lock = read(directory / "freeze.json")
    require(lock["schema"] == SCHEMA and lock["criteria"] == GATE and lock["operating_point_sha256"] == sha(POINT)
            and lock["weight_receipt_sha256"] == sha(WEIGHTS) and lock["experiment_sha256"] == sha(Path(__file__)), "Frozen PNG method changed")
    panel = []
    for item in lock["items"]:
        path = (directory / item["path"]).resolve()
        require(path.is_relative_to(directory) and sha(path) == item["sha256"], "Frozen PNG input bytes changed")
        panel.append({**item, "input_path": str(path), "input_sha256": item["sha256"]})
    require(Counter(row["family"] for row in panel) == COUNTS, "Incomplete held PNGs")
    return lock, panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("acquire", "freeze", "candidate", "freeze-baseline", "baseline"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    runner.prepared = prepared
    if args.stage in {"candidate", "freeze-baseline", "baseline"}:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    {"acquire": acquire, "freeze": freeze, "candidate": runner.candidate, "freeze-baseline": runner.freeze_baseline,
     "baseline": runner.baseline}[args.stage](args.directory.resolve())


if __name__ == "__main__":
    main()
