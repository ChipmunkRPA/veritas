# SPDX-License-Identifier: Apache-2.0
"""Challenge the existing PNG graphic and screen-size rules with exact paired controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
PNG = PANELS / "image-whole-png-transfer-2026-11-12.json"
JPEG = PANELS / "image-whole-fresh-sources-2026-11-12.json"
LAYOUTS = ("gallery", "photo-editor", "street-map", "article", "dense-plot", "calendar")
SIZES = ((1366, 768), (1408, 768))
RULES = ("screen_size", "flat_default", "flat_floor", "default_or_screen_size", "floor_or_screen_size")
POINTS = {"flat_default": .5346, "flat_floor": .25, "block_pixels": 8}
ADMISSION = {"minimum_both_sizes_screen_groups": 5, "maximum_any_exclusion_photo_groups": 1}
SCHEMA = "chipmunk.image-whole-png-scope-probe/1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))


def prepare(directory):
    from PIL import Image, ImageOps

    from scripts.image_whole_fresh_sources import ARCHIVE, ordinary_plan, rank
    from scripts.image_whole_png_transfer import png_identity

    directory.mkdir(mode=0o700)
    archive = ROOT / ARCHIVE
    excluded = {r.get("publisher_sha256") for r in read(PNG)["freeze"]["items"]}
    excluded.update(r["sha256"] for r in read(JPEG)["sources"])
    rows = sorted(ordinary_plan(archive)["sources"], key=lambda row: rank("scope/" + row["title"]))
    cases = []
    for row in rows:
        if row["archive_sha256"] in excluded:
            continue
        source = archive / row["file"]
        require(file_sha(source) == row["archive_sha256"], "Photo archive source changed")
        with Image.open(source) as original:
            photo = ImageOps.exif_transpose(original).convert("RGB")
        if photo.width < 1408 or photo.height < 768:
            continue
        layout = LAYOUTS[len(cases) // len(SIZES)]
        for width, height in SIZES:
            x, y = (photo.width - width) // 2, (photo.height - height) // 2
            name = f"{layout}-{width}-photo.png"
            path = directory / name
            photo.crop((x, y, x + width, y + height)).save(path, "PNG", icc_profile=None)
            require(png_identity(path.read_bytes())[0] == [width, height], "Native photo crop changed")
            cases.append({"layout": layout, "dimensions": [width, height], "crop": [x, y, width, height], "photo": name,
                          "author": row["author"], "license": row["license"], "source_url": row["page"],
                          "source_sha256": row["archive_sha256"], "photo_sha256": file_sha(path),
                          "screen": f"{layout}-{width}-screen.png"})
        if len(cases) == len(LAYOUTS) * len(SIZES):
            break
    require(len({row["source_sha256"] for row in cases}) == len(LAYOUTS), "Incomplete independent photo roster")
    save(directory / "plan.json", {"cases": cases, "admission": ADMISSION, "existing_points": POINTS,
                                   "browser_capture_sha256": file_sha(Path(__file__).with_name("capture_image_whole_png_scope_controls.cjs"))})


def freeze(directory):
    from scripts.image_whole_png_transfer import png_identity

    plan = read(directory / "plan.json")
    cases = []
    for row in plan["cases"]:
        for role in ("photo", "screen"):
            data = (directory / row[role]).read_bytes()
            require(png_identity(data)[0] == row["dimensions"] and (role != "photo" or sha(data) == row["photo_sha256"]),
                    "Exact PNG delivery changed")
            cases.append({"id": row[role], "sha256": sha(data), "role": role, "layout": row["layout"], "dimensions": row["dimensions"]})
    save(directory / "freeze.json", {"schema": SCHEMA, "plan": plan, "cases": cases,
                                     "browser": read(directory / "browser.json"), "script_sha256": file_sha(Path(__file__))})


def rule_flags(width, height, flat, blocks):
    size = (width, height) == SIZES[0]
    default = flat / blocks > POINTS["flat_default"]
    floor = flat / blocks > POINTS["flat_floor"]
    return dict(zip(RULES, (size, default, floor, size or default, size or floor), strict=True))


def summary(frozen, rows):
    require(frozen["schema"] == SCHEMA and frozen["plan"]["existing_points"] == POINTS
            and frozen["plan"]["admission"] == ADMISSION, "Frozen coverage rules changed")
    wanted = {(layout, size, role) for layout in LAYOUTS for size in SIZES for role in ("photo", "screen")}
    expected = {(row["layout"], tuple(row["dimensions"]), row["role"]) for row in rows}
    require(len(rows) == len(frozen["cases"]) == len(wanted) and expected == wanted, "Missing matched controls")
    for case, row in zip(frozen["cases"], rows, strict=True):
        require(all(row[k] == v for k, v in case.items()), "Scope result belongs to a different PNG")
        w, h = row["dimensions"]
        block = POINTS["block_pixels"]
        blocks = (w // block) * (h // block)
        require(row["blocks"] == blocks and 0 <= row["flat_blocks"] <= blocks and row["has_exif"] is False
                and row["rules"] == rule_flags(w, h, row["flat_blocks"], blocks), "Coverage rule or observed block count differs")
    output = {}
    for rule in RULES:
        excluded = [row for row in rows if row["rules"][rule]]
        photo = [row for row in excluded if row["role"] == "photo"]
        screen = [row for row in excluded if row["role"] == "screen"]
        photo_groups = {row["layout"] for row in photo}
        complete = {layout for layout in LAYOUTS if sum(row["layout"] == layout for row in screen) == len(SIZES)}
        output[rule] = {"photo_views_excluded": len(photo), "photo_groups_excluded": sorted(photo_groups),
                        "screens_excluded": len(screen), "screen_groups_both_sizes": sorted(complete),
                        "passed": len(photo_groups) <= ADMISSION["maximum_any_exclusion_photo_groups"]
                        and len(complete) >= ADMISSION["minimum_both_sizes_screen_groups"]}
    return output


def measure(directory):
    from PIL import Image

    from app.engine.fusion import abstain, content_kinds

    freeze_path = directory / "freeze.json"
    frozen = read(freeze_path)
    require(frozen["script_sha256"] == file_sha(Path(__file__)) and POINTS["flat_floor"] == content_kinds.FLOOR
            and POINTS["block_pixels"] == content_kinds.BLOCK
            and next(iter(content_kinds.limits().values()))["flat_blocks"]["limit"] == POINTS["flat_default"], "Existing app rules changed")
    rows = []
    for case in frozen["cases"]:
        path = directory / case["id"]
        require(file_sha(path) == case["sha256"], "Frozen scope PNG changed")
        width, height = case["dimensions"]
        with Image.open(path, formats=["PNG"]) as image:
            has_exif = bool(image.getexif())
            screenshot = bool(abstain.screenshot_likelihood(image.format, width, height, has_exif))
        measured = content_kinds.measure(path)
        blocks = (width // content_kinds.BLOCK) * (height // content_kinds.BLOCK)
        flat = round(measured["flat_block_share"] * blocks)
        rules = rule_flags(width, height, flat, blocks)
        require(rules["screen_size"] == screenshot and flat / blocks == measured["flat_block_share"], "App rule response differs")
        rows.append({**case, "flat_blocks": flat, "blocks": blocks, "has_exif": has_exif, "rules": rules})
    output = summary(frozen, rows)
    save(directory / "result.json", {"freeze": frozen, "freeze_sha256": file_sha(freeze_path), "rows": rows, "summary": output})
    print(json.dumps(output, sort_keys=True), flush=True)


def review(ledger):
    result = read(ledger)
    require(sha(encoded(result["freeze"])) == result["freeze_sha256"], "Frozen scope commitment changed")
    actual = summary(result["freeze"], result["rows"])
    require(actual == result["summary"], "Portable scope totals differ")
    print(json.dumps(actual, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "measure", "review"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "measure": measure, "review": review}[args.stage](args.path.resolve())
