# SPDX-License-Identifier: Apache-2.0
"""Evaluate a frozen small PNG boundary fallback on licensed photos and authored graphics."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from PIL import __version__ as pillow_version

from app.detectors.image import sharp_boundary
from scripts import image_boundary_extent_transfer as prior
from scripts import image_small_boundary_fallback_transfer as fallback

SCHEMA = "chipmunk.image-small-boundary-control-stress/1"
POINT = 1.2805722063718288
PHOTO_GROUPS = 16
GRAPHICS_GROUPS = 12
PHOTO_ROLES = ("original", "global_tone", "feathered_blur", "resize_roundtrip", "self_paste", "other_paste")
GRAPHIC_ROLES = ("authored_graphic", "resized_graphic")
PASTES = frozenset({"self_paste", "other_paste"})
SIZES = ((704, 544), (640, 608), (768, 512), (512, 704))
TARGET_SIZES = ((64, 77), (76, 64), (70, 86), (85, 71))
PROTOCOL = {
    "point": POINT,
    "minimum_side": 56,
    "maximum_axis_fraction": 0.18,
    "useful_iou": 0.30,
    "useful_recall": 0.50,
    "minimum_additional_useful": 8,
    "minimum_additional_groups": 6,
    "minimum_additional_per_paste_type": 2,
    "maximum_added_control_views": 3,
    "maximum_added_control_groups": 2,
    "maximum_added_untouched_photo_groups": 1,
    "maximum_added_graphic_groups": 1,
    "maximum_wrong_added_paste_views": 2,
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def require(valid, message: str) -> None:
    if not valid:
        raise ValueError(message)


def photo(cache: Path, source: dict, size: tuple[int, int]) -> Image.Image:
    encoded = (cache / f"{source['encoded_sha256']}.bin").read_bytes()
    require(digest(encoded) == source["encoded_sha256"], "Different encoded source")
    with Image.open(io.BytesIO(encoded)) as image:
        require(image.format == "JPEG" and image.getexif().get(274, 1) == 1, "Unexpected source delivery")
        rgb = image.convert("RGB")
    require(list(rgb.size) == source["size"] and digest(rgb.tobytes()) == source["rgb_sha256"], "Different native source pixels")
    return ImageOps.fit(rgb, size, Image.Resampling.LANCZOS)


def roi(image: Image.Image, slot: int, role: str) -> tuple[int, int, int, int]:
    width, height = image.size
    index = slot + (role == "other_paste")
    box_width, box_height = TARGET_SIZES[index % len(TARGET_SIZES)]
    x = 82 + (slot * 97 + index * 17) % (width - box_width - 164)
    y = 80 + (slot * 59 + index * 19) % (height - box_height - 160)
    return x, y, x + box_width, y + box_height


def photo_views(original: Image.Image, donor: Image.Image, slot: int):
    width, height = original.size
    yield "original", original, None
    yield "global_tone", ImageEnhance.Contrast(original).enhance(1.14), None
    mask = Image.new("L", original.size)
    ImageDraw.Draw(mask).ellipse((width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=180)
    yield "feathered_blur", Image.composite(original.filter(ImageFilter.GaussianBlur(1.25)), original,
                                           mask.filter(ImageFilter.GaussianBlur(12))), None
    yield "resize_roundtrip", resize(original), None
    for role, other in (("self_paste", original), ("other_paste", donor)):
        x, y, right, bottom = roi(original, slot, role)
        bw, bh = right - x, bottom - y
        dx = width - bw - 20 if x < width / 2 else 20
        dy = height - bh - 20 if y < height / 2 else 20
        edited = original.copy()
        edited.paste(other.crop((dx, dy, dx + bw, dy + bh)), (x, y))
        yield role, edited, (x, y, right, bottom)


def resize(image: Image.Image) -> Image.Image:
    width, height = image.size
    return image.resize((width * 7 // 10, height * 7 // 10), Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.LANCZOS)


def graphic(slot: int) -> Image.Image:
    """Draw ordinary screenshots/graphics without assets or third-party text."""
    width, height = SIZES[slot % len(SIZES)]
    white = (245, 247, 249)
    dark = (27, 36, 51)
    accent = ((22, 86, 151), (115, 50, 116), (13, 103, 78))[slot % 3]
    image = Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image)
    if slot < 4:
        draw.rectangle((0, 0, width - 1, 42), fill=dark)
        draw.rectangle((0, 43, 132, height - 1), fill=(204, 216, 230))
        draw.text((22, 17), f"Local project dashboard {slot + 1}", fill=white)
        for row in range(4):
            for col in range(3):
                x, y = 163 + col * ((width - 168) // 3), 74 + row * 106
                draw.rectangle((x + 3, y + 3, x + 78, y + 82), fill=(172, 179, 191))
                draw.rectangle((x, y, x + 74, y + 78), fill=white, outline=accent, width=2)
                draw.rectangle((x + 5, y + 6, x + 68, y + 30), fill=accent)
                draw.text((x + 12, y + 48), f"Item {row * 3 + col + 1}", fill=dark)
    elif slot < 8:
        for row in range(2 + height // 65):
            for col in range(2 + width // 68):
                x, y = 21 + col * 68, 18 + row * 65
                fill = (dark, accent, (180, 190, 201), white)[(row * 3 + col * 2 + slot) % 4]
                draw.rectangle((x, y, x + 60, y + 58), fill=fill, outline=(95, 111, 124), width=2)
    else:
        draw.rectangle((24, 30, width - 26, height - 29), outline=dark, width=3)
        draw.text((45, 47), "Measured values", fill=dark)
        base = height - 65
        draw.line((66, 90, 66, base, width - 52, base), fill=dark, width=3)
        for i in range(5):
            x = 82 + i * ((width - 128) // 5)
            length = 63 + (slot * 31 + i * 43) % (height - 194)
            draw.rectangle((x, base - length, x + 62, base), fill=accent)
            draw.text((x + 7, base + 11), f"Q{i + 1}", fill=dark)
    return image


def check(image: Image.Image) -> tuple[dict, dict]:
    pixels = np.asarray(image)
    incumbent = sharp_boundary.measure(pixels)
    fixed = {"score": incumbent.score, "proposal": list(incumbent.box)}
    candidate = prior.propose(pixels)["distributed"]
    return fixed, {"score": candidate["score"], "proposal": candidate["box"]}


def selected_boxes(image: Image.Image, fixed: dict, candidate: dict):
    original_box = fixed["proposal"] if fixed["score"] > sharp_boundary.REVIEW_POINT else None
    x, y, right, bottom = candidate["proposal"]
    width, height = image.size
    eligible = (candidate["score"] > POINT and fallback.MIN_SHORT_EDGE <= right - x <= width * fallback.MAX_AXIS_FRACTION
                and fallback.MIN_SHORT_EDGE <= bottom - y <= height * fallback.MAX_AXIS_FRACTION)
    return original_box, original_box or (candidate["proposal"] if eligible else None)


def evidence(image: Image.Image, mask: np.ndarray, box: list[int] | None) -> dict | None:
    if box is None:
        return None
    x, y, right, bottom = box
    pad = max(24, round(.12 * max(right - x, bottom - y)))
    crop = (max(0, x - pad), max(0, y - pad), min(image.width, right + pad), min(image.height, bottom + pad))
    return {"box": box, "crop": list(crop), "crop_rgb_sha256": digest(image.crop(crop).tobytes()),
            "area": (right - x) * (bottom - y), "overlap": int(mask[y:bottom, x:right].sum())}


def observe(run: Path, cohort: str, slot: int, role: str, image: Image.Image, source: Image.Image, target):
    sample = f"{cohort}-{slot:02d}-{role}"
    path = run / "inputs" / f"{sample}.png"
    image.save(path, format="PNG")
    with Image.open(path) as delivered:
        delivered.load()
        require(delivered.mode == "RGB" and delivered.tobytes() == image.tobytes(), "PNG delivery changed source RGB")
        fixed, candidate = check(delivered)
        base, combined = selected_boxes(delivered, fixed, candidate)
        mask = np.any(np.asarray(delivered) != np.asarray(source), axis=2) if target else np.zeros((image.height, image.width), dtype=bool)
        if target:
            x, y, right, bottom = target
            require(int(mask.sum()) == int(mask[y:bottom, x:right].sum()) > 0, "The conventional paste differs outside its target")
        return {"cohort": cohort, "slot": slot, "role": role, "size": list(image.size), "target": target,
                "target_pixels": int(mask.sum()), "truth_sha256": digest(mask.tobytes()),
                "input_sha256": digest(path.read_bytes()), "fixed": fixed | {"selected": evidence(delivered, mask, base)},
                "combined": candidate | {"selected": evidence(delivered, mask, combined)}}


def freeze(cache: Path, roster: dict):
    prior_receipt = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/image-boundary-extent-transfer-2026-11-12.json"
    require(read(prior_receipt)["points"]["distributed"]["point"] == POINT
            and PROTOCOL["minimum_side"] == fallback.MIN_SHORT_EDGE
            and PROTOCOL["maximum_axis_fraction"] == fallback.MAX_AXIS_FRACTION, "The published fallback changed")
    sources = roster["sources"]
    require(len(sources) == 2 * PHOTO_GROUPS and len({row["author_profile"] for row in sources}) == len(sources)
            and len({row["encoded_sha256"] for row in sources}) == len(sources)
            and len({row["rgb_sha256"] for row in sources}) == len(sources)
            and all(row["license_url"] == "https://creativecommons.org/licenses/by/2.0/" for row in sources),
            "Need distinct recorded CC BY 2.0 targets and donors")
    for row in sources:
        photo(cache, row, SIZES[0])
    paths = (Path(__file__), Path(sharp_boundary.__file__), Path(prior.__file__), Path(fallback.__file__))
    return {"schema": SCHEMA, "roster": roster, "photo_roles": PHOTO_ROLES, "graphic_roles": GRAPHIC_ROLES,
            "graphics_groups": GRAPHICS_GROUPS, "method": PROTOCOL, "fixed_point": sharp_boundary.REVIEW_POINT,
            "prior_panel_sha256": digest(prior_receipt.read_bytes()),
            "code_sha256": {path.name: digest(path.read_bytes()) for path in paths},
            "runtime": {"opencv": cv2.__version__, "numpy": np.__version__, "pillow": pillow_version}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for arg in ("cache", "roster", "run"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    roster = read(args.roster)
    frozen = freeze(args.cache, roster)
    path = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(path, frozen)
        print(digest(path.read_bytes()), flush=True)
        return
    require(read(path) == json.loads(json.dumps(frozen)), "Frozen inputs or operating rules changed")
    (args.run / "inputs").mkdir()
    rows = []
    sources = roster["sources"]
    for slot in range(PHOTO_GROUPS):
        size = SIZES[slot % len(SIZES)]
        source = photo(args.cache, sources[slot], size)
        donor = photo(args.cache, sources[slot + PHOTO_GROUPS], size)
        rows.extend(observe(args.run, "photo", slot, role, image, source, target)
                    for role, image, target in photo_views(source, donor, slot))
    for slot in range(GRAPHICS_GROUPS):
        source = graphic(slot)
        rows.extend(observe(args.run, "graphic", slot, role, image, source, None)
                    for role, image in zip(GRAPHIC_ROLES, (source, resize(source)), strict=True))
    write(args.run / "rows.json", rows)
    print(f"{len(rows)} PNG views", flush=True)


if __name__ == "__main__":
    main()
