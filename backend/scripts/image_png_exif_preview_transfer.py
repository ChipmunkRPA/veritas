# SPDX-License-Identifier: Apache-2.0
"""Replay exact JPEG preview fixtures as upright PNGs with standard eXIf."""

from __future__ import annotations

import argparse
import io
import json
import struct
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageEnhance

from app.evidence import image_thumbnail_evidence as exhibits
from app.provenance import image_thumbnail_localization as localization
from app.provenance import png_exif_thumbnail
from scripts import image_local_residual_covariance as sources
from scripts import image_thumbnail_extent_transfer as earlier

SCHEMA = "chipmunk.image-png-exif-preview-transfer/1"
GATE = {"minimum_useful_lama_feather": 6, "minimum_useful_flux_feather": 3,
        "maximum_ordinary_delivery_flags": 1, "maximum_unrelated_admitted": 0,
        "maximum_malformed_admitted": 0}
read, write, sha, need = sources.read, sources.write, sources.sha, sources.need


def png(image, metadata=None, **options):
    with io.BytesIO() as stream:
        image.save(stream, format="PNG", compress_level=1, **({"exif": metadata} if metadata else {}), **options)
        return stream.getvalue()


def corrupt_crc(data, chunk):
    offset = 8
    while offset + 12 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, offset)
        end = offset + length + 12
        if kind == chunk:
            return data[:end - 1] + bytes((data[end - 1] ^ 1,)) + data[end:]
        offset = end
    raise ValueError("Fixture chunk was not found")


def xmp_oriented_png(image, metadata):
    # Hide the IFD0 orientation tag while preserving the same retained JPEG.
    absent_orientation = metadata[:16] + b"\x31\x01" + metadata[18:]
    encoded = png(image, absent_orientation)
    value = b"XML:com.adobe.xmp\0\0\0\0\0<rdf:Description tiff:Orientation=\"6\"/>"
    chunk = b"iTXt" + value
    field = struct.pack(">I", len(value)) + chunk + struct.pack(">I", zlib.crc32(chunk))
    with Image.open(io.BytesIO(encoded[:-12] + field + encoded[-12:]), formats=["PNG"]) as decoded:
        decoded.load()
        need(decoded.getexif().get(274, 1) == 6, "Fixture must have a nonupright effective orientation")
    return encoded[:-12] + field + encoded[-12:]


def cases(args):
    portraits = list(sources.sources(args, "held"))
    prior_thumbnail = earlier.earlier.thumbnail
    for i, (family, source, _, original, edited) in enumerate(portraits):
        before = Image.fromarray(original)
        preview = prior_thumbnail(before, (168, 192, 216)[i % 3])
        meta = earlier.earlier.exif_bytes(preview)
        foreign = prior_thumbnail(Image.fromarray(portraits[(i + 1) % len(portraits)][3]), 192)
        for role, variant, positive in sources.variants(original, edited, i):
            truth = (variant != original).any(axis=2) if positive or role in ("local_blur", "local_tone") else None
            yield family, source, role, png(Image.fromarray(variant), meta), truth
        tone = ImageEnhance.Brightness(before).enhance(.91)
        yield family, source, "global_tone", png(tone, meta), None
        w, h = before.size
        resampled = before.resize((round(w * .72), round(h * .72)), Image.Resampling.BICUBIC).resize(before.size, Image.Resampling.LANCZOS)
        yield family, source, "resampled", png(resampled, meta), None
        yield family, source, "missing", png(before), None
        yield family, source, "foreign", png(before, earlier.earlier.exif_bytes(foreign)), None
    family, source, _, original, edited = portraits[0]
    main = Image.fromarray(edited)
    meta = earlier.earlier.exif_bytes(prior_thumbnail(Image.fromarray(original), 192))
    canonical = png(main, meta)
    start = canonical.index(b"eXIf") - 4
    size = int.from_bytes(canonical[start:start + 4], "big") + 12
    end = start + size
    block = canonical[start:end]
    offset_iend = len(canonical) - 12
    tiff = bytearray(meta[6:])
    thumbnail_offset_tag = struct.pack("<HHI", 0x0201, 4, 1)
    at = tiff.index(thumbnail_offset_tag) + len(thumbnail_offset_tag)
    tiff[at:at + 4] = struct.pack("<I", len(tiff))
    chunk = b"eXIf" + tiff
    missing_bytes = struct.pack(">I", len(tiff)) + chunk + struct.pack(">I", zlib.crc32(chunk))
    def ancillary(kind, value):
        payload = kind + value
        block = struct.pack(">I", len(value)) + payload + struct.pack(">I", zlib.crc32(payload))
        return canonical[:start] + block + canonical[start:]

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    # Profile creation timestamps vary. Replace the creation fields with a fixed date.
    icc = icc[:24] + struct.pack(">6H", 2026, 11, 12, 0, 0, 0) + icc[36:]
    large_preview = prior_thumbnail(Image.fromarray(original), 513)
    rgba = main.convert("RGBA")
    animated = png(rgba, meta, save_all=True, append_images=[Image.fromarray(original).convert("RGBA")], duration=100)
    for role, body in {"explicit_srgb": ancillary(b"sRGB", b"\0"),
                       "standard_gamma": ancillary(b"gAMA", struct.pack(">I", 45455))}.items():
        yield "encoding", source, role, body, (edited != original).any(axis=2)
    unsupported = {
        "duplicate": canonical[:end] + block + canonical[end:],
        "late": canonical[:start] + canonical[end:offset_iend] + block + canonical[offset_iend:],
        "bad_crc": corrupt_crc(canonical, b"eXIf"),
        "bad_idat_crc": corrupt_crc(canonical, b"IDAT"),
        "bad_iend_crc": corrupt_crc(canonical, b"IEND"),
        "bad_pointer": canonical[:start] + missing_bytes + canonical[end:],
        "alpha": png(rgba, meta),
        "animated": animated,
        "palette": png(main.quantize(), meta),
        "rgb_icc": png(main, meta, icc_profile=icc),
        "nonstandard_gamma": ancillary(b"gAMA", struct.pack(">I", 22000)),
        "oversized_preview": png(main, earlier.earlier.exif_bytes(large_preview)),
        "nonupright": png(main, meta[:24] + b"\x06\x00" + meta[26:]),
        "effective_xmp_rotation": xmp_oriented_png(main, meta),
    }
    thumb_with_icc = io.BytesIO()
    Image.fromarray(original).resize((192, 150)).save(thumb_with_icc, "JPEG", icc_profile=icc)
    unsupported["jpeg_preview_icc"] = png(main, earlier.earlier.exif_bytes(thumb_with_icc.getvalue()))
    for role, body in unsupported.items():
        yield "malformed", source, role, body, None


def binding(case):
    family, source, role, body, truth = case
    return {"family": family, "source": source, "role": role, "sha256": sha(body),
            "truth": sha(truth.tobytes()) if truth is not None else None, "target": int(truth.sum()) if truth is not None else 0}


def candidate(body):
    try:
        return localization.measure_png(body, minimum_component_fraction=localization.CONFIRMATION_COMPONENT_FRACTION)
    except (OSError, SyntaxError, ValueError, struct.error):
        return {"status": "unsupported_carrier"}, None


def measure(item, pinned):
    need(binding(item) == pinned, "Input carrier or exact changed pixels differ")
    _, _, _, body, truth = item
    result, masks = candidate(body)
    if pinned["family"] == "encoding":
        need(result["status"] == "comparable", "Standard PNG color declaration was rejected")
    mask = masks[0] if masks is not None else None
    areas = earlier.earlier.area(mask, truth)
    regions = [{**region, "preview_bounds": exhibits._stored_preview_bounds(region, result)} for region in result.get("regions", ())]
    if regions:
        preview = png_exif_thumbnail.extract(body)
        with (Image.open(io.BytesIO(body), formats=["PNG"]) as native,
              Image.open(io.BytesIO(preview), formats=["JPEG"]) as thumb):
            for region in regions:
                pins = {}
                for key, source in (("submitted", native), ("preview_bounds", thumb)):
                    bounds = region[key]
                    x, y, w, h = (bounds[k] for k in exhibits.BOX_KEYS)
                    need(0 <= x < x + w <= source.width and 0 <= y < y + h <= source.height, "Crop outside native bytes")
                    body, size = exhibits._crop(source, bounds)
                    expected = source.crop((x, y, x + w, y + h)).convert("RGB")
                    if expected.size != size:
                        expected.thumbnail((exhibits.DISPLAY_EDGE, exhibits.DISPLAY_EDGE), Image.Resampling.LANCZOS)
                    with Image.open(io.BytesIO(body), formats=["PNG"]) as cropped:
                        need(cropped.size == expected.size and cropped.tobytes() == expected.tobytes(), "Decoded crop pixels differ")
                    pins[key] = sha(body)
                region["crop_sha256"] = pins
    return {**pinned, "status": result["status"], "regions": regions, "area": areas,
            "mask": sha(mask.tobytes()) if mask is not None else None}


def summary(rows):
    counts = {}
    for row in rows:
        tally = counts.setdefault(f"{row['family']}/{row['role']}", Counter())
        tally.update(total=1, comparable=int(row["status"] == "comparable"), flags=int(bool(row["regions"])), useful=int(row["area"]["useful"]))
    actual = {"minimum_useful_lama_feather": counts["lama/feather"]["useful"],
              "minimum_useful_flux_feather": counts["flux/feather"]["useful"],
              "maximum_ordinary_delivery_flags": sum(tally["flags"] for key, tally in counts.items()
                                                     if key.split("/")[1] in ("ordinary", "global_tone", "resampled")),
              "maximum_unrelated_admitted": sum(tally["comparable"] for key, tally in counts.items()
                                                 if key.split("/")[1] in ("missing", "foreign")),
              "maximum_malformed_admitted": sum(tally["comparable"] for key, tally in counts.items()
                                                  if key.startswith("malformed/"))}
    return {"counts": counts, "gate_actual": actual,
            "passed": all(actual[key] >= n if key.startswith("minimum") else actual[key] <= n for key, n in GATE.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    parser.add_argument("run", type=Path)
    data = Path("backend/data/independent-evaluation")
    for name, path in (("lama", "image-bg-real-acquisition-v225"),
                       ("flux-earlier", "image-residual-patch-held-2026-11-11/plan.json"),
                       ("flux-cache", "image-fresh-source-cache-v39/private-groups.json")):
        parser.add_argument("--" + name, type=Path, default=data / path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    modules = (localization, png_exif_thumbnail, sources, earlier)
    path = args.run / "freeze.json"
    frozen = {"schema": SCHEMA, "code": {Path(m.__file__).name: sha(Path(m.__file__).read_bytes()) for m in modules}
              | {Path(__file__).name: sha(Path(__file__).read_bytes())}, "cases": [binding(case) for case in cases(args)],
              "runtime": {"numpy": np.__version__, "opencv": cv2.__version__}, "gate": GATE, "confidence_weight": 0}
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        write(path, frozen)
        print("Frozen", len(frozen["cases"]), sha(path.read_bytes()), flush=True)
        return
    need(read(path) == frozen, "Frozen inputs or implementation differ")
    rows = [measure(item, committed) for item, committed in zip(cases(args), frozen["cases"], strict=True)]
    actual = {"freeze_sha256": sha(path.read_bytes()), "summary": summary(rows), "rows": rows}
    outcome = args.run / "result.json"
    if args.stage == "review":
        need(read(outcome) == actual, "Native replay differs")
    else:
        write(outcome, actual)
    print(json.dumps(actual["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
