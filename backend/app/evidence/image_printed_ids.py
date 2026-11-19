"""Read two explicitly linked identifiers in a submitted screenshot."""
from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict

from PIL import Image, ImageChops

from app.evidence import image_png_layout
from app.evidence import image_printed_ids_binding as binding

LINE = re.compile(rf"(?P<confirm>confirm\s+)?(?P<field>(?:reference|record|invoice)\s+id)\s*:\s*(?P<value>{binding.IDENTIFIER})",
                  re.IGNORECASE | re.ASCII)
MAX_OUTPUT = 128_000
TIMEOUT = 10.


def _ocr(binary, image, mode, end):
    png = io.BytesIO()
    image.save(png, format="PNG")
    remaining = end - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Printed identifier OCR exceeded its time limit")
    with tempfile.TemporaryFile() as output:
        subprocess.run([binary, "stdin", "stdout", "--tessdata-dir", str(image_png_layout.TESSDATA), "-l", "eng",
                        "--psm", str(mode), "-c", "tessedit_create_tsv=1"], input=png.getvalue(),
                       stdout=output, stderr=subprocess.DEVNULL, check=True, timeout=min(6., remaining),
                       env={**os.environ, "OMP_THREAD_LIMIT": "1"})
        output.seek(0)
        raw = output.read(MAX_OUTPUT + 1)
    if len(raw) > MAX_OUTPUT:
        raise ValueError("Printed identifier OCR output exceeds its size limit")
    lines = defaultdict(list)
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row["level"] != "5" or not (text := row["text"].strip()):
            continue
        x, y, width, height = (int(row[key]) for key in ("left", "top", "width", "height"))
        if not (0 <= x < x + width <= image.width and 0 <= y < y + height <= image.height):
            raise ValueError("Printed identifier OCR box is outside the source")
        lines[tuple(row[key] for key in ("block_num", "par_num", "line_num"))].append(
            ((x, y, x + width, y + height), text, float(row["conf"])))
    for line in lines.values():
        line.sort(key=lambda row: row[0][0])
        expression = " ".join(text for _, text, _ in line)
        match = LINE.fullmatch(expression)
        if match is None:
            continue
        left, top, right, bottom = zip(*(box for box, _, _ in line), strict=True)
        box = [min(left), min(top), max(right), max(bottom)]
        scale = 2 if mode == 7 else 1
        eligible = (all(confidence >= 80 for _, _, confidence in line)
                    and 12 * scale <= box[3] - box[1] <= 64 * scale and box[2] - box[0] <= 920 * scale)
        yield (" ".join(match["field"].lower().split()), bool(match["confirm"]), match["value"].upper(), box, eligible)


def _find(binary, image, end):
    grouped = defaultdict(list)
    for field, confirmation, value, box, eligible in _ocr(binary, image, 11, end):
        grouped[field].append((confirmation, value, box, eligible))
    pairs = []
    for field, rows in grouped.items():
        if len(rows) != 2 or {row[0] for row in rows} != {False, True} or not all(row[3] for row in rows):
            continue
        (_, first, upper, _), (_, second, lower, _) = sorted(rows)
        if first != second and upper[3] + 5 <= lower[1]:
            pairs.append((field, ((first, upper), (second, lower))))
    if len(pairs) != 1:
        return None
    field, pair = pairs[0]
    values = {}
    for confirmation, (role, (value, box)) in enumerate(zip(binding.PLATES, pair, strict=True)):
        x, y, right, bottom = box
        with image.crop((max(0, x - 7), max(0, y - 7), min(image.width, right + 7), min(image.height, bottom + 7))) as crop:
            with crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC) as enlarged:
                repeated = [(kind, is_confirmation, printed) for kind, is_confirmation, printed, _, eligible
                            in _ocr(binary, enlarged, 7, end) if eligible]
        if repeated != [(field, bool(confirmation), value)]:
            return None
        values[role] = {"value": value, "region": box}
    return {"field": binding.FIELDS[field], "values": values}


def inspect(path, sha256, time_budget):
    binary = shutil.which("tesseract")
    if binary is None or time_budget <= 0:
        return None
    end = time.monotonic() + min(TIMEOUT, time_budget)
    try:
        with Image.open(path, formats=["PNG", "JPEG"]) as source:
            if (source.mode not in {"RGB", "L"} or source.getexif().get(274, 1) != 1 or getattr(source, "n_frames", 1) != 1
                    or "transparency" in source.info or not binding.size_supported(source.size)):
                return None
            with (image_png_layout.TESSDATA / "eng.traineddata").open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != image_png_layout.ENGLISH_SHA256:
                    return None
            with source.convert("RGB") as image:
                comparison = _find(binary, image, end)
    except (OSError, ValueError, KeyError, TypeError, Image.DecompressionBombError, subprocess.SubprocessError):
        return None
    return ({"schema": binding.SCHEMA, "source_sha256": sha256, "affects_overall_confidence": False, **comparison}
            if comparison else None)


def verify_crops(source, evidence, read_asset):
    for role, item in evidence["record"]["values"].items():
        with (source.crop(tuple(item["region"])) as expected,
              Image.open(io.BytesIO(read_asset(evidence["artifacts"][role]["sha256"])), formats=["PNG"]) as crop):
            if crop.size != expected.size or crop.mode != "RGB" or ImageChops.difference(crop, expected).getbbox():
                raise ValueError("Printed identifier crop differs from the original image")
