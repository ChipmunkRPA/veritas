"""Read a subtotal, tax and total from an opaque screenshot."""

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
from typing import NamedTuple

from PIL import Image, ImageChops, ImageOps

from app.evidence import image_png_layout
from app.evidence import image_printed_totals_binding as binding

LABELS = {
    "subtotal": "subtotal",
    "subtotal before tax": "subtotal",
    "tax": "tax",
    "sales tax": "tax",
    "total": "total",
    "grand total": "total",
    "invoice total": "total",
}
ROLES = ("subtotal", "tax", "total")
AMOUNT = rf"(?P<currency>[{binding.CURRENCIES}])\s*(?P<amount>{binding.MONEY})"
LABEL = "|".join(sorted(LABELS, key=len, reverse=True))
LINE = re.compile(rf"(?P<label>{LABEL})\s*:?\s+{AMOUNT}", re.I | re.ASCII)
PREFIX = re.compile(rf"(?P<label>{LABEL})\s*:?\s+(?=\S)", re.I | re.ASCII)
MONEY_SEEN = re.compile(r"[$£€]\s*[0-9]", re.ASCII)
TAX_INCLUDED = re.compile(
    r"\b(?:tax\s+included|included\s+tax|includes\s+tax|inclusive\s+(?:of\s+)?tax|"
    r"VAT\s+included|includes\s+VAT|inc\.?\s+VAT)\b",
    re.I | re.ASCII,
)
ADJUSTMENT = re.compile(
    r"\b(?:discounts?|coupons?|credits?|shipping|delivery|duty|tips?|fees?|charges?|"
    r"rounding|adjustments?|refunds?|balance|amount\s+due|previous\s+payment)\b",
    re.I | re.ASCII,
)

TIMEOUT = 8.0


class Word(NamedTuple):
    text: str
    box: tuple[int, int, int, int]
    confidence: float


class Reading(NamedTuple):
    role: str
    currency: str | None
    printed: str | None
    cents: int | None
    crop: tuple[int, int, int, int]
    amount: tuple[int, int, int, int]
    label: tuple[int, int, int, int]
    minimum_quality: float
    amount_quality: float


def union(words):
    l, t, r, b = zip(*(w.box for w in words), strict=True)
    return min(l), min(t), max(r), max(b)


def ocr(binary, image, psm, end):
    remaining = end - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Local text reading exceeded its time budget")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    with tempfile.TemporaryFile() as f:
        subprocess.run(
            [
                binary,
                "stdin",
                "stdout",
                "--tessdata-dir",
                str(image_png_layout.TESSDATA),
                "-l",
                "eng",
                "--psm",
                str(psm),
                "-c",
                "tessedit_create_tsv=1",
            ],
            input=buf.getvalue(),
            stdout=f,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=remaining,
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},
        )
        f.seek(0)
        data = f.read(192_001)
    if len(data) > 192_000:
        raise ValueError("Local text reader output exceeds its bound")
    out = []
    for row in csv.DictReader(io.StringIO(data.decode()), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row["level"] != "5" or not (text := row["text"].strip()):
            continue
        x, y, w, h = [int(row[key]) for key in ("left", "top", "width", "height")]
        if not (0 <= x < x + w <= image.width and 0 <= y < y + h <= image.height):
            raise ValueError("OCR box lies outside source image")
        out.append(Word(text, (x, y, x + w, y + h), float(row["conf"])))
    if len(out) > 420:
        raise ValueError("Local text reader produced too many words")
    return out


def center(word):
    return (word.box[1] + word.box[3]) / 2


def rows(words):
    grouped = []
    for item in sorted(words, key=center):
        prior = grouped[-1][0] if grouped else None
        item_height = item.box[3] - item.box[1]
        if prior and center(item) - center(prior) <= max(
            4, min(item_height, prior.box[3] - prior.box[1]) * 0.56
        ):
            grouped[-1].append(item)
        else:
            grouped.append([item])
    return [sorted(row, key=lambda w: w.box[0]) for row in grouped]


def transcript(row):
    return " ".join(w.text for w in row)


def field(row):
    text = transcript(row)
    prefix = PREFIX.match(text)
    if prefix is None:
        return None
    matched = LINE.fullmatch(text)
    label = []
    amount = []
    start = 0
    for word in row:
        stop = start + len(word.text)
        if start < prefix.end("label"):
            label.append(word)
        if start >= prefix.end() and not word.text.isspace():
            amount.append(word)
        start = stop + 1
    if not label or not amount:
        return None
    label_box, amount_box = union(label), union(amount)
    if amount_box[0] - label_box[2] < max(21, label_box[3] - label_box[1]):
        return None
    symbol = matched["currency"] if matched else None
    printed = matched["amount"] if matched else None
    return Reading(
        LABELS[prefix["label"].lower()],
        symbol,
        printed,
        binding.cash_value(printed) if printed else None,
        union(row),
        amount_box,
        label_box,
        min(w.confidence for w in label),
        min(w.confidence for w in amount),
    )


def adjustments(lines, fields):
    texts = [transcript(line) for line in lines]
    if TAX_INCLUDED.search(" ".join(texts)) or any(
        ADJUSTMENT.search(t) and MONEY_SEEN.search(t) for t in texts
    ):
        return True
    if not fields:
        return False
    known = {f.crop for f in fields}
    after = min(f.crop[1] for f in fields)
    edges = [f.amount[2] for f in fields]
    widest = max(f.crop[3] - f.crop[1] for f in fields)
    for text, row in zip(texts, lines, strict=True):
        box = union(row)
        if box in known or box[1] < after:
            continue
        if MONEY_SEEN.search(text):
            return True
        if (
            len(fields) >= 2
            and box[2] - box[0] >= 2 * widest
            and min(abs(box[2] - edge) for edge in edges) <= max(12, widest)
            and box[0] < min(f.amount[0] for f in fields) - widest
        ):
            return True
    return False


def find(binary, image, end, *, native_lines=None):
    lines = rows(ocr(binary, image, 11, end)) if native_lines is None else native_lines
    fields = [item for row in lines if (item := field(row)) is not None]
    if adjustments(lines, fields) or len(fields) != len({f.role for f in fields}):
        return "not_supported", None
    if [item.role for item in fields] != list(ROLES) or min(f.minimum_quality for f in fields) < 78:
        if image.width * image.height > 1_500_000:
            return "not_supported", None
        base = {
            f.role: (f.currency, f.printed)
            for f in fields
            if f.printed is not None and f.amount_quality >= 87
        }
        with ImageOps.grayscale(image) as gray:
            level = gray.resize((1, 1)).getpixel((0, 0))
            with ImageOps.autocontrast(gray, cutoff=0.2) as contrasted:
                normal = ImageOps.invert(contrasted) if level < 128 else contrasted.copy()
                with normal:
                    with normal.resize(
                        (2 * normal.width, 2 * normal.height), Image.Resampling.BICUBIC
                    ) as large:
                        words = ocr(binary, large, 11, end)
        native = [
            Word(
                w.text, (w.box[0] // 2, w.box[1] // 2, (w.box[2] + 1) // 2, (w.box[3] + 1) // 2), w.confidence
            )
            for w in words
        ]
        lines = rows(native)
        fields = [item for line in lines if (item := field(line)) is not None]
        if adjustments(lines, fields) or any(
            f.role in base
            and f.printed is not None
            and f.amount_quality >= 87
            and (f.currency, f.printed) != base[f.role]
            for f in fields
        ):
            return "not_supported", None
    if [item.role for item in fields] != list(ROLES):
        return "not_supported", None
    heights = [b - t for _, t, _, b in (f.crop for f in fields)]
    max_height = max(heights)
    if min(heights) < 12 or max_height > 72 or min(f.minimum_quality for f in fields) < 78:
        return "not_supported", None
    l, t, r, b = union([Word("", item.crop, 100.0) for item in fields])
    if (
        r - l > 1000
        or b - t > 6 * max_height
        or any(
            a.crop[3] + 4 > z.crop[1] or z.crop[1] - a.crop[3] > 2 * max_height
            for a, z in zip(fields, fields[1:])
        )
        or max(f.label[0] for f in fields) - min(f.label[0] for f in fields) > max(9, max_height * 0.55)
        or max(f.amount[2] for f in fields) - min(f.amount[2] for f in fields) > max(9, max_height * 0.55)
    ):
        return "not_supported", None
    candidate_box = set(item.crop for item in fields)
    if any(
        t < union(line)[3]
        and union(line)[1] < b
        and l < union(line)[2]
        and union(line)[0] < r
        and union(line) not in candidate_box
        for line in lines
    ):
        return "not_supported", None
    verified = []
    for item in fields:
        x, top, right, bottom = item.amount
        base = (
            (item.currency, item.printed) if item.amount_quality >= 87 and item.printed is not None else None
        )
        good = []
        with image.crop(
            (max(0, x - 6), max(0, top - 6), min(image.width, right + 6), min(image.height, bottom + 6))
        ) as crop:
            for scale, segmenter in ((2, 8), (3, 7)):
                with crop.resize((scale * crop.width, scale * crop.height), Image.Resampling.BICUBIC) as zoom:
                    words = ocr(binary, zoom, segmenter, end)
                reading = re.fullmatch(AMOUNT, transcript(words))
                quality = min((w.confidence for w in words), default=0.0)
                if reading is None:
                    continue
                current = (reading["currency"], reading["amount"])
                if quality >= 62 and (
                    (base is not None and base != current) or (good and good[0] != current)
                ):
                    return "not_supported", None
                if quality >= 83:
                    good.append(current)
                    if base is not None:
                        break
        if len(good) < (1 if base is not None else 2):
            return "not_supported", None
        symbol, printed = good[0]
        verified.append(item._replace(currency=symbol, printed=printed, cents=binding.cash_value(printed)))
    fields = verified
    if len({item.currency for item in fields}) != 1:
        return "not_supported", None
    delta = fields[2].cents - fields[0].cents - fields[1].cents
    return ("mismatch" if abs(delta) >= binding.MIN_DELTA_CENTS else "match"), {
        "currency": fields[0].currency,
        "calculated_cents": fields[0].cents + fields[1].cents,
        "delta_cents": delta,
        "fields": {
            f.role: {"printed": f.printed, "source_region": list(f.crop), "amount_region": list(f.amount)}
            for f in fields
        },
    }


def inspect(path, source_sha256, time_budget):
    from app.evidence import image_printed_euro_totals

    binary = shutil.which("tesseract")
    if binary is None or time_budget <= 0:
        return None
    end = time.monotonic() + min(TIMEOUT, time_budget)
    try:
        with (image_png_layout.TESSDATA / "eng.traineddata").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != image_png_layout.ENGLISH_SHA256:
                return None
        with Image.open(path, formats=["PNG"]) as original:
            if (
                original.mode not in ("RGB", "L")
                or "transparency" in original.info
                or original.getexif().get(274, 1) != 1
                or getattr(original, "n_frames", 1) > 1
                or not binding.size_supported(original.size)
            ):
                return None
            with original.convert("RGB") as photo:
                native_lines = rows(ocr(binary, photo, 11, end))
                if record := image_printed_euro_totals.find(binary, photo, native_lines, end):
                    if abs(record["delta_cents"]) < binding.MIN_DELTA_CENTS:
                        return None
                    return {"source_sha256": source_sha256, "affects_overall_confidence": False, **record}
                status, reading = find(binary, photo, end, native_lines=native_lines)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        Image.DecompressionBombError,
        subprocess.SubprocessError,
    ):
        return None
    return (
        {
            "schema": binding.SCHEMA,
            "source_sha256": source_sha256,
            "affects_overall_confidence": False,
            **reading,
        }
        if status == "mismatch"
        else None
    )


def verify_crops(source, evidence, read_asset):
    for role, field in evidence["record"]["fields"].items():
        digest = evidence["artifacts"][role]["sha256"]
        with (
            source.crop(tuple(field["source_region"])) as expected,
            Image.open(io.BytesIO(read_asset(digest)), formats=["PNG"]) as crop,
        ):
            if (
                crop.size != expected.size
                or crop.mode != "RGB"
                or ImageChops.difference(crop, expected).getbbox()
            ):
                raise ValueError("Printed total crop differs from the original image")
