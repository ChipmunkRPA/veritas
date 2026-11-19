# SPDX-License-Identifier: Apache-2.0
"""Compare a static local form value with the literal text painted by its saved widget."""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import zlib
from contextlib import suppress
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pymupdf as pdf
from PIL import Image, ImageChops

from app.evidence import document_pdf_form_binding as binding

MAX_BYTES = 2 * 1024 * 1024
MAX_OBJECTS = 1000
MAX_APPEARANCE_BYTES = 4096
DIFFERENCE = 50
PAINT = re.compile(rb"\(([A-Za-z0-9 ]{5,54})\)[\t\r\n ]+Tj\b")
TEXT_OPERATOR = re.compile(rb"(?<![A-Za-z0-9/])(?:Tj|TJ|Tf|Tr|BT|ET)\b")
SAFE_FONT = re.compile(rb"/([A-Za-z0-9.-]{1,30})\s+[1-9][0-9]*(?:\.[0-9]+)?\s+Tf\b")
REFERENCE = re.compile(r"(\d+) 0 R\Z")
ARRAY_NUMBER = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)")
ALLOW_FLAGS = 12582919  # read-only, required, no export, do not spellcheck/scroll
BASE_FONTS = ("/Helvetica", "/Helvetica-Bold", "/Courier", "/Courier-Bold", "/Times-Roman", "/Times-Bold")


def ref(value):
    kind, raw = value
    match = REFERENCE.fullmatch(raw) if kind == "xref" else None
    return int(match[1]) if match else None


def array(value):
    kind, raw = value
    if kind != "array" or not re.fullmatch(r"\[[ 0-9.\-]+\]", raw):
        return ()
    return tuple(float(number) for number in ARRAY_NUMBER.findall(raw))


def allowed_document(doc):
    catalog = doc.pdf_catalog()
    if (
        doc.needs_pass
        or doc.is_repaired
        or not 1 <= len(doc) <= binding.MAX_PAGES
        or doc.xref_length() > MAX_OBJECTS
        or doc.xref_get_key(catalog, "AcroForm")[0] not in {"dict", "xref"}
    ):
        return False
    if doc.xref_get_key(catalog, "AcroForm/NeedAppearances") not in {("null", "null"), ("bool", "false")}:
        return False
    if any(
        doc.xref_get_key(catalog, key)[0] != "null"
        for key in ("AcroForm/XFA", "AcroForm/CO", "Names/JavaScript", "AA", "OCProperties")
    ):
        return False
    goto = doc.xref_get_key(catalog, "OpenAction")
    if goto[0] not in ("null", "array"):
        navigation = ref(goto)
        if (
            navigation is None
            or set(doc.xref_get_keys(navigation)) != {"S", "D"}
            or doc.xref_get_key(navigation, "S") != ("name", "/GoTo")
            or doc.xref_get_key(navigation, "D")[0] != "array"
        ):
            return False
    # A first bounded static version does not consider any scripted document.
    for xref in range(1, doc.xref_length()):
        with suppress(RuntimeError, ValueError):
            if (
                doc.xref_get_key(xref, "S") == ("name", "/JavaScript")
                or doc.xref_get_key(xref, "JS")[0] != "null"
            ):
                return False
    return True


def references(value):
    kind, raw = value
    if kind != "array" or not re.fullmatch(r"\[(?:\s*\d+ 0 R)*\s*\]", raw):
        return ()
    return tuple(int(match) for match in re.findall(r"(\d+) 0 R", raw))


def effective(doc, xref, roots):
    # Accept a merged field/widget or an exact one-child terminal field.
    parent = doc.xref_get_key(xref, "Parent")
    where = xref
    if parent[0] != "null":
        where = ref(parent)
        if (
            where is None
            or where not in roots
            or doc.xref_get_key(xref, "V")[0] != "null"
            or references(doc.xref_get_key(where, "Kids")) != (xref,)
            or doc.xref_get_key(where, "FT") != ("name", "/Tx")
            or any(doc.xref_get_key(where, key)[0] != "null" for key in ("AA", "A", "Parent", "OC", "RV"))
        ):
            return ("null", "null")
    return doc.xref_get_key(where, "V") if roots.count(where) == 1 else ("null", "null")


def compatible_encoding(doc, font):
    encoding = doc.xref_get_key(font, "Encoding")
    if encoding in {("null", "null"), ("name", "/WinAnsiEncoding"), ("name", "/StandardEncoding")}:
        return True
    enc = ref(encoding)
    if enc is None or doc.xref_get_key(enc, "BaseEncoding") not in {
        ("null", "null"),
        ("name", "/WinAnsiEncoding"),
        ("name", "/StandardEncoding"),
    }:
        return False
    kind, raw = doc.xref_get_key(enc, "Differences")
    if (
        kind != "array"
        or len(raw) > 8192
        or not re.fullmatch(r"\[(?:\s*(?:[0-9]+|/[A-Za-z0-9_.-]+))*\s*\]", raw)
    ):
        return False
    pin = {32, *range(48, 58), *range(65, 91), *range(97, 123)}
    cursor = None
    for item in re.findall(r"\d+|/[A-Za-z0-9_.-]+", raw):
        if item.isdecimal():
            cursor = int(item)
        elif cursor is None or cursor not in range(256) or cursor in pin:
            return False
        else:
            cursor += 1
    return True


def rect_contains(inner, outer):
    (a, b, c, d), (x, y, z, w) = inner, outer
    return x - 0.03 <= a < c <= z + 0.03 and y - 0.03 <= b < d <= w + 0.03


def found_trace(traces, rectangle, painted):
    spans = []
    for span in traces:
        if not rect_contains(span["bbox"], rectangle):
            continue
        if (
            span["type"] != 0
            or span["opacity"] != 1.0
            or span["dir"] != (1.0, 0.0)
            or span["layer"]
            or span["size"] < 7
            or not span["chars"]
        ):
            continue
        if "".join(chr(item[0]) for item in span["chars"]) == painted:
            spans.append(span)
    if len(spans) == 1:
        return spans[0]["chars"]
    return None


def pdf_crop(page, rectangle):
    pixmap = page.get_pixmap(
        matrix=pdf.Matrix(binding.SCALE, binding.SCALE), clip=rectangle, colorspace=pdf.csGRAY, alpha=False
    )
    return Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples), pixmap.x, pixmap.y


def pixel_proof(source, page, traces, box, ap, stream, match, painted):
    chars = found_trace(traces, box, painted)
    if chars is None:
        return None
    initial, x, y = pdf_crop(page, box)
    # Only in a copy, remove the string-drawing operation from the same AP stream.
    # Its local clipping, border, background, text placement and visibility remain.
    with pdf.open(stream=source, filetype="pdf") as counterfactual:
        counterfactual.update_stream(ap, stream[: match.start()] + b"() Tj" + stream[match.end() :])
        blank, x_blank, y_blank = pdf_crop(counterfactual[page.number], box)
    if (x_blank, y_blank) != (x, y) or initial.size != blank.size:
        return None
    diff = ImageChops.difference(initial, blank).point([0] * DIFFERENCE + [255] * (256 - DIFFERENCE))
    if diff.getbbox() is None:
        return None
    for letter, shape in zip(painted, chars, strict=True):
        if letter == " ":
            continue
        a, b, c, d = shape[3]
        w = c - a
        bounds = (
            int((a + 0.10 * w) * binding.SCALE) - x,
            int(b * binding.SCALE) - y,
            int((c - 0.10 * w) * binding.SCALE) - x,
            int(d * binding.SCALE) - y,
        )
        if diff.crop(bounds).histogram()[255] < 4:
            return None
    return initial


def _appearance_bytes(doc, xref):
    if doc.xref_get_key(xref, "DecodeParms")[0] != "null":
        return None
    encoding = doc.xref_get_key(xref, "Filter")
    if encoding not in {("null", "null"), ("name", "/FlateDecode")}:
        return None
    raw = doc.xref_stream_raw(xref)
    if encoding[0] != "null":
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(raw, MAX_APPEARANCE_BYTES)
        except zlib.error:
            return None
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            return None
    return raw if 0 < len(raw) < MAX_APPEARANCE_BYTES else None


def appearance(doc, widget, rectangle):
    xref = widget.xref
    ap = ref(doc.xref_get_key(xref, "AP/N"))
    if ap is None or not doc.xref_is_stream(ap) or doc.xref_get_key(ap, "Subtype") != ("name", "/Form"):
        return None
    if any(
        doc.xref_get_key(ap, key)[0] != "null"
        for key in ("Group", "OC", "Resources/ExtGState", "Resources/XObject")
    ):
        return None
    bbox = array(doc.xref_get_key(ap, "BBox"))
    if (
        len(bbox) != 4
        or bbox[:2] != (0.0, 0.0)
        or abs(bbox[2] - rectangle.width) > 0.05
        or abs(bbox[3] - rectangle.height) > 0.05
        or (matrix := doc.xref_get_key(ap, "Matrix"))[0] != "null"
        and array(matrix) != (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    ):
        return None
    stream = _appearance_bytes(doc, ap)
    if stream is None or TEXT_OPERATOR.findall(stream) != [b"BT", b"Tf", b"Tj", b"ET"]:
        return None
    fields = list(PAINT.finditer(stream))
    fonts = SAFE_FONT.findall(stream)
    if len(fields) != 1 or len(fonts) != 1:
        return None
    font = ref(doc.xref_get_key(ap, "Resources/Font/" + fonts[0].decode("ascii")))
    if (
        font is None
        or doc.xref_get_key(font, "Subtype") != ("name", "/Type1")
        or doc.xref_get_key(font, "BaseFont")[1] not in BASE_FONTS
        or not compatible_encoding(doc, font)
        or any(doc.xref_get_key(font, key)[0] != "null" for key in ("ToUnicode", "FontDescriptor"))
    ):
        return None
    return ap, stream, fields[0], fields[0][1].decode("ascii")


def _png(crop):
    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    return buffer.getvalue()


def _field(doc, widget, others, page_size, roots):
    xref, box = widget.xref, widget.rect
    if (
        widget.field_type != pdf.PDF_WIDGET_TYPE_TEXT
        or widget.field_flags & ~ALLOW_FLAGS
        or not binding.rectangles(list(box), page_size)
        or doc.xref_get_key(xref, "F") not in {("int", "4"), ("int", "0"), ("null", "null")}
        or any(
            doc.xref_get_key(xref, name)[0] != "null"
            for name in ("AA", "A", "OC", "RV", "Kids", "AP/D", "AP/R")
        )
        or any(other != xref and box.intersects(rect) for other, rect in others)
    ):
        return None
    kind, saved = effective(doc, xref, roots)
    if (
        kind != "string"
        or len(saved) > 54
        or not binding.WORDING.fullmatch(saved.strip(" "))
        or (paint := appearance(doc, widget, box)) is None
    ):
        return None
    render_text = paint[-1]
    if (
        not binding.WORDING.fullmatch(render_text.strip(" "))
        or len(saved.split()) != len(render_text.split())
        or binding.normalized(saved) == binding.normalized(render_text)
    ):
        return None
    return saved, paint


def capture(path: Path, *, source_sha256: str):
    """Return a bounded, positive-only inspection and in-memory field crops."""
    if not 0 < path.stat().st_size <= MAX_BYTES:
        return None, {}
    with path.open("rb") as stream:
        source = stream.read(MAX_BYTES + 1)
    if len(source) > MAX_BYTES or sha256(source).hexdigest() != source_sha256:
        return None, {}
    rows, crops, candidates = [], {}, 0
    with pdf.open(stream=source, filetype="pdf") as doc:
        if not allowed_document(doc):
            return None, {}
        roots = references(doc.xref_get_key(doc.pdf_catalog(), "AcroForm/Fields"))
        if not roots or len(set(roots)) != len(roots):
            return None, {}
        for page in doc:
            if (
                page.rotation
                or page.cropbox_position != pdf.Point(0, 0)
                or doc.xref_get_key(page.xref, "AA")[0] != "null"
            ):
                continue
            page_size = [round(page.rect.width, 6), round(page.rect.height, 6)]
            widgets, annots = list(page.widgets()), list(page.annots())
            if len(widgets) + len(annots) > binding.MAX_WIDGETS:
                continue
            others = [(item.xref, item.rect) for item in (*widgets, *annots)]
            traces = None
            for widget in sorted(widgets, key=lambda field: field.xref):
                if (values := _field(doc, widget, others, page_size, roots)) is None:
                    continue
                candidates += 1
                if candidates > binding.MAX_REVIEWED:
                    return None, {}
                if traces is None:
                    traces = page.get_texttrace()
                saved, paint = values
                if (crop := pixel_proof(source, page, traces, widget.rect, *paint)) is None:
                    continue
                encoded = _png(crop)
                if len(encoded) > binding.MAX_CROP_BYTES:
                    return None, {}
                info = {
                    "sha256": sha256(encoded).hexdigest(),
                    "pixels_sha256": sha256(crop.tobytes()).hexdigest(),
                    "width": crop.width,
                    "height": crop.height,
                }
                crops[binding.crop_role(len(rows))] = encoded
                rows.append(
                    {
                        "page": page.number + 1,
                        "xref": widget.xref,
                        "page_size": page_size,
                        "box": [round(value, 6) for value in widget.rect],
                        "saved_value": saved,
                        "display_value": paint[-1],
                        "crop": info,
                    }
                )
    if not rows:
        return None, {}
    record = {"schema": binding.SCHEMA, "affects_overall_confidence": False, "rows": rows}
    binding.validate_record(record)
    return record, crops


def inspect_within(path: Path, *, timeout: float, source_sha256: str):
    """A timeout or unsupported form withholds this optional observation."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed module and server-staged source file
            [sys.executable, "-I", "-m", "app.evidence.document_pdf_form_values", str(path), source_sha256],
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        record, crops = json.loads(result.stdout)
        if not record:
            return None, {}
        binding.validate_record(record)
        encoded = {role: base64.b64decode(data, validate=True) for role, data in crops.items()}
        expected = {binding.crop_role(i): row["crop"]["sha256"] for i, row in enumerate(record["rows"])}
        if {role: sha256(data).hexdigest() for role, data in encoded.items()} != expected:
            raise ValueError("Rendered form crops differ from the isolated read")
        return record, encoded
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, {}


def _native_record(record):
    if record is None:
        return None
    return {
        **record,
        "rows": [
            {**row, "crop": {key: value for key, value in row["crop"].items() if key != "sha256"}}
            for row in record["rows"]
        ],
    }


def verify_original(document, archive, original):
    record = document[binding.KEY]["inspection"]
    reread, _ = inspect_within(original, timeout=8, source_sha256=document["source"]["sha256"])
    if _native_record(reread) != _native_record(record):
        raise ValueError("Saved PDF form fields or their original raster coordinates differ")
    for row in record["rows"]:
        crop = row["crop"]
        data = archive.read(f"assets/{crop['sha256']}.png")
        verify_crop(crop, data)


def verify_crop(crop, data):
    with Image.open(BytesIO(data)) as image:
        if (
            image.format != "PNG"
            or image.mode != "L"
            or image.size != (crop["width"], crop["height"])
            or sha256(image.tobytes()).hexdigest() != crop["pixels_sha256"]
        ):
            raise ValueError("A saved PDF form raster differs from the original pixels")


if __name__ == "__main__":
    inspection, artefacts = capture(Path(sys.argv[1]), source_sha256=sys.argv[2])
    print(
        json.dumps(
            (inspection, {role: base64.b64encode(data).decode("ascii") for role, data in artefacts.items()})
        )
    )
