"""Document text extraction preserving paragraph boundaries and character offsets.

This file imports nothing of the application: `pipeline.extract_within` runs it as a script in a child process
(`python -I extract.py <path> <ext>`), which prints the document, or the error, as one JSON object (N-10; SEC-04)."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from app.detectors.text import html_article
from app.evidence import document_native_locations, document_pdf_visibility
from app.evidence.document_scope import EMAIL_HTML_COORDINATES, valid_html_article_scope


@dataclass
class Document:
    text: str
    paragraphs: list[tuple[int, int]]  # (start, end) char offsets into text
    meta: dict = field(default_factory=dict)
    source_map: list[dict] = field(default_factory=list)
    source_rasters: dict[str, str] = field(default_factory=dict)  # SHA256 -> base64 exact OCR input PNG, child transport only
    word_review: dict | None = None


SOURCE_MAP_VERSION = "document-source/1"
EXTRACTION_VERSION = "document-extraction/4"
WORD_CONTENT_CONTROLS_KEY = "word_content_controls"
WORD_CONTENT_CONTROLS = "stored-text/1"
OCR_VERSION = "local-ocr/1"
OCR_MIN_MEAN_CONFIDENCE = 75.0
OCR_MAX_PAGES = 12
OCR_MAX_PIXELS = 4_000_000
OCR_MAX_RASTER_BYTES = 24 * 1024 * 1024
OCR_BUDGET_SECONDS = 70.0
NATIVE_COMPARISON_MAX_PAGES = 12


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _normalise(raw: str, *, dehyphenate: bool = False) -> tuple[str, list[tuple[int, int]]]:
    """Normalise whitespace while retaining each output codepoint's source range."""
    omitted = {m.start() + len(m.group(1)) for m in re.finditer(r"(\w)-\n(?=\w)", raw)} if dehyphenate else set()
    omitted |= {i + 1 for i in list(omitted)}
    chars: list[str] = []
    origins: list[tuple[int, int]] = []
    for i, ch in enumerate(raw):
        if i in omitted:
            continue
        if ch.isspace():
            if chars and chars[-1] == " ":
                origins[-1] = (origins[-1][0], i + 1)
            else:
                chars.append(" ")
                origins.append((i, i + 1))
        else:
            chars.append(ch)
            origins.append((i, i + 1))
    start, end = (1 if chars and chars[0] == " " else 0), len(chars)
    if end and chars[-1] == " ":
        end -= 1
    return "".join(chars[start:end]), origins[start:end]


def _source_ranges(raw, text, origins, pos, *, base=0, html_origins=None):
    ranges = []
    previous_direct = False
    for i, (a, b) in enumerate(origins):
        lo, hi = (html_origins[a][0], html_origins[b - 1][1]) if html_origins is not None else (base + a, base + b)
        row = [pos + i, pos + i + 1, lo, hi]
        direct = b - a == hi - lo == 1 and raw[a:b] == text[i]
        if ranges and previous_direct and direct and ranges[-1][1] == row[0] and ranges[-1][3] == row[2]:
            ranges[-1][1], ranges[-1][3] = row[1], row[3]
        else:
            ranges.append(row)
        previous_direct = direct
    return ranges


def _paragraph_groups(parts):
    group = []
    for part in parts:
        group.append(part)
        if not part[1].get("join_following_paragraph"):
            yield group
            group = []
    if group:
        raise ValueError("A deleted Word paragraph mark has no following paragraph")


def _joined_word_mapping(group, text, origins, pos):
    source_start = first = 0
    for raw, source in group:
        source_end = source_start + len(raw)
        last = first
        while last < len(origins) and origins[last][0] < source_end:
            last += 1
        if first < last:
            # When whitespace straddles the deleted break, the first actual
            # source of that space anchors the single normalized character.
            selected = [(a - source_start, min(b, source_end) - source_start) for a, b in origins[first:last]]
            entry = {key: value for key, value in source.items() if key != "join_following_paragraph"}
            yield {**entry, "start": pos + first, "end": pos + last, "offset_unit": "unicode_codepoint",
                   "source_ranges": _source_ranges(raw, text[first:last], selected, pos + first)}
        first, source_start = last, source_end


def _join_mapped(parts: list[tuple[str, dict]], *, dehyphenate: bool = False, map_paragraphs: bool = False) -> Document:
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    mapping: list[dict] = []
    pos = 0
    for group in _paragraph_groups(parts):
        raw, source = group[0]
        if len(group) > 1:
            raw = "".join(piece for piece, _ in group)
        text, origins = _normalise(raw, dehyphenate=dehyphenate)
        if not text:
            continue
        if texts:
            pos += 2
        spans.append((pos, pos + len(text)))
        texts.append(text)
        if len(group) > 1:
            mapping.extend(_joined_word_mapping(group, text, origins, pos))
            pos += len(text)
            continue
        entry = {k: v for k, v in source.items() if k not in {"glyphs", "ocr_words", "html_origins"}}
        entry.update(start=pos, end=pos + len(text), offset_unit="unicode_codepoint")
        if map_paragraphs or source.get("kind") in {"text", "email", "docx"} or source.get("html_origins") is not None:
            # Consecutive 1:1 stretches are compacted; collapsed whitespace keeps
            # its entire source range. These are decoded codepoints, not bytes.
            entry["source_ranges"] = _source_ranges(raw, text, origins, pos, base=source.get("source_start", 0),
                                                     html_origins=source.get("html_origins"))
        glyphs = source.get("glyphs")
        if glyphs is not None:
            fragments = []
            for match in re.finditer(r"\S+", text):
                a, b = origins[match.start()][0], origins[match.end() - 1][1]
                boxes = [box for box in glyphs[a:b] if box is not None]
                # A dehyphenated word can span two lines. Keep separate boxes;
                # their bounding union would highlight unrelated words between.
                lines: list[list] = []
                for box in boxes:
                    if not lines or abs(lines[-1][0][1] - box[1]) > 1:
                        lines.append([])
                    lines[-1].append(box)
                for line in lines:
                    fragment = {"start": pos + match.start(), "end": pos + match.end(),
                                "bbox": [min(x[0] for x in line), min(x[1] for x in line),
                                         max(x[2] for x in line), max(x[3] for x in line)]}
                    recognized = [w for w in source.get("ocr_words", []) if w["start"] < b and w["end"] > a]
                    if recognized:
                        fragment.update(text=match.group(), recognition_confidence=min(w["confidence"] for w in recognized))
                    fragments.append(fragment)
            entry["fragments"] = fragments
            entry["geometry_granularity"] = "word"
        mapping.append(entry)
        pos += len(text)
    return Document("\n\n".join(texts), spans, source_map=mapping)


def _seal(doc: Document, *, original_sha256: str | None = None) -> Document:
    doc.meta.update(source_map_version=SOURCE_MAP_VERSION, extraction_version=EXTRACTION_VERSION, offset_unit="unicode_codepoint",
                    extracted_text_sha256=text_hash(doc.text), normalization="whitespace-and-paragraphs/1")
    if original_sha256:
        doc.meta["original_sha256"] = original_sha256
    return doc


_WS = re.compile(r"[ \t ]+")
# Every pattern is linear in the length of the file (N-10; SEC-04). A bracket class that could run to the end of the file
# from each `[`, or `\s*` that could run over every blank line from each line start, made three of them quadratic: 40,000
# characters of `[` took 2.6 s and 200,000 took 69 s, inside one call that holds the interpreter. The text of a link or an
# image cannot hold `[`, and its address cannot either, so the scan from one opener ends at the next; the rule and table
# patterns stay on their line.
_MD_INLINE = [
    (re.compile(r"```.*?```", re.DOTALL), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"!\[[^\[\]]*\]\([^)\[]*\)"), " "),
    (re.compile(r"\[([^\[\]]+)\]\([^)\[]*\)"), r"\1"),
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.MULTILINE), ""),
    (re.compile(r"^[ \t]{0,3}(?:[-*+]|\d+[.)])[ \t]+", re.MULTILINE), ""),
    (re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE), ""),
    (re.compile(r"(\*\*|__)(.*?)\1"), r"\2"),
    (re.compile(r"(\*|_)(.*?)\1"), r"\2"),
    (re.compile(r"^[ \t]*[-*_]{3,}[ \t]*$", re.MULTILINE), ""),
    (re.compile(r"^\|.*\|[ \t]*$", re.MULTILINE), lambda m: m.group(0).replace("|", " ")),
]


def _join_paragraphs(paras: list[str]) -> Document:
    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for p in paras:
        p = _WS.sub(" ", p).strip()
        if not p:
            continue
        if text_parts:
            text_parts.append("\n\n")
            pos += 2
        text_parts.append(p)
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return Document("".join(text_parts), spans)


def _dehyphenate(s: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", s)


def _ocr_runtime() -> dict:
    """Only use the provisioned, hash-pinned local language data; never download at inference."""
    binary = shutil.which("tesseract")
    if not binary:
        return {"available": False, "reason": "Local OCR executable is not installed."}
    root = Path(__file__).resolve().parents[4] / "models"
    try:
        manifest = json.loads((root / "ocr-manifest.json").read_text())
        directory = root / "weights/ocr/tessdata"
        hashes = {}
        for item in manifest["files"]:
            name = item["file"]
            if name not in {"eng.traineddata", "chi_sim.traineddata", "chi_tra.traineddata", "osd.traineddata"}:
                raise ValueError("Unexpected OCR model file")
            data = (directory / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("OCR language data hash mismatch")
            hashes[name] = item["sha256"]
        if len(hashes) != 4:
            raise ValueError("OCR language data incomplete")
        version = subprocess.run([binary, "--version"], capture_output=True, timeout=5, check=True).stdout.decode().splitlines()[0]
        return {"available": True, "binary": binary, "directory": str(directory), "engine_version": version,
                "language_data_sha256": hashes, "languages": "eng+chi_sim+chi_tra"}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return {"available": False, "reason": "Verified local OCR language data or executable is unavailable."}


_LANGUAGE_IDENTIFIER = None  # Published only after construction. Classify reads the model without mutating it.


def _ocr_language(text: str) -> dict:
    global _LANGUAGE_IDENTIFIER
    if sum(ch.isalpha() for ch in text) < 25:
        return {"code": None, "confidence": None, "status": "unknown", "reason": "Too little recognized text to assess language."}
    try:
        if _LANGUAGE_IDENTIFIER is None:
            from langid.langid import LanguageIdentifier, model
            _LANGUAGE_IDENTIFIER = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        code, confidence = _LANGUAGE_IDENTIFIER.classify(text)
        supported = code in {"en", "zh"} and confidence >= 0.8
        return {"code": code, "confidence": float(confidence), "status": "supported" if supported else "unknown_or_unsupported",
                "method": "statistical-language-identification", "reason": None if supported else "Recognized language is uncertain or outside local OCR coverage."}
    except (ImportError, ValueError):
        return {"code": None, "confidence": None, "status": "unknown", "reason": "Local language coverage check is unavailable."}


def _ocr_call(runtime: dict, raster: bytes, *, osd=False, timeout=10.0) -> str:
    # Subprocess receives only locally generated raster bytes and fixed options.
    # Paths and document contents are never interpolated into a shell command.
    with tempfile.TemporaryDirectory(prefix="chipmunk-ocr-") as temporary:
        path = Path(temporary) / "input.png"
        path.write_bytes(raster)
        command = [runtime["binary"], str(path), "stdout", "--tessdata-dir", runtime["directory"],
                   "-l", "osd" if osd else runtime["languages"], "--psm", "0" if osd else "3"]
        if not osd:
            command += ["--oem", "1", "-c", "tessedit_create_tsv=1"]
        completed = subprocess.run(command, capture_output=True, timeout=timeout, check=True)
        return completed.stdout.decode("utf-8", "replace")


def _intersects(first, second) -> bool:
    return (max(first[0], second[0]) < min(first[2], second[2])
            and max(first[1], second[1]) < min(first[3], second[3]))


def _ocr_image_words(words: list[dict], image_box: list[float]) -> list[dict] | None:
    """Keep intact OCR paragraphs inside the image. Do not splice partial text."""
    paragraphs: dict[tuple[int, int], list[dict]] = {}
    for word in words:
        paragraphs.setdefault(tuple(word["paragraph"]), []).append(word)
    selected = []
    for group in paragraphs.values():
        if not any(_intersects(word["bbox"], image_box) for word in group):
            continue
        if not all(image_box[0] <= x0 < x1 <= image_box[2] and image_box[1] <= y0 < y1 <= image_box[3]
                   for x0, y0, x1, y1 in (word["bbox"] for word in group)):
            return None
        selected.extend(group)
    return selected


def _ocr_page(page, runtime: dict, remaining: float, *,
              image_box: list[float] | None = None) -> tuple[list[tuple[str, dict]], dict, bytes | None]:
    import pymupdf
    from PIL import Image

    info = {"page": page.number + 1, "status": "unchecked", "version": OCR_VERSION,
            "page_rotation": page.rotation, "confidence_semantics": "OCR recognition quality, not AI-generation confidence"}
    scope = "image area" if image_box is not None else "page"
    if image_box is not None:
        info.update(coverage_scope="image_region", image_box=image_box)
    if not runtime.get("available"):
        return [], {**info, "reason": runtime["reason"]}, None
    if remaining < 2:
        return [], {**info, "reason": f"Local OCR time budget exhausted. This {scope} was not checked."}, None
    rotation = page.rotation
    try:
        page.set_rotation(0)
        width, height = page.rect.width, page.rect.height
        scale = min(200 / 72, math.sqrt(OCR_MAX_PIXELS / (width * height)))
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False)
    except (RuntimeError, ValueError):
        return [], {**info, "reason": f"Local OCR could not render its source. This {scope} was not checked."}, None
    finally:
        page.set_rotation(rotation)
    original = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    raw = io.BytesIO()
    original.save(raw, format="PNG")
    clockwise, osd_info = 0, {"status": "unavailable"}
    started = time.monotonic()
    try:
        detected = _ocr_call(runtime, raw.getvalue(), osd=True, timeout=min(5, remaining / 3))
        fields = dict(re.findall(r"^([^:\n]+):\s*([^\n]+)", detected, re.MULTILINE))
        candidate = int(fields.get("Rotate", "0"))
        orientation_confidence = float(fields.get("Orientation confidence", "0"))
        if candidate in {0, 90, 180, 270} and orientation_confidence >= 2:
            clockwise = candidate
        osd_info = {"status": "estimated", "script": fields.get("Script"),
                    "orientation_confidence": orientation_confidence,
                    "script_confidence": float(fields.get("Script confidence", "0")),
                    "applied_rotation_clockwise": clockwise}
    except (ValueError, subprocess.SubprocessError):
        pass  # OCR can still recognize short upright pages; language/quality gates remain mandatory.
    rotated = original.rotate(-clockwise, expand=True)
    raw = io.BytesIO()
    rotated.save(raw, format="PNG")
    raster = raw.getvalue()
    sx, sy = original.width / width, original.height / height
    transforms = {0: [1 / sx, 0, 0, 1 / sy, 0, 0],
                  90: [0, -1 / sy, 1 / sx, 0, 0, height],
                  180: [-1 / sx, 0, 0, -1 / sy, width, height],
                  270: [0, 1 / sy, -1 / sx, 0, width, 0]}
    raster_info = {"sha256": hashlib.sha256(raster).hexdigest(), "width": rotated.width, "height": rotated.height,
                   "base_width": original.width, "base_height": original.height,
                   "page_width": width, "page_height": height, "page_rotation_removed": rotation,
                   "rotation_clockwise": clockwise, "input_to_page": transforms[clockwise],
                   "measured_dpi_x": sx * 72, "measured_dpi_y": sy * 72,
                   "colorspace": "grayscale", "renderer": f"pymupdf-{pymupdf.VersionBind}"}
    info.update(raster=raster_info, orientation=osd_info, engine_version=runtime["engine_version"],
                language_data_sha256=runtime["language_data_sha256"], languages_requested=runtime["languages"])
    try:
        tsv = _ocr_call(runtime, raster, timeout=max(1, min(10, remaining - (time.monotonic() - started))))
        words = []
        for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
            if row.get("level") != "5" or not (row.get("text") or "").strip():
                continue
            x, y, w, h = [int(row[k]) for k in ("left", "top", "width", "height")]
            confidence = float(row["conf"])
            if not (0 <= x < x + w <= rotated.width and 0 <= y < y + h <= rotated.height and 0 <= confidence <= 100):
                raise ValueError("OCR returned invalid word coordinates or quality")
            a, b, c, d, e, f = transforms[clockwise]
            points = [(a * px + c * py + e, b * px + d * py + f) for px, py in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))]
            words.append({"text": row["text"].strip(), "confidence": confidence, "input_bbox": [x, y, x + w, y + h],
                          "bbox": [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)],
                          "paragraph": [int(row["block_num"]), int(row["par_num"])], "line": int(row["line_num"])})
    except (ValueError, KeyError, subprocess.SubprocessError):
        return [], {**info, "reason": f"Local OCR failed or exceeded its per-page time limit. This {scope} was not checked."}, raster
    if image_box is not None:
        selected = _ocr_image_words(words, image_box)
        if selected is None:
            return [], {**info, "reason": "A recognized paragraph crosses the image boundary. The image area remains unchecked."}, raster
        words = selected
    # Preserve recognized words in the requested source scope even when quality is too low.
    cursor, recognized = 0, []
    for word in words:
        if recognized:
            cursor += 1
        word.update(start=cursor, end=cursor + len(word["text"]), offset_unit="unicode_codepoint")
        recognized.append(word["text"])
        cursor = word["end"]
    recognized_text = " ".join(recognized)
    language = _ocr_language(recognized_text)
    mean = sum(w["confidence"] * len(w["text"]) for w in words) / max(1, sum(len(w["text"]) for w in words))
    low_fraction = sum(w["confidence"] < 50 for w in words) / max(1, len(words))
    info.update(recognized_text=recognized_text, words=words, language=language,
                mean_recognition_confidence=mean, low_confidence_word_fraction=low_fraction,
                quality_policy={"mean_minimum": OCR_MIN_MEAN_CONFIDENCE, "low_word_threshold": 50, "maximum_low_word_fraction": 0.15})
    if not words or len(recognized_text.strip()) < 25:
        return [], {**info, "reason": f"Too little readable OCR text. This {scope} remains unchecked."}, raster
    if mean < OCR_MIN_MEAN_CONFIDENCE or low_fraction > 0.15:
        return [], {**info, "reason": f"OCR recognition quality is too low. This {scope} remains unchecked."}, raster
    if language["status"] != "supported":
        return [], {**info, "reason": language["reason"] + f" This {scope} remains unchecked."}, raster
    parts = []
    ocr = {"raster": raster_info, "language": language, "mean_recognition_confidence": mean,
           "confidence_semantics": info["confidence_semantics"]}
    if image_box is not None:
        ocr.update(coverage_scope="image_region", image_box=image_box)
    groups: dict[tuple, list[dict]] = {}
    for word in words:
        groups.setdefault(tuple(word["paragraph"]), []).append(word)
    for index, group in enumerate(groups.values()):
        raw, glyphs, mapped, position = [], [], [], 0
        for word in group:
            if raw:
                raw.append(" ")
                glyphs.append(None)
                position += 1
            raw.append(word["text"])
            glyphs.extend([word["bbox"]] * len(word["text"]))
            mapped.append({**word, "start": position, "end": position + len(word["text"])})
            position += len(word["text"])
        boxes = [word["bbox"] for word in group]
        parts.append(("".join(raw), {"kind": "pdf", "extraction": "ocr", "page": page.number + 1, "block": index,
                      "page_rotation": rotation, "page_width": width, "page_height": height,
                      "coordinate_space": "unrotated-page-points", "glyphs": glyphs, "ocr_words": mapped,
                      "bbox": [min(x[0] for x in boxes), min(x[1] for x in boxes), max(x[2] for x in boxes), max(x[3] for x in boxes)],
                      "ocr": ocr}))
    return parts, {**info, "status": "extracted", "reason": None}, raster


def _unread_pdf_image_boxes(blocks: list[dict], width: float, height: float,
                           native_lines: list[list[float]]) -> list[list[float]]:
    """Locate dominant raster areas with too little mapped text to cover them."""
    boxes = set()
    centres = [((a + c) / 2, (b + d) / 2) for a, b, c, d in native_lines]
    for block in blocks:
        if block["type"] != 1:
            continue
        x0, y0, x1, y1 = block["bbox"]
        box = x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) < width * height / 2:
            continue
        # A caption, page number or very sparse overlay cannot establish text
        # coverage for the rest of the raster. Avoid flagging ordinary small figures.
        covered_rows = {round(y / 5) for x, y in centres if x0 <= x <= x1 and y0 <= y <= y1}
        if len(covered_rows) < 3:
            boxes.add(box)
    return [list(box) for box in sorted(boxes)]


def partial_pdf_notice(pages: list[dict]) -> str:
    if len(pages) == 1:
        return f"Page {pages[0]['page']} was partially checked. Only its mapped text was retained for analysis. Large image areas were not checked for text."
    if pages:
        numbers = ", ".join(str(page["page"]) for page in pages)
        return f"Pages {numbers} were partially checked. Only their mapped text was retained for analysis. Large image areas were not checked for text."
    return ""


def hybrid_ocr_notice(ocr_pages: list[dict]) -> str:
    pages = [str(page["page"]) for page in ocr_pages
             if page.get("coverage_scope") == "image_region" and page["status"] == "extracted"]
    if not pages:
        return ""
    subject = f"Page {pages[0]} combines" if len(pages) == 1 else f"Pages {', '.join(pages)} combine"
    return f"{subject} mapped text with text transcribed from image areas. Authorship confidence has not been validated for this mixed extraction."


def extract_pdf(path: Path, *, allow_ocr: bool = True, inspect_pdf_covers: bool = True) -> Document:
    import pymupdf

    page_parts: list[list[tuple[str, dict]]] = []
    empty_pages: list[int] = []
    partial_pages: list[dict] = []
    hybrid_pages: list[dict] = []
    ocr_pages, rasters, raster_bytes = [], {}, 0
    percentage_text_pages = []
    visibility_examined, invisible_text = [], []
    replacement_examined, replacement_differences = [], []
    covered_examined, covered_locations = [], []
    replacement_flag = getattr(pymupdf, "TEXT_IGNORE_ACTUALTEXT", 0)
    glyph_flags = (pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES) | replacement_flag
    runtime = None
    started = time.monotonic()

    def recognize(page, image_box=None):
        nonlocal runtime, raster_bytes
        if not allow_ocr:
            raise ValueError("Original-document replay supports native PDF text only. This PDF requires OCR.")
        scope = "image area" if image_box is not None else "page"
        if len(ocr_pages) >= OCR_MAX_PAGES or raster_bytes >= OCR_MAX_RASTER_BYTES:
            parts, info, raster = [], {"page": page.number + 1, "status": "unchecked",
                "reason": f"Local OCR page or raster budget exhausted. This {scope} was not checked."}, None
        else:
            if runtime is None:
                runtime = _ocr_runtime()
            parts, info, raster = _ocr_page(page, runtime, OCR_BUDGET_SECONDS - (time.monotonic() - started),
                                            image_box=image_box)
        if raster is not None:
            if raster_bytes + len(raster) > OCR_MAX_RASTER_BYTES:
                parts, info = [], {"page": page.number + 1, "status": "unchecked",
                    "reason": f"OCR raster storage budget exhausted. This {scope} was not checked."}
            else:
                rasters[hashlib.sha256(raster).hexdigest()] = base64.b64encode(raster).decode("ascii")
                raster_bytes += len(raster)
        if image_box is not None:
            info.update(coverage_scope="image_region", image_box=image_box)
        ocr_pages.append(info)
        return parts, info

    with pymupdf.open(path) as doc:
        for page in doc:
            parts = []
            page_parts.append(parts)
            blocks = page.get_text("rawdict")["blocks"]
            if replacement_flag:
                try:
                    glyph_blocks = page.get_text("rawdict", flags=glyph_flags)["blocks"]
                except (ValueError, RuntimeError):
                    pass
                else:
                    replacement_examined.append(page.number + 1)
                    if difference := document_pdf_visibility.inspect_replacement(page, blocks, glyph_blocks):
                        replacement_differences.append(difference)
            try:
                spans = page.get_texttrace()
            except (ValueError, RuntimeError):
                spans = None
            if spans is not None:
                visibility_examined.append(page.number + 1)
                if location := document_pdf_visibility.inspect_page(page, spans):
                    invisible_text.append(location)
                if inspect_pdf_covers:
                    try:
                        covered = document_pdf_visibility.inspect_covered(page, spans, blocks)
                    except (ValueError, RuntimeError):
                        pass
                    else:
                        covered_examined.append(page.number + 1)
                        if covered is not None:
                            covered_locations.append(covered)
                            continue
            if len(doc) <= NATIVE_COMPARISON_MAX_PAGES and spans and all(
                    span["type"] != 3 and span["opacity"] > 0 for span in spans):
                percentage_text_pages.append(page.number + 1)
            blocks.sort(key=lambda b: (round(b["bbox"][1] / 5), b["bbox"][0]))
            page_chars = 0
            native_lines = []
            for b in blocks:
                if b["type"] != 0:
                    continue
                chars, glyphs = [], []
                for line in b["lines"]:
                    line_start = len(chars)
                    for span in line["spans"]:
                        for char in span["chars"]:
                            chars.append(char["c"])
                            glyphs.extend([list(char["bbox"])] * len(char["c"]))
                    if sum(ch.isalnum() for ch in "".join(chars[line_start:])) >= 2:
                        native_lines.append(list(line["bbox"]))
                    chars.append("\n")
                    glyphs.append(None)
                raw = "".join(chars)
                if sum(ch.isalnum() for ch in raw) >= 2:
                    page_chars += len(raw.strip())
                    parts.append((raw, {"kind": "pdf", "page": page.number + 1, "block": b["number"],
                                        "bbox": list(b["bbox"]), "page_rotation": page.rotation,
                                        "page_width": page.cropbox.width, "page_height": page.cropbox.height,
                                        "coordinate_space": "unrotated-page-points", "extraction": "text_layer", "glyphs": glyphs}))
            if page_chars < 2:
                local_parts, info = recognize(page)
                parts.extend(local_parts)
                if info["status"] != "extracted":
                    empty_pages.append(page.number + 1)
            else:
                boxes = _unread_pdf_image_boxes(blocks, page.cropbox.width, page.cropbox.height, native_lines)
                if boxes:
                    review = {"page": page.number + 1, "image_boxes": boxes,
                              "coordinate_space": "unrotated-page-points"}
                    if len(boxes) == 1 and not any(_intersects(boxes[0], line) for line in native_lines):
                        hybrid_pages.append(review)
                    else:
                        partial_pages.append({**review, "reason": "OCR requires one dominant image area separate from mapped text."})
        # Preserve the existing OCR budget for pages with no native text before
        # spending any remaining capacity on supplemental hybrid-page text.
        for review in hybrid_pages:
            index = review["page"] - 1
            local_parts, info = recognize(doc[index], review["image_boxes"][0])
            if info["status"] != "extracted":
                partial_pages.append({**review, "reason": info["reason"]})
                continue
            parts = page_parts[index]
            parts.extend(local_parts)
            parts.sort(key=lambda part: (round(part[1]["bbox"][1] / 5), part[1]["bbox"][0]))
    pages = len(page_parts)
    partial_pages.sort(key=lambda page: page["page"])
    ocr_pages.sort(key=lambda page: page["page"])
    d = _join_mapped([part for parts in page_parts for part in parts], dehyphenate=True)
    d.source_rasters = rasters
    d.meta = {"pages": pages, "method": "pymupdf", "unmapped_pages": empty_pages,
              "percentage_text_pages": percentage_text_pages,
              document_pdf_visibility.KEY: document_pdf_visibility.build(pages, visibility_examined, invisible_text,
                  replacement_examined, replacement_differences,
                  covered=(covered_examined, covered_locations) if inspect_pdf_covers else None),
              "partially_checked_pages": partial_pages,
              "ocr": {"version": OCR_VERSION, "status": "partial" if empty_pages or partial_pages else "complete" if ocr_pages else "not_needed",
                      "pages": ocr_pages, "pages_without_text": [p["page"] for p in ocr_pages if p.get("coverage_scope") != "image_region"], "unchecked_pages": empty_pages,
                      "confidence_semantics": "OCR recognition quality, not AI-generation confidence",
                      "max_pages": OCR_MAX_PAGES, "max_pixels_per_page": OCR_MAX_PIXELS}}
    warnings = []
    if empty_pages:
        warnings.append(f"Pages {', '.join(map(str, empty_pages))} could not be reliably extracted. They were not checked and must not be treated as human-authored.")
    elif any(page["status"] == "extracted" for page in ocr_pages):
        warnings.append("Scanned pages were transcribed locally. OCR recognition quality is not AI-generation confidence; language identification is heuristic. Compare all excerpts with their frozen page images.")
    if hybrid_warning := hybrid_ocr_notice(ocr_pages):
        warnings.append(hybrid_warning)
    if partial_warning := partial_pdf_notice(partial_pages):
        warnings.append(partial_warning)
        d.meta["warning"] = partial_warning
    elif len(d.text.split()) < 5 and pages:
        d.meta["warning"] = "The PDF contains little or no reliably extracted text. Unreadable pages remain unchecked."
    if warnings:
        d.meta["coverage_warning"] = " ".join(warnings)
    return d


def docx_body(path: Path, *, project_word_revisions: bool = True, project_word_breaks: bool = True,
              include_word_content_controls: bool = True):
    from app.detectors.text.strict_word import open_document
    from app.evidence.document_tracked_changes import PART_NAME, inspect

    document = open_document(path)
    part = str(document.part.partname).removeprefix("/")
    joined = set() if project_word_revisions and project_word_breaks else None
    review = (inspect(document.element.body, part, project_inline=project_word_revisions, joined_paragraphs=joined,
                      include_word_content_controls=include_word_content_controls)
              if PART_NAME.fullmatch(part) else None)
    return document, part, review, joined


def _word_control_content(element):
    from docx.oxml.ns import qn

    placeholder = element.find(f"{qn('w:sdtPr')}/{qn('w:showingPlcHdr')}")
    if placeholder is not None and placeholder.get(qn("w:val")) not in {"0", "false", "off"}:
        return None
    return element.find(qn("w:sdtContent"))


def _project_word_inline_controls(container):
    """Use saved visible runs from filled controls in their paragraph order."""
    from docx.oxml.ns import qn

    changed = False
    for child in list(container):
        if child.tag == qn("w:hyperlink"):
            changed |= _project_word_inline_controls(child)
        elif child.tag == qn("w:sdt"):
            content = _word_control_content(child)
            if content is not None:
                _project_word_inline_controls(content)
                for run in list(content):
                    if run.tag in {qn("w:r"), qn("w:hyperlink")}:
                        child.addprevious(run)
            container.remove(child)
            changed = True
    return changed


def docx_paragraphs(parent, elements, body_path=(), *, include_content_controls=True):
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for index, element in enumerate(elements):
        path = [*body_path, index]
        if element.tag == qn("w:p"):
            projected = include_content_controls and _project_word_inline_controls(element)
            yield Paragraph(element, parent), path, projected
        elif include_content_controls and element.tag == qn("w:sdt"):
            if (content := _word_control_content(element)) is not None:
                yield from docx_paragraphs(parent, content, [*path, "control"])
        elif element.tag == qn("w:tbl"):
            # A wrapped row or cell is omitted by python-docx's table grid.
            # Reading the siblings would assign their text the wrong locations.
            if include_content_controls and element.xpath("./w:sdt | ./w:tr/w:sdt"):
                continue
            table = Table(element, parent)
            seen = set()
            for row_i, row in enumerate(table.rows):
                for col_i, cell in enumerate(row.cells):
                    if cell._tc in seen:
                        continue
                    seen.add(cell._tc)
                    yield from docx_paragraphs(cell, cell._tc, [*path, "row", row_i, "cell", col_i],
                                               include_content_controls=include_content_controls)


def extract_docx(path: Path, *, project_word_revisions: bool = True, project_word_breaks: bool = True,
                 include_word_content_controls: bool = True) -> Document:
    from app.evidence import document_word_visibility
    from app.evidence.document_archive_word import ordinary_body_paragraph
    from app.evidence.document_tracked_changes import INPUT_NOTICES, JOINED_PARAGRAPHS, PART_NAME

    d, part, review, joined = docx_body(path, project_word_revisions=project_word_revisions,
                                      project_word_breaks=project_word_breaks,
                                      include_word_content_controls=include_word_content_controls)
    parts: list[tuple[str, dict]] = []
    percentage_body_paragraphs = []
    hidden_locations = []
    checked_paragraphs = hidden_runs = 0

    continued = False
    for paragraph, path_here, projected_control in docx_paragraphs(
            d, d.element.body, include_content_controls=include_word_content_controls):
        text = paragraph.text
        top_level = len(path_here) == 1
        body_paragraph = len(percentage_body_paragraphs) + 1 if top_level else None
        if text.strip():
            checked_paragraphs += 1
            for location in document_word_visibility.inspect_paragraph(
                    paragraph, path_here, body_paragraph, projected_control=projected_control):
                hidden_runs += 1
                if len(hidden_locations) < document_word_visibility.MAX_LOCATIONS:
                    hidden_locations.append(location)
        joins_next = joined is not None and paragraph._p in joined
        if top_level:
            style = paragraph.style
            eligible = style is not None and style.style_id.casefold() in {"normal", "bodytext", "bodytext2", "bodytext3"}
            percentage_body_paragraphs.append(path_here[0] if eligible and not (continued or joins_next or projected_control)
                                              and ordinary_body_paragraph(paragraph) else None)
            continued = joins_next
        cursor, runs = 0, []
        for ri, run in enumerate(paragraph.runs):
            length = len(run.text)
            runs.append({"run": ri, "start": cursor, "end": cursor + length})
            cursor += length
        source = {"kind": "docx", "body_path": path_here, "paragraph_index": len(parts),
                  "runs": [] if projected_control else runs,
                  "run_mapping_complete": not projected_control and cursor == len(text)}
        if joins_next:
            source["join_following_paragraph"] = True
        parts.append((text, source))
    doc = _join_mapped(parts)
    doc.word_review = review
    word_warning = ""
    if review and review["status"] != "no_match":
        word_warning = " " + INPUT_NOTICES.get(review.get("writing_input"),
            "The file contains Word revision marks. The writing checks may omit text within those marks.")
    doc.meta = {"method": "python-docx", "paragraph_count": len(parts) - len(joined or ()),
                "percentage_body_paragraphs": percentage_body_paragraphs,
                "coverage_warning": ("Body paragraphs and tables were extracted in document order. Headers, footers, text boxes and "
                                     "embedded attachments were not checked; DOCX has no stable page numbers." + word_warning)}
    if include_word_content_controls:
        doc.meta[WORD_CONTENT_CONTROLS_KEY] = WORD_CONTENT_CONTROLS
        doc.meta["coverage_warning"] += (" Filled paragraph and inline controls use their saved text. This check excludes "
                                         "placeholders, external data and tables with controls around entire rows or cells.")
    if joined:
        doc.meta[JOINED_PARAGRAPHS] = len(joined)
    if PART_NAME.fullmatch(part):
        doc.meta[document_word_visibility.KEY] = document_word_visibility.build(part, checked_paragraphs, hidden_runs, hidden_locations)
    return doc


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


class _MappedEmailHTML(HTMLParser):
    """Locate text inside simple HTML paragraphs in the decoded MIME part."""

    INLINE = frozenset({"b", "em", "i", "span", "strong"})
    PARENTS = {"html": {None}, "body": {"html"}, "blockquote": {None, "body"},
               "p": {None, "body", "blockquote"}}

    def __init__(self, raw):
        super().__init__(convert_charrefs=False)
        self.raw = raw
        self.lines = [0, *(match.end() for match in re.finditer("\n", raw))]
        self.stack: list[str] = []
        self.parts: list[tuple[str, dict]] = []
        self.words: list[str] = []
        self.ranges: list[tuple[int, int]] = []

    def _offset(self):
        line, col = self.getpos()
        return self.lines[line - 1] + col

    def handle_starttag(self, tag, attrs):
        parent = self.stack[-1] if self.stack else None
        if attrs or not (parent in self.PARENTS.get(tag, ()) or tag in self.INLINE and "p" in self.stack):
            raise ValueError("Unsupported email HTML structure")
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            raise ValueError("Unbalanced email HTML")
        if tag == "p":
            role = "quoted" if "blockquote" in self.stack else "body"
            text = "".join(self.words)
            if role == "body" and any(re.match(r"\s*(?:>|--\s*$|-----Original Message-----|On .{1,300} wrote:\s*$)", line, re.I)
                                      for line in text.splitlines()):
                raise ValueError("Unmarked quotation or signature in email HTML")
            self.parts.append((text, {"kind": "email", "quote_role": role, "html_origins": self.ranges}))
            self.words, self.ranges = [], []

    def handle_data(self, data):
        if "p" not in self.stack:
            if data.strip():
                raise ValueError("Email HTML text must appear in a paragraph")
            return
        start = self._offset()
        self.words.append(data)
        self.ranges.extend((start + i, start + i + 1) for i in range(len(data)))

    def _entity(self, token):
        start = self._offset()
        value = unescape(token)
        if "p" not in self.stack or self.raw[start:start + len(token)] != token or len(value) != 1 or value == "\ufffd":
            raise ValueError("Unsupported email HTML entity")
        self.words.append(value)
        self.ranges.append((start, start + len(token)))

    def handle_entityref(self, name):
        self._entity(f"&{name};")

    def handle_charref(self, name):
        self._entity(f"&#{name};")

    def handle_decl(self, decl):
        if decl.lower() != "doctype html":
            raise ValueError("Unsupported email HTML declaration")

    def handle_pi(self, data):
        raise ValueError("Unsupported email HTML instruction")


def _mapped_email_html(raw, mime_part, text):
    if len(raw) > 64 * 1024 or "\ufffd" in raw:
        return None
    parser = _MappedEmailHTML(raw)
    try:
        parser.feed(raw)
        parser.close()
    except (ValueError, AssertionError):
        return None
    if parser.stack:
        return None
    candidate = _join_mapped([(value, {**source, "mime_part": mime_part}) for value, source in parser.parts])
    return candidate if candidate.text == text and candidate.source_map else None


def _html_text(raw: str) -> str:
    parser = _HTMLText()
    parser.feed(raw)
    return "".join(parser.parts)


def text_like_content(raw: str, ext: str) -> tuple[str, str]:
    """Project text-like input without constructing a per-character source map."""
    if ext in (".html", ".htm"):
        return _html_text(raw), "html"
    if ext in (".md", ".markdown"):
        for rx, rep in _MD_INLINE:
            raw = rx.sub(rep, raw)
        return raw, "markdown"
    return raw, "plain"


def rtf_text(data: bytes) -> str:
    from striprtf.striprtf import rtf_to_text

    from app.storage.files import RTF_HEADER, rtf_body

    body = rtf_body(data)
    if not RTF_HEADER.match(body):
        raise ValueError("This file is not a readable Rich Text (RTF) document.")
    depth = 0
    for token in re.finditer(rb"\\[\\{}]|\\bin(?=-?[0-9])|[{}]", body):
        match token.group():
            case b"\\bin":
                raise ValueError("This RTF contains binary content. Save it as DOCX or PDF for analysis.")
            case b"{":
                depth += 1
            case b"}":
                depth -= 1
                if depth == 0:
                    body = body[:token.end()]
                    break
    if depth:
        raise ValueError("This RTF is incomplete. Save it again as RTF, DOCX or PDF.")
    # Present raw ANSI bytes as hex escapes so the parser applies the declared codepage.
    encoded = re.sub(r"[\x80-\xff]", lambda match: f"\\'{ord(match[0]):02x}", body.decode("latin-1"))
    try:
        text = rtf_to_text(encoded)
        if re.search(r"[\ud800-\udfff]", text):
            text = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
        return text
    except (LookupError, ValueError) as error:
        raise ValueError("The RTF text could not be decoded. Save it as DOCX or PDF.") from error


def _attach_html_source_ranges(doc: Document, raw: str) -> None:
    candidate = html_article.parts(raw)
    if candidate is None or not valid_html_article_scope(candidate[1]):
        return
    parts, scope = candidate
    mapped = _join_mapped(parts)
    texts = [doc.text[block["start"]:block["end"]] for block in doc.source_map]
    body = [mapped.text[block["start"]:block["end"]] for block in mapped.source_map]
    full_counts, body_counts = Counter(texts), Counter(body)
    indexed = {value: block for value, block in zip(body, mapped.source_map, strict=True) if body_counts[value] == 1}
    attached = False
    for value, block in zip(texts, doc.source_map, strict=True):
        match = indexed.get(value)
        if match is None or full_counts[value] != 1:
            continue
        source_text = "".join(unescape(raw[a:b]) for _, _, a, b in match["source_ranges"])
        if _normalise(source_text)[0] != value:
            continue
        shift = block["start"] - match["start"]
        block["source_ranges"] = [[a + shift, b + shift, lo, hi] for a, b, lo, hi in match["source_ranges"]]
        attached = True
    if attached:
        doc.meta.update({html_article.KEY: scope, "decoded_source_sha256": text_hash(raw), "coverage_warning": (
            "Original HTML source positions are available for uniquely matched article passages. Other positions refer to the "
            "extracted text, which can include page navigation and hidden markup text. Scripts and linked content were not loaded.")})


def extract_text_like(path: Path, ext: str, *, include_html_article: bool = True, map_body_paragraphs: bool = True) -> Document:
    data = path.read_bytes()
    if ext == ".rtf":
        # RTF \par becomes a newline. Keep those breaks as passage boundaries.
        paragraphs = rtf_text(data).splitlines()
        doc = _join_mapped([(value, {"kind": "markup", **({"paragraph_index": index} if map_body_paragraphs else {})})
                            for index, value in enumerate(paragraphs)], map_paragraphs=map_body_paragraphs)
        doc.meta = {"method": "rtf", "coverage_warning": (
            "Rich Text was converted to text. Passage locations refer to the extracted text. "
            "Layout, formatting and embedded objects were outside this check.")}
        if map_body_paragraphs:
            doc.meta[document_native_locations.PARAGRAPH_MAP_KEY] = document_native_locations.PARAGRAPH_MAP
        return doc
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    elif data.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"
    decoded = data.decode(encoding, errors="replace")
    raw, method = text_like_content(decoded, ext)
    d = _join_mapped(_text_parts(raw, kind="text" if method == "plain" else "markup"))
    d.meta = {"method": method, "encoding": encoding,
              "decoded_source_sha256": text_hash(raw) if method == "plain" else None}
    if method != "plain":
        d.meta["coverage_warning"] = "Markup was converted to text. Excerpts map to the extracted text, not exact HTML or Markdown source positions."
    if method == "html" and include_html_article:
        d.meta[html_article.READER] = html_article.POLICY
        _attach_html_source_ranges(d, decoded)
    return d


def _text_parts(raw: str, *, kind: str = "text", **anchor) -> list[tuple[str, dict]]:
    parts, cursor = [], 0
    for blank in re.finditer(r"(?:\r?\n)[ \t]*(?:\r?\n)", raw):
        parts.append((raw[cursor:blank.start()], {"kind": kind, "source_start": cursor, **anchor}))
        cursor = blank.end()
    parts.append((raw[cursor:], {"kind": kind, "source_start": cursor, **anchor}))
    return parts


def extract_eml(path: Path) -> Document:
    from email import policy
    from email.parser import BytesParser

    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())

    def leaves(part, part_id="1"):
        yield part_id, part
        if part.is_multipart():
            for index, child in enumerate(part.iter_parts(), 1):
                yield from leaves(child, f"{part_id}.{index}")

    indexed = list(leaves(message))
    body = message.get_body(preferencelist=("plain", "html"))
    if body is None:
        raise ValueError("This email has no readable plain-text or HTML body. Attachments are not analysed as its body.")
    mime_part = next(part_id for part_id, part in indexed if part is body)
    raw = body.get_content()
    if not isinstance(raw, str):
        raise ValueError("This email body could not be decoded as text.")
    is_html = body.get_content_type() == "text/html"
    if is_html:
        parts = _text_parts(_html_text(raw), kind="markup", mime_part=mime_part, quote_role="unknown")
    else:
        # Quotation detection is a navigation aid, never an authorship assertion.
        # Keep the '>' prefixes and signature text rather than silently deleting it.
        parts, accumulated, start, cursor, previous_role = [], [], 0, 0, None
        quoted_tail, signature = False, False
        for line in raw.splitlines(keepends=True):
            stripped = line.strip()
            if re.match(r"On .{1,300} wrote:\s*$", stripped, re.IGNORECASE) or stripped.startswith("-----Original Message-----"):
                quoted_tail = True
            if stripped == "--":
                signature = True
            role = "quoted" if quoted_tail or stripped.startswith(">") else "signature" if signature else "body"
            if accumulated and (role != previous_role or not stripped):
                parts.append(("".join(accumulated), {"kind": "email", "source_start": start,
                                                    "mime_part": mime_part, "quote_role": previous_role}))
                accumulated = []
            if stripped:
                if not accumulated:
                    start = cursor
                accumulated.append(line)
                previous_role = role
            cursor += len(line)
        if accumulated:
            parts.append(("".join(accumulated), {"kind": "email", "source_start": start,
                                                "mime_part": mime_part, "quote_role": previous_role}))
    document = _join_mapped(parts)
    mapped = _mapped_email_html(raw, mime_part, document.text) if is_html else None
    if mapped is not None:
        document = mapped
    attachments = [{"mime_part": part_id, "filename": part.get_filename(), "content_type": part.get_content_type()}
                   for part_id, part in indexed if part.get_content_disposition() == "attachment" or part.get_filename()]
    document.meta = {"method": "email", "email": {"body_part": mime_part, "body_content_type": body.get_content_type(),
                     "quote_detection": "heuristic", "headers_authenticated": False, "attachments": attachments},
                     "coverage_warning": "The extracted email text comes from one selected message part. Quote and signature labels are inferred from that text. Attachments and sender authenticity are outside the analysis scope."}
    if mapped is not None:
        document.meta["email"].update(body_coordinates=EMAIL_HTML_COORDINATES, quote_detection="html-blockquote")
        document.meta["coverage_warning"] = (
            "The extracted email text maps to decoded HTML in one selected message part. Quote labels follow explicit blockquotes. "
            "Attachments and sender authenticity are outside the analysis scope."
        )
    return document


def extract(path: Path, ext: str, *, allow_ocr: bool = True, inspect_pdf_covers: bool = True,
            project_word_revisions: bool = True, project_word_breaks: bool = True,
            include_word_content_controls: bool = True, include_html_article: bool = True,
            map_body_paragraphs: bool = True) -> Document:
    ext = ext.lower()
    if ext == ".pdf":
        d = extract_pdf(path, allow_ocr=allow_ocr, inspect_pdf_covers=inspect_pdf_covers)
    elif ext == ".docx":
        d = extract_docx(path, project_word_revisions=project_word_revisions, project_word_breaks=project_word_breaks,
                         include_word_content_controls=include_word_content_controls)
    elif ext == ".odt":
        from app.detectors.text.opendocument import body_paragraphs

        paragraphs = body_paragraphs(path)
        d = _join_mapped([(value, {"kind": "markup", "paragraph_index": index}) for index, value in enumerate(paragraphs)],
                         map_paragraphs=map_body_paragraphs)
        d.meta = {"method": "odt", "paragraph_count": len(paragraphs), "coverage_warning":
                  "ODT body paragraphs, headings, lists and tables were read in order. Locations refer to the extracted text. "
                  "Headers, footers, notes, tracked revisions and embedded objects were outside this check."}
        if map_body_paragraphs:
            d.meta[document_native_locations.PARAGRAPH_MAP_KEY] = document_native_locations.PARAGRAPH_MAP
    elif ext == ".eml":
        d = extract_eml(path)
    else:
        d = extract_text_like(path, ext, include_html_article=include_html_article, map_body_paragraphs=map_body_paragraphs)
    d.meta.update({"chars": len(d.text), "words": len(d.text.split()), "paragraphs": len(d.paragraphs)})
    return _seal(d, original_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def from_string(text: str) -> Document:
    d = _join_mapped(_text_parts(text))
    d.meta = {"method": "string", "chars": len(d.text), "words": len(d.text.split()), "paragraphs": len(d.paragraphs)}
    return _seal(d)


def _child(argv: list[str]) -> int:
    """`python -I extract.py <path> <ext>`: the document as JSON on stdout, or the error it raised (its class name, its text
    and whether it is a `ValueError`, which the application shows to the person as it is)."""
    import json
    import sys

    try:
        path, ext = Path(argv[1]), argv[2]
        if argv[3:] == ["--native-pdf"] and ext == ".pdf":
            import pymupdf

            with pymupdf.open(path) as pdf:
                page_count = len(pdf)
                if not pdf.is_pdf or pdf.needs_pass or not 1 <= page_count <= NATIVE_COMPARISON_MAX_PAGES:
                    raise ValueError(f"Use an unlocked PDF with up to {NATIVE_COMPARISON_MAX_PAGES} pages in each version.")
                hidden_text_pages = [page.number + 1 for page in pdf
                                     if any(span["type"] == 3 or span["opacity"] == 0 for span in page.get_texttrace())]
            d = extract(path, ext, allow_ocr=False)
            d.meta["hidden_text_pages"] = hidden_text_pages
        else:
            d = extract(path, ext)
        out: dict = {"text": d.text, "paragraphs": d.paragraphs, "meta": d.meta, "source_map": d.source_map,
                     "source_rasters": d.source_rasters, "word_review": d.word_review}
    except Exception as e:  # noqa: BLE001 - handed to the parent, which raises it again
        out = {"error": {"type": type(e).__name__, "message": str(e), "value_error": isinstance(e, ValueError)}}
    sys.stdout.write(json.dumps(out))  # ASCII only: a lone surrogate of a damaged file crosses as an escape
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_child(sys.argv))
