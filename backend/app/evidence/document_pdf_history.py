# SPDX-License-Identifier: Apache-2.0
"""Inspect native page wording in the previous saves retained inside a PDF."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from itertools import islice
from math import isfinite
from pathlib import Path

from app.core.canonical import canonical_hash
from app.evidence import document_pdf_visibility
from app.evidence.document_pdf_history_binding import (
    MAX_BYTES,
    MAX_CHANGES,
    MAX_EXCERPT_CHARS,
    MAX_EXCERPT_WORDS,
    MAX_PAGES,
    MAX_SAVES,
    MAX_WORDS,
    SCHEMA,
    _has_changes,
    _sha,
    current_locations,
)

_WS = b"\x00\t\n\x0c\r "
_END = re.compile(rb"(?<![^\r\n])[ \t]*startxref[ \t]*[\r\n]+[ \t]*([0-9]{1,10})[ \t]*[\r\n]+%%EOF[ \t]*(?:\r\n|\r|\n|$)")
_OBJECT = re.compile(rb"([1-9][0-9]{0,8})[ \t]+[0-9]{1,5}[ \t]+obj[\x00\t\n\x0c\r ]")
def _xref_kind(pdf, data, offset):
    if data[offset:offset + 4] == b"xref" and data[offset + 4:offset + 5] in _WS:
        return "table"
    header = _OBJECT.match(data, offset)
    if header and pdf.xref_get_key(int(header[1]), "Type") == ("name", "/XRef"):
        return "stream"
    raise ValueError("Unsupported PDF cross-reference section")


def _revision(pdf, data, ending):
    if not pdf.is_pdf or pdf.needs_pass or pdf.is_encrypted or pdf.is_repaired:
        raise ValueError("The retained PDF required repair or a password")
    start, end = int(ending[1]), ending.end()
    if not 0 < start < ending.start():
        raise ValueError("A PDF cross-reference offset is outside its saved revision")
    return {"version": pdf.version_count, "xref_offset": start, "prefix_bytes": end, "sha256": _sha(data[:end]),
            "xref_kind": _xref_kind(pdf, data, start), "pages": len(pdf)}


def _previous(pdf, endings, revision):
    kind, offset = pdf.xref_get_key(-1, "Prev")
    if kind != "int" or not offset.isdecimal():
        raise ValueError("An earlier PDF save has no usable cross-reference pointer")
    candidates = [end for end in endings if int(end[1]) == int(offset) and end.end() <= revision["xref_offset"]]
    if len(candidates) != 1:
        raise ValueError("An earlier cross-reference pointer has no unique file boundary")
    return candidates[0]


def _normalize(word):
    return unicodedata.normalize("NFC", word.translate(document_pdf_visibility.LIGATURES).replace("\u00ad", ""))


def _page_words(page):
    import pymupdf

    rotation = page.rotation
    if rotation:
        page.set_rotation(0)
    try:
        return _unrotated_words(page, pymupdf)
    finally:
        if rotation:
            page.set_rotation(rotation)


def _unrotated_words(page, pymupdf):
    text_flags = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    spans = page.get_texttrace()
    if not spans or len(spans) > document_pdf_visibility.MAX_PAINT_OPERATIONS:
        return None
    if document_pdf_visibility.inspect_page(page, spans):
        return None
    display = page.get_displaylist(annots=False)
    blocks = pymupdf.TextPage(display.get_textpage(text_flags)).extractRAWDICT()["blocks"]
    if document_pdf_visibility.inspect_covered(page, spans, blocks):
        return None
    if flag := getattr(pymupdf, "TEXT_IGNORE_ACTUALTEXT", 0):
        glyphs = pymupdf.TextPage(display.get_textpage(text_flags | flag)).extractRAWDICT()["blocks"]
        if document_pdf_visibility.inspect_replacement(page, blocks, glyphs):
            return None
    words = pymupdf.TextPage(display.get_textpage(pymupdf.TEXTFLAGS_WORDS)).extractWORDS()
    if not 0 < len(words) <= MAX_WORDS:
        return None
    result = []
    for x0, y0, x1, y1, literal, *_ in words:
        bbox = [round(value, 4) for value in (x0, y0, x1, y1)]
        if (not all(isfinite(value) for value in bbox) or len(literal) > MAX_EXCERPT_CHARS or literal.split() != [literal]
                or not 0 <= bbox[0] < bbox[2] <= page.cropbox.width + .01
                or not 0 <= bbox[1] < bbox[3] <= page.cropbox.height + .01):
            return None
        result.append((literal, bbox))
    return result


def _location(page, words, low, high, *, publish_text):
    if low == high:
        return None
    sample = words[low:min(high, low + MAX_EXCERPT_WORDS)]
    literal = " ".join(word for word, _ in sample)
    while len(literal) > MAX_EXCERPT_CHARS:
        sample.pop()
        literal = " ".join(word for word, _ in sample)
    side = {"page": page.number + 1, "page_xref": page.xref, "rotation": page.rotation,
            "width": round(page.cropbox.width, 4), "height": round(page.cropbox.height, 4),
            "word_range": [low, high], "excerpt_word_range": [low, low + len(sample)],
            "boxes": [bbox for _, bbox in sample]}
    if publish_text:
        side["excerpt"] = literal
        side["excerpt_sha256"] = _sha(literal.encode())
    return side


def _differences(earlier, current, before_words, now_words):
    children = []
    old = [_normalize(word) for word, _ in before_words]
    new = [_normalize(word) for word, _ in now_words]
    if old == new:
        return children, False
    if Counter(old) == Counter(new):
        return children, True
    for op, a, b, c, d in SequenceMatcher(None, old, new).get_opcodes():
        if op != "equal":
            children.append({"kind": op, "earlier": _location(earlier, before_words, a, b, publish_text=False),
                             "current": _location(current, now_words, c, d, publish_text=True)})
    return children, False


def _compare(previous, latest, current_pages):
    earlier_pages = {previous.page_xref(index): index for index in range(len(previous))}
    current = {latest.page_xref(index): index for index in range(len(latest))}
    matched = sorted(earlier_pages.keys() & current.keys(), key=current.__getitem__)
    report = {"earlier_version": previous.version_count, "checked": [], "unassessed": [], "order_only": [],
              "earlier_only_pages": sorted(earlier_pages[xref] + 1 for xref in earlier_pages.keys() - current.keys()),
              "current_only_pages": sorted(current[xref] + 1 for xref in current.keys() - earlier_pages.keys()),
              "changed_ranges": 0, "changes": []}
    if len(earlier_pages) != len(previous) or len(current) != len(latest):
        raise ValueError("Page identities are ambiguous in a retained save")
    for xref in matched:
        a, b = earlier_pages[xref], current[xref]
        pair = [a + 1, b + 1]
        try:
            first = _page_words(previous[a])
            if b not in current_pages:
                current_pages[b] = _page_words(latest[b])
        except (RuntimeError, ValueError):
            report["unassessed"].append(pair)
            continue
        if first is None or current_pages[b] is None:
            report["unassessed"].append(pair)
            continue
        changes, reordered = _differences(previous[a], latest[b], first, current_pages[b])
        report["checked"].append(pair)
        if reordered:
            report["order_only"].append(pair)
        report["changed_ranges"] += len(changes)
        report["changes"].extend(changes[:max(0, MAX_CHANGES - len(report["changes"]))])
    return report


def inspect(data):
    """Compare recent byte-exact saves with the submitted PDF's current pages."""
    import pymupdf

    result = _unavailable(len(data), None)
    if len(data) > MAX_BYTES:
        return result | {"reason": "size_limit"}
    endings = list(islice(_END.finditer(data), 65))
    if not endings or len(endings) > 64 or data[endings[-1].end():].strip(_WS):
        return result | {"reason": "unverified_chain"}
    try:
        with pymupdf.open(stream=data, filetype="pdf") as latest:
            ending = endings[-1]
            revision = _revision(latest, data, ending)
            result["saves"].append(revision)
            if revision["version"] == 1:
                return result | {"status": "no_history"}
            if not 1 <= len(latest) <= MAX_PAGES:
                return result | {"reason": "page_limit"}
            current_pages = {}
            ending = _previous(latest, endings, revision)
            for _ in range(min(revision["version"], MAX_SAVES) - 1):
                with pymupdf.open(stream=data[:ending.end()], filetype="pdf") as previous:
                    earlier = _revision(previous, data, ending)
                    if earlier["version"] != revision["version"] - 1:
                        raise ValueError("The saved PDF revision order could not be verified")
                    result["saves"].append(earlier)
                    if not 1 <= len(previous) <= MAX_PAGES:
                        return result | {"reason": "page_limit"}
                    comparison = _compare(previous, latest, current_pages)
                    result["comparisons"].append(comparison)
                    if _has_changes(comparison):
                        return result | {"status": "changed"}
                    revision = earlier
                    if revision["version"] == 1:
                        break
                    ending = _previous(previous, endings, revision)
    except (RuntimeError, ValueError, OverflowError):
        return result | {"reason": "unverified_chain"}
    if any(row["unassessed"] or row["order_only"] or not row["checked"] for row in result["comparisons"]):
        return result | {"reason": "page_text_incomplete"}
    if result["saves"][-1]["version"] != 1:
        return result | {"reason": "save_limit"}
    return result | {"status": "no_change"}


def _unavailable(size, reason):
    return {"schema": SCHEMA, "affects_overall_confidence": False, "source_bytes": size,
            "status": "unavailable", "reason": reason, "saves": [], "comparisons": []}


def inspect_path(path):
    size = path.stat().st_size
    if size > MAX_BYTES:
        return _unavailable(size, "size_limit")
    return inspect(path.read_bytes())


def inspect_within(path, timeout):
    try:
        child = subprocess.run(  # noqa: S603 - fixed interpreter, module, and staged source path
            [sys.executable, "-I", "-m", "app.evidence.document_pdf_history", str(path)],
            check=True, capture_output=True, timeout=timeout)
        return json.loads(child.stdout)
    except subprocess.TimeoutExpired:
        return _unavailable(path.stat().st_size, "time_limit")
    except (subprocess.CalledProcessError, ValueError):
        return _unavailable(path.stat().st_size, "reader_unavailable")


def previews(data, record):
    from app.evidence.document_packets import _pdf_plates, _png

    for index, side in current_locations(record):
        page = side["page"]
        anchor = {"kind": "pdf", "page": page, "page_width": side["width"], "page_height": side["height"],
                  "page_rotation": side["rotation"], "highlight_boxes": side["boxes"]}
        plates = _pdf_plates(data, [anchor])
        for role in ("crop", "overlay"):
            image, geometry = plates[f"page{page}_{role}"]
            payload = _png(image)
            yield f"change{index}_{role}", payload, {"media_type": "image/png", "sha256": _sha(payload),
                                                    "width": image.width, "height": image.height, **geometry}


def verify_assets(data, review, read_asset):
    for role, payload, metadata in previews(data, review["inspection"]):
        recorded = review["artifacts"][role]
        if (canonical_hash({k: v for k, v in recorded.items() if k != "url"}) != canonical_hash(metadata)
                or read_asset(recorded["sha256"]) != payload):
            raise ValueError("A saved wording crop does not reproduce from the submitted PDF")


if __name__ == "__main__":
    print(json.dumps(inspect_path(Path(sys.argv[1]))))
