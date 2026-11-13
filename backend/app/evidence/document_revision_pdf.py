"""Locate paired native PDF text on the same page geometry as document evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from app.core.config import settings
from app.core.deadline import StepTimeout
from app.detectors.text.extract import NATIVE_COMPARISON_MAX_PAGES
from app.evidence.document_packets import DocumentPacketIntegrityError, _pdf_plates, _png, _rect

COORDINATES = "pdf_native_text_unicode_codepoint"
SCOPE = ("Compares wording and punctuation in the extracted PDF text layers. Page previews highlight source word boxes. "
         "Boxes may include adjacent text. Scanned pages, text in images, and formatting are outside this comparison.")
MAX_PREVIEWS_PER_VERSION = 6
MAX_PREVIEW_BYTES = 1024 * 1024
MULTIPAGE_LEFT_MARGIN_SPREAD = 36  # Points, allowing ordinary first-line indents.


def extract_native(path: Path, data: bytes, *, mixed_draft: bool = False, visible_text_only: bool = False):
    from app.detectors.text.pipeline import extract_within

    path.write_bytes(data)
    message = f"Use unlocked PDFs with mapped text on every page and at most {NATIVE_COMPARISON_MAX_PAGES} pages per version."
    try:
        document = extract_within(path, ".pdf", native_pdf_only=True)
    except (RuntimeError, ValueError, StepTimeout):
        raise ValueError(message) from None
    if (not document.source_map or "\ufffd" in document.text or "\x00" in document.text
            or document.meta.get("unmapped_pages") or document.meta.get("partially_checked_pages")
            or any(row.get("extraction") != "text_layer" for row in document.source_map)):
        raise ValueError(message)
    hidden_pages = document.meta.get("hidden_text_pages")
    if visible_text_only and hidden_pages:
        raise ValueError("The PDF contains hidden or fully transparent text.")
    if mixed_draft and document.meta["pages"] > 1 and hidden_pages:
        numbers = ", ".join(map(str, hidden_pages))
        raise ValueError(f"Multi-page draft comparison requires visible PDF text. Hidden or fully transparent text was detected on page(s) {numbers}.")
    return document


def page_locations(extent, changed_tokens, source_map):
    """Use normalized source offsets, including both boxes for a wrapped word."""
    a, b = extent
    if a == b:
        # Insertions/deletions have a point in the other source. Prefer the
        # next block at a boundary, then use the last page at end of document.
        sources = [next((row for row in source_map if row["end"] > a), source_map[-1])]
    else:
        sources = [row for row in source_map if row["start"] < b and row["end"] > a]
    pages = {}
    for source in sources:
        page = pages.setdefault(source["page"], {key: source[key] for key in ("page", "page_width", "page_height", "page_rotation")}
                                | {"word_regions": []})
        for fragment in source.get("fragments", ()):
            start, end = fragment["start"], fragment["end"]
            if not any(start < to and end > first for first, to in changed_tokens):
                continue
            try:
                box = _rect(fragment["bbox"], page["page_width"], page["page_height"])
            except DocumentPacketIntegrityError:
                raise ValueError("A changed text box could not be placed on its source PDF page.") from None
            page["word_regions"].append({"start": start, "end": end, "bbox": list(box)})
    words = [word for page in pages.values() for word in page["word_regions"]]
    if any(not any(word["start"] <= a and word["end"] >= b for word in words) for a, b in changed_tokens):
        raise ValueError("Some changed tokens could not be located in the source PDF text layer.")
    return list(pages.values())


def ordered_text(text: str, source_map: list[dict], *, page_limit: int = 1, single_column: bool = False) -> str:
    """Require upright native pages with contiguous, top-to-bottom word boxes."""
    if not source_map:
        return ""
    page = cursor = 0
    previous_bottom = 0.
    # Restrict multi-page comparisons to a single column on each page.
    single_column = single_column or source_map[-1]["page"] > 1
    left_min = left_max = None
    for index, block in enumerate(source_map):
        number = block.get("page")
        if number != page:
            if number != page + 1 or number > page_limit:
                return ""
            page, previous_bottom = number, 0.
            left_min = left_max = None
        separator = "\n\n" if index else ""
        if (block.get("page_rotation") != 0 or block.get("extraction") != "text_layer"
                or block["start"] != cursor + len(separator) or text[cursor:block["start"]] != separator):
            return ""
        start, end = block["start"], block["end"]
        width, height = block["page_width"], block["page_height"]
        words = [(start + match.start(), start + match.end()) for match in re.finditer(r"\S+", text[start:end])]
        fragments = block.get("fragments") or []
        if not words or words != [(item["start"], item["end"]) for item in fragments]:
            return ""
        try:
            _, top, _, bottom = _rect(block["bbox"], width, height)
            if top < previous_bottom - .5:
                return ""
            line_top = line_bottom = previous_right = None
            for item in fragments:
                x0, y0, x1, y1 = _rect(item["bbox"], width, height)
                if line_top is None or y0 > line_top + 2:
                    if line_bottom is not None and y0 < line_bottom - .5:
                        return ""
                    if single_column:
                        left_min = x0 if left_min is None else min(left_min, x0)
                        left_max = x0 if left_max is None else max(left_max, x0)
                        if left_max - left_min > MULTIPAGE_LEFT_MARGIN_SPREAD:
                            return ""
                    line_top, line_bottom = y0, y1
                else:
                    if (y0 < line_top - 2 or x0 < previous_right - .5
                            or x0 - previous_right > 2 * max(y1 - y0, line_bottom - line_top)):
                        return ""
                    line_bottom = max(line_bottom, y1)
                previous_right = x1
        except DocumentPacketIntegrityError:
            return ""
        cursor, previous_bottom = end, bottom
    return text if cursor == len(text) else ""


def render_previews(changes: list[dict], earlier: Path, later: Path) -> list[dict]:
    if not changes:
        return changes
    try:
        completed = subprocess.run(  # noqa: S603 - fixed interpreter and private staged paths
            [sys.executable, "-m", "app.evidence.document_revision_pdf", str(earlier), str(later)],
            input=json.dumps(changes).encode(), capture_output=True, check=True,
            cwd=Path(__file__).resolve().parents[2], timeout=settings.document_decode_timeout_s,
        )
        return json.loads(completed.stdout)
    except (subprocess.SubprocessError, ValueError):
        raise ValueError("The local service could not render the recorded PDF source regions.") from None


def _add_previews(changes: list[dict], earlier: bytes, later: bytes) -> None:
    """Embed bounded original and highlighted crops from the existing renderer."""
    remaining = MAX_PREVIEW_BYTES
    used = {"earlier": 0, "later": 0}
    for change in changes:
        for side, source in (("earlier", earlier), ("later", later)):
            for page in (change[side] or {}).get("pdf_pages", ()):
                if (not page["word_regions"] or used[side] >= MAX_PREVIEWS_PER_VERSION or remaining <= 0):
                    continue
                used[side] += 1
                anchor = {**page, "kind": "pdf", "highlight_boxes": [row["bbox"] for row in page["word_regions"]]}
                try:
                    plates = _pdf_plates(source, [anchor])
                except DocumentPacketIntegrityError:
                    raise ValueError("A preview could not be tied to the recorded PDF page geometry.") from None
                number = page["page"]
                crop, geometry = plates[f"page{number}_crop"]
                overlay, _ = plates[f"page{number}_overlay"]
                raw, marked = _png(crop), _png(overlay)
                size = len(raw) + len(marked)
                if size > remaining:
                    continue
                remaining -= size
                page["preview"] = {
                    "source_bbox": geometry["source_bbox"], "render_scale": geometry["render_scale"],
                    "source_crop_pixels": geometry["source_crop_pixels"],
                    "crop_png": base64.b64encode(raw).decode("ascii"), "crop_sha256": hashlib.sha256(raw).hexdigest(),
                    "highlighted_png": base64.b64encode(marked).decode("ascii"), "highlighted_sha256": hashlib.sha256(marked).hexdigest(),
                }


if __name__ == "__main__":
    located = json.load(sys.stdin)
    old_path, new_path = map(Path, sys.argv[1:])
    _add_previews(located, *(path.read_bytes() if path.suffix == ".pdf" else b"" for path in (old_path, new_path)))
    json.dump(located, sys.stdout)
