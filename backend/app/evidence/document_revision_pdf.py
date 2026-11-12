"""Locate paired native PDF text on the same page geometry as document evidence."""

from __future__ import annotations

import base64
import hashlib
import json
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


def extract_native(path: Path, data: bytes):
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
    _add_previews(located, old_path.read_bytes(), new_path.read_bytes())
    json.dump(located, sys.stdout)
