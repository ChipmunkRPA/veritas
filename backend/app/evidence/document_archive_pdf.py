"""Bind public archive comparisons to visible, ordered PDF text."""

from pathlib import Path
from tempfile import TemporaryDirectory

from app.evidence import document_archive, document_gao_archive, document_revision_pdf
from app.evidence.document_revision import MAX_CHARS

SCOPE = ("The PDF comparison reads visible selectable text in a clear reading order, using one column per page. "
         "Image text and formatting are outside this comparison. Page crops highlight source word boxes.")


def compare(data: bytes, upload_sha256: str, *, gao_archive=None):
    with TemporaryDirectory(prefix="chipmunk-archive-pdf-") as directory:
        path = Path(directory) / "retained.pdf"
        doc = document_revision_pdf.extract_native(path, data, visible_text_only=True)
        limit = document_revision_pdf.NATIVE_COMPARISON_MAX_PAGES
        if len(doc.text) > MAX_CHARS or not document_revision_pdf.ordered_text(
            doc.text, doc.source_map, page_limit=limit, single_column=True,
        ):
            raise ValueError("Use a PDF with visible selectable text in a clear reading order.")
        result = (document_gao_archive.compare(doc, upload_sha256, *gao_archive) if gao_archive is not None
                  else document_archive.compare(doc, upload_sha256))
        result["scope"] += " " + SCOPE
        if result["status"] != "reference_bound":
            return result

        if gao_archive is not None:
            match_start, match_end = result["reference"]["submitted_range"]
            for number, block in enumerate(doc.source_map, 1):
                start, end = max(match_start, block["start"]), min(match_end, block["end"])
                if start < end:
                    result["aligned_paragraphs"].append({"submitted_paragraph": number, "submitted_range": [start, end],
                        "archive_paragraph": None, "alignment": "bounded_pair"})
        page_extents = {}
        for block in doc.source_map:
            extent = page_extents.setdefault(block["page"], [block["start"], block["end"]])
            extent[1] = block["end"]
        for row in result["aligned_paragraphs"]:
            start, end = row["submitted_range"]
            row["pdf_pages"] = [{"page": page, "extracted_range": [max(start, first), min(end, last)]}
                                for page, (first, last) in page_extents.items() if start < last and first < end]

        for change in result["changes"]:
            excerpt = change["submitted"]
            marked = excerpt["changed_tokens"]
            if gao_archive is not None:
                marked = [(first + a, first + b) for first, end in marked
                          for _, a, b in document_gao_archive.token_stream(doc.text[first:end])]
            excerpt["pdf_pages"] = document_revision_pdf.page_locations(
                (excerpt["start"], excerpt["end"]), marked, doc.source_map,
            )
        if result["changes"]:
            locations = [{"earlier": None, "later": change["submitted"]} for change in result["changes"]]
            previews = document_revision_pdf.render_previews(locations, path.with_name("archive.txt"), path)
            for change, preview in zip(result["changes"], previews, strict=True):
                change["submitted"] = preview["later"]
        return result
