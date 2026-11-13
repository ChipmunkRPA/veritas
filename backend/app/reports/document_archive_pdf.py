"""Render source comparison evidence without fetching external resources."""

from __future__ import annotations

from html import escape

from app.evidence.document_archive import RECOVERED_BOUNDARIES, STATUS
from app.evidence.document_gao_archive import SCHEMA as GAO_SCHEMA
from app.reports.document_revision_pdf import STYLE, _side
from app.reports.engine import load_weasyprint


def render(comparison: dict) -> bytes:
    library, reference, coverage = (comparison[key] for key in ("library", "reference", "coverage"))
    source = ""
    joined = comparison.get("layout") == RECOVERED_BOUNDARIES
    gao = comparison["schema"] == GAO_SCHEMA
    terms = "body tokens" if gao else "body words"
    if reference:
        source = (f'<p><a href="{escape(reference["url"])}">{escape(reference["id"])}</a> · '
                  f'<a href="{escape(reference["version_url"])}">Pinned source version</a></p>'
                  f'<p>{coverage["matched_words"]} of {coverage["body_words"]} {terms} support the source match.</p>')
        if gao:
            submitted, transcript = reference["submitted_range"], reference["transcript_range"]
            source += (f'<p>Published {escape(reference["published_date"])}. Matched submitted range '
                       f'[{submitted[0]}, {submitted[1]}). Transcript range [{transcript[0]}, {transcript[1]}).</p>'
                       f'<p>Publisher record SHA-256 <span class="hash">{escape(reference["source_record_sha256"])}</span></p>'
                       f'<p>Extracted transcript SHA-256 <span class="hash">{escape(reference["transcript_sha256"])}</span></p>')
        if joined:
            source += "<p>The submitted text joins several source paragraphs. The comparison recovered their boundaries from the archive.</p>"
        word_passages = [row["docx_location"] for row in comparison["aligned_paragraphs"] if row.get("docx_location")]
        if word_passages:
            source += "<h2>Matched Word paragraphs</h2><ul>" + "".join(
                f'<li>Word body paragraph {location["paragraph"]}, characters '
                f'[{location["paragraph_range"][0]}, {location["paragraph_range"][1]}).</li>'
                for location in word_passages
            ) + "</ul>"
    changes = []
    for change in comparison["changes"]:
        a = change["archive_paragraph"]
        b = change["submitted_paragraph"]
        label = "Published highlights transcript" if gao else f"Archive body paragraph {a}" if a is not None else "Archive version"
        submitted_label = f"Passage in submitted paragraph {b}" if joined else f"Submitted paragraph {b}"
        location = change["submitted"].get("docx_location")
        native = ""
        if location is not None:
            submitted_label = f"Word body paragraph {location['paragraph']}"
            start, end = location["paragraph_range"]
            native = f'<p class="offset">Word paragraph characters [{start}, {end}).</p>'
        changes.append(f'<section><h2>{submitted_label}</h2>'
                       f'{_side(label, change["archive"])}{_side("Submitted text", change["submitted"])}{native}</section>')
    if reference and not changes:
        changes.append("<p>No normalized wording differences in the compared passage.</p>" if gao else
                       "<p>No word or punctuation differences in the compared passages.</p>")
    sentences = comparison.get("submitted_sentence_ranges") or ()
    sentence_ranges = ("<p>Submitted sentences containing differences "
                       + ", ".join(f"[{start}, {end})" for start, end in sentences) + ".</p>") if sentences else ""
    coordinate_label = "relative to the full highlights transcript" if gao else "relative to their body paragraph"
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Public archive comparison</title>
<meta name="author" content="Chipmunk Verify"><style>{STYLE}</style></head><body>
<p class="brand">CHIPMUNK VERIFY</p><h1>Public archive comparison</h1>
<p>{escape(STATUS[comparison["status"]])}</p><p>{escape(comparison["scope"])}</p>
<p>{escape(comparison["interpretation"])}</p>
<p><strong>{escape(library["title"])}</strong> · {escape(library["publisher"])} ·
<a href="{escape(library["license_url"])}">{escape(library["license"])}</a></p>{source}
<p>Submitted ranges use extracted Unicode text offsets. Archive ranges are {coordinate_label}.
Each range includes its start and excludes its end.</p>
<div class="sources"><p>Submitted SHA-256 <span class="hash">{escape(comparison["submitted_sha256"])}</span></p>
<p>Extracted text SHA-256 <span class="hash">{escape(comparison["extracted_text_sha256"])}</span></p>
<p>Archive library SHA-256 <span class="hash">{escape(library["sha256"])}</span></p></div>{sentence_ranges}{''.join(changes)}
</body></html>'''

    def no_external_resources(_url):
        raise ValueError("Archive comparison PDFs use inline content only.")

    return load_weasyprint().HTML(string=html, url_fetcher=no_external_resources).write_pdf()
