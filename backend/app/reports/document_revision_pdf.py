"""Render an earlier-draft comparison from its source-bound JSON."""

from __future__ import annotations

from html import escape

from app.evidence.document_revision import HTML_COORDINATES
from app.reports.engine import _restricted_fetcher, load_weasyprint

STYLE = """
@page { size: Letter; margin: 0.65in; @bottom-right { content: counter(page); color: #526658; font-size: 9pt; } }
body { color: #24382a; font: 10pt/1.4 sans-serif; }
.brand { font-weight: 700; letter-spacing: .12em; font-size: 9pt; border-bottom: 1px solid #bdc9be; padding-bottom: 8pt; }
h1 { margin: 14pt 0 6pt; font-size: 22pt; line-height: 1.15; }
h2 { margin: 13pt 0 4pt; font-size: 12pt; break-after: avoid; }
p { margin: 4pt 0; }
.sources { margin: 9pt 0; font-size: 8pt; }
.hash { font-family: monospace; overflow-wrap: anywhere; }
.side { margin: 5pt 0; padding-left: 8pt; border-left: 2px solid #bdc9be; }
.coordinates { font: 8.5pt/1.25 monospace; color: #526658; break-after: avoid; }
.excerpt { white-space: pre-wrap; overflow-wrap: anywhere; }
.word-paired-change { break-inside: avoid; }
mark { color: inherit; background: #e3e9e0; text-decoration: underline; text-decoration-color: #285f50; }
a { color: #285f50; }
.page-preview { margin: 5pt 0 8pt; break-inside: avoid; }
.page-preview img { display: block; max-width: 100%; max-height: 2.2in; object-fit: contain; object-position: left top; }
.page-preview figcaption { color: #526658; font-size: 8.5pt; }
"""
LABELS = {"body": "Message body", "quoted": "Quoted text", "signature": "Signature"}
FILE_LABELS = {"plain_text": "Plain text", "docx": "Word body text", "pdf": "PDF text layer"}
PNG_URL = "data:image/png;base64,"
TABLE_SKIP = {"no_matching_position": "No table at the same body position.",
              "unsupported_layout": "Outside the supported table layout.",
              "changed_structure": "The table shape, headings or row labels changed.",
              "comparison_limit": "The table exceeded the comparison limit.",
              "caption_unmatched": "The table caption has no unique match in both drafts."}


def _word_position(item):
    return f'Paragraph {item["paragraph"]} [{item["start"]}, {item["end"]})'


def _page_previews(excerpt):
    output = []
    for page in excerpt.get("pdf_pages") or ():
        number = page["page"]
        if preview := page.get("preview"):
            url = PNG_URL + escape(preview["highlighted_png"], quote=True)
            output.append(f'<figure class="page-preview"><img alt="Highlighted source text on page {number}" src="{url}">'
                          f'<figcaption>Page {number} · highlighted source crop</figcaption></figure>')
        else:
            output.append(f'<p class="coordinates">Page {number}</p>')
    return "".join(output)


def _side(label, excerpt):
    if excerpt is None:
        return f'<div class="side"><p class="coordinates">{label}</p><p>No corresponding section.</p></div>'
    start, end = excerpt["start"], excerpt["end"]
    text = excerpt["text"]
    segments = []
    cursor = 0
    for changed_from, changed_to in excerpt["changed_tokens"]:
        token_start, token_end = changed_from - start, changed_to - start
        segments.append(escape(text[cursor:token_start]))
        segments.append(f"<mark>{escape(text[token_start:token_end])}</mark>")
        cursor = token_end
    segments.append(escape(text[cursor:]))
    value = "".join(segments) or "No text at this location."
    anchors = " · ".join(_word_position(item) for item in excerpt.get("paragraphs") or ())
    paragraph_ranges = f'<p class="coordinates">{anchors}</p>' if anchors else ""
    html_positions = []
    for token in excerpt.get("html_tokens") or ():
        ranges = ", ".join(f"[{a}, {b})" for a, b in token["source_ranges"])
        html_positions.append(f'[{token["start"]}, {token["end"]}) → {ranges}')
    html_ranges = f'<p class="coordinates">Decoded HTML positions · {" · ".join(html_positions)}</p>' if html_positions else ""
    return (f'<div class="side"><p class="coordinates">{label} [{start}, {end})</p>'
            f'{paragraph_ranges}<p class="excerpt">{value}</p>{html_ranges}{_page_previews(excerpt)}</div>')


def _paired(review: dict | None) -> str:
    if review is None:
        return ""
    output = [f'<section><h2>Experimental paired writing check</h2><p>{escape(review["summary"])}</p>'
              f'<p>{escape(review["scope"])}</p>']
    if review.get("responses"):
        links = ", ".join(f'<a href="#draft-change-{escape(value)}">{escape(value)}</a>' for value in review["change_ids"])
        output.append(f'<p class="coordinates">Linked wording changes {links}</p>')
        for side, label in (("earlier", "Earlier"), ("later", "Analyzed")):
            row = review[side]
            unit = "sentences" if len(row["context_spans"]) == 4 else "sentence"
            ranges = "".join(f'<p class="coordinates">{name} · {_word_position(item)}</p>'
                             for name, item in zip(("Previous sentence", "Changed sentence", "Next sentence"),
                                                   row.get("context_paragraphs", ())))
            output.append(f'<div class="side"><p class="coordinates">{label} {unit} [{row["start"]}, {row["end"]})</p>'
                          f'{ranges}<p class="excerpt">{escape(row["text"])}</p></div>')
        output.append("<p>Uncalibrated responses from the earlier and analyzed versions</p>")
        center = review.get("center_responses")
        for i, row in enumerate(review["responses"], 1):
            reading = f'{row["earlier"]:.4g} → {row["later"]:.4g}'
            if center:
                exact = center[i - 1]
                reading = f'With neighbors {reading}. Changed sentence alone {exact["earlier"]:.4g} → {exact["later"]:.4g}.'
            elif review.get("neighboring_contexts"):
                reading = f"Centered context {reading}"
            output.append(f'<p>Check {i} · {reading}</p>')
        if review.get("neighboring_contexts"):
            output.append("<h2>Additional scored contexts</h2>")
        for context in review.get("neighboring_contexts", ()):
            reading = ("Both responses increased enough for review." if context["status"] == "joint_increase"
                       else "The responses did not show a shared increase at the review level.")
            output.append(f"<p>{reading}</p>")
            for side, label in (("earlier", "Earlier"), ("later", "Analyzed")):
                row = context[side]
                output.append(f'<div class="side"><p class="coordinates">{label} scored context [{row["start"]}, '
                              f'{row["end"]})</p><p class="excerpt">{escape(row["text"])}</p></div>')
            for i, row in enumerate(context["responses"], 1):
                output.append(f'<p>Check {i} · {row["earlier"]:.4g} → {row["later"]:.4g}</p>')
    return "".join(output) + "</section>"


def _word_tables(coverage):
    if not coverage:
        return ""
    output = ["<section><h2>Word table wording changes</h2>",
              "<p>Table, row and column numbers start at one. Character positions start within each cell.</p>"]
    for table in coverage["compared"]:
        old_number, new_number = table["earlier_table"], table["later_table"]
        if table.get("match") == "caption":
            output.append(f'<p>Earlier table {old_number} and analyzed table {new_number} were matched by a unique Word caption.</p>')
        for key, label in (("unchecked_rows", "rows"), ("unchecked_columns", "columns")):
            if (counts := table.get(key)) and any(counts.values()):
                output.append(f'<p>Unchecked {label}: {counts["earlier"]} earlier, {counts["later"]} analyzed.</p>')
        if not table["changes"]:
            output.append(f'<p>No wording changes in the compared cells of earlier table {old_number} and analyzed table {new_number}.</p>')
        for change in table["changes"]:
            ident = escape(change["id"])
            old_pos = (f' · Row {change["earlier_row"]}, column {change["earlier_column"]}' if "earlier_row" in change else "")
            new_pos = f' · Row {change["row"]}, column {change["column"]}' if old_pos else ""
            output.append(f'<section><h2 id="draft-change-{ident}">{ident} · Row {change["row"]}, column {change["column"]}</h2>'
                          f'{_side(f"Earlier table {old_number}{old_pos}", change["earlier"])}'
                          f'{_side(f"Analyzed table {new_number}{new_pos}", change["later"])}</section>')
    for table in coverage["skipped"]:
        numbers = [f"{label} table {table[key]}" for key, label in (("earlier_table", "Earlier"), ("later_table", "Analyzed"))
                   if table[key] is not None]
        output.append(f'<p>{" / ".join(numbers)}. {TABLE_SKIP[table["reason"]]}</p>')
    if coverage["omitted_positions"]:
        output.append(f'<p>{coverage["omitted_positions"]} further table positions exceed the reporting limit.</p>')
    return "".join(output) + "</section>"


def render(comparison: dict) -> bytes:
    changes = []
    docx = comparison["format"] == "docx"
    paired = comparison.get("paired_check")
    change_class = ' class="word-paired-change"' if paired and paired.get("earlier", {}).get("context_paragraphs") else ""
    html_reply = comparison["coordinate_space"] == HTML_COORDINATES
    mixed = comparison["format"] == "mixed"
    format_label = {"plain_text": "Text", "docx": "Word body text", "pdf": "PDF text", "mixed": "Wording change"}.get(comparison["format"])
    for change in comparison["changes"]:
        label = format_label or LABELS[change["role"]]
        anchor = escape(change["id"])
        changes.append(f'<section{change_class}><h2 id="draft-change-{anchor}">{anchor} · {label}</h2>'
                       f'{_side("Earlier draft", change["earlier"])}{_side("Analyzed version", change["later"])}</section>')
    sources = []
    for key, label in (("earlier", "Earlier draft"), ("later", "Analyzed version")):
        source = comparison[key]
        part = f' · MIME part {escape(source["mime_part"])}' if source["mime_part"] else ""
        if mixed:
            part = f' · {FILE_LABELS[source["format"]]}'
        sources.append(f'<p><strong>{label}</strong>{part}<br>SHA-256 <span class="hash">{escape(source["sha256"])}</span></p>')
    positions = ("Word offsets count Unicode code points in the body text. Two line breaks separate included paragraphs. "
                 "Each paragraph range uses offsets within that paragraph." if docx else "Ranges use decoded Unicode code points.")
    if comparison["format"] == "pdf":
        positions = ("Ranges use Unicode code points in the extracted text. Source boxes use unrotated PDF page points. "
                     "Previews show selected source regions.")
    if html_reply:
        positions = ("The first range gives the position in the extracted reply. The ranges after the arrow give the positions "
                     "in the decoded HTML MIME part. Both count Unicode code points.")
    if mixed:
        positions = ("Each side uses its own Unicode character offsets. Word paragraph offsets start within the paragraph. "
                     "PDF source boxes use unrotated page coordinates.")
    word_tables = comparison.get("word_tables")
    no_changes = ("No word or punctuation changes found in the top-level body paragraphs." if word_tables else
                  "No word or punctuation changes found in the reply body." if html_reply else "No word or punctuation changes found.")
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Earlier-draft comparison</title>
<meta name="author" content="Chipmunk Verify"><style>{STYLE}</style></head><body>
<p class="brand">CHIPMUNK VERIFY</p><h1>Earlier-draft comparison</h1>
<p>{escape(comparison['scope'])}</p><p>{escape(comparison['interpretation'])}</p>
<p>Highlights mark changed words and punctuation. {positions} Each range includes its start and excludes its end.</p>
<div class="sources">{''.join(sources)}</div>{_paired(paired)}
{''.join(changes) or f'<p>{no_changes}</p>'}{_word_tables(word_tables)}
</body></html>'''

    return load_weasyprint().HTML(string=content, url_fetcher=_restricted_fetcher(())).write_pdf()
