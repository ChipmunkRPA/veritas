"""Render an earlier-draft comparison from its source-bound JSON."""

from __future__ import annotations

from html import escape

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
mark { color: inherit; background: #e3e9e0; text-decoration: underline; text-decoration-color: #285f50; }
a { color: #285f50; }
.page-preview { margin: 5pt 0 8pt; break-inside: avoid; }
.page-preview img { display: block; max-width: 100%; max-height: 2.2in; object-fit: contain; object-position: left top; }
.page-preview figcaption { color: #526658; font-size: 8.5pt; }
"""
LABELS = {"body": "Message body", "quoted": "Quoted text", "signature": "Signature"}
PNG_URL = "data:image/png;base64,"


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
    anchors = " · ".join(f'Paragraph {item["paragraph"]} [{item["start"]}, {item["end"]})'
                         for item in excerpt.get("paragraphs") or ())
    paragraph_ranges = f'<p class="coordinates">{anchors}</p>' if anchors else ""
    return (f'<div class="side"><p class="coordinates">{label} [{start}, {end})</p>'
            f'{paragraph_ranges}<p class="excerpt">{value}</p>{_page_previews(excerpt)}</div>')


def _paired(review: dict | None) -> str:
    if review is None:
        return ""
    output = [f'<section><h2>Experimental paired writing check</h2><p>{escape(review["summary"])}</p>'
              f'<p>{escape(review["scope"])}</p>']
    if review.get("responses"):
        links = ", ".join(f'<a href="#draft-change-{escape(value)}">{escape(value)}</a>' for value in review["change_ids"])
        output.append(f'<p class="coordinates">Linked wording changes {links}</p>')
        for side, label in (("earlier", "Earlier sentence"), ("later", "Analyzed sentence")):
            row = review[side]
            output.append(f'<div class="side"><p class="coordinates">{label} [{row["start"]}, {row["end"]})</p>'
                          f'<p class="excerpt">{escape(row["text"])}</p></div>')
        output.append("<p>Uncalibrated responses from the earlier and analyzed versions</p>")
        output.extend(f'<p>Check {i} · {row["earlier"]:.4g} → {row["later"]:.4g}</p>'
                      for i, row in enumerate(review["responses"], 1))
    return "".join(output) + "</section>"


def render(comparison: dict) -> bytes:
    changes = []
    docx = comparison["format"] == "docx"
    format_label = {"plain_text": "Text", "docx": "Word body text", "pdf": "PDF text"}.get(comparison["format"])
    for change in comparison["changes"]:
        label = format_label or LABELS[change["role"]]
        anchor = escape(change["id"])
        changes.append(f'<section><h2 id="draft-change-{anchor}">{anchor} · {label}</h2>'
                       f'{_side("Earlier draft", change["earlier"])}{_side("Analyzed version", change["later"])}</section>')
    sources = []
    for key, label in (("earlier", "Earlier draft"), ("later", "Analyzed version")):
        source = comparison[key]
        part = f' · MIME part {escape(source["mime_part"])}' if source["mime_part"] else ""
        sources.append(f'<p><strong>{label}</strong>{part}<br>SHA-256 <span class="hash">{escape(source["sha256"])}</span></p>')
    positions = ("Word offsets count Unicode code points in the body text. Two line breaks separate included paragraphs. "
                 "Each paragraph range uses offsets within that paragraph." if docx else "Ranges use decoded Unicode code points.")
    if comparison["format"] == "pdf":
        positions = ("Ranges use Unicode code points in the extracted text. Source boxes use unrotated PDF page points. "
                     "Previews show selected source regions.")
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Earlier-draft comparison</title>
<meta name="author" content="Chipmunk Verify"><style>{STYLE}</style></head><body>
<p class="brand">CHIPMUNK VERIFY</p><h1>Earlier-draft comparison</h1>
<p>{escape(comparison['scope'])}</p><p>{escape(comparison['interpretation'])}</p>
<p>Highlights mark changed words and punctuation. {positions} Each range includes its start and excludes its end.</p>
<div class="sources">{''.join(sources)}</div>{_paired(comparison.get('paired_check'))}
{''.join(changes) or '<p>No word or punctuation changes found.</p>'}
</body></html>'''

    return load_weasyprint().HTML(string=content, url_fetcher=_restricted_fetcher(())).write_pdf()
