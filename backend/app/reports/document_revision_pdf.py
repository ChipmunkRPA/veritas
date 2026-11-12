"""Render an earlier-draft comparison from its source-bound JSON."""

from __future__ import annotations

from html import escape

from app.reports.engine import load_weasyprint

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
"""
LABELS = {"body": "Message body", "quoted": "Quoted text", "signature": "Signature"}


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
    return (f'<div class="side"><p class="coordinates">{label} [{start}, {end})</p>'
            f'<p class="excerpt">{value}</p></div>')


def render(comparison: dict) -> bytes:
    changes = []
    for change in comparison["changes"]:
        label = "Text" if comparison["format"] == "plain_text" else LABELS[change["role"]]
        changes.append(f'<section><h2>{escape(change["id"])} · {label}</h2>'
                       f'{_side("Earlier draft", change["earlier"])}{_side("Analyzed version", change["later"])}</section>')
    sources = []
    for key, label in (("earlier", "Earlier draft"), ("later", "Analyzed version")):
        source = comparison[key]
        part = f' · MIME part {escape(source["mime_part"])}' if source["mime_part"] else ""
        sources.append(f'<p><strong>{label}</strong>{part}<br>SHA-256 <span class="hash">{escape(source["sha256"])}</span></p>')
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Earlier-draft comparison</title>
<meta name="author" content="Chipmunk Verify"><style>{STYLE}</style></head><body>
<p class="brand">CHIPMUNK VERIFY</p><h1>Earlier-draft comparison</h1>
<p>{escape(comparison['scope'])}</p><p>{escape(comparison['interpretation'])}</p>
<p>Highlights mark changed words and punctuation. Ranges use decoded Unicode code points. Each range includes its start and excludes its end.</p>
<div class="sources">{''.join(sources)}</div>{''.join(changes) or '<p>No word or punctuation changes found.</p>'}
</body></html>'''

    def no_external_resources(_url):
        raise ValueError("Earlier-draft reports use inline content only.")

    return load_weasyprint().HTML(string=content, url_fetcher=no_external_resources).write_pdf()
