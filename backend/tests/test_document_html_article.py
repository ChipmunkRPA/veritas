from __future__ import annotations

from copy import deepcopy
from email.message import EmailMessage
from html import unescape
from io import BytesIO

import pytest

from app.billing.preflight import PreflightError, validate_upload
from app.detectors.text import html_article
from app.detectors.text.extract import extract
from app.evidence import document_native_locations, document_scope
from app.evidence.document_packets import _source_anchors
from app.storage.files import validate_kind


@pytest.mark.parametrize("extension", [".htm", ".xhtml"])
def test_static_article_adds_exact_source_ranges_without_changing_writing_input(tmp_path, extension):
    raw = ('<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Navigation title</title></head><body>\r\n'
           '<nav><article><p>Other teaser.</p></article></nav><div role="main">\r\n'
           '<header><p>Article heading boilerplate.</p></header><article><p>Read &amp; <em>review</em> the article.</p>\r\n'
           '<p style="display: none !important">Hidden text.</p><p hidden="hidden">Placeholder text.</p>\r\n'
           '<blockquote><p>Earlier quote should stay outside.</p></blockquote>'
           '<p>The exact passage contains a &#233; and another line.</p></article></div>\r\n'
           '<footer><p>Unrelated facts.</p></footer></body></html>')
    source = tmp_path / f"article{extension.upper()}"
    source.write_bytes(raw.encode())
    assert validate_kind("document", raw.encode(), source.name)[1] == extension
    doc = extract(source, extension, use_static_html=False)
    old = extract(source, extension, include_html_article=False, use_static_html=False)
    assert doc.text == old.text and doc.paragraphs == old.paragraphs
    assert "Navigation title" in doc.text and "Hidden text" in doc.text
    assert doc.meta[html_article.KEY]["element"] == "article"
    a = doc.text.index("Read & review the article.")
    b = a + len("Read & review the article.")
    evidence = {"source": {"method": "html"}, "coverage": {html_article.KEY: doc.meta[html_article.KEY]},
                "packets": [{"id": "T01", "anchors": _source_anchors(doc.source_map, a, b, len(doc.text))}]}
    key = document_native_locations.KEY
    evidence["coverage"][key] = document_native_locations.build(evidence)
    (location,) = evidence["coverage"][key]["packets"]["T01"]
    assert location["coordinate"] == "decoded_html"
    assert "".join(unescape(raw[start:end]) for start, end in location["ranges"]) == doc.text[a:b]
    document_native_locations.validate(evidence)
    tampered = deepcopy(evidence)
    tampered["coverage"][html_article.KEY]["scope"][0] = raw.index("The exact passage")
    with pytest.raises(ValueError, match="outside the saved article"):
        document_native_locations.validate(tampered)
    assert html_article.KEY not in old.meta
    assert all("source_ranges" not in block for block in old.source_map)
    assert [doc.text[block["start"]:block["end"]] for block in doc.source_map if "source_ranges" in block] == [
        "Read & review the article.", "The exact passage contains a é and another line.",
    ]


@pytest.mark.parametrize("raw", [
    "<main><article><p>One article.</p></article><article><p>Another article.</p></article></main>",
    "<div><main/><p>Neither scoped nor missing.</p></div>",
    "<nav><p>Repeated wording.</p></nav><main><p>Repeated wording.</p></main>",
    "<main><p>Paragraph <script>ignored()</script> words.</p></main>",
])
def test_ambiguous_or_inexact_html_keeps_extracted_positions(tmp_path, raw):
    source = tmp_path / "articles.html"
    source.write_text(raw)
    new = extract(source, ".html", use_static_html=False)
    old = extract(source, ".html", include_html_article=False, use_static_html=False)
    assert new.text == old.text and new.source_map == old.source_map
    assert html_article.KEY not in new.meta


@pytest.mark.parametrize("extension", [".html", ".xhtml", ".eml"])
def test_static_projection_excludes_omissions_without_breaking_words_or_coordinates(tmp_path, extension):
    paragraph = ('<p>Careful co<span hidden="false">UNSEEN_INLINE</span>operation &amp; '
                 'word <span hidden>UNSEEN_GAP</span> boundaries stay intact in the report.</p>')
    blocks = ('<nav>UNSEEN_NAV</nav><div role="navigation">UNSEEN_ROLE</div>'
              '<p style="color: red; DISPLAY: none !important">UNSEEN_CSS</p>'
              '<template><p>UNSEEN_TEMPLATE</p></template><p>Next visible paragraph records enough detail.</p>')
    raw = '<html><head><title>UNSEEN_TITLE</title></head><body>'
    raw += paragraph + blocks if extension == ".eml" else "<main>" + paragraph + blocks + "</main>"
    raw += '<blockquote><p>Visible quoted reply remains in the text.</p></blockquote></body></html>'
    source = tmp_path / ("writing" + extension)
    if extension == ".eml":
        message = EmailMessage()
        message.set_content(raw, subtype="html", cte="base64")
        source.write_bytes(message.as_bytes())
    else:
        source.write_bytes(raw.encode("utf-16"))
    doc = extract(source, extension)
    assert "UNSEEN_" not in doc.text
    assert "UNSEEN_INLINE" in extract(source, extension, use_static_html=False).text
    assert "cooperation & word boundaries" in doc.text
    assert "Visible quoted reply" in doc.text
    expected = document_scope.XHTML_STATIC_POLICY if extension == ".xhtml" else document_scope.HTML_STATIC_POLICY
    assert doc.meta[document_scope.HTML_STATIC_KEY] == expected
    first = doc.source_map[0]
    ranges = first["source_ranges"]
    excerpt = "".join(unescape(raw[lo:hi]) for _, _, lo, hi in ranges)
    assert excerpt == doc.text[first["start"]:first["end"]]
    assert all("UNSEEN_" not in raw[lo:hi] for _, _, lo, hi in ranges)
    if extension == ".eml":
        assert first["mime_part"] == "1"
        assert doc.source_map[-1]["quote_role"] == "quoted"


def test_static_projection_restores_optional_paragraph_after_hidden_paragraph(tmp_path):
    raw = ('<head><title>UNSEEN_TITLE</title><body><main><p hidden>UNSEEN_PARAGRAPH'
           '<p>Visible paragraph survives.<input hidden> Surrounding prose remains.'
           '<p style="display:none;display:block" aria-hidden="true">Visible competing style survives.</main>')
    source = tmp_path / "optional.html"
    source.write_text(raw)
    assert extract(source, ".html").text == (
        "Visible paragraph survives. Surrounding prose remains.\n\nVisible competing style survives.")


@pytest.mark.parametrize("raw", [
    '<!doctype html><html><head><title>UNSEEN_TITLE</title><main><p>{visible}</p></main></html>',
    '<table><tbody hidden><tr><td>UNSEEN_ROW<tbody><tr><td>{visible}</table>',
    '<table><colgroup hidden><col><tbody><tr><td>{visible}</table>',
    '<div hidden><textarea>UNSEEN </div> token. UNSEEN_PROOF.</textarea></div><p>{visible}</p>',
])
def test_static_projection_restores_writing_after_an_implicitly_closed_omission(tmp_path, raw):
    visible = "This visible writing must survive the ordinary optional HTML endings used in saved articles and simple tables."
    source = tmp_path / "optional.html"
    data = raw.format(visible=visible).encode()
    source.write_bytes(data)
    validate_upload(BytesIO(data), source.name, "document")
    assert extract(source, ".html").text == visible


@pytest.mark.parametrize("extension", [".html", ".eml"])
def test_hidden_only_html_rejects_before_checkout(extension):
    html = ('<p hidden>This long and ordinary saved text would otherwise be counted as writing if it were visible.</p>'
            '<nav>Another paragraph contains enough words to incorrectly accept an empty page for review.</nav>')
    if extension == ".eml":
        message = EmailMessage()
        message.set_content(html, subtype="html")
        data = message.as_bytes()
    else:
        data = html.encode()
    with pytest.raises(PreflightError, match="no readable"):
        validate_upload(BytesIO(data), "only-hidden" + extension, "document")
