from __future__ import annotations

from copy import deepcopy
from html import unescape

import pytest

from app.detectors.text import html_article
from app.detectors.text.extract import extract
from app.evidence import document_native_locations
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
    doc = extract(source, extension)
    old = extract(source, extension, include_html_article=False)
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
    new, old = extract(source, ".html"), extract(source, ".html", include_html_article=False)
    assert new.text == old.text and new.source_map == old.source_map
    assert html_article.KEY not in new.meta
