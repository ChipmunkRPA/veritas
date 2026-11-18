from __future__ import annotations

from copy import deepcopy
from email.message import EmailMessage
from zipfile import ZipFile

import pymupdf
import pytest
from docx import Document

from app.detectors.text.extract import extract
from app.evidence import document_publication_ids as records
from app.storage.files import ODT_MIME

VALUE = "978-1-5286-4958-3"
BODY = "The library copied the identifier from the printed report. ISBN " + VALUE + " identifies the edition in the submitted note."
SECOND = "The reviewer checked the available copies and made a note about the formats at the reading room. The catalogue is dated."
THIRD = "Staff log each request when the library opens for the first shift in the morning. The log records the official reference text."
HTML = "<html><body><article>" + "".join(f"<p>{line}</p>" for line in (BODY, SECOND, THIRD)) + "</article></body></html>"


def _write(path, variant):
    if variant == "docx":
        document = Document()
        document.add_paragraph(BODY)
        document.save(path)
    elif variant == "pdf":
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 560, 780), BODY, fontsize=10)
            document.save(path)
    elif variant == "odt":
        with ZipFile(path, "w") as archive:
            archive.writestr("mimetype", ODT_MIME)
            archive.writestr("content.xml",
                             '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
                             'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" office:version="1.3">'
                             f"<office:body><office:text><text:p>{BODY}</text:p></office:text></office:body></office:document-content>")
    elif variant == "rtf":
        path.write_text(r"{\rtf1\ansi " + BODY + r"\par}")
    elif variant == "html":
        path.write_text(HTML.replace(VALUE, "978-1-<em>5286</em>-4958-3"))
    elif variant == "eml":
        message = EmailMessage()
        message["From"] = "writer@example.invalid"
        message["To"] = "editor@example.invalid"
        message["Subject"] = "A public catalogue reference"
        message.set_content(BODY + "\n\n> ISSN 0317-8472")
        path.write_bytes(message.as_bytes())
    else:
        path.write_text(BODY)


@pytest.mark.parametrize(("ext", "kind"), [
    ("txt", "text"), ("docx", "docx"), ("pdf", "pdf"), ("odt", "odt_paragraph"),
    ("rtf", "rtf_paragraph"), ("html", "decoded_html"), ("eml", "email"),
])
def test_publisher_digit_disagreement_preserves_native_location(ext, kind, tmp_path):
    file = tmp_path / f"source.{ext}"
    _write(file, ext)
    doc = extract(file, f".{ext}", allow_ocr=False)
    source = {"method": doc.meta["method"], "chars": len(doc.text)}
    evidence = records.build(source, doc.text, doc.source_map, doc.meta)
    assert evidence["checked"] == evidence["disagreements"] == 1
    assert evidence["affects_overall_confidence"] is False
    row, = evidence["locations"]
    assert row["kind"] == "ISBN-13" and row["expected"] == "2" and row["written"] == "3"
    assert row["anchor"]["kind"] == kind and row["text"] == doc.text[row["start"]:row["end"]] == VALUE
    if kind == "pdf":
        assert row["anchor"]["page"] == 1
        with pymupdf.open(file) as original:
            assert any(pymupdf.Rect(box).intersects(bounds) for box in row["location"]["boxes"]
                       for bounds in original[0].search_for(VALUE))
    elif ext == "html":
        original = file.read_text()
        assert len(row["location"]["ranges"]) == 3
        assert "".join(original[start:end] for start, end in row["location"]["ranges"]) == VALUE
    else:
        assert row["location"]["ranges"] == [[BODY.index(VALUE), BODY.index(VALUE) + len(VALUE)]]
    document = {"source": source, records.KEY: evidence}
    records.validate(document)
    records.verify_source(document, doc.text, doc.source_map, doc.meta)
    altered = deepcopy(document)
    altered[records.KEY]["locations"][0]["expected"] = "1"
    with pytest.raises(ValueError, match="computed check digit"):
        records.validate(altered)
    if ext == "html":
        # An older extraction without exact native HTML ranges must abstain.
        prior = extract(file, ".html", include_html_article=False)
        assert records.build(source, prior.text, prior.source_map, prior.meta) is None


def test_numeric_scheme_and_context_controls():
    text = (
        "10 Digit ISBN: 92-95055-02-5 13 Digit ISBN: 978-92-95055-02-5. "
        "ISSN 1050-124X. ISBN 0-16-047213-X. Unknown number 978-1-5286-4958-3. "
        "ISBN (invalid) 978-1-5286-4958-3. Incorrect ISSN 0046-2254. "
        "ISBN 978-1-5286-4958-3 (canceled). ISBN 979-0-006-57321-5."
    )
    rows = list(records.readings(text, 0, len(text)))
    assert [(kind, written, expected) for kind, _, written, expected in rows] == [
        ("ISBN-10", "5", "0"), ("ISBN-13", "5", "5"), ("ISSN", "X", "X"), ("ISBN-10", "X", "X"),
    ]
