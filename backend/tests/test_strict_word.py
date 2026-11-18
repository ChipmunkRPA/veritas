from io import BytesIO
from zipfile import ZipFile

import pytest
from docx import Document

from app.billing.preflight import _document
from app.detectors.text.extract import extract
from app.detectors.text.strict_word import NAMESPACES
from app.evidence import document_revision


def _strict_package(data, *, relationship_target=None, macro_enabled=False):
    buffer = BytesIO()
    with ZipFile(BytesIO(data)) as source, ZipFile(buffer, "w") as output:
        for name in source.namelist():
            part = source.read(name)
            if name.endswith((".xml", ".rels")):
                for strict, usual in NAMESPACES.items():
                    part = part.replace(usual.encode(), strict.encode())
            if name == "_rels/.rels" and relationship_target is not None:
                part = part.replace(b'Target="word/document.xml"', relationship_target)
            if name == "[Content_Types].xml" and macro_enabled:
                part = part.replace(b"openxmlformats-officedocument.wordprocessingml.document.main+xml",
                                    b"ms-word.document.macroEnabled.main+xml")
            output.writestr(name, part)
    return buffer.getvalue()


def _word(paragraph):
    word = Document()
    word.add_paragraph(paragraph)
    word.add_table(rows=1, cols=1).cell(0, 0).text = "Body table entry."
    word.sections[0].header.paragraphs[0].text = "Header outside the body."
    out = BytesIO()
    word.save(out)
    return out.getvalue()


def test_strict_word_preserves_native_locations_in_analysis_and_drafts(tmp_path):
    old_bytes = _word("The case arrived on Monday.")
    new_bytes = _word("The case arrived on Tuesday.")
    regular = tmp_path / "default.docx"
    regular.write_bytes(old_bytes)
    strict = tmp_path / "strict.docx"
    strict.write_bytes(_strict_package(old_bytes))
    _document(strict, ".docx")
    expected = extract(regular, ".docx")
    actual = extract(strict, ".docx")
    assert actual.text == "The case arrived on Monday.\n\nBody table entry."
    assert actual.source_map == expected.source_map
    assert actual.meta["percentage_body_paragraphs"] == expected.meta["percentage_body_paragraphs"] == [0]
    assert actual.meta["original_sha256"] != expected.meta["original_sha256"]
    assert [row["body_path"] for row in actual.source_map] == [[0], [1, "row", 0, "cell", 0, 1]]

    result, old_text, new_text = document_revision.compare_with_text(strict.read_bytes(), _strict_package(new_bytes), ".docx")
    assert result["earlier"]["sha256"] == actual.meta["original_sha256"]
    assert result["format"] == "docx"
    change, = result["changes"]
    for side, source, word in (("earlier", old_text, "Monday"), ("later", new_text, "Tuesday")):
        (start, end), = change[side]["changed_tokens"]
        assert source[start:end] == word


def test_earlier_standard_word_still_reads_only_its_declared_body():
    out = BytesIO()
    with ZipFile(BytesIO(_word("Recorded wording."))) as source, ZipFile(out, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"<incomplete" if name == "_rels/.rels" else source.read(name))

    (_, _, paragraph), = document_revision.docx_body_paragraphs(out.getvalue())
    assert paragraph.text == "Recorded wording."


@pytest.mark.parametrize(("target", "macro_enabled", "message"), [
    (b'TargetMode="External" Target="https://example.invalid/document.xml"', False, "invalid package relationship"),
    (b'Target="../document.xml"', False, "unsupported XML part path"),
    (None, True, "saved in .docx format"),
])
def test_strict_word_refuses_external_invalid_and_macro_packages(tmp_path, target, macro_enabled, message):
    path = tmp_path / "invalid.docx"
    path.write_bytes(_strict_package(_word("Paragraph."), relationship_target=target, macro_enabled=macro_enabled))
    with pytest.raises(ValueError, match=message):
        _document(path, ".docx")
    with pytest.raises(ValueError, match=message):
        list(document_revision.docx_body_paragraphs(path.read_bytes()))
