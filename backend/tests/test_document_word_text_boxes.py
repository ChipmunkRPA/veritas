import pytest
from docx import Document
from lxml import etree
from test_strict_word import _strict_package

from app.billing.preflight import PreflightError, _document
from app.detectors.text.extract import extract
from app.detectors.text.word_text_boxes import MC, TRANSITIONAL, WPS
from app.evidence import document_native_locations as locations
from app.evidence.document_packets import _source_anchors

BOX_TEXT = ("The museum retained a  résumé and a signed inspection note in its archive. "
            "The conservator checked each item alongside the photographs and recorded the condition before packing.")


def _box(path, *, strict=False, hidden=False, linked=False, direct=False):
    word = Document()
    run = word.add_paragraph().add_run()
    word_namespace = run._r.nsmap["w"]
    choice = '<mc:AlternateContent><mc:Choice Requires="wps">' if not direct else ""
    fallback = ('</mc:Choice><mc:Fallback><w:pict><w:txbxContent><w:p><w:r><w:t>'
                'Fallback must stay outside the writing check.</w:t></w:r></w:p></w:txbxContent>'
                '</w:pict></mc:Fallback></mc:AlternateContent>') if choice else ""
    markup = f'''<fixture xmlns:w="{word_namespace}" xmlns:wp="{TRANSITIONAL}wordprocessingDrawing"
        xmlns:a="{TRANSITIONAL}main" xmlns:wps="{WPS}" xmlns:mc="{MC[1:-1]}">
      {choice}<w:drawing><wp:inline>
        <wp:extent cx="1500" cy="900"/><wp:docPr id="1" name="Box" hidden="{int(hidden)}"/>
        <a:graphic><a:graphicData uri="{WPS}"><wps:wsp>
          <wps:txbx id="{int(linked)}"><w:txbxContent>
            <w:p><w:r><w:t>{BOX_TEXT}</w:t></w:r></w:p>
            <w:p><w:r><w:rPr><w:vanish/></w:rPr><w:t>Hidden boxed text.</w:t></w:r></w:p>
          </w:txbxContent></wps:txbx>
        </wps:wsp></a:graphicData></a:graphic>
      </wp:inline></w:drawing>{fallback}
    </fixture>'''
    run._r.append(etree.fromstring(markup.encode())[0])
    word.sections[0].header.paragraphs[0].text = "Header outside the selected document text."
    word.save(path)
    if strict:
        path.write_bytes(_strict_package(path.read_bytes()))
    return path


@pytest.mark.parametrize("strict,direct", [(False, False), (True, False), (False, True)])
def test_box_only_checkout_and_original_unicode_range(tmp_path, strict, direct):
    path = _box(tmp_path / "box.docx", strict=strict, direct=direct)
    _document(path, ".docx")
    doc = extract(path, ".docx")
    assert doc.text == BOX_TEXT.replace("retained a  ", "retained a ")
    assert doc.meta[locations.WORD_TEXT_BOX_KEY] == {"schema": locations.WORD_TEXT_BOX, "boxes": 1, "paragraphs": 1}
    start = doc.text.index("résumé")
    anchor, = _source_anchors(doc.source_map, start, start + len("résumé"), len(doc.text))
    assert locations.package_anchor_label(anchor) == "Word text box at body block 1, drawing 1, paragraph 1"
    coverage = {locations.WORD_TEXT_BOX_KEY: doc.meta[locations.WORD_TEXT_BOX_KEY]}
    evidence = {"source": {"method": "python-docx"}, "coverage": coverage,
                "packets": [{"id": "T01", "anchors": [anchor]}]}
    coverage[locations.KEY] = locations.build(evidence)
    locations.validate(evidence)
    row, = coverage[locations.KEY]["packets"]["T01"]
    assert row["coordinate"] == "word_text_box_paragraph"
    (lo, hi), = row["ranges"]
    assert BOX_TEXT[lo:hi] == "résumé"
    legacy = extract(path, ".docx", include_word_text_boxes=False)
    assert not legacy.text and locations.WORD_TEXT_BOX_KEY not in legacy.meta


@pytest.mark.parametrize("hidden,linked", [(True, False), (False, True)])
def test_unsupported_boxes_do_not_satisfy_checkout(tmp_path, hidden, linked):
    path = _box(tmp_path / "omitted.docx", hidden=hidden, linked=linked)
    assert not extract(path, ".docx").text
    with pytest.raises(PreflightError, match="no readable body text"):
        _document(path, ".docx")
