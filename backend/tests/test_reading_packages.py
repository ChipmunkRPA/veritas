from __future__ import annotations

from html import escape
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest

from app.billing.preflight import PreflightError, _document
from app.detectors.text.extract import extract
from app.detectors.text.reading_packages import (
    CONTAINER,
    CONTENT_TYPES,
    ENCRYPTION,
    PACKAGE,
    PRESENTATION_NAMESPACES,
    PRESENTATION_TYPE,
    RELATIONSHIPS,
    XHTML,
)
from app.evidence import document_native_locations
from app.evidence.document_packets import _source_anchors
from app.storage.files import EPUB_MIME, PPTX_MIME, validate_kind

PASSAGE = ("The community library stays open until eight on Thursday evenings. "
           "The volunteers use a separate entrance near the parking lot to deliver new books.")


def powerpoint(path, passages=(PASSAGE, PASSAGE), *, strict=False, external=False, macro=False):
    p, (a, r) = list(PRESENTATION_NAMESPACES.items())[int(strict)]
    p, a = p[1:-1], a[1:-1]
    main_type = PRESENTATION_TYPE if not macro else "application/vnd.ms-powerpoint.presentation.macroEnabled.main+xml"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml",
                         f'<Types xmlns="{CONTENT_TYPES[1:-1]}"><Override PartName="/ppt/presentation.xml" '
                         f'ContentType="{main_type}"/></Types>')
        archive.writestr("_rels/.rels",
                         f'<Relationships xmlns="{RELATIONSHIPS[1:-1]}"><Relationship Id="main" '
                         f'Type="{r}/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
        archive.writestr("ppt/presentation.xml", f'<p:presentation xmlns:p="{p}" xmlns:r="{r}"><p:sldIdLst>'
                         '<p:sldId id="256" r:id="two"/><p:sldId id="257" r:id="hidden"/><p:sldId id="258" r:id="one"/>'
                         '</p:sldIdLst></p:presentation>')
        first_target = 'TargetMode="External" Target="https://example.invalid/slide.xml"' if external else 'Target="slides/slide1.xml"'
        archive.writestr("ppt/_rels/presentation.xml.rels", f'<Relationships xmlns="{RELATIONSHIPS[1:-1]}">'
                         f'<Relationship Id="one" Type="{r}/slide" {first_target}/>'
                         f'<Relationship Id="two" Type="{r}/slide" Target="slides/slide2.xml"/>'
                         f'<Relationship Id="hidden" Type="{r}/slide" Target="slides/slide3.xml"/>'
                         '</Relationships>')
        for index, value in enumerate((*passages, "Hidden slide text"), 1):
            stanza = f'<a:p><a:r><a:t>{escape(value)}</a:t></a:r></a:p>'
            shapes = (f'<p:sp><p:txBody>{stanza}</p:txBody></p:sp>' if index != 1 else
                      f'<p:graphicFrame><a:graphic><a:graphicData><a:tbl><a:tr><a:tc><a:txBody>{stanza}'
                      '</a:txBody></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>')
            show = ' show="0"' if index == 3 else ''
            archive.writestr(f"ppt/slides/slide{index}.xml", f'<p:sld xmlns:p="{p}" xmlns:a="{a}"{show}>'
                             f'<p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>')
        archive.writestr("ppt/notesSlides/notesSlide1.xml", "<notes>Private speaker notes</notes>")


def epub(path, passages=(PASSAGE, PASSAGE), *, protected=False):
    html, opf, container = XHTML[1:-1], PACKAGE[1:-1], CONTAINER[1:-1]
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", EPUB_MIME, compress_type=ZIP_STORED)
        archive.writestr("META-INF/container.xml", f'<container xmlns="{container}"><rootfiles>'
                         '<rootfile full-path="OPS/book.opf"/></rootfiles></container>')
        archive.writestr("OPS/book.opf", f'<package xmlns="{opf}"><manifest>'
                         '<item id="one" href="chapters/first%20page.xhtml" media-type="application/xhtml+xml"/>'
                         '<item id="two" href="chapters/two.xhtml" media-type="application/xhtml+xml"/>'
                         '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
                         '</manifest><spine><itemref idref="two"/><itemref idref="nav" linear="no"/>'
                         '<itemref idref="one"/></spine></package>')
        for name, passage in zip(("first page", "two"), passages, strict=True):
            archive.writestr(f"OPS/chapters/{name}.xhtml", f'<!DOCTYPE html><html xmlns="{html}">'
                             f'<head><title>Title metadata</title></head><body><script>Skipped text</script>'
                             f'<p hidden="hidden">Hidden text</p><p>{escape(passage)}</p></body></html>')
        if protected:
            archive.writestr("META-INF/encryption.xml",
                             f'<encryption xmlns:e="{ENCRYPTION[1:-1]}">'
                             '<e:EncryptedData><e:CipherData><e:CipherReference URI="OPS/chapters/two.xhtml"/>'
                             '</e:CipherData></e:EncryptedData></encryption>')


@pytest.mark.parametrize(("extension", "make", "mime", "part_kind", "coordinate"), [
    (".pptx", powerpoint, PPTX_MIME, "slide", "pptx_block"),
    (".epub", epub, EPUB_MIME, "epub_entry", "epub_block"),
])
def test_order_scope_and_exact_positions(tmp_path, extension, make, mime, part_kind, coordinate):
    path = tmp_path / f"source{extension.upper()}"
    make(path, ("First table/chapter. " + PASSAGE, "Second   text 🦊. " + PASSAGE))
    assert validate_kind("document", path.read_bytes()[:8192], path.name, path.stat().st_size) == (mime, extension)
    _document(path, extension)
    doc = extract(path, extension)
    assert doc.text == "Second text 🦊. " + PASSAGE + "\n\nFirst table/chapter. " + PASSAGE
    assert [(row["part_kind"], row["part_index"], row["paragraph_index"]) for row in doc.source_map] == [
        (part_kind, 0, 0), (part_kind, 2, 0),
    ]
    start, end = doc.text.index("text"), doc.text.index(".") + 1
    anchors = _source_anchors(doc.source_map, start, end, len(doc.text))
    packet_set = {"source": {"method": extension[1:]}, "coverage": doc.meta,
                  "packets": [{"id": "T01", "anchors": anchors}]}
    record = document_native_locations.build(packet_set)
    assert record["packets"]["T01"] == [{"anchor_index": 0, "coordinate": coordinate, "ranges": [[9, 16]]}]


def test_strict_powerpoint_body(tmp_path):
    path = tmp_path / "strict.pptx"
    powerpoint(path, strict=True)
    assert extract(path, ".pptx").text == f"{PASSAGE}\n\n{PASSAGE}"


@pytest.mark.parametrize(("extension", "make", "options", "message"), [
    (".pptx", powerpoint, {"external": True}, "internal slide relationship"),
    (".pptx", powerpoint, {"macro": True}, "macro-free"),
    (".pptx", powerpoint, {"passages": ("", "")}, "at least 15 readable words"),
    (".epub", epub, {"protected": True}, "protected reading text"),
    (".epub", epub, {"passages": ("", "")}, "at least 15 readable words"),
])
def test_package_preflight_rejects_unreadable_or_unsupported_input(tmp_path, extension, make, options, message):
    path = tmp_path / f"bad{extension}"
    make(path, **options)
    with pytest.raises(PreflightError, match=message):
        _document(path, extension)
