"""Read simple saved Word text boxes at their main-body drawing anchors."""

from __future__ import annotations

from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from app.detectors.text import word_saved_fields
from app.evidence.document_archive_word import ordinary_body_paragraph

WPS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
TRANSITIONAL = "http://schemas.openxmlformats.org/drawingml/2006/"
STRICT = "http://purl.oclc.org/ooxml/drawingml/"
MAX_PARAGRAPHS = 4096
_HIDDEN = "./w:rPr/w:vanish | ./w:rPr/w:specVanish | ./w:rPr/w:webHidden"


def _visible_run(element, paragraph):
    return (not element.xpath(_HIDDEN)
            and word_saved_fields.style_allows_visible_runs(Run(element, paragraph).style))


def _direct_drawing(item):
    if item.tag == qn("w:drawing"):
        return item
    if item.tag != MC + "AlternateContent":
        return None
    # Word commonly stores the same box in a modern choice and a VML fallback.
    # Read one unambiguous choice using its namespace, independent of the prefix.
    choices = item.findall(MC + "Choice")
    if len(choices) == 1 and len(choices[0]) == 1:
        choice = choices[0]
        prefix = choice.get("Requires", "").strip()
        if prefix and choice.nsmap.get(prefix) == WPS and choice[0].tag == qn("w:drawing"):
            return choice[0]
    return None


def _contents(drawing):
    if len(drawing) != 1:
        return None
    frame = drawing[0]
    for base in (TRANSITIONAL, STRICT):
        wp, a = "{" + base + "wordprocessingDrawing}", "{" + base + "main}"
        if frame.tag not in {wp + "inline", wp + "anchor"}:
            continue
        properties = frame.findall(wp + "docPr")
        if len(properties) != 1 or properties[0].get("hidden") not in {None, "0", "false"}:
            return None
        extents = frame.findall(wp + "extent")
        if len(extents) != 1 or not all(size.isascii() and size.isdecimal() and size.lstrip("0")
                                        for size in (extents[0].get("cx", ""), extents[0].get("cy", ""))):
            return None
        graphics = frame.findall(f"{a}graphic/{a}graphicData")
        if len(graphics) != 1 or graphics[0].get("uri") != WPS or len(graphics[0]) != 1:
            return None
        shape = graphics[0][0]
        if shape.tag != f"{{{WPS}}}wsp" or shape.find(f"{{{WPS}}}linkedTxbx") is not None:
            return None
        boxes = shape.findall(f"{{{WPS}}}txbx")
        if len(boxes) == 1 and boxes[0].get("id", "0") == "0":
            content = boxes[0].findall(qn("w:txbxContent"))
            if len(content) == 1:
                return content[0]
        return None
    return None


def paragraphs(document, paragraph, body_index, limit):
    if not word_saved_fields.paragraph_allows_visible_runs(paragraph):
        return
    shape_index = 0
    for run in paragraph._p.findall(qn("w:r")):
        visible = _visible_run(run, paragraph)
        for item in run:
            if item.tag not in {qn("w:drawing"), MC + "AlternateContent"}:
                continue
            index_here = shape_index
            shape_index += 1
            if not visible or (drawing := _direct_drawing(item)) is None or (content := _contents(drawing)) is None:
                continue
            # Tables and other story-level wrappers change reading order and are outside this reader.
            if any(child.tag != qn("w:p") for child in content):
                continue
            for index, child in enumerate(content):
                saved = Paragraph(child, document)
                if (not ordinary_body_paragraph(saved) or not word_saved_fields.paragraph_allows_visible_runs(saved)
                        or any(not _visible_run(r, saved) for r in child.findall(qn("w:r")))):
                    continue
                if (text := saved.text).strip():
                    yield text, {"kind": "markup", "part_kind": "word_text_box", "reference_body_index": body_index,
                                 "part_index": index_here, "part_paragraph_index": index}
                    limit -= 1
                    if limit == 0:
                        return
