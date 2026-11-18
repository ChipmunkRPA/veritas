"""Read saved ODP slide paragraphs without opening or rendering the presentation."""

import re

from app.detectors.text.opendocument import BODY_CONTAINERS, OFFICE, PARAGRAPHS, STYLE, TEXT, _visible_text
from app.storage.files import ODP_MIME

DRAW = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}"
PRESENTATION = "{urn:oasis:names:tc:opendocument:xmlns:presentation:1.0}"
TEXT_SHAPES = {DRAW + name for name in (
    "a", "caption", "circle", "connector", "custom-shape", "ellipse", "frame", "g", "line", "path",
    "polygon", "polyline", "rect", "regular-polygon", "text-box",
)}


def _drawing_pages(roots):
    styles = {}
    for root in roots:
        for group in ("styles", "automatic-styles"):
            for style in root.iterfind(f"{OFFICE}{group}/{STYLE}style"):
                if style.get(STYLE + "family") == "drawing-page" and (name := style.get(STYLE + "name")):
                    styles[name] = style
    return styles


def _visible_slide(slide, styles):
    name = slide.get(DRAW + "style-name")
    seen = set()
    while name in styles:
        if name in seen or len(seen) == 64:
            raise ValueError("The ODP presentation has unsupported nested slide styles")
        seen.add(name)
        style = styles[name]
        properties = style.find(STYLE + "drawing-page-properties")
        if properties is not None and (visibility := properties.get(PRESENTATION + "visibility")) is not None:
            return visibility == "visible"
        name = style.get(STYLE + "parent-style-name")
    return True


def _layers(roots, slide):
    layer_sets = []
    master_name = slide.get(DRAW + "master-page-name")
    for root in roots:
        master_styles = root.find(OFFICE + "master-styles")
        if master_styles is not None:
            layer_sets.extend(master_styles.findall(DRAW + "layer-set"))
            for master in master_styles.findall(STYLE + "master-page"):
                if master.get(STYLE + "name") == master_name:
                    layer_sets.extend(master.findall(DRAW + "layer-set"))
    layer_sets.extend(slide.findall(DRAW + "layer-set"))
    return {layer.get(DRAW + "name"): layer.get(DRAW + "display")
            for layers in layer_sets for layer in layers.findall(DRAW + "layer") if layer.get(DRAW + "name")}


def _slide_paragraphs(container, layers):
    for child in container:
        if child.tag in PARAGRAPHS:
            yield child
        elif child.tag in TEXT_SHAPES or child.tag in BODY_CONTAINERS:
            if (layers.get(child.get(DRAW + "layer")) in {"none", "printer"}
                    or child.get(PRESENTATION + "placeholder") in {"true", "1"}
                    or child.tag == TEXT + "section" and child.get(TEXT + "display") in {"none", "condition"}):
                continue
            yield from _slide_paragraphs(child, layers)


def _paragraph_text(paragraph, limit):
    def spacing(element):
        if element.tag == TEXT + "tab":
            return "\t"
        if element.tag == TEXT + "line-break":
            return "\n"
        count = element.get(TEXT + "c", "1").strip()
        if not re.fullmatch(r"\+?[0-9]{1,7}", count) or not 1 <= int(count) <= limit:
            raise ValueError("The ODP slide exceeds the writing limit for whitespace")
        return " " * int(count)

    fragments = []
    for fragment in _visible_text(paragraph, spacing):
        limit -= len(fragment)
        if limit < 0:
            raise ValueError("The ODP slide exceeds the writing limit for a text block")
        fragments.append(fragment)
    return "".join(fragments)


def slide_text(package, *, max_slides, max_chars):
    with package.archive.open("mimetype") as mime:
        if mime.read(len(ODP_MIME) + 1) != ODP_MIME.encode("ascii"):
            raise ValueError("This archive is not an OpenDocument Presentation file")
    content = package.xml("content.xml")
    body = content.find(f"{OFFICE}body/{OFFICE}presentation")
    if content.tag != OFFICE + "document-content" or body is None:
        raise ValueError("The ODP archive has no supported presentation body")
    slides = body.findall(DRAW + "page")
    if not 0 < len(slides) <= max_slides:
        raise ValueError(f"Upload an ODP presentation with between 1 and {max_slides} slides")

    roots = [content]
    if "styles.xml" in package.archive.NameToInfo:
        styles = package.xml("styles.xml")
        if styles.tag != OFFICE + "document-styles":
            raise ValueError("The ODP presentation has unreadable styles")
        roots.insert(0, styles)
    page_styles = _drawing_pages(roots)
    for index, slide in enumerate(slides):
        if _visible_slide(slide, page_styles):
            for paragraph_index, paragraph in enumerate(_slide_paragraphs(slide, _layers(roots, slide))):
                yield _paragraph_text(paragraph, 4 * max_chars), index, paragraph_index
