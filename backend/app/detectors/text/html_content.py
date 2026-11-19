# SPDX-License-Identifier: Apache-2.0
"""Exclude a bounded set of non-content HTML while preserving source offsets."""

from __future__ import annotations

import re
from array import array
from html.parser import HTMLParser

VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})
OMIT = frozenset({"head", "title", "nav", "script", "style", "template", "noscript"})
CLOSE_PARAGRAPH = frozenset({"address", "article", "aside", "blockquote", "details", "dialog", "div", "dl", "fieldset", "figure",
                             "figcaption", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "main", "menu", "nav",
                             "ol", "p", "pre", "search", "section", "table", "ul"})
HEAD_BODY = CLOSE_PARAGRAPH | {"body"}
DISPLAY = re.compile(r"^\s*display\s*:\s*(.*?)\s*$", re.IGNORECASE | re.ASCII)
NONE = re.compile(r"none(?:\s*!\s*important)?", re.IGNORECASE | re.ASCII)
COMPLEX_CSS = frozenset("'\"(){}\\/")
BREAKER = frozenset({"button", "caption", "html", "table", "td", "th", "template"})
TABLE_SECTIONS = frozenset({"thead", "tbody", "tfoot"})
OPTIONAL_ENDS = {
    "li": ({"li"}, {"ol", "ul", "menu"}),
    "dt": ({"dt", "dd"}, {"dl"}),
    "dd": ({"dt", "dd"}, {"dl"}),
    "tr": ({"tr"}, {"table"}),
    "td": ({"td", "th"}, {"tr", "table"}),
    "th": ({"td", "th"}, {"tr", "table"}),
    **dict.fromkeys(TABLE_SECTIONS, (TABLE_SECTIONS, {"table"})),
}


def _outside(tag, attrs):
    values = dict(reversed(attrs))  # The first occurrence is the saved HTML attribute.
    if tag in OMIT or "hidden" in values or (values.get("role") or "").strip().casefold() == "navigation":
        return True
    style = values.get("style") or ""
    if not style or any(ch in COMPLEX_CSS for ch in style):
        return False
    display = [match[1] for declaration in style.split(";") if (match := DISPLAY.fullmatch(declaration))]
    return len(display) == 1 and NONE.fullmatch(display[0]) is not None


class _Omissions(HTMLParser):
    def __init__(self, raw, *, xhtml):
        super().__init__(convert_charrefs=False)
        self.raw = raw
        self.lines = array("I", [0])
        self.lines.extend(match.end() for match in re.finditer("\n", raw))
        self.xhtml = xhtml
        if not xhtml:
            # Textarea and title treat apparent closing tags as text until
            # their own end tag. The same boundary rule applies to raw text.
            self.CDATA_CONTENT_ELEMENTS = (*HTMLParser.CDATA_CONTENT_ELEMENTS, "textarea", "title", "xmp",
                                           "iframe", "noembed", "noframes", "noscript")
        self.stack = []
        self.open = None
        self.ranges = []

    def source_position(self):
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def finish(self, depth, end):
        if self.open is not None and self.open[0] >= depth:
            self.ranges.append((self.open[1], end))
            self.open = None
        del self.stack[depth:]

    def _end_in_scope(self, tags, barriers, pos):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index] in tags:
                self.finish(index, pos)
                break
            if self.stack[index] in barriers:
                break

    def _start(self, tag, attrs, self_closes):
        pos = self.source_position()
        if not self.xhtml:
            if tag in HEAD_BODY and self.stack and self.stack[-1] == "head":
                self.finish(len(self.stack) - 1, pos)
            if tag not in {"col", "template"} and self.stack and self.stack[-1] == "colgroup":
                self.finish(len(self.stack) - 1, pos)
            if tag in CLOSE_PARAGRAPH or tag in {"li", "dt", "dd"}:
                self._end_in_scope({"p"}, BREAKER, pos)
            if tag in OPTIONAL_ENDS:
                tags, barriers = OPTIONAL_ENDS[tag]
                self._end_in_scope(tags, barriers | {"template"}, pos)
        outside = self.open is None and _outside(tag, attrs)
        if tag in VOID or self_closes:
            if outside:
                self.ranges.append((pos, pos + len(self.get_starttag_text())))
            return
        if len(self.stack) >= 256:
            raise ValueError("HTML nesting exceeds the static text reader limit")
        if outside:
            self.open = (len(self.stack), pos)
        self.stack.append(tag)

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs, False)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs, self.xhtml)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index] == tag:
                pos = self.source_position()
                if self.open is not None and self.open[0] == index:
                    pos = self.raw.index(">", pos) + 1
                self.finish(index, pos)
                break


def mask(raw: str, *, xhtml: bool = False) -> str:
    """Replace omitted ranges with ignored markup at the same decoded offsets."""
    reader = _Omissions(raw, xhtml=xhtml)
    reader.feed(raw)
    reader.close()
    reader.finish(0, len(raw))
    if not reader.ranges:
        return raw
    parts = []
    cursor = 0
    for start, end in reader.ranges:
        # Ignored markup inserts no space into a visible word, and the exact
        # replacement length preserves all following absolute source offsets.
        length = end - start
        blank = "<!--" + " " * (length - 7) + "-->" if length >= 7 else "<!" + " " * (length - 3) + ">"
        parts.extend((raw[cursor:start], blank))
        cursor = end
    parts.append(raw[cursor:])
    return "".join(parts)
