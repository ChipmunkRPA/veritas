# SPDX-License-Identifier: Apache-2.0
"""Extract exact text positions from a bounded, static HTML article."""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser

from app.evidence.document_scope import HTML_ARTICLE_KEY as KEY
from app.evidence.document_scope import HTML_ARTICLE_MAX_SOURCE as MAX_HTML
from app.evidence.document_scope import HTML_ARTICLE_POLICY as POLICY

__all__ = ["KEY", "LIMITATION", "POLICY", "READER", "parts", "read"]

READER = "html_article_reader"
LIMITATION = ("Writing checks read paragraphs, headings, lists and descriptions in one marked HTML article or main area. "
              "Page controls, quotations, figures, tables and explicitly hidden elements were outside this check. "
              "Scripts and linked content were not loaded. Positions count characters in the decoded HTML source.")
MAX_TEXT = 80_000
VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})
OMIT = frozenset({"aside", "audio", "blockquote", "button", "canvas", "code", "details", "dialog", "figure", "footer", "form",
                  "head", "header", "iframe", "math", "nav", "noscript", "pre", "script", "style", "svg", "table", "template", "video"})
HIDDEN_STYLE = re.compile(r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*(?:hidden|collapse))\s*(?:!important\s*)?(?:;|$)", re.I)
BLOCKS = frozenset({"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd"})
TABLE_CELLS = frozenset({"td", "th"})


def _hidden(tag, attrs, *, include_table_cells=False):
    values = dict(attrs)
    return ((tag in OMIT and not (include_table_cells and tag == "table")) or "hidden" in values or "inert" in values
            or (values.get("role") or "").strip().casefold() in {"navigation", "contentinfo", "banner", "complementary"}
            or (values.get("aria-hidden") or "").strip().casefold() == "true"
            or bool(HIDDEN_STYLE.search(values.get("style") or "")))


class _StaticHTML(HTMLParser):
    def __init__(self, raw):
        super().__init__(convert_charrefs=False)
        self.raw = raw
        self.lines = [0, *(match.end() for match in re.finditer("\n", raw))]
        self.stack = []

    def position(self):
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def hidden(self):
        return bool(self.stack and self.stack[-1][1])

    def end_index(self, tag):
        return next((index for index in range(len(self.stack) - 1, -1, -1) if self.stack[index][0] == tag), None)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)


class _Regions(_StaticHTML):
    """Find explicit body boundaries without running the document or its resources."""

    def __init__(self, raw):
        super().__init__(raw)
        self.regions = []

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        hidden = self.hidden() or _hidden(tag, attrs)
        region = None
        role = (dict(attrs).get("role") or "").strip().casefold()
        scope = "main" if role == "main" else tag
        if scope in {"article", "main"} and not hidden:
            region = {"tag": scope, "start": self.position() + len(self.get_starttag_text()), "end": None}
            self.regions.append(region)
        if len(self.stack) >= 256 or len(self.regions) > 32:
            raise ValueError("HTML structure exceeds the supported article limit")
        self.stack.append((tag, hidden, region))

    def handle_endtag(self, tag):
        if (position := self.end_index(tag)) is not None:
            if region := self.stack[position][2]:
                region["end"] = self.position()
            del self.stack[position:]

    def selected(self):
        mains = [row for row in self.regions if row["tag"] == "main"]
        articles = [row for row in self.regions if row["tag"] == "article"]
        if len(mains) > 1 or any(row["end"] is None for row in self.regions):
            return None
        if mains:
            main = mains[0]
            articles = [row for row in articles if main["start"] <= row["start"] < row["end"] <= main["end"]]
        if len(articles) == 1:
            return articles[0]
        return mains[0] if not articles and mains else None


class _Paragraphs(_StaticHTML):
    def __init__(self, raw, base, *, include_table_cells):
        super().__init__(raw)
        self.base = base
        self.include_table_cells = include_table_cells
        self.blocks = BLOCKS | TABLE_CELLS if include_table_cells else BLOCKS
        self.parts = []
        self.words = []
        self.origins = []
        self.chars = 0
        self.unmapped = False

    def flush(self):
        if self.origins:
            self.parts.append(("".join(self.words), {"kind": "markup", "html_origins": self.origins}))
        self.words, self.origins = [], []

    def handle_starttag(self, tag, attrs):
        hidden = self.hidden() or _hidden(tag, attrs, include_table_cells=self.include_table_cells)
        if not hidden and (tag in self.blocks or tag in {"br", "hr"}):
            self.flush()
        if tag not in VOID:
            self.stack.append((tag, hidden))

    def handle_endtag(self, tag):
        if (position := self.end_index(tag)) is not None:
            if not self.hidden() and any(name in self.blocks for name, _ in self.stack[position:]):
                self.flush()
            del self.stack[position:]

    def reading(self):
        return not self.hidden() and any(tag in self.blocks for tag, _ in self.stack)

    def _append(self, value, origins):
        self.chars += len(value)
        if self.chars > MAX_TEXT:
            raise ValueError("The selected article exceeds the supported text limit")
        self.words.append(value)
        self.origins.extend(origins)

    def handle_data(self, data):
        if self.reading():
            start = self.base + self.position()
            self._append(data, ((start + i, start + i + 1) for i in range(len(data))))
        elif not self.hidden() and data.strip():
            self.unmapped = True

    def _entity(self, token):
        if not self.reading():
            return
        start, value = self.position(), unescape(token)
        if self.raw[start:start + len(token)] != token or len(value) != 1 or value == "\ufffd":
            raise ValueError("An HTML character could not be mapped to its exact source")
        self._append(value, [(self.base + start, self.base + start + len(token))])

    def handle_entityref(self, name):
        self._entity(f"&{name};")

    def handle_charref(self, name):
        self._entity(f"&#{name};")


def read(raw, *, include_table_cells=False):
    """Return literal text ranges and any unmapped text inside the selected area."""
    if len(raw) > MAX_HTML or "\ufffd" in raw or "\x00" in raw:
        return None
    try:
        regions = _Regions(raw)
        regions.feed(raw)
        regions.close()
        selected = regions.selected()
        if selected is None:
            return None
        start, end = selected["start"], selected["end"]
        parser = _Paragraphs(raw[start:end], start, include_table_cells=include_table_cells)
        parser.feed(raw[start:end])
        parser.close()
        parser.flush()
    except (ValueError, AssertionError):
        return None
    if parser.unmapped and not any(text.strip() for text, _ in parser.parts):
        return None
    return parser.parts, {"policy": POLICY, "element": selected["tag"], "scope": [start, end]}, parser.unmapped


def parts(raw, *, include_table_cells=False):
    result = read(raw, include_table_cells=include_table_cells)
    return result[:2] if result is not None else None
