# SPDX-License-Identifier: Apache-2.0
"""Compare an explicit internal quotation of a reference number with its saved target."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

from app.detectors.text import html_content
from app.evidence import document_reference_links_binding as binding

NUMBER = re.compile(r"[1-9][0-9]{0,2}\Z")
DECIMAL = r"[1-9][0-9]{0,2}"
BRACKETS = re.compile(rf"\[{DECIMAL}(?:\s*[,–-]\s*{DECIMAL})*\]")
LABEL = re.compile(rf"^\s*(?:\[({DECIMAL})\]|({DECIMAL})[.)])(?:\s+|$)")
FRAGMENT = re.compile(r"#[A-Za-z][A-Za-z0-9_.:-]{0,239}\Z")


def _number(text, offset=0):
    matched = LABEL.match(text)
    if matched is None:
        return None
    group = 1 if matched[1] else 2
    return int(matched[group]), (offset + matched.start(group), offset + matched.end(group))


def _local_target(href):
    return href[1:] if FRAGMENT.fullmatch(href or "") else None


class _Positions(HTMLParser):
    """Collect exact positions separately from structural HTML inspection."""

    CDATA_CONTENT_ELEMENTS = (*HTMLParser.CDATA_CONTENT_ELEMENTS, "textarea", "title", "xmp",
                              "iframe", "noembed", "noframes", "noscript")

    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.lines = [0, *(line.end() for line in re.finditer("\n", source))]
        self.link = None
        self.links = defaultdict(list)
        self.items = defaultdict(list)
        self.list_items = []
        self.count = 0

    def position(self):
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag, attrs):
        values = dict(reversed(attrs))
        if tag == "li":
            self.list_items.append({"targets": set(), "label": None})
        if self.list_items and values.get("id"):
            self.list_items[-1]["targets"].add(values["id"])
        if self.link is not None:
            self.link["simple"] = False
        elif tag == "a" and (target := _local_target(values.get("href"))) and sum(name == "href" for name, _ in attrs) == 1:
            self.link = {"target": target, "simple": True, "number": None}

    def handle_endtag(self, tag):
        if tag == "a" and self.link is not None:
            link, self.link = self.link, None
            if link["simple"] and link["number"] is not None:
                self.count += 1
                if self.count > binding.MAX_LINKS:
                    raise ValueError("HTML reference links exceed the supported count")
                self.links[link["number"], link["target"]].append(link["range"])
        if tag == "li" and self.list_items:
            item = self.list_items.pop()
            if item["label"]:
                for target in item["targets"]:
                    self.items[target].append(item["label"])

    def handle_data(self, text):
        if self.link is not None:
            if self.link["number"] is not None or not NUMBER.fullmatch(text):
                self.link["simple"] = False
            else:
                self.link["number"] = int(text)
                self.link["range"] = (self.position(), self.position() + len(text))
        if self.list_items and text.strip() and self.list_items[-1]["label"] is None:
            self.list_items[-1]["label"] = _number(text, self.position()) or False


def _reference_targets(tree):
    headings = [h for h in tree.xpath("//h2 | //h3 | //h4")
                if " ".join(h.text_content().split()).casefold() == "references"]
    if len(headings) != 1:
        return None
    heading = headings[0]
    lists = heading.getparent().xpath("./ol")
    if len(lists) != 1 or lists[0].getprevious() is not heading:
        return None
    listing = lists[0]
    entries = listing.xpath("./li")
    if (not binding.MIN_REFERENCES <= len(entries) <= binding.MAX_REFERENCES
            or listing.get("start", "1") != "1" or listing.get("reversed") is not None):
        return None
    targets = {}
    for expected, item in enumerate(entries, 1):
        label = _number(item.text_content())
        if label is None or label[0] != expected or item.get("value") is not None:
            return None
        for node in (item, *item.xpath(".//*[@id]")):
            if identifier := node.get("id"):
                targets[identifier] = expected
    return len(entries), targets


def _paragraph_links(tree):
    for paragraph in tree.xpath("//p[not(ancestor::ol)]"):
        markers = BRACKETS.finditer(paragraph.text_content())
        marker = next(markers, None)
        before = len(paragraph.text or "")
        for child in paragraph:
            while marker is not None and marker.end() <= before:
                marker = next(markers, None)
            if marker is None:
                break
            text = child.text_content() if isinstance(child.tag, str) else ""
            if (child.tag == "a" and NUMBER.fullmatch(text) and (target := _local_target(child.get("href")))
                    and marker.start() < before and before + len(text) < marker.end()):
                yield int(text), target
            before += len(text) + len(child.tail or "")


def capture(raw):
    """Return comparisons bound to visible label and target codepoints, with no network access."""
    if not 0 < len(raw) <= binding.MAX_CHARS or "\x00" in raw or "\ufffd" in raw:
        return None
    from lxml import etree, html

    parser = html.HTMLParser(no_network=True)
    try:
        original = html.fromstring(raw, parser=parser)
        if original.xpath("//base[@href]"):
            return None
        ids = Counter(original.xpath("//*[@id]/@id"))
        visible = html_content.mask(raw)
        tree = html.fromstring(visible, parser=parser)
        selected = _reference_targets(tree)
        if selected is None:
            return None
        references, targets = selected
        positions = _Positions(visible)
        positions.feed(visible)
        positions.close()
    except (ValueError, AssertionError, etree.Error):
        return None
    observations = []
    checked = 0
    for (label, target), occurrences in Counter(_paragraph_links(tree)).items():
        if occurrences != 1 or ids[target] != 1 or target not in targets:
            continue
        markers, entries = positions.links[label, target], positions.items[target]
        if len(markers) != 1 or len(entries) != 1 or entries[0][0] != targets[target]:
            continue
        (a, b), = markers
        number, (c, d) = entries[0]
        if raw[a:b] != str(label) or raw[c:d] != str(number):
            continue
        checked += 1
        if label != number:
            observations.append({"label_number": label, "target_number": number,
                                 "label_range": [a, b], "target_range": [c, d]})
    if not observations:
        return None
    observations.sort(key=lambda item: item["label_range"])
    return {"schema": binding.SCHEMA, "decoded_characters": len(raw), "references": references,
            "checked_links": checked, "mismatches": len(observations), "findings": observations[:binding.MAX_FINDINGS],
            "authorship_confidence_contribution": 0, "destinations_contacted": 0}


def inspect(path: Path, extension: str, *, source_sha256: str, allow_legacy_html: bool = True):
    if extension not in binding.EXTENSIONS:
        return None
    with path.open("rb") as source:
        data = source.read(binding.MAX_BYTES + 1)
    if len(data) > binding.MAX_BYTES:
        return None
    if hashlib.sha256(data).hexdigest() != source_sha256:
        raise ValueError("Submitted HTML differs from the reference link source")
    try:
        raw, _ = html_content.decode_saved(data, allow_legacy=allow_legacy_html)
        return capture(raw)
    except UnicodeError:
        return None
