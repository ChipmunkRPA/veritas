"""Whitespace-only paired audit views with exact original-character correspondence.

Research preparation only. A format response is not an AI-authorship fact.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.detectors.text.extract import from_string


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass(frozen=True)
class View:
    name: str
    input_text: str
    text: str
    source_positions: tuple[int | None, ...]

    def project(self, start, end):
        """Actual original non-whitespace positions, not a filled bounding interval."""
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.text):
            raise ValueError("Nonempty Unicode-codepoint interval required")
        return frozenset(p for p in self.source_positions[start:end] if p is not None)


def correspondence(source, rendered):
    """Preserve every non-whitespace codepoint AND every existing token boundary."""
    first = [(i, c) for i, c in enumerate(source) if not c.isspace()]
    second = [(i, c) for i, c in enumerate(rendered) if not c.isspace()]
    if [c for _, c in first] != [c for _, c in second]:
        raise ValueError("Non-whitespace content changed")
    before = [b - a > 1 for (a, _), (b, _) in zip(first, first[1:])]
    after = [b - a > 1 for (a, _), (b, _) in zip(second, second[1:])]
    if before != after:
        raise ValueError("Existing non-whitespace token boundaries changed")
    positions = [None] * len(rendered)
    for (source_index, _), (view_index, _) in zip(first, second, strict=True):
        positions[view_index] = source_index
    return tuple(positions)


def rewrap(source, width):
    if type(width) is not int or width <= 0:
        raise ValueError("Positive codepoint wrap width required")
    output, line_length = [], 0
    for word in re.findall(r"\S+", source):
        if output:
            line_break = line_length + 1 + len(word) > width
            output.append("\n" if line_break else " ")
            line_length = 0 if line_break else line_length + 1
        output.append(word)
        line_length += len(word)
    return "".join(output)


def ingress_views(source):
    words = re.findall(r"\S+", source)
    paragraphs = [" ".join(words[i : i + 40]) for i in range(0, len(words), 40)]
    horizontal = re.sub(r"[^\S\r\n]+", "  ", source)
    variants = {
        "original": source,
        "horizontal_space_control": horizontal,
        "single_spaces": " ".join(words),
        "wrap_48": rewrap(source, 48),
        "paragraph_40_words": "\n\n".join(paragraphs),
    }
    output = []
    for name, text in variants.items():
        correspondence(source, text)
        actual = from_string(text).text
        output.append(View(name, text, actual, correspondence(source, actual)))
    return tuple(output)


def project_labels(source, labels, view):
    if len(labels) != len(source) or correspondence(source, view.text) != view.source_positions:
        raise ValueError("Exact source labels and validated view mapping required")
    if any(
        type(v) is not int or v not in (0, 1) for c, v in zip(source, labels, strict=True) if not c.isspace()
    ):
        raise ValueError("Every non-whitespace source character needs explicit provenance")
    return [labels[p] if p is not None else -1 for p in view.source_positions]
