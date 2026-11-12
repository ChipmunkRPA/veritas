"""Recover archive paragraph boundaries when a plain-text copy joins them."""

from __future__ import annotations

from collections import defaultdict

from app.evidence.document_revision import tokens

EDGE = 8
MIN_PARAGRAPHS = 3


def index_edges(catalog):
    by_source, starts = {}, defaultdict(list)
    for source, row in catalog.items():
        edges = []
        for i, paragraph in enumerate(row["paragraphs"]):
            text = paragraph["text"]
            values = tuple(value for value, _, _ in tokens(text, [(0, len(text))]))
            prefix, suffix = values[:EDGE], values[-EDGE:]
            edges.append((prefix, suffix, len(values)))
            if len(values) >= 2 * EDGE and len(row["paragraphs"]) - i >= MIN_PARAGRAPHS:
                starts[prefix].append((source, i))
        by_source[source] = edges
    return by_source, dict(starts)


def _runs(words, edges, positions):
    """Match consecutive complete paragraphs using their exact token edges."""
    left = 0
    spans = []
    for prefix, suffix, count in edges:
        if count < 2 * EDGE or words[left:left + EDGE] != prefix:
            return None
        ends = [end + EDGE for end in positions.get(suffix, ())
                if end >= left + EDGE and 65 * count <= 100 * (end + EDGE - left) <= 135 * count]
        if len(ends) != 1:
            return None
        right = ends[0]
        spans.append((left, right))
        left = right
        if right == len(words):
            return spans if len(spans) >= MIN_PARAGRAPHS else None
    return None


def joined_spans(text, span, edge_index):
    """Return a unique complete chain of exact paragraph edges in submitted coordinates."""
    rows = tokens(text, [span])
    if len(rows) < 2 * EDGE * MIN_PARAGRAPHS:
        return None
    values = tuple(value for value, _, _ in rows)
    positions = defaultdict(list)
    for i in range(len(values) - EDGE + 1):
        positions[values[i:i + EDGE]].append(i)
    found = None
    by_source, starts = edge_index
    for source, first in starts.get(values[:EDGE], ()):
        run = _runs(values, by_source[source][first:], positions)
        if run is None:
            continue
        if found is not None:
            return None
        found = (source, first, [(rows[a][1], rows[b - 1][2]) for a, b in run])
    return found
