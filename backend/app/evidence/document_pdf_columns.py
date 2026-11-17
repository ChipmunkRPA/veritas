"""Project separated PDF columns onto a text view with exact original offsets."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby, pairwise
from statistics import median

from app.evidence.document_interval_locations import locate

MIN_GUTTER = 8.0  # PDF points
MIN_SHARED_HEIGHT = 36.0
MAX_BLOCKS_PER_PAGE = 96
SIDES = ("left", "right", "both")


@dataclass(frozen=True)
class ColumnView:
    page: int
    side: str
    text: str
    # Position in this column view, followed by position in the original extraction.
    spans: tuple[tuple[int, int, int], ...]

    @property
    def paragraphs(self):
        return [(start, end) for start, end, _ in self.spans]

    def scope(self):
        return {"page": self.page, "column": self.side,
                "source_blocks": [[source, source + end - start] for start, end, source in self.spans]}

    def places(self, mapping, start, end):
        if start == end:
            end += 1
        locations = []
        for a, b, source in self.spans:
            lo, hi = max(start, a), min(end, b)
            if lo < hi:
                places = locate("pymupdf", mapping, source + lo - a, source + hi - a)
                if not places:
                    return []
                locations.extend(places)
        return locations

    def standalone_label(self, start, end):
        excerpt = self.text[start:end]
        words = excerpt.split()
        if not words or len(excerpt) > 200 or len(words) > 28 or excerpt.endswith((".", "!", "?")):
            return False
        if len(words) > 14 and sum(word[0].isupper() for word in words) * 5 < len(words) * 2:
            return False
        return any(a <= start < end <= b and not (self.text[a:start] + self.text[end:b]).strip() for a, b, _ in self.spans)


def _view(text, page, side, blocks):
    pieces, spans = [], []
    cursor = 0
    for block in blocks:
        a, b = block["start"], block["end"]
        pieces.append(text[a:b])
        spans.append((cursor, cursor + b - a, a))
        cursor += b - a + 2
    return ColumnView(page, side, "\n\n".join(pieces), tuple(spans))


def _ordered_column(blocks):
    """Require monotonic baselines even where font boxes overlap vertically."""
    left_edges = []
    preceding_bottom = 0.
    for block in blocks:
        boxes = [word["bbox"] for word in block["fragments"]]
        height = median(y1 - y0 for _, y0, _, y1 in boxes)
        if block["bbox"][1] < preceding_bottom - .30 * height:
            return False
        preceding_bottom = block["bbox"][3]
        line_y = previous_right = None
        for x0, y0, x1, y1 in boxes:
            centre = (y0 + y1) / 2
            tolerance = max(2., .22 * height)
            if line_y is None or centre - line_y > tolerance:
                if line_y is not None and centre - line_y < .52 * height:
                    return False
                line_y = centre
                left_edges.append(x0)
            elif (centre < line_y - tolerance or x0 < previous_right - .75
                  or x0 - previous_right > 2.5 * height):
                return False
            previous_right = x1
    return max(left_edges) - min(left_edges) <= 36.


def page_views(text, page, blocks):
    """Find one continuous whitespace gutter between two non-overlapping flows."""
    if not 4 <= len(blocks) <= MAX_BLOCKS_PER_PAGE:
        return ()
    x_edges = sorted({x for block in blocks for x in block["bbox"][::2]})
    width = blocks[0]["page_width"]
    candidates = {}
    for a, b in pairwise(x_edges):
        cut = (a + b) / 2
        if b - a < MIN_GUTTER or not .22 * width <= cut <= .78 * width:
            continue
        left = [block for block in blocks if block["bbox"][2] <= cut]
        right = [block for block in blocks if block["bbox"][0] >= cut]
        if min(len(left), len(right)) < 2:
            continue
        envelopes = [[min(row["bbox"][i] for row in lane) if i < 2 else max(row["bbox"][i] for row in lane)
                      for i in range(4)] for lane in (left, right)]
        left_box, right_box = envelopes
        height = min(left_box[3], right_box[3]) - max(left_box[1], right_box[1])
        if height < MIN_SHARED_HEIGHT or min(box[2] - box[0] for box in envelopes) < .14 * width:
            continue
        column_top, column_bottom = min(box[1] for box in envelopes), max(box[3] for box in envelopes)
        spanning = [row for row in blocks if row["bbox"][0] < cut < row["bbox"][2]]
        if any(row["bbox"][3] > column_top + .5 and row["bbox"][1] < column_bottom - .5 for row in spanning):
            continue
        first, second = _view(text, page, "left", left), _view(text, page, "right", right)
        if not (_ordered_column(left) and _ordered_column(right)):
            continue
        sizes = [box[2] - box[0] for box in envelopes]
        # A substantially narrower sidebar has its own reading stream.
        combined = (_view(text, page, "both", left + right),) if min(sizes) >= .75 * max(sizes) else ()
        candidates[first.spans, second.spans] = (*combined, first, second)
    return next(iter(candidates.values())) if len(candidates) == 1 else ()


def views(text, mapping, page_limit):
    if not mapping or mapping[-1]["page"] > page_limit:
        return ()
    return tuple(page_views(text, page, list(blocks)) for page, blocks in groupby(mapping, key=lambda block: block["page"]))
