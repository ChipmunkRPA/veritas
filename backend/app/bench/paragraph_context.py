"""Portable paragraph response arithmetic for an isolated text experiment."""

from __future__ import annotations

from math import isfinite, log, log1p


def responses(paragraphs: list[tuple[int, int, int, float]]) -> list[dict]:
    """Measure eligible interior paragraphs from ordered spans, word counts and responses."""
    logits = []
    previous_end = 0
    for start, end, words, response in paragraphs:
        if (type(start) is not int or type(end) is not int or not previous_end <= start < end
                or type(words) is not int or words < 0 or type(response) not in (float, int)
                or not isfinite(response) or not 0 <= response <= 1):
            raise ValueError("Expected ordered source spans, word counts and finite responses in [0, 1]")
        previous_end = end
        probability = min(max(response, 1e-6), 1 - 1e-6)
        logits.append(log(probability) - log1p(-probability))
    return [{"span": (start, end), "isolated": response,
             "two_sided": logits[i] - max(logits[i - 1], logits[i + 1])}
            for i, (start, end, _, response) in enumerate(paragraphs[1:-1], 1)
            if all(row[2] >= 20 for row in paragraphs[i - 1:i + 2])]


def selected_span(measured: list[dict], points: dict[str, float], rule: str) -> tuple[int, int] | None:
    """Pick the unique strongest qualifying span. Empty measurements mean unassessed."""
    if rule not in ("isolated", "two_sided", "both"):
        raise ValueError("Unknown paragraph comparison rule")
    filters = ("isolated", "two_sided") if rule == "both" else (rule,)
    rank = "two_sided" if rule == "both" else rule
    qualified = [row for row in measured if all(row[key] > points[key] for key in filters)]
    if not qualified:
        return None
    strongest = max(row[rank] for row in qualified)
    spans = [tuple(row["span"]) for row in qualified if row[rank] == strongest]
    return spans[0] if len(spans) == 1 else None
