"""Allocate a bounded set of writing cards using recorded directions and source blocks."""

from __future__ import annotations

from collections import Counter

from app.evidence import document_scope

DEFAULT_POLICY = (
    "Up to half the displayed passages are reserved for disagreement; remaining slots use other candidate "
    "passages, then any remaining disagreements. Each group follows source order; incomparable score kinds are not pooled."
)
BALANCED_POLICY = (
    "Up to half the displayed passages are reserved for disagreement. Within each group, the next card favors "
    "a source block with fewer displayed cards. Classifier-backed passages break ties, followed by source order."
)


def select(candidates, disagreements, classifier_support, priority, blocks):
    """Keep the existing disagreement allocation and balance locations within each group."""
    limit = document_scope.MAX_PASSAGES
    selected = list(priority)[:limit]
    remaining = set(candidates).difference(selected)
    conflicts = sorted(remaining & disagreements)
    other = sorted(remaining - disagreements)
    balance = len(candidates) > limit and len(blocks) == len(candidates) and len(set(blocks.values())) > 1
    if not balance:
        reserved = min(limit // 2, limit - len(selected))
        selected.extend(conflicts[:reserved])
        selected.extend(other[:limit - len(selected)])
        selected.extend(conflicts[reserved:reserved + limit - len(selected)])
        return sorted(selected), None

    counts = Counter(blocks[location] for location in selected)
    for options, allowance in ((conflicts, limit // 2), (other, limit), (conflicts, limit)):
        for _ in range(min(len(options), allowance, limit - len(selected))):
            location = min(options, key=lambda location: (
                counts[blocks[location]], location not in classifier_support, location,
            ))
            selected.append(location)
            counts[blocks[location]] += 1
            options.remove(location)
    return sorted(selected), {
        "schema": document_scope.CARD_SELECTION_SCHEMA,
        "candidate_blocks": len(set(blocks.values())), "displayed_blocks": len(counts),
        "displayed_passages": len(selected),
    }


def directions(groups, candidates):
    disagreements, classifier = set(), set()
    for key in candidates:
        for window in groups[key]:
            if document_scope.inspection_direction(window) is False:
                disagreements.add(key)
            if (window.get("score_kind") == "classifier_probability" and document_scope.supports_inspection(window)):
                classifier.add(key)
    return disagreements, classifier
