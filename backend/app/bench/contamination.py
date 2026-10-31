"""Explicit source aliases for future admission evidence; no fuzzy training-overlap claims.

The source catalogue calls COCO ``detection-datasets/coco`` while detector cards
use ``MSCOCO``. Only these documented collection names are equivalent. Qualified
names such as ``MSCOCO train2017`` remain exact: collection identity does not
establish overlap with a particular upstream split. ``coco-val`` is a source_id,
not a source_dataset alias. Unknown labels retain exact matching semantics.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

POLICY_VERSION = "explicit-source-aliases/1"
_COCO_ALIASES = frozenset({"mscoco", "coco", "detection-datasets/coco"})


def _key(source: str) -> tuple[str, str]:
    return ("family", "coco") if source.strip().casefold() in _COCO_ALIASES else ("exact", source)


def canonical_exclusions(exclusions: Sequence[str]) -> frozenset[tuple[str, str]]:
    """Validated identities for equality/subset checks; use the same aliases as filtering."""
    if (not isinstance(exclusions, Sequence) or isinstance(exclusions, str | bytes)
            or any(not isinstance(value, str) or not value.strip() for value in exclusions)):
        raise ValueError("contamination exclusions must be nonblank source names in a sequence")
    return frozenset(_key(value) for value in exclusions)


def card_contamination_sets(detector_id: str) -> list[str] | None:
    """Card exclusions, or the source card for a derived feature; None means no card."""
    from app.engine.fusion import derived
    from app.engine.registry import get_registry

    manifest = get_registry().manifest
    card = manifest.detector(detector_id)
    if card is None and detector_id in derived.BY_ID:
        card = manifest.detector(derived.BY_ID[detector_id].source)
    return None if card is None else list(card.contamination_sets)


def contamination_filter(sources: Sequence[object], exclusions: Sequence[str]) -> tuple[list[bool], dict]:
    """Return the drop mask and auditable counts, before any fitting or evaluation.

    Missing source identity is tolerated only when no exclusions were declared.
    Callers must first restrict to their eligible fit/cal population, so unrelated
    test/wild or refused records cannot affect this check.
    """
    keys = canonical_exclusions(exclusions)
    declared = sorted(set(exclusions))
    if declared and any(not isinstance(value, str) or not value.strip() for value in sources):
        raise ValueError("declared contamination exclusions require nonblank source_dataset for every eligible fit/cal item")
    mask = [_key(value) in keys if isinstance(value, str) and keys else False for value in sources]
    counts = Counter(value for value, dropped in zip(sources, mask, strict=True) if dropped)
    return mask, {"contamination_policy": POLICY_VERSION, "contamination_sets": declared,
                  "contaminated_dropped": sum(mask), "contaminated_source_counts": dict(sorted(counts.items()))}
