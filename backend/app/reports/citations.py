"""Pure selection of a finding's first cited heatmap from a frozen result.

Selection depends only on citation order and unique evidence IDs, never preview/payload/mask availability. Rendering and
capture share this walk so an unavailable selected map cannot be replaced by a different, more convenient map.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def _index(value: Any) -> tuple[dict[str, dict[str, Any]], set[str], set[str]]:
    records = [r for r in value if isinstance(r, dict) and isinstance(r.get("id"), str) and r["id"]] if isinstance(value, list) else []
    counts = Counter(r["id"] for r in records)
    repeated = {ident for ident, count in counts.items() if count != 1}
    return {r["id"]: r for r in records if r["id"] not in repeated}, repeated, set(counts)


def _references(value: Any) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(ident, str) or not ident for ident in value):
        return None
    return value


def _finding_references(finding: dict[str, Any]) -> list[str] | None:
    direct = _references(finding.get("evidence_ids"))
    sentences = finding.get("sentences")
    if direct is None or (sentences is not None and not isinstance(sentences, list)):
        return None
    cited = list(direct)
    for sentence in sentences or []:
        if not isinstance(sentence, dict):
            return None
        refs = _references(sentence.get("evidence_ids"))
        if refs is None:
            return None
        cited.extend(refs)
    return list(dict.fromkeys(cited))


SUMMARY_FAMILIES = frozenset({"coverage", "F9_distribution_conditions"})  # findings about the analysis, not about a check's map


def finding_heatmaps(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Finding ID → its first cited heatmap exhibit, preserving recorded citation order and without modifying ``result``.

    Walk finding-level references before sentence-level references, expanding a cited signal's exhibit IDs in their recorded
    order. References to other evidence kinds are skipped; signals are not recursively followed. Duplicate IDs within a
    collection and IDs shared between finding/signal/exhibit collections are ambiguous. A finding that references ambiguity
    or malformed citation lists is omitted rather than selecting a later map. Unrelated ambiguous IDs affect no other finding.
    Returned dictionaries are the original exhibit objects; callers must treat them as read-only.
    """
    if not isinstance(result, dict):
        return {}
    findings, duplicate_findings, finding_ids = _index(result.get("findings"))
    exhibits, duplicate_exhibits, exhibit_ids = _index(result.get("exhibits"))
    signals, duplicate_signals, signal_ids = _index(result.get("signals"))
    ambiguous = (duplicate_findings | duplicate_exhibits | duplicate_signals
                 | (finding_ids & exhibit_ids) | (finding_ids & signal_ids) | (exhibit_ids & signal_ids))
    selected = {}
    for finding_id, finding in findings.items():
        if finding_id in ambiguous:
            continue
        # N-19q (N-20b's read of the reports, finding 13): the findings on the checks that ran and on the input cite the
        # signals they count, and drew the map of one of them ("Checks that ran" beside the error-level miniature of F5)
        if finding.get("family") in SUMMARY_FAMILIES:
            continue
        cited = _finding_references(finding)
        if cited is None:
            continue
        candidates = []
        valid = True
        for ref in cited:
            if ref in ambiguous:
                valid = False
                break
            if ref in exhibits:
                candidates.append(ref)
            elif ref in signals:
                expanded = _references(signals[ref].get("exhibit_ids"))
                if expanded is None or any(ident in ambiguous for ident in expanded):
                    valid = False
                    break
                candidates.extend(expanded)
        if not valid:
            continue
        chosen = next((exhibits[ident] for ident in candidates
                       if ident in exhibits and exhibits[ident].get("kind") == "heatmap"), None)
        if chosen is not None:
            selected[finding_id] = chosen
    return selected
