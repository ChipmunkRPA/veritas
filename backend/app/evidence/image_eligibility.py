"""Public, dependency-free validation of a recorded image candidate gate.

This gate is a selection rule, never a calibration or a pixel reliability mask.
Exact hexadecimal scores survive the canonical JSON six-decimal presentation.
"""
from __future__ import annotations

import math
import re

_SHA = re.compile(r"[a-f0-9]{64}")
RESPONSE_GATE = "image-candidate-eligibility/1"
AREA_GATE = "image-candidate-eligibility/2"


def area_counts(gate: dict) -> tuple[int, int, int]:
    active, minimum, total = (gate.get(key) for key in ("active_cells", "minimum_active_cells", "input_cells"))
    if (any(type(count) is not int for count in (active, minimum, total))
            or not 0 <= active <= total or not 1 <= minimum <= total):
        raise ValueError("The image candidate gate has invalid response cell counts")
    return active, minimum, total


def area_gate(active: int, minimum: int, total: int) -> dict:
    counts = {"active_cells": active, "minimum_active_cells": minimum, "input_cells": total}
    area_counts(counts)
    score, threshold = active / total, minimum / total
    return {"schema": AREA_GATE, "status": "evaluated", "scope": "whole_image",
            "score_kind": "native_response_area_fraction", "comparison": "greater_or_equal",
            "score": score, "score_hex": score.hex(), "threshold": threshold, "threshold_hex": threshold.hex(),
            "eligible": active >= minimum, "role": "candidate_selection_only", "raw_mask_retained": True,
            **counts}


def validate_gate(gate: dict, *, source_sha256: str | None = None,
                  scalar_map_sha256: str | None = None) -> None:
    if not isinstance(gate, dict):
        raise ValueError("The image candidate gate is missing")
    schema = gate.get("schema")
    if (schema not in {RESPONSE_GATE, AREA_GATE} or gate.get("status") != "evaluated"
            or gate.get("scope") != "whole_image" or gate.get("comparison") != "greater_or_equal"):
        raise ValueError("The image candidate gate has an unsupported rule")
    try:
        score = float.fromhex(gate["score_hex"])
        threshold = float.fromhex(gate["threshold_hex"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The image candidate gate has no exact response") from exc
    if schema == AREA_GATE:
        active, minimum, total = area_counts(gate)
        if (gate.get("score_kind") != "native_response_area_fraction" or score != active / total
                or threshold != minimum / total):
            raise ValueError("The image candidate gate contradicts its native response cell counts")
    elif threshold != 0.5:
        raise ValueError("The image candidate gate has an unsupported response threshold")
    if (not math.isfinite(score) or not 0 <= score <= 1
            or type(gate.get("eligible")) is not bool or gate["eligible"] != (score >= threshold)):
        raise ValueError("The image candidate eligibility contradicts its recorded response")
    for name, value in (("score", score), ("threshold", threshold)):
        displayed = gate.get(name)
        if (not isinstance(displayed, int | float) or isinstance(displayed, bool)
                or not math.isfinite(displayed) or round(displayed, 6) != round(value, 6)):
            raise ValueError("The image candidate gate has a contradictory presentation value")
    for field, value in (("score_exact_decimal", score), ("threshold_exact_decimal", threshold)):
        if field in gate and gate[field] != repr(value):
            raise ValueError("The exact image response presentation differs")
    for field, expected in (("source_sha256", source_sha256), ("scalar_map_sha256", scalar_map_sha256)):
        if expected is not None and (not _SHA.fullmatch(str(gate.get(field, ""))) or gate[field] != expected):
            raise ValueError("The image candidate gate is bound to a different source or response map")


def validate_inspection_coverage(document: dict) -> None:
    """Check exact counts/bindings without importing inference or image libraries."""
    if document.get("version") != "image-evidence-3":
        return
    checks = (document.get("coverage") or {}).get("inspection_checks")
    if not isinstance(checks, list):
        raise ValueError("Image inspection coverage is missing")
    seen = set()
    for check in checks:
        ref = check.get("exhibit_ref")
        if not isinstance(ref, str) or ref in seen:
            raise ValueError("Image inspection coverage has ambiguous map references")
        seen.add(ref)
        if not _SHA.fullmatch(str(check.get("scalar_map_sha256", ""))):
            raise ValueError("Image inspection coverage has an invalid response map hash")
        names = ("raw_components", "eligible_components", "selected_components", "omitted_components")
        if any(type(check.get(k)) is not int or check[k] < 0 for k in names):
            raise ValueError("Image inspection coverage has invalid counts")
        raw, eligible, selected, omitted = (check[k] for k in names)
        gate = check.get("candidate_eligibility")
        if gate is not None:
            validate_gate(gate, source_sha256=document["source"]["sha256"], scalar_map_sha256=check.get("scalar_map_sha256"))
        expected = raw if gate is None or gate["eligible"] else 0
        if eligible != expected or selected > eligible or omitted != eligible - selected:
            raise ValueError("Image inspection coverage contradicts candidate selection")
        packets = [p for p in document.get("packets", []) if p.get("evidence_type") == "localisation"
                   and ref in p.get("exhibit_refs", [])]
        if len(packets) != selected:
            raise ValueError("Image inspection coverage differs from the displayed regions")
        for packet in packets:
            if packet.get("localisation", {}).get("candidate_eligibility") != gate:
                raise ValueError("A selected region has different image candidate eligibility")
            if packet.get("anchor", {}).get("scalar_sha256") != check.get("scalar_map_sha256"):
                raise ValueError("A selected region uses a different response map")
        overview = document.get("overview") or {}
        if not any(a.get("sha256") == check.get("scalar_map_sha256") for a in overview.values()):
            raise ValueError("The original measured image response is not included")
        reliability = check.get("reliability_mask_sha256")
        if reliability is not None and not any(a.get("sha256") == reliability for a in overview.values()):
            raise ValueError("The recorded image quality exclusions are not included")
