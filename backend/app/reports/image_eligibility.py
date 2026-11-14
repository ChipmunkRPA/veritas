"""Neutral customer-facing display of already validated image eligibility metadata."""
from __future__ import annotations

import math
import re

from app.evidence.image_eligibility import AREA_GATE, QUALITY_LABELS, area_counts, validate_gate


def _response(gate: dict, key: str) -> float | None:
    try:
        if isinstance(gate.get(key + "_hex"), str):
            value = float.fromhex(gate[key + "_hex"])
        elif isinstance(gate.get(key + "_exact_decimal"), str):
            value = float(gate[key + "_exact_decimal"])
        elif type(gate.get(key)) in (int, float):
            value = float(gate[key])
        else:
            return None
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def eligibility_view(gate: dict | None) -> dict:
    if not gate:
        return {"status": "not_recorded", "title": "No whole-image gate recorded", "score": None, "threshold": None,
                "explanation": "This record does not document a whole-image eligibility gate. Missing eligibility information does not establish authenticity."}
    if gate.get("schema") == AREA_GATE:
        try:
            validate_gate(gate)
            active, minimum, _ = area_counts(gate)
        except ValueError:
            return {"status": "unavailable", "title": "Image eligibility unavailable", "score": None, "threshold": None,
                    "explanation": "The recorded response cell counts could not be verified."}
        return {"status": "eligible" if gate["eligible"] else "withheld", "score_kind": "area_cells",
                "title": "Eligible for region inspection" if gate["eligible"] else "Regions withheld by the image gate",
                "score": str(active), "threshold": str(minimum),
                "explanation": ("The map met the minimum area rule for showing candidates. Normal processing can also create a highlight."
                                if gate["eligible"] else
                                "The map was below the minimum area rule. Small or weak edits may be missed.")}
    score, threshold = _response(gate, "score"), _response(gate, "threshold")
    if (gate.get("schema") != "image-candidate-eligibility/1" or gate.get("status") != "evaluated"
            or gate.get("scope") != "whole_image" or gate.get("comparison") != "greater_or_equal"
            or type(gate.get("eligible")) is not bool or score is None or threshold is None
            or gate["eligible"] != (score >= threshold)):
        return {"status": "unavailable", "title": "Image eligibility unavailable", "score": None, "threshold": None,
                "explanation": "An exact eligibility response could not be displayed. This does not establish that the image is unaltered."}
    return {"status": "eligible" if gate["eligible"] else "withheld",
            "title": "Eligible for region inspection" if gate["eligible"] else "Regions withheld by the image gate",
            "score": repr(score), "threshold": repr(threshold),
            "explanation": "The whole-image response meets the display threshold. Region masks still require inspection and can contain false alarms or miss edits." if gate["eligible"] else
                           "The whole-image response is below the display threshold, so this check contributes no candidate regions. This does not establish that the image is unaltered."}


def inspection_check_view(check: dict) -> dict:
    labels = list(dict.fromkeys(ref.get("display_id") for ref in check.get("signal_refs", [])
                               if isinstance(ref.get("display_id"), str) and re.fullmatch(r"S\d+", ref["display_id"])))
    counts = [check.get(key) for key in ("raw_components", "eligible_components", "selected_components", "omitted_components")]
    valid = all(type(value) is int and value >= 0 for value in counts)
    raw, eligible, selected, omitted = counts
    valid = valid and eligible <= raw and selected <= eligible and omitted == eligible - selected
    exclusion = check.get("excluded_area_fraction")
    eligibility = eligibility_view(check.get("candidate_eligibility"))
    has_flip = "flip_consistency" in check
    flip = check.get("flip_consistency")
    flip_view = None
    if valid and isinstance(flip, dict) and flip.get("schema") == "image-flip-consistency/1":
        inspected, supported, withheld = (flip.get(key) for key in
                                          ("inspected_components", "supported_components", "withheld_components"))
        candidates = flip.get("candidates")
        if (all(type(value) is int for value in (inspected, supported, withheld))
                and eligible <= inspected <= min(2, raw) and supported == eligible and withheld == inspected - supported
                and isinstance(candidates, list) and len(candidates) == inspected):
            flip_view = {"inspected": inspected, "withheld": withheld,
                         "outside_limit": raw - inspected if eligibility["status"] != "withheld" else 0,
                         "candidates": candidates}
    suppressed = 0
    if valid:
        suppressed = raw - eligible if not has_flip else raw if eligibility["status"] == "withheld" else 0
    eligible_label = "Eligible after image gate" if eligibility["status"] in {"eligible", "withheld"} else "Available for region selection"
    shares = (check.get("source_quality") or {}).get("native_pixel_shares") or {}
    return {"label": "Check " + ", ".join(labels) if labels else "Recorded image check",
            "eligibility": eligibility,
            "eligible_count_label": "Supported by the flip check" if has_flip else eligible_label,
            "counts": {"raw": raw, "eligible": eligible, "selected": selected, "omitted": omitted,
                       "suppressed": suppressed} if valid else None,
            "has_flip": has_flip, "flip": flip_view,
            "map_grid": check.get("map_grid"),
            "quality_reasons": [{"label": label, "percent": round(shares[key] * 100, 2)} for key, label in QUALITY_LABELS.items()
                                if shares.get(key, 0) > 0],
            "excluded_percent": round(exclusion * 100, 3) if type(exclusion) in (int, float) and math.isfinite(exclusion) and 0 <= exclusion <= 1 else None}
