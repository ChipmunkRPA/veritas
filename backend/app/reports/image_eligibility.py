"""Neutral customer-facing display of already validated image eligibility metadata."""
from __future__ import annotations

import math
import re


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
    return {"label": "Check " + ", ".join(labels) if labels else "Recorded image check",
            "eligibility": eligibility,
            "eligible_count_label": "Eligible after image gate" if eligibility["status"] in {"eligible", "withheld"} else "Available for region selection",
            "counts": {"raw": raw, "eligible": eligible, "selected": selected, "omitted": omitted,
                       "suppressed": raw - eligible} if valid else None,
            "map_grid": check.get("map_grid"),
            "excluded_percent": round(exclusion * 100, 3) if type(exclusion) in (int, float) and math.isfinite(exclusion) and 0 <= exclusion <= 1 else None}
