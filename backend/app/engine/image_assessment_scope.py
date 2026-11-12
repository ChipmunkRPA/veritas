"""Scope image estimates to contributing checks; regional cues never become AI probabilities."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from app.evidence.image_whole_review import from_result as whole_image_review


def _counted(row):
    value = row.get("logit")
    return row.get("counted") is True or (row.get("counted") is not False and isinstance(value, int | float)
                                        and not isinstance(value, bool) and math.isfinite(value) and abs(value) > 1e-9)


def image_scope(result: Mapping[str, Any], answer: dict) -> dict:
    """Change presentation only. Preserve the saved decision, calibrations and raw evidence."""
    if result.get("kind") != "image" or not answer["applies"] or answer["direction"] not in {"ai", "original"}:
        return answer
    rows = [r for r in (result.get("decision") or {}).get("contributions") or [] if r.get("kind") == "detector"]
    included = set()
    local_cues = False
    for signal in result.get("signals") or []:
        if signal.get("status", "ok") != "ok":
            continue
        refs = {str(ref) for ref in (signal.get("id"), (signal.get("legacy") or {}).get("v1_id"),
                                    f"detector:{signal['detector_id']}" if signal.get("detector_id") else None) if ref}
        family = str(signal.get("family") or "")
        rule = (signal.get("details") or {}).get("localisation")
        regions = (signal.get("metrics") or {}).get("regions")
        gate = rule.get("candidate_eligibility") if isinstance(rule, dict) else None
        if (family == "localiser" and isinstance(regions, int | float) and not isinstance(regions, bool)
                and math.isfinite(regions) and regions > 0 and (gate is None or gate.get("eligible") is True)):
            local_cues = True
        matched = [r for r in rows if r.get("signal_id") in refs]
        omitted = (bool(signal.get("outside_estimate")) or signal.get("effective_weight") == 0
                   or any(r.get("counted") is False for r in matched))
        if omitted or not any(_counted(r) for r in matched):
            continue
        hypothesis = signal.get("hypothesis")
        if hypothesis == "H1_whole_content_synthesis" or family == "learned-whole-image":
            included.add("generation")
        elif "face" in family:
            included.add("face")
        elif hypothesis == "H2_localised_manipulation" and family == "localiser":
            included.add("editing")
    if not included:
        return {**answer, "direction": "inconclusive", "headline": "Inconclusive: image-check coverage is not established",
                "confidence_pct": None, "confidence_label": None, "probability_ai_pct": None,
                "interval_pct": None, "interval_label": None,
                "meaning": "The saved result does not identify usable contributing image checks. It cannot establish original or AI authorship."}
    labels = [label for key, label in (("generation", "whole-image generation"), ("face", "face-pattern"),
                                      ("editing", "local-editing")) if key in included]
    subject = " and ".join(labels)
    label = subject[0].upper() + subject[1:] + " checks"
    note = (f"The percentage covers {subject} checks. It does not verify every pixel or rule out localized AI edits. "
            "Regional editing cues are assessed separately and do not enter this percentage.")
    scoped = {**answer, "scope": "image_components", "scope_label": label, "scope_note": note,
              "confidence_label": "Scoped image-check estimate", "interval_label": "90% interval for these image checks"
              if answer["interval_pct"] is not None else None,
              "headline": f"{label} lean towards AI generation or alteration" if answer["direction"] == "ai" else
                          f"{label} lean original; local edits remain unresolved",
              "meaning": "The calibrated estimate applies to the listed image checks, not a finding that every part of the file is original or AI-generated."}
    whole_cue = (whole_image_review(result) or {}).get("status") == "flagged"
    if answer["direction"] == "original" and (local_cues or whole_cue):
        component = {k: scoped[k] for k in ("direction", "headline", "confidence_pct", "confidence_label",
                                           "probability_ai_pct", "interval_pct", "interval_label", "scope_label")}
        interval = answer["interval_pct"]
        uncertainty = (f" (90% reference-fit interval {interval[0]:g}-{interval[1]:g}%)" if interval else "")
        estimate = f"The separate {subject} estimate leans original at {answer['confidence_pct']:g}%{uncertainty}. "
        if whole_cue:
            note = (estimate + "A supplementary check found a pattern associated with fully generated images. "
                    "It carries zero weight in the calibrated estimate. No whole-file confidence percentage is assigned.")
            headline = "Inconclusive: image checks disagree"
            meaning = ("The calibrated checks and the supplementary whole-image cue point in different directions. "
                       "Review the inspected image views and limitations before drawing a conclusion about the file.")
        else:
            note = (estimate +
                "That percentage does not cover the highlighted editing cues. Ordinary edits and AI edits can produce "
                "such cues; their AI origin remains inconclusive. No whole-file confidence percentage is assigned.")
            headline = "Local editing cues found; AI origin is inconclusive"
            meaning = ("A local check highlighted editing cues that the whole-image estimate cannot resolve. Inspect the source crops, masks "
                       "and limitations; neither AI editing nor an entirely original file is established.")
        return {**scoped, "direction": "inconclusive", "headline": headline,
                "confidence_pct": None, "confidence_label": None, "probability_ai_pct": None,
                "interval_pct": None, "interval_label": None, "scope_note": note,
                "component_estimate": component,
                "meaning": meaning}
    return scoped
