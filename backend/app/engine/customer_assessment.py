"""Customer-facing tendency from an existing result; never changes a frozen decision.

The 80% rule reads the saved calibrated *point estimate*, not a detector score or
the end of its bootstrap interval. The output is a screening interpretation,
not a new calibration claim. The browser mirrors this function for old servers.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from app.engine.fusion import abstain

RULE = "80-percent-leaning/v1"
THRESHOLD = 0.8
INTERVAL_NOTE = (
    "This range describes uncertainty in the combined estimate. It does not include every detection error "
    "or unfamiliar type of content; a 90% interval does not mean the conclusion is 90% accurate."
)
ESTIMATE_NOTE = (
    "This is the system's estimated likelihood, not an independently established accuracy rate or proof of origin. "
    "An original-file tendency does not establish that every part of the file is unedited."
)


def _video_scope(result: Mapping[str, Any]) -> dict[str, Any]:
    scope = {"scope": "whole_file", "scope_label": "Combined file assessment", "scope_note": None, "excluded_channels": []}
    if result.get("kind") != "video":
        return scope
    if not result.get("signals"):
        return {"scope": "assessed_components", "scope_label": "Check coverage not yet available",
                "scope_note": "The contribution of picture, voice and timing checks has not been established from the available record. This estimate must not be read as clearing the whole video.",
                "excluded_channels": []}
    findings = result.get("findings") or []
    rows = [r for r in (result.get("decision") or {}).get("contributions") or [] if r.get("kind") == "detector"]
    channels: dict[str, str] = {}
    excluded: set[str] = set()
    for signal in result.get("signals") or []:
        refs = {signal.get("id"), (signal.get("legacy") or {}).get("v1_id"), f"detector:{signal.get('detector_id')}"} - {None}
        families = {f.get("family") for f in findings if refs.intersection(f.get("evidence_ids") or [])}
        family, modality, lane = signal.get("family"), signal.get("modality"), signal.get("lane")
        detector = str(signal.get("detector_id") or "")
        if family == "av-sync" or lane == "lip_sync" or "V2_lip_sync" in families or detector.startswith("vid.lipsync."):
            channel = "audio_picture_timing"
        elif family == "voice" or modality == "audio" or "A1_voice" in families or detector.startswith("aud."):
            channel = "voice"
        elif modality in {"image", "video"} or detector.startswith(("img.", "vid.", "face.")) or families.intersection({
                "F3_whole_image_generation", "F4_localised_manipulation", "F10_face", "V1_face_track", "V3_synthetic_video"}):
            channel = "visual"
        else:
            channel = "other"
        for ref in refs:
            channels[str(ref)] = channel
        omitted = (bool(signal.get("outside_estimate")) or signal.get("effective_weight") == 0
                   or any(r.get("signal_id") in refs and r.get("counted") is False for r in rows))
        if signal.get("status", "ok") == "ok" and omitted and channel in {"voice", "audio_picture_timing"}:
            excluded.add(channel)
    if not excluded:
        return scope
    counted = [r for r in rows if r.get("counted") is True or (r.get("counted") is not False
               and isinstance(r.get("logit"), (int, float)) and abs(r["logit"]) > 1e-9)]
    visual_only = bool(counted) and all(channels.get(str(r.get("signal_id"))) == "visual" for r in counted)
    omitted_labels = [label for channel, label in [("voice", "voice/audio evidence"), ("audio_picture_timing", "audio/picture timing evidence")]
                      if channel in excluded]
    note = ("This percentage applies to the visual checks that contributed to the estimate. " if visual_only
            else "This percentage applies only to the checks that contributed to the estimate. ")
    verb = "were" if len(omitted_labels) > 1 else "was"
    note += "The " + " and ".join(omitted_labels) + f" {verb} checked separately but {verb} not included in this percentage."
    if "voice" in excluded:
        note += " The audio has not been cleared as original."
    return {"scope": "visual_content" if visual_only else "assessed_components",
            "scope_label": "Visual content only" if visual_only else "Assessed components only", "scope_note": note,
            "excluded_channels": [key for key in ("voice", "audio_picture_timing") if key in excluded]}


def _probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and 0 <= value <= 1 else None


def _pct(value: float) -> float:
    # Do not round 79.99% up to the 80% boundary or print a statistical estimate as 100%.
    return math.floor(round(value * 1000, 8)) / 10


def customer_assessment(result: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only interpretation shared by API, JSON and report presentation.

    ``result`` is an analysis-detail-shaped mapping with decision, condition_profile,
    caveats and optional status. A provenance-led result retains its existing words
    (applies=False), since documentary floors/caps are not statistical confidence.
    """
    decision = result.get("decision") or {}
    profile = result.get("condition_profile") or {}
    answer: dict[str, Any] = {
        "rule": RULE, "threshold_pct": 80, "applies": True,
        "direction": "unavailable", "headline": "No reliable automated conclusion",
        "confidence_pct": None, "confidence_label": None, "probability_ai_pct": None,
        "interval_pct": None, "interval_label": None, "interval_note": INTERVAL_NOTE,
        "meaning": "The analysis did not establish a usable whole-file estimate.",
        "limitation": "Review the recorded observations and any checks that could not run. Missing evidence does not favour either answer.",
        **_video_scope(result),
    }
    if decision.get("provenance_record") or result.get("stored_record") or decision.get("driving_hypothesis") == "H3_provenance_integrity":
        return {**answer, "applies": False, "direction": "record", "headline": None,
                "meaning": "The conclusion follows the file's provenance record, not a confidence percentage."}
    if result.get("status") not in (None, "done") or not decision or decision.get("code") == "not_assessable":
        return answer
    from app.detectors.audio.calibration import SHA256, NOTE

    if result.get("kind") == "audio" and decision.get("fuser_sha256") == SHA256:
        p = _probability(decision.get("probability_ai_calibrated"))
        direction = ("ai" if decision.get("code") == "ai_supported" and p is not None and p >= .8 else
                     "original" if decision.get("code") == "no_ai_indicators" and p is not None and p <= .2 + 1e-12 else "inconclusive")
        return {**answer, "direction": direction, "headline": decision.get("verbal"),
                "confidence_pct": min(99.9, _pct(p if direction == "ai" else 1-p)) if direction != "inconclusive" else None,
                "confidence_label": "Reference-calibrated estimate" if direction != "inconclusive" else None,
                "probability_ai_pct": _pct(p) if p is not None else None,
                "meaning": decision.get("reason", ""), "limitation": NOTE,
                "scope_label": "Speech assessment", "scope_note": "Local experimental audio screening; the percentage concerns speech, not every sound or edit in the recording.",
                "interval_note": "No per-file confidence interval has been established for this audio calibration."}
    if decision.get("fusion_mode") != "artefact":
        return {**answer, "headline": "Inconclusive: no calibrated estimate",
                "meaning": "The available checks provide inspection scores, but those scores have not been calibrated into a reliable whole-file likelihood.",
                "limitation": "An inspection score of 80 out of 100 does not mean 80% confidence. Use the linked evidence to examine the file further."}
    if abstain.outside_validated(decision, profile) or profile.get("within_validated_envelope") is not True:
        return {**answer, "headline": "Inconclusive: this input is outside the verified scope",
                "meaning": "The saved validation does not cover this kind of input well enough to apply the 80% leaning rule.",
                "limitation": "A score can still be recorded for inspection, but it cannot support a confidence percentage for this file."}
    p = _probability(decision.get("probability_ai_calibrated"))
    if p is None:
        return answer
    answer.update(direction="inconclusive", headline="Inconclusive: neither tendency reaches 80%",
                  probability_ai_pct=_pct(p), meaning="Neither the AI-generated nor original-file estimate reaches the 80% threshold.",
                  limitation=ESTIMATE_NOTE)
    ci = decision.get("ci90")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        lo, hi = _probability(ci[0]), _probability(ci[1])
        if lo is not None and hi is not None and lo <= hi:
            answer.update(interval_pct=[_pct(lo), _pct(hi)], interval_label="90% interval for AI likelihood")
    # A high prior, opposing checks or a strong disagreement must not become confidence
    # merely because the final point estimate crosses a display threshold.
    rule = abstain.rule_taken(decision)
    caveats = result.get("caveats") or []
    disagree = any("strongly disagree" in str(c).lower() for c in caveats)
    if rule is not None or disagree:
        return {**answer, "headline": "Inconclusive: the evidence does not support a reliable tendency",
                "meaning": "The recorded checks are conflicting or do not provide enough support for the direction of the combined estimate."}
    direction = "ai" if p >= THRESHOLD else "original" if p <= 0.2 else None
    if direction is None:
        return answer
    ai = direction == "ai"
    answer.update(direction=direction,
                  headline="Leaning towards an AI-generated or AI-altered file" if ai else "Leaning towards an original file",
                  confidence_pct=min(99.9, _pct(p if ai else 1 - p)), confidence_label="Automated estimate",
                  meaning="The combined automated estimate reaches the 80% threshold for AI generation or AI alteration."
                  if ai else "The combined automated estimate reaches the 80% threshold in favour of original content among the checks that contributed to this estimate. Local flags and checks outside the estimate require separate consideration.")
    if not ai and answer["interval_pct"] is not None:
        answer.update(interval_pct=[_pct(1 - float(ci[1])), _pct(1 - float(ci[0]))],
                      interval_label="90% interval for original-file likelihood")
    if answer["scope"] != "whole_file":
        subject = "visual content" if answer["scope"] == "visual_content" else "content in the assessed components"
        answer["headline"] = (f"Leaning towards AI-generated or AI-altered {subject}" if ai
                              else f"Leaning towards original {subject}")
        answer["meaning"] = (f"Among the checks included in this estimate, the combined estimate reaches the 80% threshold "
                             f"in favour of {'AI generation or alteration of' if ai else 'original'} {subject}. "
                             "Local flags and excluded checks require separate consideration.")
        if answer["interval_pct"] is not None:
            answer["interval_label"] = "90% interval for this scoped estimate"
    return answer
