"""Customer-facing tendency from an existing result; never changes a frozen decision.

The evidence-led presentation reads the saved calibrated *point estimate*, not a detector score or
the end of its bootstrap interval. The output is a screening interpretation,
not a new calibration claim. The browser mirrors this function for old servers.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from app.engine.finding_assessment import finding_assessments
from app.engine.fusion import abstain
from app.engine.service_policy import RULE, THRESHOLD, standard_service_policy

ARTICLE_REFERENCE_SLICE = "document/local-article-v1"
EMAIL_REFERENCE_SLICE = "document/local-email-v1"
INTERVAL_NOTE = (
    "This range describes uncertainty in the combined estimate. It does not include every detection error "
    "or unfamiliar type of content; a 90% interval does not mean the conclusion is 90% accurate."
)
ESTIMATE_NOTE = (
    "This is the system's estimated likelihood, not an independently established accuracy rate or proof of origin. "
    "An original-file tendency does not establish that every part of the file is unedited."
)


def _signal_refs(signal: Mapping[str, Any]) -> set[str]:
    return {str(ref) for ref in (signal.get("id"), (signal.get("legacy") or {}).get("v1_id"),
                                f"detector:{signal['detector_id']}" if signal.get("detector_id") else None) if ref}


def _counted(row: Mapping[str, Any]) -> bool:
    value = row.get("logit")
    return row.get("counted") is True or (row.get("counted") is not False and isinstance(value, int | float)
                                        and not isinstance(value, bool) and math.isfinite(value) and abs(value) > 1e-9)


def _video_scope(result: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    scope = {"scope": "whole_file", "scope_label": "Assessed content", "scope_note": "The overall estimate covers only the checks included in it. Findings assessed separately do not enter this percentage; unassessed regions and types of editing remain unresolved.", "excluded_channels": []}
    if result.get("kind") != "video":
        return scope, True
    findings = result.get("findings") or []
    rows = [r for r in (result.get("decision") or {}).get("contributions") or [] if r.get("kind") == "detector"]
    included: set[str] = set()
    statuses: dict[str, set[str]] = {channel: set() for channel in ("visual", "voice", "audio_picture_timing")}
    for signal in result.get("signals") or []:
        refs = _signal_refs(signal)
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
            continue
        status = signal.get("status") or "unknown"
        statuses[channel].add(status)
        matched = [r for r in rows if r.get("signal_id") in refs]
        omitted = (bool(signal.get("outside_estimate")) or signal.get("effective_weight") == 0
                   or any(r.get("counted") is False for r in matched))
        if status == "ok" and not omitted and any(_counted(r) for r in matched):
            included.add(channel)
    labels = {"visual": "Visual content", "voice": "Voice/audio", "audio_picture_timing": "Audio/picture timing"}
    subjects = [label for channel, label in (("visual", "sampled visual content"), ("voice", "speech"),
                                            ("audio_picture_timing", "audio/picture timing")) if channel in included]
    notes = [f"The main assessment uses {' and '.join(subjects)}." if subjects else
             "The saved result does not establish which content checks contributed to the main assessment."]
    for channel, states in statuses.items():
        label = labels[channel]
        if "ok" in states and channel not in included:
            notes.append(f"{label} was checked separately from the main assessment.")
        if states.intersection({"failed", "timeout", "timed_out"}):
            notes.append(f"Some {label.lower()} checks could not complete.")
        if "skipped" in states:
            notes.append(f"Some {label.lower()} checks were skipped.")
        if "not_applicable" in states:
            notes.append(f"Some {label.lower()} checks were unavailable or not applicable.")
        if not states or states == {"unknown"}:
            notes.append(f"{label} check coverage is unavailable in the saved result.")
    visual_only = included == {"visual"}
    return {"scope": "visual_content" if visual_only else "assessed_components",
            "scope_label": "Visual content only" if visual_only else "Assessed components only" if included else "Check coverage not yet available",
            "scope_note": " ".join(notes),
            "excluded_channels": [key for key in ("voice", "audio_picture_timing") if key not in included]}, bool(included)


def _unavailable_contributor(result: Mapping[str, Any]) -> bool:
    rows = [r for r in (result.get("decision") or {}).get("contributions") or [] if r.get("kind") == "detector" and _counted(r)]
    available = {ref for signal in result.get("signals") or [] if signal.get("status") == "ok" for ref in _signal_refs(signal)}
    return any(row.get("signal_id") not in available for row in rows)


def _probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and 0 <= value <= 1 else None


def _pct(value: float) -> float:
    # Do not round 59.99% up to the 60% boundary or print a statistical estimate as 100%.
    return math.floor(round(value * 1000, 8)) / 10


def _video_soundtrack(result: Mapping[str, Any]) -> dict[str, Any] | None:
    if result.get("kind") != "video" or result.get("status") not in (None, "done"):
        return None
    from app.detectors.audio.calibration import NOTE, SHA256

    for signal in result.get("signals") or []:
        if signal.get("status") != "ok" or signal.get("modality") != "audio" or signal.get("scope") != "global":
            continue
        screen = (signal.get("details") or {}).get("audio_screening") or {}
        if signal.get("calibration_ref") != SHA256 or screen.get("calibration_sha256") != SHA256:
            continue
        p = _probability(screen.get("probability_ai")) if screen.get("scope_supported") is True else None
        direction, headline, confidence = "inconclusive", "No reliable AI or original speech tendency", None
        if p is not None and screen.get("decision") == "ai" and p >= THRESHOLD:
            direction, headline, confidence = "ai", "Leaning towards AI-generated speech", min(99.9, _pct(p))
        elif (p is not None and screen.get("decision") == "original" and screen.get("original_stability_veto") is False
                and p <= 1 - THRESHOLD + 1e-12):
            direction, headline, confidence = "original", "Leaning towards original speech", min(99.9, _pct(1 - p))
        elif screen.get("scope_supported") is not True:
            headline = "No calibrated speech tendency for this soundtrack duration"
        return {"direction": direction, "headline": headline, "confidence_pct": confidence,
                "limitation": f"{NOTE} The speech assessment is separate from the visual and audio-picture timing results."}
    return None


def text_reference_rate(kind: str, profile: Mapping[str, Any] | None, calibration: list[dict]) -> str | None:
    """Read the actual screening-threshold result from the saved text-reference evaluation."""
    if kind != "document":
        return None
    context = {
        ARTICLE_REFERENCE_SLICE: ("Held-out English news and articles", "AI articles", "human articles"),
        EMAIL_REFERENCE_SLICE: ("Held-out plain-text business email", "AI-rewritten emails", "human emails"),
    }.get((profile or {}).get("slice_id"))
    if context is None:
        return None
    label, ai, human = context
    for calibration_row in calibration:
        for row in calibration_row.get("error_rates") or []:
            if (row.get("context") == label
                    and row.get("threshold_kind") == "fixed" and row.get("threshold") == THRESHOLD
                    and row.get("customer_rule") == RULE):
                n_ai, n_human = int(row["n_pos"]), int(row["n_neg"])
                flagged_ai = round(float(row["tpr"]) * n_ai)
                flagged_human = round(float(row["fpr"]) * n_human)
                return (f"In the recorded reference comparison, {flagged_ai} of {n_ai} {ai} and {flagged_human} of "
                        f"{n_human} {human} received an AI-writing tendency.")
    return None


def _customer_assessment(result: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only interpretation shared by API, JSON and report presentation.

    ``result`` is an analysis-detail-shaped mapping with decision, condition_profile,
    caveats and optional status. A provenance-led result retains its existing words
    (applies=False), since documentary floors/caps are not statistical confidence.
    """
    decision = result.get("decision") or {}
    profile = result.get("condition_profile") or {}
    soundtrack = _video_soundtrack(result)
    scope, content_assessed = _video_scope(result)
    answer: dict[str, Any] = {
        "rule": RULE, "finding_assessments": finding_assessments(dict(result)), "applies": True,
        "direction": "unavailable", "headline": "No reliable automated conclusion",
        "confidence_pct": None, "confidence_label": None, "probability_ai_pct": None,
        "interval_pct": None, "interval_label": None, "interval_note": INTERVAL_NOTE,
        "meaning": "The analysis did not establish a usable whole-file estimate.",
        "limitation": "Review the recorded observations and any checks that could not run. Missing evidence does not favour either answer.",
        **scope, "service_policy": standard_service_policy(),
    }
    content_findings = [f for f in answer["finding_assessments"] if f["direction"] != "context"]
    included = list(dict.fromkeys(f["scope_label"] for f in content_findings if f["included_in_overall"] is True))
    separate = list(dict.fromkeys(f["scope_label"] for f in content_findings if f["included_in_overall"] is False))
    if included or separate:
        notes = [answer.get("scope_note") or ""]
        if included:
            notes.append("Included finding scopes: " + "; ".join(included) + ".")
        if separate:
            notes.append("Finding scopes assessed separately from this percentage: " + "; ".join(separate) + ".")
        answer["scope_note"] = " ".join(note for note in notes if note)
    if soundtrack is not None:
        answer["soundtrack"] = soundtrack
    from app.reports.view import stored_record

    origin_record = (decision.get("provenance_record") or result.get("stored_record")
                     or stored_record(dict(decision), result.get("assessments"), result.get("signals")))
    if origin_record and origin_record.get("kind") == "self_declared_ai":
        return {**answer, "direction": "inconclusive", "headline": "Inconclusive: unverified origin marker",
                "meaning": "This saved result used a retired rule that let an unverified marker decide the answer. The marker does not authenticate AI generation.",
                "limitation": "The original calculation is retained in the analysis record. Run a new analysis for a content-based assessment; no confidence percentage is assigned to the old marker-led verdict."}
    if origin_record or decision.get("driving_hypothesis") == "H3_provenance_integrity":
        return {**answer, "applies": False, "direction": "record", "headline": None,
                "meaning": "The conclusion follows the file's provenance record, not a confidence percentage."}
    if result.get("status") not in (None, "done") or not decision or decision.get("code") == "not_assessable":
        return answer
    if _unavailable_contributor(result):
        return {**answer, "direction": "inconclusive", "headline": "Inconclusive: contributing checks could not complete",
                "meaning": "A check required by this estimate failed or was unavailable. Review the recorded check coverage."}
    from app.detectors.audio.calibration import NOTE, SHA256

    if result.get("kind") == "audio" and decision.get("fuser_sha256") == SHA256:
        p = _probability(decision.get("probability_ai_calibrated"))
        direction = ("ai" if decision.get("code") == "ai_supported" and p is not None and p >= THRESHOLD else
                     "original" if decision.get("code") == "no_ai_indicators" and p is not None and p <= 1 - THRESHOLD + 1e-12 else "inconclusive")
        return {**answer, "direction": direction,
                "headline": "Leaning towards AI-generated speech" if direction == "ai" else "Leaning towards original speech" if direction == "original" else "Inconclusive: no reliable speech tendency",
                "confidence_pct": min(99.9, _pct(p if direction == "ai" else 1-p)) if direction != "inconclusive" else None,
                "confidence_label": "Reference-calibrated estimate" if direction != "inconclusive" else None,
                "probability_ai_pct": _pct(p) if p is not None else None,
                "meaning": "The recorded speech checks support this tendency only within the assessed speech and its calibrated scope." if direction != "inconclusive" else "The recorded speech checks do not support a reliable directional conclusion in their calibrated scope.", "limitation": NOTE,
                "scope_label": "Speech assessment", "scope_note": "Local experimental audio screening; the percentage concerns speech, not every sound or edit in the recording.",
                "interval_note": "No per-file confidence interval has been established for this audio calibration."}
    if decision.get("fusion_mode") != "artefact":
        return {**answer, "headline": "Inconclusive: no calibrated estimate",
                "meaning": "The available checks provide inspection scores, but those scores have not been calibrated into a reliable whole-file likelihood.",
                "limitation": "An inspection response is not a calibrated confidence percentage. Use the linked evidence to examine the file further."}
    if abstain.outside_validated(decision, profile) or profile.get("within_validated_envelope") is not True:
        return {**answer, "headline": "Inconclusive: this input is outside the verified scope",
                "meaning": "The saved validation does not cover this kind of input well enough to support a reliable combined assessment.",
                "limitation": "A score can still be recorded for inspection, but it cannot support a confidence percentage for this file."}
    p = _probability(decision.get("probability_ai_calibrated"))
    email_reference = result.get("kind") == "document" and profile.get("slice_id") == EMAIL_REFERENCE_SLICE
    if email_reference:
        ci = decision.get("ci90")
        ai = decision.get("code") == "ai_supported" and p is not None and p >= THRESHOLD
        valid_interval = (ai and isinstance(ci, (list, tuple)) and len(ci) == 2
                          and all(_probability(x) is not None for x in ci) and ci[0] <= ci[1])
        return {**answer, "direction": "ai" if ai else "inconclusive",
                "headline": "Leaning towards AI-written or AI-rewritten email text" if ai else "Inconclusive: no reliable AI-writing tendency",
                "confidence_pct": min(99.9, _pct(p)) if ai else None,
                "confidence_label": "Reference-calibrated estimate" if ai else None,
                "probability_ai_pct": _pct(p) if ai else None,
                "interval_pct": [_pct(x) for x in ci] if valid_interval else None,
                "interval_label": "90% reference-fit interval" if valid_interval else None,
                "interval_note": "The range reflects uncertainty in the fitted reference. It does not measure all possible detector errors.",
                "meaning": "The reference-calibrated email-body check supports an AI-writing tendency for the selected message body." if ai else "The recorded email-body check does not establish a reliable authorship tendency.", "limitation": "Some AI rewrites are missed, so this check can leave email authorship inconclusive.",
                "scope_label": "Selected email body",
                "scope_note": "This assessment covers writing in the message body. Sender identity and attachments require separate verification."}
    if p is None:
        return answer
    text_reference = result.get("kind") == "document" and profile.get("slice_id") == ARTICLE_REFERENCE_SLICE
    answer.update(direction="inconclusive", headline="Inconclusive: no reliable overall tendency",
                  probability_ai_pct=_pct(p), meaning="The recorded estimate does not establish a reliable tendency. Review the observations, source links and check coverage.",
                  limitation=ESTIMATE_NOTE)
    if text_reference:
        answer.update(scope_label="Extracted writing",
                      scope_note="Reference based on English news and articles. The estimate covers the extracted writing.",
                      interval_note="The range reflects uncertainty in the fitted reference. It does not measure all possible detector errors.")
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
    if decision.get("code") == "inconclusive" or rule is not None or disagree or not content_assessed:
        return {**answer, "headline": "Inconclusive: the checks do not support a reliable tendency",
                "meaning": "The recorded result remains inconclusive. Review the linked findings and the coverage of checks that could not complete."}
    direction = ("ai" if decision.get("code") in {"ai_supported", "ai_supported_strong"} and p >= THRESHOLD else
                 "original" if decision.get("code") in {"no_ai_indicators", "no_ai_indicators_strong"} and p <= 1 - THRESHOLD + 1e-12 else None)
    if direction is None:
        if p >= THRESHOLD or p <= 1 - THRESHOLD + 1e-12:
            return {**answer, "headline": "Inconclusive: the checks do not support a reliable tendency",
                    "meaning": "The recorded decision does not support the direction suggested by the estimate."}
        return answer
    ai = direction == "ai"
    answer.update(direction=direction,
                  headline="Leaning towards AI-generated or AI-altered content" if ai else "Leaning towards original content in the assessed scope",
                  confidence_pct=min(99.9, _pct(p if ai else 1 - p)), confidence_label="Reference-calibrated overall estimate",
                  meaning="The applicable, reference-calibrated checks lean towards AI generation or alteration. Review the supporting findings and their scope."
                  if ai else "The applicable, reference-calibrated checks lean towards original content in the assessed scope. Local flags and checks outside the estimate require separate consideration.")
    if not ai and answer["interval_pct"] is not None:
        answer.update(interval_pct=[_pct(1 - float(ci[1])), _pct(1 - float(ci[0]))],
                      interval_label="90% interval for original-file likelihood")
    if text_reference:
        answer.update(headline="Leaning towards AI-generated or AI-rewritten text" if ai else "Leaning towards original writing",
                      confidence_label="Reference-calibrated estimate", meaning="The reference-calibrated writing check supports this tendency for the assessed writing, not the authorship of every highlighted passage.",
                      interval_label="90% reference-fit interval" if answer["interval_pct"] is not None else None)
    if answer["scope"] != "whole_file":
        subject = "visual content" if answer["scope"] == "visual_content" else "content in the assessed components"
        answer["headline"] = (f"Leaning towards AI-generated or AI-altered {subject}" if ai
                              else f"Leaning towards original {subject}")
        answer["meaning"] = (f"The reference-calibrated checks included in this estimate lean "
                             f"towards {'AI generation or alteration of' if ai else 'original'} {subject}. "
                             "Local flags and excluded checks require separate consideration.")
        if answer["interval_pct"] is not None:
            answer["interval_label"] = "90% interval for this scoped estimate"
    return answer


def customer_assessment(result: Mapping[str, Any]) -> dict[str, Any]:
    from app.engine.image_assessment_scope import image_scope

    return image_scope(result, _customer_assessment(result))
