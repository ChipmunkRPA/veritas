"""Read-only finding interpretations. Never turn raw scores or fusion effects into confidence."""
from __future__ import annotations

import math
import re

VERSION = "finding-interpretation/v1"
CONTEXT = {"coverage", "F1_provenance", "F2_self_declared_metadata", "F5_compression_history",
           "F6_capture_pipeline", "F7_duplication", "F8_metadata_consistency", "F9_distribution_conditions",
           "F11_physical_scene", "V2_lip_sync", "V4_codec"}
SCOPES = {"F3_whole_image_generation": "Whole-image generation indicators",
          "F4_localised_manipulation": "Local editing indicators", "F10_face": "Assessed face content",
          "V1_face_track": "Sampled face content", "V3_synthetic_video": "Sampled visual content",
          "A1_voice": "Speech in the assessed recording", "T1_ai_text": "Assessed writing"}
LABELS = {"ai": "Leans AI", "original": "Leans original", "inconclusive": "Inconclusive", "context": "Context only"}


def _probability(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def _pct(value):
    return min(99.9, math.floor(round(value * 1000, 8)) / 10)


def _refs(finding):
    return set(finding.get("evidence_ids") or []) | {
        ref for sentence in finding.get("sentences") or [] for ref in sentence.get("evidence_ids") or []}


def linked_signals(finding, result):
    refs = _refs(finding)
    return [signal for signal in result.get("signals") or []
            if refs.intersection({signal.get("id"), (signal.get("legacy") or {}).get("v1_id"),
                                  f"detector:{signal.get('detector_id')}"})]


def finding_assessment(finding: dict, result: dict) -> dict:
    family = finding.get("family")
    signals = linked_signals(finding, result)
    scope = finding.get("scope")
    scope_label = SCOPES.get(family, "Recorded observation")
    if isinstance(scope, dict):
        scope_label = ("Selected image region" if scope.get("kind") in {"bbox", "polygon", "mask"} else
                       "Selected time interval" if "start_s" in scope else "Selected face track")
    elif len(signals) == 1:
        scope_label = {"window": "Selected recording window", "segment": "Selected time interval",
                       "sentence": "Selected writing context", "region": "Selected image region",
                       "face": "Selected face input", "track": "Selected face track"}.get(signals[0].get("scope"), scope_label)
    rows = (result.get("decision") or {}).get("contributions")
    refs = {ref for s in signals for ref in (s.get("id"), (s.get("legacy") or {}).get("v1_id"), f"detector:{s.get('detector_id')}") if ref}
    included = None if rows is None else any(
        r.get("kind") == "detector" and r.get("signal_id") in refs and r.get("counted") is not False
        and (r.get("counted") is True or (isinstance(r.get("logit"), (int, float)) and abs(r["logit"]) > 1e-9)) for r in rows)
    base = {"version": VERSION, "finding_id": finding.get("id"), "direction": "inconclusive",
            "included_in_overall": included,
            "label": LABELS["inconclusive"], "confidence_pct": None, "confidence_label": None,
            "scope_label": scope_label,
            "confidence_note": "A calibrated confidence percentage has not been established for this finding. Raw responses and contribution strengths are not AI probabilities.",
            "interpretation": "The recorded evidence does not establish an AI or original tendency for this finding."}
    from app.engine.local_editing_assessment import local_editing_assessment

    local = local_editing_assessment(finding, result, signals, base)
    if local is not None:
        return local
    learned = any(s.get("category") == "model" or str(s.get("family") or "").startswith("learned-")
                  or (family == "A1_voice" and (s.get("details") or {}).get("audio_screening")) for s in signals)
    if family in CONTEXT or (signals and not learned):
        return {**base, "direction": "context", "label": LABELS["context"], "included_in_overall": False,
                "interpretation": "This records provenance, timing, file history or conditions. It does not by itself establish AI generation or original authorship."}
    if result.get("status") not in (None, "done") or not signals or any(s.get("status", "ok") != "ok" for s in signals):
        return {**base, "interpretation": "The checks needed for this finding are missing, unavailable or incomplete. Missing evidence does not favour original content."}
    standalone = finding.get("standalone_direction") or finding.get("direction")
    strength = finding.get("standalone_strength") or finding.get("strength")
    individual = {(s.get("legacy") or {}).get("direction") for s in signals}
    if {"supports_ai", "supports_authentic"}.issubset(individual) or finding.get("dissent_note"):
        return {**base, "interpretation": "The linked checks disagree. Their responses are not averaged into a finding confidence."}
    direction = {"for": "ai", "against": "original"}.get(standalone) if strength not in (None, "none") else None
    if direction:
        base.update(direction=direction, label=LABELS[direction],
                    interpretation="The recorded check provides an AI-leaning inspection clue; it does not prove AI origin."
                    if direction == "ai" else "The recorded check leans original in its assessed scope. Absence of an AI flag does not prove the file is original.")
    # A whole-input calibration must never be borrowed for a crop, word span or interval.
    if len(signals) != 1 or finding.get("scope") is not None or signals[0].get("scope", "global") != "global":
        return base
    signal = signals[0]
    p = signal.get("calibrated_probability")
    ref = signal.get("calibration_ref")
    supported = (result.get("condition_profile") or {}).get("within_validated_envelope") is True
    calibrated = False
    audio = (signal.get("details") or {}).get("audio_screening")
    if family == "A1_voice" and isinstance(audio, dict):
        from app.detectors.audio.calibration import SHA256
        from app.detectors.audio.reference import COMBINED_SHA256, combined_tendency

        if ref == COMBINED_SHA256:
            p, direction = combined_tendency(audio)
            supported = direction in {"ai", "original"}
        else:
            supported = (audio.get("scope_supported") is True and ref == SHA256
                         and audio.get("calibration_sha256") == SHA256)
            audio_direction = audio.get("decision")
            direction = audio_direction if audio_direction in {"ai", "original"} else None
            if direction == "original" and audio.get("original_stability_veto") is not False:
                direction = None
            p = audio.get("probability_ai")
        if not direction or not supported:
            return {**base, "direction": "inconclusive", "label": LABELS["inconclusive"],
                    "interpretation": "The saved speech calibration does not support a reliable tendency for this recording."}
        base.update(direction=direction, label=LABELS[direction])
        base["interpretation"] = ("The calibrated speech check leans AI within the assessed recording."
                                  if direction == "ai" else "The calibrated speech check leans original within the assessed recording.")
        calibrated = True
        if result.get("kind") == "audio" and (result.get("decision") or {}).get("fuser_sha256") == ref:
            base["included_in_overall"] = True
    else:
        subject = f"{signal.get('detector_id')}@{signal.get('detector_version')}"
        calibrated = any(c.get("subject") == subject and c.get("method") in {"platt", "isotonic"}
                         and ref == f"{subject}/{c.get('bench_version')}" and c.get("bench_version")
                         and re.fullmatch(r"[0-9a-f]{64}", str(c.get("artefact_sha256") or ""))
                         and re.fullmatch(r"[0-9a-f]{64}", str(c.get("bench_manifest_sha256") or ""))
                         for c in result.get("calibration") or [])
    if (not calibrated or not supported or not direction or not _probability(p)
            or (signal.get("reliability") or {}).get("factor", 1) == 0):
        return base
    if (direction == "ai" and p <= .5) or (direction == "original" and p >= .5):
        return {**base, "direction": "inconclusive", "label": LABELS["inconclusive"],
                "interpretation": "The recorded tendency and calibrated estimate disagree. No finding confidence is assigned."}
    return {**base, "confidence_pct": _pct(p if direction == "ai" else 1 - p),
            "confidence_label": "Reference-calibrated estimate for this check",
            "confidence_note": "This estimate applies to the stated check and assessed input. It is not measured accuracy for this file or a probability for every highlighted pixel, frame, word or sound."}


def finding_assessments(result: dict) -> list[dict]:
    return [finding_assessment(finding, result) for finding in result.get("findings") or []
            if re.fullmatch(r"F[1-9]\d*", str(finding.get("id") or ""))]
