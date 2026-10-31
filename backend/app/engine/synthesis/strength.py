"""Evidential strength and review priority of a finding (IF §6.1–6.2, §6.4; D-16).

- **Likelihood-ratio bands** on the natural-log LR of a signal: strong |LR| ≥ 100, moderate 10–100, weak 2–10, none below 2.
  The band edges are the IF §6.2 proposal, "to be confirmed (ENFSI)" (D-16). The direction is the sign.
- **Decisive** is documentary only: a C2PA declaration validated against the trust list (Stage 0).
- **Family caps** (IF §6.1): F4 localisation is moderate alone and strong when corroborated by an independent family with
  IoU ≥ 0.3, which upgrades one level (IF §6.4). Families IF does not cover follow their closest analogue: learned lanes
  (V3 synthetic video, A1 voice, T1 text) like F3, face tracks and lip-sync like F10, codec like F5.
- **Uncalibrated scores** (the fallback path, or a detector outside the fuser artefact) get an LR from the raw probability
  p/(1−p) and are capped at moderate: without calibration a stated strong band would not be a measured likelihood ratio.
- **Review priority** is triage, separate from strength (IF §6.1). Findings against the headline hypothesis are low when at
  least moderate and info otherwise, so strong benign evidence does not alarm.
"""
from __future__ import annotations

import math

STRENGTHS = ("none", "weak", "moderate", "strong", "decisive")
RANK = {s: i for i, s in enumerate(STRENGTHS)}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
LN2, LN10, LN100 = math.log(2.0), math.log(10.0), math.log(100.0)
UNCALIBRATED_CAP = "moderate"
FAMILY_CAP = {
    "F1_provenance": "strong", "F2_self_declared_metadata": "strong", "F3_whole_image_generation": "strong",
    "F4_localised_manipulation": "moderate", "F5_compression_history": "moderate", "F6_capture_pipeline": "moderate",
    "F7_duplication": "moderate", "F8_metadata_consistency": "moderate", "F9_distribution_conditions": "none",
    "F10_face": "moderate", "F11_physical_scene": "none", "V1_face_track": "moderate", "V2_lip_sync": "moderate",
    "V3_synthetic_video": "strong", "V4_codec": "moderate", "A1_voice": "strong", "T1_ai_text": "strong", "coverage": "none",
}
CORROBORATED_CAP = {"F4_localised_manipulation": "strong", "F7_duplication": "strong", "F5_compression_history": "moderate"}
HIGH_FAMILIES = frozenset({"F1_provenance", "F2_self_declared_metadata", "F3_whole_image_generation", "F4_localised_manipulation"})
REGION_HIGH_FAMILIES = frozenset({"F4_localised_manipulation", "F7_duplication"})


def lower(a: str, b: str) -> str:
    return a if RANK[a] <= RANK[b] else b


def band(log_lr: float) -> str:
    magnitude = abs(log_lr)
    if magnitude >= LN100:
        return "strong"
    if magnitude >= LN10:
        return "moderate"
    return "weak" if magnitude >= LN2 else "none"


def direction(log_lr: float | None, strength: str) -> str:
    if log_lr is None or strength == "none":
        return "neutral"
    return "for" if log_lr > 0 else "against"


def cap(strength: str, family: str, *, corroborated: bool = False, uncalibrated: bool = False) -> str:
    limit = CORROBORATED_CAP.get(family, FAMILY_CAP.get(family, "moderate")) if corroborated else FAMILY_CAP.get(family, "moderate")
    if uncalibrated:
        limit = lower(limit, UNCALIBRATED_CAP)
    return lower(strength, limit)


def upgrade(strength: str) -> str:
    """One level up, never beyond strong (decisive is documentary only)."""
    if strength in ("none", "strong", "decisive"):
        return strength
    return STRENGTHS[RANK[strength] + 1]


def priority(family: str, strength: str, direction_: str, *, c2pa_invalid: bool = False, region_on_face: bool = False,
             weak_pair: bool = False) -> str:
    if c2pa_invalid:
        return "high"
    if family in ("F9_distribution_conditions", "coverage") or strength == "none":
        return "info"
    if direction_ == "against":
        return "low" if RANK[strength] >= RANK["moderate"] else "info"
    if strength in ("decisive", "strong") and family in HIGH_FAMILIES:
        return "high"
    if family in REGION_HIGH_FAMILIES and region_on_face and RANK[strength] >= RANK["moderate"]:
        return "high"
    if RANK[strength] >= RANK["moderate"] or weak_pair:
        return "medium"
    return "low"
