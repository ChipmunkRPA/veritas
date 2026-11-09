"""Explain recorded local editing observations without assigning authorship probabilities."""
from __future__ import annotations

import math


def _rule(signal: dict) -> dict:
    value = (signal.get("details") or {}).get("localisation")
    return value if isinstance(value, dict) else {}


def local_editing_assessment(finding: dict, result: dict, signals: list[dict], base: dict) -> dict | None:
    if finding.get("family") != "F4_localised_manipulation" or not signals:
        return None
    regional = [s for s in signals if s.get("family") == "localiser"
                or _rule(s).get("schema") == "recorded-mask/1"]
    physical = [s for s in signals if (s.get("details") or {}).get("display_mapping_version")
                and any(k in (s.get("metrics") or {}) for k in ("ela_mean", "noise_std"))]
    if not regional and not physical:
        return None
    answer = {**base, "version": "finding-interpretation/v2", "included_in_overall": False}
    if result.get("status") not in (None, "done") or any(s.get("status", "ok") != "ok" for s in signals):
        return {**answer, "interpretation": "The local editing checks for this finding are incomplete or unavailable. Missing evidence does not favour original content."}
    if regional:
        eligible, withheld = [], False
        for signal in regional:
            rule = _rule(signal)
            gate = rule.get("candidate_eligibility")
            count = (signal.get("metrics") or {}).get("regions")
            if gate is not None and gate.get("eligible") is not True:
                withheld = True
                continue
            if (isinstance(count, int | float) and not isinstance(count, bool)
                    and math.isfinite(count) and count >= 0 and float(count).is_integer()):
                eligible.append(count)
        if any(count > 0 for count in eligible):
            return {**answer, "direction": "context", "label": "Possible local editing", "scope_label": "Selected image regions",
                    "interpretation": "Local editing cues were recorded in the selected regions. Inspect the linked original crops, measured masks and excluded areas. Ordinary processing can produce similar cues; whether AI caused them is inconclusive. The response is not an AI probability."}
        if withheld:
            return {**answer, "scope_label": "Local editing check coverage",
                    "interpretation": "A selection rule withheld this check's candidate regions. The measured response is retained, but withheld regions are not cleared as original and do not establish an absence of editing."}
        if len(eligible) != len(regional):
            return {**answer, "interpretation": "The saved local editing evidence does not establish usable region coverage. It cannot support an AI or original conclusion."}
        return {**answer, "direction": "context", "label": "No selected editing region",
                "interpretation": "No region met the recorded local editing selection rules. This check can miss edits; it does not establish that the image is unaltered or original."}
    return {**answer, "direction": "context", "label": "Compression and noise context",
            "interpretation": "The linked maps measure compression differences or residual noise variation. Inspect the corresponding source regions. Texture, lighting, re-saving, sharpening and denoising can cause these differences without AI editing; these measurements do not establish AI origin."}
