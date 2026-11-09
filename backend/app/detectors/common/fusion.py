"""Evidence fusion.

The legacy fallback averages statistical log-odds with hand-set weights. This is
not an empirically calibrated ensemble, and different categories need not be
independent. Metadata/provenance records do not enter the statistical score.
Trusted explicit-AI provenance is handled by the staged provenance gate, which
verifies trust and exact-file binding rather than accepting a Signal flag.
"""
from __future__ import annotations

import math

from app.core.config import settings
from app.core.evidence import BAND_LABELS, Signal, Verdict, band_for


def _logit(p: float) -> float:
    p = min(max(p, settings.prob_floor), settings.prob_ceiling)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def fuse(signals: list[Signal], *, subject: str = "content", min_informative: int = 2) -> Verdict:
    # every weighted probability takes part in the log-odds average (a near-0.5 signal contributes ~0 logit);
    # strength only affects how the evidence is described, never whether it is counted
    informative = [s for s in signals if s.ai_probability is not None and s.weight > 0
                   and s.category not in {"metadata", "provenance"}]

    if informative:
        num = sum(s.weight * _logit(s.ai_probability) for s in informative)
        den = sum(s.weight for s in informative)
        p = _sigmoid(num / den)
    else:
        p = 0.5

    reasoning: list[str] = []
    caveats: list[str] = []

    # --- agreement / confidence (near-neutral signals neither agree nor disagree) ---
    decided = [s for s in informative if s.strength != "none"]
    for_ai = [s for s in decided if s.direction == "supports_ai"]
    for_auth = [s for s in decided if s.direction == "supports_authentic"]
    families = {s.category for s in decided}
    n = len(decided)
    if n < min_informative:
        confidence = "low"
    else:
        probs = [s.ai_probability for s in decided]
        spread = max(probs) - min(probs)
        agree = max(len(for_ai), len(for_auth)) / max(n, 1)
        if agree >= 0.75 and spread < 0.6 and len(families) >= 2 and abs(p - 0.5) > 0.25:
            confidence = "high"
        elif agree >= 0.6 and abs(p - 0.5) > 0.15:
            confidence = "medium"
        else:
            confidence = "low"

    p = min(max(p, settings.prob_floor), settings.prob_ceiling)
    band = band_for(p)

    # --- reader-facing reasoning: a few sentences that add information; per-signal detail stays in signals[] ---
    ranked = sorted(informative, key=lambda s: abs(s.ai_probability - 0.5) * s.weight, reverse=True)
    strong_ai = [s for s in ranked if s.direction == "supports_ai" and s.strength in ("strong", "decisive")]
    strong_auth = [s for s in ranked if s.direction == "supports_authentic" and s.strength in ("strong", "decisive")]

    def _short(x: Signal) -> str:
        name = x.name
        if name.startswith("Classifier — "):
            name = name.split(" — ", 1)[1].rsplit("/", 1)[-1]  # model id without the org prefix
        elif " — " in name:
            name = name.split(" — ", 1)[0]
        return name.split(" (")[0].strip()

    def _names(sigs: list[Signal], k: int = 3) -> str:
        parts = [f"{_short(x)} ({x.ai_probability:.2f})" for x in sigs[:k]]
        if not parts:
            return ""
        return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]

    if strong_ai:
        reasoning.append(f"The strongest evidence for AI generation or manipulation comes from {_names(strong_ai)}.")
    if strong_auth:
        reasoning.append(f"Evidence for authenticity comes from {_names(strong_auth)}.")
    if p >= 0.5 and for_auth and not strong_auth:
        reasoning.append("The signals pointing to authenticity are weak or concern a different hypothesis and do not outweigh the positive findings.")
    if p < 0.5 and for_ai and not strong_ai:
        reasoning.append("The signals pointing to AI are weak and are outweighed by the authentic-leaning evidence.")
    if strong_ai and strong_auth:
        reasoning.append(f"Strong signals disagree ({len(for_ai)} for AI, {len(for_auth)} for authentic); the combined score uses declared weights, whose reliability has not been established for this setting.")
    if ranked and not (strong_ai or strong_auth):
        reasoning.append(f"No single detector is decisive; the estimate aggregates {n} moderate or weak signals, led by {_names(ranked, 2)}.")
    if n < min_informative:
        caveats.append("Few informative content checks were available; treat the score as indicative only.")
    if any(s.category in {"metadata", "provenance"} for s in signals):
        caveats.append("Metadata and parsed provenance records are advisory here; they do not change the content score or establish originality.")
    caveats.append(
        "This combined score uses hand-set weights and is not a calibrated probability of AI authorship. "
        "Its confidence label describes signal agreement, not measured accuracy. Human review is required before any consequential decision."
    )

    summary = (
        f"The {subject} is assessed as \u201c{BAND_LABELS[band]}\u201d "
        f"(uncalibrated screening score {p:.0%}, signal-agreement confidence {confidence}). "
        f"{len(for_ai)} of {n} informative signals support AI involvement; {len(for_auth)} {'supports' if len(for_auth) == 1 else 'support'} authenticity."
    )
    return Verdict(
        probability_ai=round(p, 4),
        band=band,
        confidence=confidence,
        label=BAND_LABELS[band],
        summary=summary,
        reasoning=reasoning,
        caveats=caveats,
        signals_for_ai=len(for_ai),
        signals_for_authentic=len(for_auth),
        signals_total=n,
    )


def prob_to_signal_fields(p: float) -> dict:
    return {
        "ai_probability": round(p, 4),
        "direction": Signal.direction_from_prob(p),
        "strength": Signal.strength_from_prob(p),
    }


def fuse_probability(pairs: list[tuple[float, float]]) -> float:
    """Weighted log-odds average of (probability, weight) pairs. Returns 0.5 if empty."""
    pairs = [(p, w) for p, w in pairs if w > 0]
    if not pairs:
        return 0.5
    num = sum(w * _logit(p) for p, w in pairs)
    den = sum(w for _, w in pairs)
    return _sigmoid(num / den)
