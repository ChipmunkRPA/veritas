"""Shared evidence model.

Every detector emits `Signal`s. A Signal is a single, independently auditable observation:
what was measured, what the reference/expected range is, which way it points, how strong it
is, and the plain-language reasoning. The fusion layer combines Signals into a `Verdict`,
and the report renders Signals verbatim so a reviewer can trace each conclusion to evidence.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Category = Literal["model", "forensic", "metadata", "provenance", "temporal", "stylometric", "statistical", "face"]
Direction = Literal["supports_ai", "supports_authentic", "neutral", "inconclusive"]
Strength = Literal["decisive", "strong", "moderate", "weak", "none"]
Band = Literal["very_likely_authentic", "likely_authentic", "uncertain", "likely_ai", "very_likely_ai"]
Confidence = Literal["low", "medium", "high"]


class Signal(BaseModel):
    id: str
    name: str
    category: Category
    observed: str = Field(description="Human-readable observed value, e.g. '0.93' or 'Software=Midjourney'")
    value: float | None = None
    reference: str = Field(description="Expected range / baseline the observation is compared against")
    ai_probability: float | None = Field(
        default=None, ge=0, le=1, description="This signal's stand-alone estimate that the content is AI-made"
    )
    weight: float = 1.0
    direction: Direction = "neutral"
    strength: Strength = "none"
    reasoning: str = Field(description="Why this observation supports the direction stated")
    artifacts: list[str] = Field(default_factory=list, description="Artifact file names (heatmaps, plots, crops)")
    details: dict[str, Any] = Field(default_factory=dict)
    decisive: bool = Field(default=False, description="Legacy marker retained for compatibility; it does not authorize a statistical override. Trusted provenance is verified separately.")

    @staticmethod
    def strength_from_prob(p: float) -> Strength:
        d = abs(p - 0.5)
        if d >= 0.42:
            return "strong"
        if d >= 0.25:
            return "moderate"
        if d >= 0.10:
            return "weak"
        return "none"

    @staticmethod
    def direction_from_prob(p: float) -> Direction:
        if p >= 0.60:
            return "supports_ai"
        if p <= 0.40:
            return "supports_authentic"
        return "neutral"


class Verdict(BaseModel):
    probability_ai: float | None = Field(ge=0, le=1)  # None when the decision publishes no estimate (D-49)
    band: Band
    confidence: Confidence
    label: str
    summary: str
    reasoning: list[str] = Field(default_factory=list, description="Ordered, evidence-referenced reasoning steps")
    caveats: list[str] = Field(default_factory=list)
    signals_for_ai: int = 0
    signals_for_authentic: int = 0
    signals_total: int = 0


def band_for(p: float) -> Band:
    if p < 0.15:
        return "very_likely_authentic"
    if p < 0.35:
        return "likely_authentic"
    if p < 0.65:
        return "uncertain"
    if p < 0.85:
        return "likely_ai"
    return "very_likely_ai"


# The labels of the v1 view, of `export.json` and of the fallback's sentence. D-66 (N-18): a label never claims
# authenticity: the two clean bands read as the v2 headlines do (`schemas.legacy.BAND_HEADLINE`)
BAND_LABELS: dict[Band, str] = {
    "very_likely_authentic": "No indications of AI generation or manipulation found",
    "likely_authentic": "No significant indications of AI generation or manipulation",
    "uncertain": "Inconclusive",
    "likely_ai": "Likely AI-generated / manipulated",
    "very_likely_ai": "Very likely AI-generated / manipulated",
}
