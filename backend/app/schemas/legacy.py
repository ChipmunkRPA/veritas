"""v1 ↔ v2 result adapter (ARCHITECTURE §4.9).

`upgrade_v1(result, kind, …)` turns a stored v0.3 result into `AnalysisResultV2`:
- each v1 signal → `SignalV2` (`detector_id = "v1.<id>"`, scope `global`, `calibrated_probability = ai_probability`,
  `effective_weight = weight`); v1-only values (id, value, direction, strength, decisive) go to `SignalV2.legacy`, and a
  decisive signal maps to a provenance override;
- the verdict → a fallback `Decision`: the v1 band and confidence, the uncalibrated probability, a code capped below
  `*_strong`, and the UX §7.1 headline as `verbal` (D-16); plus `Assessment(H1, applicable=False)` "v1: hypotheses not
  separated" and a `legacy-heuristic` calibration record;
- artefact files → `ImageExhibit(role="legacy")` at `legacy:<analysis_id>/<name>` (hashed when the file is available);
  `faces[]` → `RegionsExhibit`, `frames[]` → `SeriesExhibit`, `sentences[]` → `TextSpansExhibit`;
- every other v1 block (label, summary, reasoning, caveats, counts, timings, metadata, image, video, frames, text, …) is
  kept verbatim in `LegacyResult`, so the v1 UI, report and export keep reading exactly what they read before.

`downgrade_v2(result)` produces the v1 document. For an upgraded row it is exact. For a v2-native result it derives the v1
shape: verdict from `Decision` and findings, signal mirrors from probabilities, `faces` from the first regions exhibit,
`frames` from the first series exhibit with a `p_frame` series, `sentences` from the first text-spans exhibit.
"""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from app.core.evidence import BAND_LABELS, Signal
from app.core.rounding import half_up
from app.engine.detector import Hypothesis, Modality
from app.schemas.evidence import (
    AnalysisResultV2,
    Assessment,
    BBox,
    CalibrationInfo,
    Decision,
    ImageExhibit,
    InputRef,
    LegacyResult,
    LegacySignalFields,
    Region,
    RegionsExhibit,
    ReproManifest,
    ScoredRegion,
    Series,
    SeriesExhibit,
    SignalV2,
    TextSpan,
    TextSpansExhibit,
)

V1_KINDS = ("image", "video", "document")
KIND_MODALITY = {"image": Modality.IMAGE, "video": Modality.VIDEO, "document": Modality.TEXT}
KIND_MEDIA_REF = {"image": "analysis_copy", "video": "frame", "document": "document"}
VERDICT_EXTRA_KEYS = ("label", "summary", "reasoning", "caveats", "signals_for_ai", "signals_for_authentic", "signals_total")
BAND_CODE = {"very_likely_ai": "ai_supported", "likely_ai": "ai_supported", "uncertain": "inconclusive",
             "likely_authentic": "no_ai_indicators", "very_likely_authentic": "no_ai_indicators"}  # fallback: never *_strong
BAND_HEADLINE = {  # UX §7.1 L1 headlines (D-16)
    "very_likely_ai": "Strong indications of AI generation or manipulation",
    "likely_ai": "Indications of AI generation or manipulation",
    "uncertain": "Inconclusive: the checks don't point clearly either way",
    "likely_authentic": "No significant indications of AI generation or manipulation",
    "very_likely_authentic": "No indications of AI generation or manipulation found",
}
FALLBACK_CAVEAT = "Uncalibrated fallback: probability is not a frequency estimate"
# L-3 (D-31): an uncalibrated statistical fallback never publishes a "very likely" band or high confidence, in the v2 decision,
# the v1 view, the JSON download or the PDF; documentary overrides (H3) keep theirs.
FALLBACK_BAND = {"very_likely_ai": "likely_ai", "very_likely_authentic": "likely_authentic"}


def fallback_cap(band: str, confidence: str) -> tuple[str, str]:
    return FALLBACK_BAND.get(band, band), ("medium" if confidence == "high" else confidence)


def uncalibrated_score(probability: float) -> str:
    """How an uncalibrated fallback number is printed wherever it appears (D-68): a score, never a likelihood or a percentage."""
    return f"score {half_up(probability * 100)} / 100, not calibrated"


def fallback_reason(subject: str, band: str, confidence: str, verdict: Mapping[str, Any]) -> str:
    """The sentence a statistical fallback result says about itself, in `decision.reason` and in the v1 `verdict.summary`
    (N-7): the published band and confidence, which are the capped ones, and the number as an uncalibrated score. `verdict`
    is the legacy fuser's (its number and signal counts)."""
    for_ai, for_authentic = int(verdict.get("signals_for_ai") or 0), int(verdict.get("signals_for_authentic") or 0)
    return (f"The {subject} is assessed as \u201c{BAND_LABELS[band]}\u201d ({uncalibrated_score(verdict['probability_ai'])}; "
            f"confidence {confidence}). {for_ai} of {int(verdict.get('signals_total') or 0)} informative signals support AI "
            f"involvement; {for_authentic} {'supports' if for_authentic == 1 else 'support'} authenticity.")
NOT_ASSESSABLE_LABEL = "Not assessable"
V1_NOTE = "v1: hypotheses not separated"
LEGACY_PREFIX = "legacy:"
H3_VALUE = "H3_provenance_integrity"


class LegacyFormatError(ValueError):
    """The stored document is not a valid v1 result."""


def _safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.name != name or name in (".", ".."):
        raise LegacyFormatError(f"unsafe artefact name {name!r}")
    return name


def _file_sha256(artifact_dir: Path | None, name: str) -> str | None:
    if artifact_dir is None:
        return None
    path = artifact_dir / name
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upgrade_v1(result: dict[str, Any], kind: str, *, analysis_id: str, input_ref: InputRef | None = None,
               artifact_dir: Path | None = None) -> AnalysisResultV2:
    if kind not in V1_KINDS:
        raise LegacyFormatError(f"unknown v1 analysis kind {kind!r}")
    verdict = result.get("verdict")
    if not isinstance(verdict, dict) or verdict.get("band") not in BAND_CODE:
        raise LegacyFormatError("a v1 result needs verdict.band with a v1 band value")
    try:
        v1_signals = [Signal.model_validate(raw) for raw in result.get("signals") or []]
    except ValidationError as e:
        raise LegacyFormatError(f"invalid v1 signal: {e.errors()[0]['msg']}") from None

    signals: list[SignalV2] = []
    for n, s in enumerate(v1_signals, 1):
        detector_id = f"v1.{s.id}"
        signal_id = f"{detector_id}:global" if all(x.detector_id != detector_id for x in signals) else f"{detector_id}:{n}"
        override = None
        if s.decisive:
            override = {"supports_ai": "decisive_ai", "supports_authentic": "trusted_capture"}.get(s.direction)
        signals.append(SignalV2(
            id=signal_id, display_id=f"S{n:02d}", detector_id=detector_id, detector_version="0.3", name=s.name, family="v1",
            modality=KIND_MODALITY[kind], category=s.category, effective_weight=s.weight, raw_score=s.ai_probability,
            calibrated_probability=s.ai_probability, observed=s.observed, reference=s.reference, reasoning=s.reasoning,
            details=s.details, exhibit_ids=[LEGACY_PREFIX + _safe_name(a) for a in s.artifacts], provenance_override=override,
            calibration_ref="v1-fuse@0.3/legacy-heuristic",
            legacy=LegacySignalFields(v1_id=s.id, value=s.value, direction=s.direction, strength=s.strength, decisive=s.decisive)))

    names = list(dict.fromkeys([*(result.get("artifacts") or []), *(a for s in v1_signals for a in s.artifacts)]))
    exhibits: list[Any] = []
    for i, name in enumerate(names, 1):
        exhibits.append(ImageExhibit(
            id=LEGACY_PREFIX + _safe_name(name), display_id=f"Fig. {i}", title=name, created_by="v1",
            media_ref=KIND_MEDIA_REF[kind], role="legacy", payload_artefact_id=f"{LEGACY_PREFIX}{analysis_id}/{name}",
            sha256=_file_sha256(artifact_dir, name),
            source_signal_ids=[sv2.id for sv2, s in zip(signals, v1_signals) if name in s.artifacts]))
    exhibits += _legacy_exhibits(result, kind)

    band = verdict["band"]  # the recorded v0.3 band: the upgrade is a faithful view of the stored record
    probability = verdict.get("probability_ai")
    decision = Decision(code=BAND_CODE[band], verbal=BAND_HEADLINE[band], probability_ai_uncalibrated=probability, band=band,
                        confidence=verdict.get("confidence") or "low", reason=verdict.get("summary") or "", fusion_mode="fallback")
    try:
        return AnalysisResultV2(
            analysis_id=analysis_id, kind=kind, input=input_ref,
            manifest=ReproManifest(veritas_version="0.3", pipeline_version=f"v1-{kind}", profile="v1"),
            decision=decision,
            assessments=[Assessment(hypothesis=Hypothesis.H1_WHOLE_CONTENT_SYNTHESIS, applicable=False, note=V1_NOTE)],
            signals=signals, exhibits=exhibits,
            calibration=[CalibrationInfo(subject="v1-fuse@0.3", method="legacy-heuristic")],
            caveats=[FALLBACK_CAVEAT],
            legacy=LegacyResult(verdict={k: copy.deepcopy(verdict[k]) for k in VERDICT_EXTRA_KEYS if k in verdict},
                                extras={k: copy.deepcopy(v) for k, v in result.items() if k not in ("verdict", "signals")}))
    except ValidationError as e:
        raise LegacyFormatError(f"the v1 result does not map onto schema v2: {e.errors()[0]['msg']}") from None


def _legacy_exhibits(result: dict[str, Any], kind: str) -> list[Any]:
    exhibits: list[Any] = []
    faces = result.get("faces") or []
    if kind == "image" and faces:
        regions = []
        for i, entry in enumerate(faces, 1):
            box = entry.get("face") or {}
            regions.append(ScoredRegion(display_id=f"R{i}", label=f"face {i}", probability=entry.get("p_ai"),
                                        region=Region(kind="bbox", bbox=BBox(x=box.get("x", 0), y=box.get("y", 0),
                                                                             w=box.get("w", 0), h=box.get("h", 0)))))
        exhibits.append(RegionsExhibit(id="legacy:faces", title="Detected faces", created_by="v1.faces", media_ref="analysis_copy",
                                       regions=regions))
    frames = result.get("frames") or []
    if kind == "video" and frames:
        series = [Series(name=key, points=[(float(f.get("time") or 0.0), f.get(key)) for f in frames])
                  for key in ("p_frame", "p_face", "p_general")]
        exhibits.append(SeriesExhibit(id="legacy:frames", title="Per-frame scores", created_by="v1.frames", media_ref="frame",
                                      x_unit="s", series=series))
    sentences = result.get("sentences") or []
    if kind == "document" and sentences:
        spans = [TextSpan(start=s.get("start", 0), end=s.get("end", 0), sentence_idx=s.get("idx", i), probability=s.get("p_ai"),
                          label=s.get("band") or "", reasons=list(s.get("reasons") or []))
                 for i, s in enumerate(sentences)]
        exhibits.append(TextSpansExhibit(id="legacy:sentences", title="Sentence scores", created_by="v1.sentences",
                                         media_ref="document", spans=spans))
    return exhibits


def _signal_v1(s: SignalV2) -> dict[str, Any]:
    legacy = s.legacy
    prob = s.calibrated_probability if s.calibrated_probability is not None else (
        s.raw_score if s.raw_kind == "probability" else None)
    # N-19p (browser QA Q6-01; `reports.view.is_mention`): a mention of AI in a text field sets no number and points neither
    # way, in the v1 view and `export.json` as in the ledger (it gave the check's score, 0.65, and "supports_ai")
    mention = s.detector_id == "img.meta.generator" and s.provenance_override is None and float((s.metrics or {}).get("free_text_hits") or 0) > 0
    if legacy is not None:
        direction, strength, value, decisive = legacy.direction, legacy.strength, legacy.value, legacy.decisive
    else:
        direction = Signal.direction_from_prob(prob) if prob is not None else "inconclusive"
        strength = Signal.strength_from_prob(prob) if prob is not None else "none"
        value, decisive = None, s.provenance_override == "decisive_ai"
    if mention:
        prob, direction, strength = None, "neutral", "none"
    return {
        "id": legacy.v1_id if legacy is not None and legacy.v1_id else s.id, "name": s.name, "category": s.category,
        "observed": s.observed, "value": value, "reference": s.reference, "ai_probability": prob,
        "weight": s.effective_weight if s.effective_weight is not None else 1.0, "direction": direction, "strength": strength,
        "reasoning": s.reasoning, "artifacts": [e[len(LEGACY_PREFIX):] for e in s.exhibit_ids if e.startswith(LEGACY_PREFIX)],
        "details": copy.deepcopy(s.details), "decisive": decisive,
    }


def v1_label(code: str | None, band: str) -> str:
    """D-66 (N-18): the label of the v1 view and of `export.json`, from the published v2 decision after every cap: the
    label of its band, and "Not assessable" for a result that could not be assessed (its band is the middle of the scale,
    which would read "Inconclusive")."""
    return NOT_ASSESSABLE_LABEL if code == "not_assessable" else BAND_LABELS[band]


def downgrade_v2(result: AnalysisResultV2) -> dict[str, Any]:
    decision = result.decision
    signals = [_signal_v1(s) for s in result.signals]
    verdict: dict[str, Any] = {"probability_ai": decision.probability_ai, "band": decision.band, "confidence": decision.confidence}
    if result.legacy is not None:
        verdict.update(copy.deepcopy(result.legacy.verdict))
        out = copy.deepcopy(result.legacy.extras)  # a v0.3 record reproduces its stored v1 document (no L-3 read-time cap)
    else:
        if decision.fusion_mode == "fallback" and decision.driving_hypothesis != H3_VALUE:
            verdict["band"], verdict["confidence"] = fallback_cap(decision.band, decision.confidence)
        verdict.update(
            label=v1_label(decision.code, verdict["band"]), summary=decision.verbal,
            reasoning=[f.statement for f in sorted(result.findings, key=lambda f: (f.rank, f.id))] or [decision.reason],
            caveats=list(result.caveats),
            signals_for_ai=sum(1 for s in signals if s["direction"] == "supports_ai"),
            signals_for_authentic=sum(1 for s in signals if s["direction"] == "supports_authentic"),
            signals_total=len(signals))
        out = _derive_v1_blocks(result)
    out["verdict"] = verdict
    out["signals"] = signals
    return out


def _derive_v1_blocks(result: AnalysisResultV2) -> dict[str, Any]:
    out: dict[str, Any] = {"artifacts": [e.id[len(LEGACY_PREFIX):] for e in result.exhibits
                                         if isinstance(e, ImageExhibit) and e.role == "legacy" and e.id.startswith(LEGACY_PREFIX)],
                           "timings": {}}
    profile = result.condition_profile
    if result.kind == "image":
        regions = next((e for e in result.exhibits if isinstance(e, RegionsExhibit)), None)
        out["faces"] = [{"face": {"x": r.region.bbox.x, "y": r.region.bbox.y, "w": r.region.bbox.w, "h": r.region.bbox.h},
                         "scores": {}, **({"p_ai": r.probability} if r.probability is not None else {})}
                        for r in (regions.regions if regions else []) if r.region.bbox is not None]
        out["metadata"] = {}
        out["image"] = {"width": profile.width if profile else None, "height": profile.height if profile else None}
    elif result.kind == "video":
        series_exhibit = next((e for e in result.exhibits if isinstance(e, SeriesExhibit)
                               and any(s.name == "p_frame" for s in e.series)), None)
        frames: list[dict[str, Any]] = []
        if series_exhibit is not None:
            by_name = {s.name: s.points or [] for s in series_exhibit.series}
            for i, (t, p) in enumerate(by_name["p_frame"]):
                frames.append({"index": i, "time": t, "p_frame": p,
                               "p_face": by_name["p_face"][i][1] if i < len(by_name.get("p_face", [])) else None,
                               "p_general": by_name["p_general"][i][1] if i < len(by_name.get("p_general", [])) else None,
                               "faces": 0, "scores": {}})
        out["frames"] = frames
        out["video"] = {"duration": profile.duration_s if profile else None, "width": profile.width if profile else None,
                        "height": profile.height if profile else None}
    elif result.kind == "document":
        spans = next((e for e in result.exhibits if isinstance(e, TextSpansExhibit)), None)
        out["sentences"] = [{"idx": s.sentence_idx, "start": s.start, "end": s.end, "p_ai": s.probability, "band": s.label,
                             "reasons": list(s.reasons), "scores": {}} for s in (spans.spans if spans else [])]
    return out
