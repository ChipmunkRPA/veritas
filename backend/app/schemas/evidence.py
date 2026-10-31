"""Evidence data model v2 (ARCHITECTURE §4.1–4.5; PRD Appendix B; D-15, D-26).

- Geometry is in original-media pixel coordinates; times are pts with a `time_base`, plus seconds for display.
- Exhibits form a discriminated union on `kind`, so OpenAPI produces proper TS unions.
- Stable display ids (D-26: `S01…` signals, `F1…` findings, `Fig. n` exhibits, `R1…` regions, `Seg. a…` segments) are
  persisted at synthesis time; internal ids (`{detector_id}:{scope_ref}`) stay the database keys.
- `AnalysisResultV2` checks referential integrity: unique ids, signal ↔ exhibit references and finding evidence ids.
- `legacy` fields hold v1-only values verbatim so `downgrade_v2(upgrade_v1(x))` reproduces a v1 document.

Additions to ARCHITECTURE §4 (recorded in PROGRESS.md):
- `Decision.probability_ai_uncalibrated` carries the fused probability in `fusion_mode="fallback"`, where
  `probability_ai_calibrated` must stay None (an uncalibrated score is not a frequency estimate).
- `ExhibitBase.sha256` and `CalibrationInfo.artefact_sha256` may be None for legacy exhibits and the legacy heuristic.
- `Finding.sentences` holds the template output as `{text, evidence_ids}` (§4.6). Evidence ids name signals, exhibits and
  assessment hypotheses, and also `decision`, `condition_profile` and `detector:<id>` for a manifest detector (M-4: coverage
  and validation sentences cite what they describe).
- `Decision.contributions` holds the union fuser's exact rows (context prior, detectors, missing, overrides; EA-06).
"""
from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.canonical import FLOAT_DP
from app.core.evidence import Band, Category, Confidence, Direction, Strength
from app.engine.detector import DetectorStatus, Hypothesis, Lane, Modality, ProvenanceOverride, RawKind

SCHEMA_VERSION = "2.0"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TIME_BASE = re.compile(r"[1-9]\d*/[1-9]\d*")
_FINDING_ID = re.compile(r"F[1-9]\d*")
_SIGNAL_DISPLAY = re.compile(r"S\d{2,}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- 4.1 geometry and time ---------------------------------------------------------------------------------

class BBox(_Model):
    x: float
    y: float
    w: float = Field(ge=0)
    h: float = Field(ge=0)


class CoordinateSpace(_Model):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    scale_to_original: float = Field(gt=0)


class Region(_Model):
    kind: Literal["bbox", "polygon", "mask"]
    bbox: BBox | None = None
    polygon: list[tuple[float, float]] | None = None
    mask_exhibit_id: str | None = None
    frame_pts: int | None = None  # video: which frame
    track_id: str | None = None
    landmarks: list[tuple[float, float]] | None = None  # face landmarks in source px (YuNet: eyes, nose, mouth corners)

    @model_validator(mode="after")
    def _shape(self) -> Region:
        if self.kind == "bbox" and self.bbox is None:
            raise ValueError("a bbox region needs bbox")
        if self.kind == "polygon" and (self.polygon is None or len(self.polygon) < 3):
            raise ValueError("a polygon region needs at least 3 points")
        if self.kind == "mask" and not self.mask_exhibit_id:
            raise ValueError("a mask region needs mask_exhibit_id")
        return self


class TimeSegment(_Model):
    display_id: str | None = None  # D-26 "Seg. a"
    start_pts: int
    end_pts: int
    time_base: str  # e.g. "1/30000"
    start_s: float
    end_s: float
    channel: Literal["visual", "audio", "av"]
    track_id: str | None = None
    peak: float = Field(ge=0, le=1)  # calibrated lane probability at the peak
    lane: Lane | None = None
    hypothesis: Hypothesis
    sources: list[str] = Field(default_factory=list)  # signal ids that produced the segment

    @model_validator(mode="after")
    def _ordered(self) -> TimeSegment:
        if not _TIME_BASE.fullmatch(self.time_base):
            raise ValueError(f"time_base {self.time_base!r} must look like 1/30000")
        if self.end_pts < self.start_pts or self.end_s < self.start_s:
            raise ValueError("a segment must not end before it starts")
        return self


class FaceTrackRef(_Model):
    track_id: str
    first_pts: int
    last_pts: int
    n_frames: int = Field(ge=0)
    identity: str | None = None  # V-2 appearance cluster letter ("A"), shown as "Face A"; None when not established


# --- 4.2 signal --------------------------------------------------------------------------------------------

class Reliability(_Model):
    factor: float = Field(default=1.0, ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    inputs: dict[str, float | bool | str | None] = Field(default_factory=dict)  # face_px, jpeg_q, upscale, snr, duration_s


class LegacySignalFields(_Model):
    """v1 values with no v2 field, kept so a v1 signal downgrades exactly."""

    v1_id: str | None = None  # the v1 signal id (the demo tour and the ledger address signals by it)
    value: float | None = None
    direction: Direction = "neutral"
    strength: Strength = "none"
    decisive: bool = False


class OutsideEstimate(_Model):
    """N-19k (browser QA Q2-12): why a check that scored carries no weight in the estimate of this result, as the fusion
    recorded it for this input: the check is not among the checks of the fusion model; its context gate is closed (`gate`:
    the gate's literals, "!has_face"); or the model holds it, with its gate open, and its weight is zero here."""
    reason: Literal["not_in_model", "gate", "zero_weight"]
    gate: list[str] = Field(default_factory=list)


class SignalV2(_Model):
    id: str  # "{detector_id}:{scope_ref}", unique per analysis
    display_id: str | None = None  # D-26 "S01"
    detector_id: str
    detector_version: str
    model_revisions: dict[str, str] = Field(default_factory=dict)
    name: str
    family: str
    modality: Modality
    category: Category  # kept for the v1 view
    status: DetectorStatus = "ok"
    status_reason: str | None = None
    scope: Literal["global", "face", "track", "segment", "region", "window", "sentence"] = "global"
    scope_ref: str | None = None
    hypothesis: Hypothesis | None = None  # None for informational signals (JPEG quality, low resolution)
    lane: Lane | None = None
    threat: list[str] = Field(default_factory=list)
    track_id: str | None = None
    segments: list[TimeSegment] = Field(default_factory=list)
    fp_modes: list[str] = Field(default_factory=list)
    contribution_logit: float | None = None
    contribution_pp: float | None = None
    effective_weight: float | None = None  # also fills the v1 `weight` mirror
    outside_estimate: OutsideEstimate | None = None  # N-19k: None for a check in the estimate, without a score, or stored before
    raw_score: float | None = None
    raw_kind: RawKind = "probability"
    calibrated_probability: float | None = Field(default=None, ge=0, le=1)
    log_lr: float | None = None
    reliability: Reliability = Field(default_factory=Reliability)
    observed: str = ""
    reference: str = ""
    reasoning: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)  # free-form; never used by fusion
    exhibit_ids: list[str] = Field(default_factory=list)
    provenance_override: ProvenanceOverride | None = None  # D-15; v1 mirror: decisive
    calibration_ref: str | None = None  # "{detector_id}@{version}/{calibration_set}"
    legacy: LegacySignalFields | None = None

    @model_validator(mode="after")
    def _display(self) -> SignalV2:
        if self.display_id is not None and not _SIGNAL_DISPLAY.fullmatch(self.display_id):
            raise ValueError(f"signal display id {self.display_id!r} must look like S01 (D-26)")
        return self


# --- 4.3 exhibits ------------------------------------------------------------------------------------------

class ExhibitBase(_Model):
    id: str
    display_id: str | None = None  # D-26 "Fig. 3"
    title: str
    caption: str = ""
    source_signal_ids: list[str] = Field(default_factory=list)
    created_by: str  # detector id or stage id
    media_ref: Literal["original", "analysis_copy", "frame", "audio", "document"]
    frame_pts: int | None = None
    coordinate_space: CoordinateSpace | None = None
    sha256: str | None = None  # of the canonical payload; None only for legacy exhibits whose payload is unavailable

    @model_validator(mode="after")
    def _hash(self) -> ExhibitBase:
        if self.sha256 is not None and not _HEX64.fullmatch(self.sha256):
            raise ValueError("exhibit sha256 must be 64 lower-case hex characters")
        return self


class TileSet(_Model):
    tile_px: int = 512
    levels: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    manifest_artefact_id: str


class HeatmapExhibit(ExhibitBase):
    kind: Literal["heatmap"] = "heatmap"
    semantics: Literal["p_manipulated", "attribution", "residual", "error_level"]
    calibrated: bool  # True only if values are calibrated probabilities
    grid: tuple[int, int]  # width, height of the stored map
    value_range: tuple[float, float]  # PNG 0..255 ↔ value_range (linear)
    payload_artefact_id: str  # 8-bit grayscale PNG
    reliability_mask_artefact_id: str | None = None  # 8-bit PNG, 255 = reliable
    preview_artefact_id: str  # colormapped PNG for PDF and thumbnails
    tiles: TileSet | None = None  # for source images > 4096 px


class ScoredRegion(_Model):
    display_id: str | None = None  # D-26 "R1"
    region: Region
    probability: float | None = Field(default=None, ge=0, le=1)
    score: float | None = None
    label: str = ""


class RegionsExhibit(ExhibitBase):
    kind: Literal["regions"] = "regions"
    regions: list[ScoredRegion]


class TrackFrame(_Model):
    pts: int
    frame_index: int
    bbox: BBox
    landmarks: list[tuple[float, float]] | None = None
    det_score: float | None = None
    still_artefact_id: str | None = None


class FaceTrackExhibit(ExhibitBase):
    kind: Literal["face_track"] = "face_track"
    track: FaceTrackRef
    frames: list[TrackFrame]


class Series(_Model):
    name: str
    hypothesis: Hypothesis | None = None
    track_id: str | None = None
    points: list[tuple[float, float | None]] | None = None  # inline ≤ 5 000 points
    payload_artefact_id: str | None = None

    @model_validator(mode="after")
    def _one_payload(self) -> Series:
        if (self.points is None) == (self.payload_artefact_id is None):
            raise ValueError("a series has exactly one of points or payload_artefact_id")
        if self.points is not None and len(self.points) > 5000:
            raise ValueError("inline series hold at most 5 000 points; use payload_artefact_id")
        return self


class SeriesExhibit(ExhibitBase):
    kind: Literal["series"] = "series"
    x_unit: Literal["s", "pts", "sentence", "frequency"]
    series: list[Series]
    thresholds: list[float] = Field(default_factory=list)


class SegmentsExhibit(ExhibitBase):
    kind: Literal["segments"] = "segments"
    lane: str
    segments: list[TimeSegment]


class ImageExhibit(ExhibitBase):
    kind: Literal["image"] = "image"
    # thumbnail: embedded by a C2PA signer (I-2); comparison: side-by-side panels with a difference image (I-3)
    role: Literal["still", "crop", "contact_sheet", "spectrum", "thumbnail", "comparison", "legacy"]
    payload_artefact_id: str


class TableExhibit(ExhibitBase):
    kind: Literal["table"] = "table"
    columns: list[str]
    rows: list[list[str | float | None]]

    @model_validator(mode="after")
    def _rectangular(self) -> TableExhibit:
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("every table row must have one cell per column")
        return self


class TextSpan(_Model):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    sentence_idx: int = Field(ge=0)
    probability: float | None = Field(default=None, ge=0, le=1)
    label: str = ""
    reasons: list[str] = Field(default_factory=list)


class TextSpansExhibit(ExhibitBase):
    kind: Literal["text_spans"] = "text_spans"
    spans: list[TextSpan]


Exhibit = Annotated[HeatmapExhibit | RegionsExhibit | FaceTrackExhibit | SeriesExhibit | SegmentsExhibit | ImageExhibit
                    | TableExhibit | TextSpansExhibit, Field(discriminator="kind")]


# --- 4.4 fusion, calibration, abstention -------------------------------------------------------------------

class Contribution(_Model):
    signal_id: str | None = None  # None for the starting row and "missing:<detector>" rows
    label: str
    # The starting row is `neutral_start` since N-17 (D-63): the estimate with every scoring check at its neutral point, in the
    # input's context. Results published earlier keep `context_prior`: the model with every check at 50 %.
    kind: Literal["detector", "missing", "context_prior", "neutral_start", "override"]
    logit: float
    pp: float  # percentage-point effect (ML §4.6)
    approximate: bool = False
    # What a scoring check's row is measured from (D-63): its neutral point, or 50 % for a check without a usable stand-alone
    # calibration. None for the other rows and for results published before N-17.
    baseline: Literal["neutral_point", "p50"] | None = None
    # N-19i (R5-01): whether the row of a check is one of the checks the decision counts ("k of n detectors", D-64, D-76):
    # false for a row of zero and for the row of a check that declares H3, which is shown and not counted. None for the
    # other rows and for results published before N-19i, whose rows count when they are not zero.
    counted: bool | None = None


class ConditionProfile(_Model):
    width: int | None = None
    height: int | None = None
    jpeg_quality_est: int | None = None
    double_compression: bool | None = None
    screenshot_likelihood: float | None = None
    min_face_px: int | None = None
    median_face_px: int | None = None
    duration_s: float | None = None
    n_frames: int | None = None
    speech_share: float | None = None
    music_share: float | None = None
    language: str | None = None
    slice_id: str
    within_validated_envelope: bool
    validation_support: Literal["direct", "pooled", "unsupported", "legacy"] | None = None
    validation_measured_at: str | None = None
    limiting_conditions: list[str] = Field(default_factory=list)


class ErrorRate(_Model):
    context: str
    threshold_kind: Literal["conformal_0.05", "conformal_0.01", "fixed"]
    threshold: float
    tpr: float = Field(ge=0, le=1)
    fpr: float = Field(ge=0, le=1)
    tpr_ci95: tuple[float, float]
    fpr_ci95: tuple[float, float]
    n_pos: int = Field(ge=0)
    n_neg: int = Field(ge=0)
    bench_version: str  # e.g. "vb-1.0"


# Historical benchmark aggregates, never reconstructed from an individual analysis.
_Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
_Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]


class ReliabilityBin(_Model):
    bin: int = Field(ge=0, strict=True)
    n: int = Field(gt=0, strict=True)
    weight: _Probability
    mean_s: _Probability
    mean_y: _Probability
    s_min: _Probability
    s_max: _Probability

    @model_validator(mode="after")
    def _range(self) -> ReliabilityBin:
        if self.weight <= 0 or not self.s_min <= self.mean_s <= self.s_max:
            raise ValueError("invalid reliability bin range or weight")
        return self


class ReliabilityDiagram(_Model):
    version: Literal["veritas.reliability-diagram/1"]
    report_sha256: _Digest
    report_generated_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    population: Literal["image/test", "audio/test", "video/test"]
    bench_version: str = Field(min_length=1)
    manifest_sha256: _Digest
    scope_sha256: _Digest
    evaluated_artefact_sha256: _Digest
    published_artefact_sha256: _Digest
    publication_record_sha256: _Digest
    publication_binding: Literal["documented_metrics_publication"]
    view: Literal["size_matched", "full"]
    variant: Literal["norm", "native"]
    n_real: int = Field(gt=0, strict=True)
    n_fake: int = Field(gt=0, strict=True)
    ece: _Probability
    ece_ci95: tuple[_Probability, _Probability]
    ece_bootstrap_replicates: int = Field(gt=0, strict=True)
    ece_bootstrap_valid: int = Field(gt=0, strict=True)
    ece_interval_method: Literal["grouped_bootstrap_percentile"]
    weighting: Literal["class_balanced"]
    binning: Literal["equal_mass"]
    score_semantics: Literal["unclamped_fuser"]
    bins: list[ReliabilityBin] = Field(min_length=15, max_length=15)
    limitation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _aggregates(self) -> ReliabilityDiagram:
        if [b.bin for b in self.bins] != list(range(15)):
            raise ValueError("reliability bins must be the 15 ordered recorded bins")
        if sum(b.n for b in self.bins) != self.n_real + self.n_fake:
            raise ValueError("reliability bin counts disagree with population")
        # Canonical evidence rounds each float to FLOAT_DP decimals. For N bins,
        # sum(weights) can change by N*eps. A weighted gap can change by
        # N*eps + 2*eps*(1 + N*eps), plus eps for the stored ECE itself.
        eps = 0.5 * 10 ** -FLOAT_DP
        weight_error = len(self.bins) * eps
        ece_error = weight_error + 2 * eps * (1 + weight_error) + eps
        if not math.isclose(sum(b.weight for b in self.bins), 1.0, abs_tol=weight_error + 1e-12, rel_tol=0):
            raise ValueError("reliability bin weights do not sum to one")
        ece = sum(b.weight * abs(b.mean_y - b.mean_s) for b in self.bins)
        if not math.isclose(ece, self.ece, abs_tol=ece_error + 1e-12, rel_tol=0):
            raise ValueError("reliability bins disagree with recorded ECE")
        if self.ece_ci95[0] > self.ece_ci95[1] or self.ece_bootstrap_valid > self.ece_bootstrap_replicates:
            raise ValueError("invalid recorded ECE interval")
        if self.variant != ("norm" if self.population == "image/test" else "native"):
            raise ValueError("reliability population/variant mismatch")
        if (self.population == "image/test") != (self.view == "size_matched"):
            raise ValueError("reliability population/view mismatch")
        if any(a.s_max > b.s_min for a, b in zip(self.bins, self.bins[1:])):
            raise ValueError("reliability bins are not in score order")
        return self


class CalibrationInfo(_Model):
    subject: str  # detector id@version or fuser-<modality>-<semver>
    method: Literal["platt", "isotonic", "legacy-heuristic"]
    artefact_sha256: str | None = None
    bench_version: str | None = None
    bench_manifest_sha256: str | None = None
    ece: float | None = None
    brier: float | None = None
    error_rates: list[ErrorRate] = Field(default_factory=list)
    reviewed_by: str | None = None
    review_date: str | None = None  # N-19d: the ISO date of the review record that names this artefact; null while its review is pending
    test_view: Literal["size_matched"] | None = None  # J-3d: the test view of `ece` / `brier`; null when the artefact recorded none
    reliability_diagram: ReliabilityDiagram | None = None

    @model_validator(mode="after")
    def _diagram_identity(self) -> CalibrationInfo:
        diagram = self.reliability_diagram
        if diagram is not None and (diagram.published_artefact_sha256 != self.artefact_sha256
                                    or diagram.bench_version != self.bench_version
                                    or diagram.manifest_sha256 != self.bench_manifest_sha256
                                    or diagram.ece != self.ece
                                    or (diagram.view == "size_matched") != (self.test_view == "size_matched")):
            raise ValueError("reliability diagram does not belong to calibration record")
        return self


class LaneAssessment(_Model):
    lane: Lane
    probability: float = Field(ge=0, le=1)
    band: Band
    track_id: str | None = None
    segments: list[TimeSegment] = Field(default_factory=list)
    contributions: list[Contribution] = Field(default_factory=list)


class Assessment(_Model):
    hypothesis: Hypothesis
    applicable: bool
    probability: float | None = Field(default=None, ge=0, le=1)  # None for H3
    ci90: tuple[float, float] | None = None
    lanes: list[LaneAssessment] = Field(default_factory=list)  # H2 only
    c2pa_state: Literal["well-formed", "valid", "trusted", "invalid", "absent"] | None = None  # H3 only
    watermarks: list[dict[str, Any]] = Field(default_factory=list)  # H3 only
    contributions: list[Contribution] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def _hypothesis_fields(self) -> Assessment:
        h3 = self.hypothesis == Hypothesis.H3_PROVENANCE_INTEGRITY
        if h3 and self.probability is not None:
            raise ValueError("H3 provenance is documentary: it carries no probability")
        if not h3 and (self.c2pa_state is not None or self.watermarks):
            raise ValueError("c2pa_state and watermarks belong to H3")
        if self.lanes and self.hypothesis != Hypothesis.H2_LOCALISED_MANIPULATION:
            raise ValueError("lanes belong to H2")
        return self


DecisionCode = Literal["ai_supported_strong", "ai_supported", "inconclusive", "no_ai_indicators", "no_ai_indicators_strong",
                       "not_assessable"]


class ProvenanceRecord(_Model):
    """N-19h (E-21): the provenance record a decision applied, where the provenance hypothesis (H3) drives it. The page and
    the reports take the basis of the result from here, not from its findings. `kind`: the override, by precedence
    (declaration, capture, marker). `marker`: for a self-declared record, what it is (`decisive.MARKER_KINDS`). `signal_ids`:
    the signals of the record; their findings entered the decision."""
    kind: ProvenanceOverride
    marker: Literal["metadata", "watermark", "credentials_listed_signer", "credentials_unlisted_signer_declared",
                    "credentials_unlisted_signer_named", "metadata_declared", "metadata_declared_composite",
                    "metadata_parameters"] | None = None
    signal_ids: list[str] = Field(default_factory=list)


class StatisticalAlone(_Model):
    """N-19l (code review R6-05; D-68 as amended): what the statistical checks alone decided, in a result that a
    provenance record decides beside a statistical estimate. The page and the reports take the form of the number from
    here where the record did not set it: the large numeral only where the checks alone decide. `code`: the code the
    result would have without the record. `unsupported`: D-76 took a decisive code of the estimate away, and the side
    of the answer that no check supports (toward AI generation or manipulation, or away from it). `rule` (N-19m; D-77,
    D-76 as amended): the condition that took the decisive code, as the stage's output names it: `scale` (D-77: the
    estimate stands outside the part of the scale of the levels' answer: in the middle part, or under a configured prior
    in the other answer's; no side), `no_check` (D-76: no single check supports the answer),
    `together_less` and `together_against` (D-76: a check supports it, and the checks together give it less than weak
    support, or point the other way). A result stored before N-19m names no condition and reads as `no_check`."""
    code: DecisionCode
    unsupported: Literal["for", "against"] | None = None
    rule: Literal["scale", "no_check", "together_less", "together_against"] | None = None


class Decision(_Model):
    code: DecisionCode
    verbal: str
    probability_ai_calibrated: float | None = Field(default=None, ge=0, le=1)
    probability_ai_uncalibrated: float | None = Field(default=None, ge=0, le=1)  # fallback mode only
    ci90: tuple[float, float] | None = None
    band: Band
    confidence: Confidence
    driving_hypothesis: Hypothesis | None = None
    # N-19h: the record that decides, when H3 drives; None for every other result and absent in results stored earlier
    provenance_record: ProvenanceRecord | None = None
    # N-19l: what the statistical checks alone decided, where a record decides beside a statistical estimate of a fusion
    # model; None for every other result and absent in results stored earlier
    statistical_alone: StatisticalAlone | None = None
    reason: str
    statistical_estimate_before_provenance: float | None = None
    disagreement: float | None = None
    fusion_mode: Literal["artefact", "fallback"]
    fuser_sha256: str | None = None
    contributions: list[Contribution] = Field(default_factory=list)  # union fuser rows; Σ logit = logit s (EA-06)

    @model_validator(mode="after")
    def _mode(self) -> Decision:
        if self.fusion_mode == "fallback":
            if self.code.endswith("_strong"):
                raise ValueError("the uncalibrated fallback never issues a *_strong decision (ARCHITECTURE §4.4)")
            if self.probability_ai_calibrated is not None:
                raise ValueError("the fallback probability is not calibrated: use probability_ai_uncalibrated")
        elif self.fuser_sha256 is None or not _HEX64.fullmatch(self.fuser_sha256):
            raise ValueError("an artefact decision names its fuser_sha256")
        return self

    @property
    def probability_ai(self) -> float | None:
        """The probability shown to v1 readers: calibrated when available, else the fallback estimate."""
        return self.probability_ai_calibrated if self.probability_ai_calibrated is not None else self.probability_ai_uncalibrated


# --- 4.5 findings ------------------------------------------------------------------------------------------

FindingFamily = Literal["F1_provenance", "F2_self_declared_metadata", "F3_whole_image_generation", "F4_localised_manipulation",
                        "F5_compression_history", "F6_capture_pipeline", "F7_duplication", "F8_metadata_consistency",
                        "F9_distribution_conditions", "F10_face", "F11_physical_scene", "V1_face_track", "V2_lip_sync",
                        "V3_synthetic_video", "V4_codec", "A1_voice", "T1_ai_text", "coverage"]


class EvidenceSentence(_Model):
    text: str
    evidence_ids: list[str] = Field(min_length=1)  # EA-07: every sentence carries evidence ids


class ExaminerLayer(_Model):
    """Joined at read time from annotations (RW-01); never mutates the machine layer. Statuses arrive with C-4/C-5."""

    status: str | None = None
    note: str | None = None
    annotation_ids: list[str] = Field(default_factory=list)


class Finding(_Model):
    id: str  # D-26 "F1…"; the family stays a field
    family: FindingFamily
    hypothesis: Hypothesis | None = None
    lane: Lane | None = None
    strength: Strength  # evidential, LR bands (IF §6.2)
    direction: Literal["for", "against", "neutral"] | None = None  # toward AI generation or manipulation (UX §3, §5.4); None in older results
    review_priority: Literal["high", "medium", "low", "info"]
    rank: int = Field(ge=1)
    title: str
    statement: str
    sentences: list[EvidenceSentence] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    scope: Region | TimeSegment | FaceTrackRef | None = None
    probability: float | None = Field(default=None, ge=0, le=1)
    contribution_logit: float | None = None
    corroborated_by: list[str] = Field(default_factory=list)
    reference_range: str = ""
    benchmark_note: str | None = None
    fp_modes: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    dissent_note: str | None = None
    examiner: ExaminerLayer | None = None
    # N-19q (finding 1; D-63): for a check that entered an estimate measured from neutral points, its stand-alone words (its
    # own likelihood ratio, "Taken alone, the score is …"); `strength` and `direction` are then its part in the estimate.
    # None for any other finding and in results published before N-19q
    standalone_strength: Strength | None = None
    standalone_direction: Literal["for", "against", "neutral"] | None = None

    @model_validator(mode="after")
    def _id(self) -> Finding:
        if not _FINDING_ID.fullmatch(self.id):
            raise ValueError(f"finding id {self.id!r} must look like F1 (D-26)")
        return self


# --- the result document -----------------------------------------------------------------------------------

class InputRef(_Model):
    sha256: str
    md5: str | None = None
    size_bytes: int = Field(ge=0)
    mime_sniffed: str | None = None
    filename: str | None = None
    intake_event_id: str | None = None

    @model_validator(mode="after")
    def _hash(self) -> InputRef:
        if not _HEX64.fullmatch(self.sha256):
            raise ValueError("input sha256 must be 64 lower-case hex characters")
        return self


class DetectorAdmission(_Model):  # the card's admission when the analysis ran (PRD RP-04)
    decision: str
    date: str | None = None


class DetectorRef(_Model):
    id: str
    version: str
    admission: DetectorAdmission | None = None
    weights_sha256: str | None = None
    revision: str | None = None
    licence: str | None = None
    coverage_date: str | None = None
    device: str | None = None
    precision: str | None = None


class ReproManifest(_Model):  # §6.6 subset carried in the result
    veritas_version: str
    pipeline_version: str
    profile: str
    effective_profile: str | None = None
    licence_posture: str | None = None
    disclosures: list[str] = Field(default_factory=list)  # D-34: e.g. allowlisted components with unreviewed terms
    git_commit: str | None = None
    config_sha256: str | None = None
    models_manifest_sha256: str | None = None  # the model cards the recorded admissions were read from
    device: str | None = None
    precision: str | None = None
    template_set: str | None = None
    detectors: list[DetectorRef] = Field(default_factory=list)
    calibration_id: str | None = None
    policy_thresholds_id: str | None = None
    library_versions: dict[str, str] = Field(default_factory=dict)
    sampling: dict[str, Any] = Field(default_factory=dict)


class SliceMetrics(_Model):
    tpr: float | None = None
    fpr: float | None = None
    auc: float | None = None
    n_pos: int = 0
    n_neg: int = 0
    tpr_ci95: tuple[float, float] | None = None
    fpr_ci95: tuple[float, float] | None = None


class ValidationRef(_Model):
    dossier_version: str
    slice_metrics: SliceMetrics | None = None


class ReviewSummary(_Model):  # filled by C-5
    status: str
    conclusion: dict[str, Any] | None = None
    signoffs: list[dict[str, Any]] = Field(default_factory=list)


class ReportRef(_Model):
    id: str
    revision: int = Field(ge=1)
    profile: Literal["summary", "standard", "forensic", "newsroom"]  # D-10
    sha256: str
    signature: str | None = None


class LegacyResult(_Model):
    """v1 blocks without a v2 counterpart, kept verbatim for `downgrade_v2`."""

    verdict: dict[str, Any] = Field(default_factory=dict)  # label, summary, reasoning, caveats, signal counts
    extras: dict[str, Any] = Field(default_factory=dict)  # per-kind v1 blocks: image, metadata, video, frames, text, …


class AnalysisResultV2(_Model):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    analysis_id: str
    case_id: str | None = None
    revision: int = Field(default=1, ge=1)
    kind: Literal["image", "video", "document", "audio"]
    input: InputRef | None = None
    manifest: ReproManifest
    condition_profile: ConditionProfile | None = None
    decision: Decision
    assessments: list[Assessment] = Field(default_factory=list)
    signals: list[SignalV2] = Field(default_factory=list)
    exhibits: list[Exhibit] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    calibration: list[CalibrationInfo] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    validation_ref: ValidationRef | None = None
    review: ReviewSummary | None = None
    report: ReportRef | None = None
    legacy: LegacyResult | None = None

    @model_validator(mode="after")
    def _references(self) -> AnalysisResultV2:
        signal_ids = _unique("signal", [s.id for s in self.signals])
        exhibit_ids = _unique("exhibit", [e.id for e in self.exhibits])
        _unique("finding", [f.id for f in self.findings])
        _unique("signal display", [s.display_id for s in self.signals if s.display_id])
        _unique("exhibit display", [e.display_id for e in self.exhibits if e.display_id])
        _unique("assessment", [a.hypothesis.value for a in self.assessments])
        for signal in self.signals:
            missing = set(signal.exhibit_ids) - exhibit_ids
            if missing:
                raise ValueError(f"signal {signal.id} references unknown exhibits {sorted(missing)}")
        for exhibit in self.exhibits:
            missing = set(exhibit.source_signal_ids) - signal_ids
            if missing:
                raise ValueError(f"exhibit {exhibit.id} references unknown signals {sorted(missing)}")
        known = (signal_ids | exhibit_ids | {a.hypothesis.value for a in self.assessments} | {"decision"}
                 | {f"detector:{d.id}" for d in self.manifest.detectors}
                 | ({"condition_profile"} if self.condition_profile is not None else set()))
        finding_ids = {f.id for f in self.findings}
        for finding in self.findings:
            referenced = set(finding.evidence_ids) | {i for s in finding.sentences for i in s.evidence_ids}
            if referenced - known:
                raise ValueError(f"finding {finding.id} cites unknown evidence {sorted(referenced - known)}")
            if set(finding.corroborated_by) - finding_ids:
                raise ValueError(f"finding {finding.id} is corroborated by unknown findings")
        return self


def _unique(what: str, ids: list[str]) -> set[str]:
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate {what} id {i!r}")
        seen.add(i)
    return seen
