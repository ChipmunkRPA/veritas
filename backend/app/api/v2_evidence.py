"""API v2 read routes (ARCHITECTURE §5.2 read set): analyses, stages, detector runs, signals, findings, assessments,
exhibits with their binaries and tiles, stills, and resource tokens (§5.3).

- **JSON** is `no-store`. Collections are `{items, next_cursor}`: `/analyses` pages over `(created_at, id)`, newest first;
  per-analysis collections page over their natural order (signal id, finding rank, exhibit id, run start).
- **Binaries** (exhibit payload, mask, preview, tiles, stills) are re-hashed before they are served. They carry
  `ETag: "<sha256>"` and `Cache-Control: private, max-age=31536000, immutable`, and a matching `If-None-Match` gets 304.
  A missing or corrupt object is a 500 problem and is never served.
- **v1-pipeline analyses** (`result_schema_version = 1`) have no evidence rows. Their detail and collections come from
  `upgrade_v1` of the stored v1 result; their legacy exhibits have no binary routes (the v1 artifact route serves them).
- **Signals are read-only** (RW-01): other methods on the collection return 405.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Path, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, FiniteFloat
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.api import tokens
from app.api.auth import Principal
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import RequestModel
from app.api.v2 import (
    DEFAULT_LIMIT,
    NO_STORE,
    Cursor,
    Limit,
    Page,
    ProblemError,
    decode_cursor,
    encode_cursor,
    paginate,
    problem_responses,
)
from app.cases import access, intake
from app.core import times
from app.core.logging import get_logger
from app.engine import runs
from app.engine.exhibits import RASTER_TYPES
from app.engine.fusion import disclosures
from app.engine.registry import get_registry
from app.reports import view as report_view
from app.schemas.evidence import (
    Assessment,
    CalibrationInfo,
    ConditionProfile,
    Decision,
    DecisionCode,
    Exhibit,
    Finding,
    InputRef,
    ProvenanceRecord,
    ReproManifest,
    SignalV2,
    ValidationRef,
)
from app.schemas.legacy import LegacyFormatError, upgrade_v1
from app.storage.artefacts import ArtefactNotFound, get_artefact
from app.storage.db import (
    Analysis,
    AnalysisStage,
    Artefact,
    AssessmentRow,
    CaseMediaRow,
    CaseRow,
    ExhibitRow,
    FindingRow,
    Job,
    MediaObject,
    SignalRow,
    get_session,
)
from app.storage.objects import ObjectNotFound, default_store

log = get_logger(__name__)
router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
IMMUTABLE = "private, max-age=31536000, immutable"
BINARY_PARTS = ("payload", "mask", "preview", "tiles")
BinaryPart = Literal["payload", "mask", "preview", "tiles"]
AnalysisKind = Literal["image", "video", "document", "audio"]
AnalysisStatus = Literal["queued", "running", "done", "failed", "cancelled"]


def _binary_responses(media_type: str) -> dict[int | str, dict[str, Any]]:
    return {200: {"content": {media_type: {}}, "description": "Immutable binary (ETag = sha256)"},
            304: {"description": "Not modified (If-None-Match)"}, **problem_responses(401, 404, 422, 500)}


# --- models ------------------------------------------------------------------------------------------------

ScalePositionCode = Literal["ai_supported_strong", "ai_supported", "inconclusive", "no_ai_indicators", "no_ai_indicators_strong", "no_opinion"]


class AnalysisSummaryV2(BaseModel):
    id: str
    kind: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    status: str
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    error: str | None
    case_id: str | None
    media_id: str | None
    revision: int
    supersedes_id: str | None
    profile: str | None
    effective_profile: str | None
    pipeline_version: str | None
    result_schema_version: int
    decision_code: DecisionCode | None
    driving_hypothesis: str | None
    fusion_mode: str | None
    probability_ai: float | None
    band: str | None
    progress: float | None
    current_stage: str | None
    locked_at: str | None
    review_state: Literal["machine", "in_review", "reviewed", "approved", "disputed", "superseded", "withdrawn"] | None = None
    examiner_position: ScalePositionCode | None = Field(default=None, description="The current conclusion's position (L-5)")
    examiner_departs: bool | None = Field(default=None, description="That position differs from the machine decision (L-5)")
    examiner_reference: Literal["capture_record", "declaration"] | None = Field(
        default=None, description="`capture_record` where the position is compared with the step of the capture record that decides "
                                  "the analysis, not with the decision code (N-19n, D4-01); `declaration` where a trusted "
                                  "declaration decides on the fallback and the position is compared with its step, the fifth, "
                                  "not with the code, the fourth (N-19o, D5-01); null otherwise")
    deciding_record: Literal["capture_record", "declaration"] | None = Field(
        default=None, description="The trusted provenance record that decides the result, as the stored result names it: "
                                  "`capture_record` (a trusted capture), `declaration` (a trusted declaration of generative AI); "
                                  "null for a marker and where no record decides. On the fallback a declaration and a marker "
                                  "give the same code and hypothesis, so a list reads the row by this field (N-19p, D6-01)")
    case_reference: str | None = Field(default=None, description="The number of the analysis's case (VX-<year>-<nnn>); none "
                                       "for an unfiled quick analysis and for an analysis without a case (N-19a)")
    exhibit_label: str | None = Field(default=None, description="The exhibit label of the file in that case (A, B, …)")
    awaits_second_review: bool = Field(default=False, description="Signed by the examiner on a case that requires a second "
                                       "review, with no peer review recorded: not issuable, and never shown as reviewed (D-56)")


class Coverage(BaseModel):
    detectors_total: int
    by_status: dict[str, int]
    not_run_on_server: dict[str, int] = Field(
        default_factory=dict,
        description="Checks skipped for a reason of the server and not of the file, by reason: `memory budget`, `model not installed` (N-14a)")


class ReviewBrief(BaseModel):
    """The review block of an analysis (D-27, D-56 displayed state; UX §5.14, §5.16; C-5)."""

    state: Literal["machine", "in_review", "reviewed", "approved", "disputed", "superseded", "withdrawn"] = Field(
        description="approved: the peer review concurs, with or without changes; disputed: the peer review does not concur")
    issuable: bool = Field(description="A report may be issued: approved, or reviewed on a case that does not require a second "
                                       "review. A disputed revision is never issued.")
    requires_second_review: bool
    signatures_valid: bool = Field(description="The current conclusion and every sign-off pass their signature check; nothing is "
                                               "issued while one does not")
    locked_at: datetime | None
    withdrawn_reason: str | None
    conclusion_id: str | None
    conclusion_version: int | None = Field(description="The current conclusion's version (L-5): a sign-off names it")
    examiner_position: ScalePositionCode | None = Field(description="The current conclusion's verbal-scale position (L-5)")
    examiner_departs: bool | None = Field(description="The examiner's position differs from the machine decision (L-5)")
    examiner_reference: Literal["capture_record", "declaration"] | None = Field(
        default=None, description="`capture_record` where the position is compared with the step of the capture record that decides "
                                  "the analysis: the first step of the scale, on both fusion paths (N-19n, D4-01); `declaration` "
                                  "where a trusted declaration decides on the fallback and the position is compared with its "
                                  "step, the fifth (N-19o, D5-01); null otherwise")
    signoffs: list[Literal["examiner", "peer_review"]]


class NotValidatedDisclosure(BaseModel):
    id: str = Field(description="The `not_validated.yaml` entry")
    label: str
    text: str


class CustodianHashCheck(BaseModel):
    """The custodian-stated hash of this analysis's media at intake into its case (CC-01; L-5): a mismatch is recorded, not
    blocked, so the examiner sees it beside the result."""

    state: Literal["match", "mismatch", "not_provided"] = Field(description="mismatch when any intake of this file into the case differed")
    stated_hash: str | None = Field(description="The hash the custodian stated on the intake shown")
    intake_at: datetime


class Cancellation(BaseModel):
    """Who asked for the cancellation of a cancelled run, when, and the note (N-19a; design review DS-12): the newest
    cancellation request in the custody log. Absent for a run that no request cancelled (a job that ran out of attempts)."""
    by: str = Field(description="Display name of the principal that asked")
    at: str
    note: str | None


class AnalysisDetailV2(AnalysisSummaryV2):
    customer_assessment: dict[str, Any] | None = Field(default=None, description=(
        "Versioned customer interpretation of the saved estimate, including the 80% leaning rule. "
        "Derived for display; it does not replace the frozen decision or alter result_hash. "
        "Automated likelihood is not a validated accuracy rate."))
    cancellation: Cancellation | None = None
    custodian_hash: CustodianHashCheck | None = Field(default=None, description="Null when the file was not taken in through a case intake")
    cancel_requested: bool = Field(default=False, description="An active analysis/refusion job has pending cooperative cancellation")
    review: ReviewBrief | None = None
    result_hash: str | None = Field(default=None, description="SHA-256 of the original stored result, not these projected response bytes")
    input: InputRef | None = None
    manifest: ReproManifest | None = None
    decision: Decision | None = None
    stored_record: ProvenanceRecord | None = Field(default=None, description=(
        "For a result stored before its decision named the provenance record that decides it: that record, worked out "
        "from the stored signals by the rule of the reports. Null where the decision names its record and where no record "
        "decides. The stored decision and the result's hash are unchanged"))
    assessments: list[Assessment] = Field(default_factory=list)
    condition_profile: ConditionProfile | None = None
    coverage: Coverage
    calibration: list[CalibrationInfo] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    validation_ref: ValidationRef | None = None  # the dossier slice in force at analysis time (RP-04); null until one exists
    not_validated: list[NotValidatedDisclosure] = Field(default_factory=list, description=(
        "Plain-words disclosures of the unvalidated families, strata, transforms and lanes that affect this result "
        "(`app/engine/fusion/disclosures.py`, L-3)"))


class StageV2(BaseModel):
    stage: str
    status: str
    attempt: int
    started_at: str | None
    finished_at: str | None
    error: str | None


class DetectorRunV2(BaseModel):
    id: str
    detector_id: str
    # N-19k (D2-05): the check as the ledger names it (its signal's name where the result holds one for it, else the
    # registry's name of the detector); None where neither knows it. The running page and the Coverage block print this,
    # and keep the id for the tooltip
    name: str | None = None
    detector_version: str
    model_revisions: dict[str, Any]
    device: str | None
    dtype: str | None
    status: str
    reason: str | None
    started_at: str | None
    duration_ms: int | None
    peak_memory_mb: float | None
    raw_output_artefact_id: str | None


class ExhibitMeta(BaseModel):
    id: str
    analysis_id: str
    kind: str
    title: str
    sha256: str | None = None
    display_id: str | None = None
    caption: str = ""
    created_by: str | None = None
    media_ref: str | None = None
    frame_pts: int | None = None
    source_signal_ids: list[str] = Field(default_factory=list)
    role: str | None = None  # image exhibits
    semantics: str | None = None  # heatmaps
    inline: bool  # False when the canonical payload is stored as an artefact (> 64 KB): read GET /exhibits/{id}
    binaries: list[BinaryPart] = Field(default_factory=list)  # GET /exhibits/{id}/{payload|mask|preview}, tiles


class LaneWindow(BaseModel):
    """Display support for one smoothed value; points never establish an interval of scored content."""

    kind: Literal["interval", "point"]
    start_s: FiniteFloat
    end_s: FiniteFloat
    p: float | None = Field(ge=0.0, le=1.0)


class LaneIndexEntry(BaseModel):
    """One lane of the temporal stage (V-7) or the deep second pass: its series exhibit and, where flagged, its segments exhibit."""

    pass_: Literal["first", "second"] = Field(alias="pass")
    lane: str
    track_id: str | None
    channel: str
    detectors: list[str]
    series_exhibit_id: str
    segments_exhibit_id: str | None
    windows: list[LaneWindow] | None = Field(default=None, description="Observed score support; null for historical records without window bounds")


class FrameTableV2(BaseModel):
    """The per-frame pts table of a video analysis (probe, V-1) and its planned sample frames, so a player maps media
    time to the analysed frame index (ARCHITECTURE §7.5 `pts.ts`)."""

    time_base: str = Field(description="e.g. 1/30000; frame k shows at pts[k] × time_base seconds")
    pts: list[int] = Field(description="Presentation timestamps in display order")
    key_frames: list[int] = Field(description="Table indices of key frames")
    pict_types: str = Field(description="One character per frame: I, P, B, S or ?")
    width: int
    height: int
    rotation: int
    truncated: bool = Field(description="The table stops at the probe cap (108k frames)")
    interpolated: list[int] = Field(description="Table indices whose pts were missing and filled in")
    sampled: list[int] = Field(description="Table indices selected by the legacy per-frame sampling plan; this is not proof of detector success")
    tracked: list[int] = Field(description="Table indices of the sparse per-track sampler (full and deep profiles)")
    shot_cuts: list[int] = Field(description="Table index of the first frame of every shot after the first")
    shots_detected: bool = Field(description="The shots stage ran (full and deep profiles); otherwise shot_cuts is not established")
    lanes: list[LaneIndexEntry] = Field(description="Lane series and segments exhibits of the temporal stage and the deep second pass")


class TokenRequest(RequestModel):
    resource: str = Field(pattern=f"^{tokens.RESOURCE_PATTERN}$", max_length=160,
                          description="analysis:<id>, events:<analysis id>, exhibit:<id>, media:<id> or report:<id>")
    ttl_s: int = Field(default=tokens.MAX_TTL_S, ge=1, le=tokens.MAX_TTL_S)


class TokenGrant(BaseModel):
    token: str
    resource: str
    expires_at: str
    ttl_s: int


# --- helpers -----------------------------------------------------------------------------------------------

def _json(body: Any, status_code: int = 200) -> JSONResponse:
    from app.api.public_projection import project

    return JSONResponse(content=project(body), status_code=status_code,
                        headers={**NO_STORE, 'X-Chipmunk-View': 'customer-projection',
                                 'X-Result-Hash-Meaning': 'original-record'})


_iso = times.iso_z  # one form on every engine (D-72)


def _belongs(db: Session, analyses: Sequence[Analysis]) -> dict[str, tuple[str | None, str | None]]:
    """N-19a (design review DS-12): where each analysis belongs: the number of its case (not of an implicit, unfiled one)
    and the exhibit label of its file there. Two queries for a page."""
    case_ids = {a.case_id for a in analyses if a.case_id}
    if not case_ids:
        return {}
    references = {case_id: reference for case_id, reference in db.execute(
        select(CaseRow.id, CaseRow.reference).where(CaseRow.id.in_(case_ids), CaseRow.implicit.is_(False))).tuples()}
    labels = {(c, m): label for c, m, label in db.execute(select(CaseMediaRow.case_id, CaseMediaRow.media_id, CaseMediaRow.label)
                                                          .where(CaseMediaRow.case_id.in_(references))).tuples()} if references else {}
    return {a.id: (references.get(a.case_id), labels.get((a.case_id, a.media_id))) for a in analyses if a.case_id in references}


def _cancellation(db: Session, a: Analysis) -> dict[str, Any] | None:
    from app.reports.snapshot import principal_names
    from app.storage.db import AuditEvent

    event = db.scalars(select(AuditEvent).where(AuditEvent.analysis_id == a.id, AuditEvent.action == "analysis_cancel_requested")
                       .order_by(AuditEvent.seq.desc()).limit(1)).first()
    if event is None:
        return None
    actor = str(event.actor_id)
    return {"by": principal_names(db, {actor})[actor], "at": _iso(event.ts), "note": (event.details or {}).get("note")}


def _summary(a: Analysis, review_state: str | None = None, position: tuple[str, bool | None, str | None] | None = None,
             awaits_second_review: bool = False, belongs: tuple[str | None, str | None] | None = None,
             deciding_record: str | None = None) -> dict[str, Any]:
    return {"id": a.id, "kind": a.kind, "filename": a.filename, "content_type": a.content_type, "size_bytes": a.size_bytes,
            "sha256": a.sha256, "status": a.status, "created_at": _iso(a.created_at), "started_at": _iso(a.started_at),
            "finished_at": _iso(a.finished_at), "error": a.error, "case_id": a.case_id, "media_id": a.media_id,
            "revision": a.revision, "supersedes_id": a.supersedes_id, "profile": a.profile,
            "effective_profile": a.effective_profile, "pipeline_version": a.pipeline_version,
            "result_schema_version": a.result_schema_version, "decision_code": a.decision_code,
            "driving_hypothesis": a.driving_hypothesis, "fusion_mode": a.fusion_mode, "probability_ai": a.probability_ai,
            "band": a.band, "progress": a.progress, "current_stage": a.current_stage, "locked_at": _iso(a.locked_at), "review_state": review_state,
            "examiner_position": position[0] if position else None, "examiner_departs": position[1] if position else None,
            "examiner_reference": position[2] if position else None,
            "awaits_second_review": awaits_second_review,
            "deciding_record": deciding_record,
            "case_reference": belongs[0] if belongs else None, "exhibit_label": belongs[1] if belongs else None}


def _analysis(db: Session, aid: str) -> Analysis:
    analysis = db.get(Analysis, aid)
    if analysis is None:
        raise ProblemError(404, "Analysis not found", "No analysis exists with this id.")
    return analysis


def _legacy(a: Analysis) -> bool:
    return a.result_schema_version == 1 and a.result_json is not None


def _read_verified(row: Artefact) -> bytes:
    try:
        data = default_store().read_bytes(row.sha256)
    except ObjectNotFound:
        log.error("api_v2.object_missing", artefact_id=row.id, sha256=row.sha256)
        raise ProblemError(500, "Evidence object unavailable", "The stored object for this artefact is missing.") from None
    if len(data) != row.bytes or hashlib.sha256(data).hexdigest() != row.sha256:
        log.error("api_v2.object_integrity_failure", artefact_id=row.id, sha256=row.sha256)
        raise ProblemError(500, "Evidence integrity failure",
                           "The stored object no longer matches its recorded hash, so it was not served.")
    return data


def _artefact(db: Session, artefact_id: str, analysis_id: str) -> Artefact:
    try:
        return get_artefact(db, artefact_id, analysis_id=analysis_id)
    except ArtefactNotFound:
        log.error("api_v2.artefact_unindexed", artefact_id=artefact_id, analysis_id=analysis_id)
        raise ProblemError(500, "Evidence object unavailable", "The exhibit references an artefact that is not indexed.") from None


def _result(db: Session, a: Analysis) -> dict[str, Any] | None:
    """The published v2 result document, a v1 result upgraded to schema v2, or None while no result exists."""
    if a.result_schema_version == 2 and a.result_hash:
        row = db.scalars(select(Artefact).where(Artefact.analysis_id == a.id, Artefact.role == "stage_output",
                                                Artefact.sha256 == a.result_hash)).first()
        if row is None:
            log.error("api_v2.result_unindexed", analysis_id=a.id)
            raise ProblemError(500, "Stored result unavailable", "The published result document is not indexed.")
        return json.loads(_read_verified(row))
    if _legacy(a):
        try:
            return upgrade_v1(a.result or {}, a.kind, analysis_id=a.id).model_dump(mode="json")
        except LegacyFormatError as e:
            log.warning("api_v2.legacy_unmappable", analysis_id=a.id, error=str(e))
            return {"caveats": [f"The stored v1 result could not be mapped onto schema v2: {e}"]}
    return None


def _filtered(items: Sequence[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    return [x for x in items if all(v is None or x.get(k) == v for k, v in filters.items())]


def _binary_ids(payload: dict[str, Any]) -> dict[str, str]:
    kind, ids = payload.get("kind"), {}
    if kind in ("heatmap", "image") and payload.get("payload_artefact_id"):
        ids["payload"] = payload["payload_artefact_id"]
    if kind == "heatmap":
        if payload.get("reliability_mask_artefact_id"):
            ids["mask"] = payload["reliability_mask_artefact_id"]
        if payload.get("preview_artefact_id"):
            ids["preview"] = payload["preview_artefact_id"]
        if payload.get("tiles"):
            ids["tiles"] = payload["tiles"]["manifest_artefact_id"]
    return ids


def _meta(analysis_id: str, *, row: ExhibitRow | None = None, payload: dict[str, Any] | None = None,
          legacy: bool = False) -> dict[str, Any]:
    body = payload if payload is not None else (row.payload if row is not None else None) or {}
    ids = {} if legacy else _binary_ids(body)
    return {"id": row.id if row is not None else body["id"], "analysis_id": analysis_id,
            "kind": row.kind if row is not None else body["kind"], "title": row.title if row is not None else body["title"],
            "sha256": row.sha256 if row is not None else body.get("sha256"), "display_id": body.get("display_id"),
            "caption": body.get("caption", ""), "created_by": body.get("created_by"), "media_ref": body.get("media_ref"),
            "frame_pts": body.get("frame_pts"), "source_signal_ids": body.get("source_signal_ids", []),
            "role": body.get("role"), "semantics": body.get("semantics"), "inline": row is None or row.payload is not None,
            "binaries": [p for p in BINARY_PARTS if p in ids]}


def _exhibit_row(db: Session, exhibit_id: str) -> ExhibitRow:
    row = db.get(ExhibitRow, exhibit_id)
    if row is None:
        raise ProblemError(404, "Exhibit not found", "No exhibit exists with this id.")
    return row


def _exhibit_payload(db: Session, row: ExhibitRow) -> dict[str, Any]:
    if row.payload is not None:
        return row.payload
    if row.payload_artefact_id is None:
        raise ProblemError(500, "Evidence object unavailable", "The exhibit has neither an inline nor a stored payload.")
    return json.loads(_read_verified(_artefact(db, row.payload_artefact_id, row.analysis_id)))


def _etag_matches(header: str | None, sha256: str) -> bool:
    return bool(header) and any(t.strip() in ("*", f'"{sha256}"', f'W/"{sha256}"') for t in (header or "").split(","))


def _binary(request: Request, row: Artefact, *, exhibit: bool = False, byte_ranges: bool = False) -> Response:
    """A stored binary with its ETag. `exhibit`: what an exhibit route serves is shown inline only when it is a raster
    image; any other stored type (an SVG or HTML thumbnail an earlier build stored from a manifest) is a download in a
    sandbox, never a document of this origin (N-10; SEC-10)."""
    headers = {"ETag": f'"{row.sha256}"', "Cache-Control": IMMUTABLE, "X-Content-Type-Options": "nosniff"}
    if byte_ranges:
        headers["Accept-Ranges"] = "bytes"
    if exhibit and row.media_type not in RASTER_TYPES:
        headers.update({"Content-Disposition": "attachment", "Content-Security-Policy": "sandbox"})
    if _etag_matches(request.headers.get("if-none-match"), row.sha256):
        return Response(status_code=304, headers=headers)
    content = _read_verified(row)
    # Opt-in only for allowlisted presentation media, after the caller's case /
    # evidence-set checks and the complete object's hash verification. A range
    # never grants access to an undeclared source/raw object.
    requested = request.headers.get("range") if byte_ranges else None
    if requested and request.headers.get("if-range", headers["ETag"]) == headers["ETag"]:
        match = re.fullmatch(r"bytes=(\d{0,20})-(\d{0,20})", requested.strip())
        if not match or not any(match.groups()) or not content:
            return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{len(content)}"})
        first, last = match.groups()
        start = int(first) if first else max(0, len(content) - int(last))
        end = min(int(last), len(content) - 1) if first and last else len(content) - 1
        if start >= len(content) or end < start or (not first and int(last) == 0):
            return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{len(content)}"})
        headers["Content-Range"] = f"bytes {start}-{end}/{len(content)}"
        return Response(content=content[start:end + 1], status_code=206, media_type=row.media_type, headers=headers)
    return Response(content=content, media_type=row.media_type, headers=headers)


# --- analyses ----------------------------------------------------------------------------------------------

@router.get("/analyses", response_model=Page[AnalysisSummaryV2], responses=problem_responses(400, 401, 422))
def list_analyses(request: Request, cursor: Cursor = None, limit: Limit = DEFAULT_LIMIT, case_id: str | None = None,
                  kind: AnalysisKind | None = None, status: AnalysisStatus | None = None, decision: str | None = None,
                  hypothesis: str | None = None, awaiting_second_review: bool | None = None):
    """Analyses, newest first. `decision` filters on the decision code, `hypothesis` on the driving hypothesis;
    `awaiting_second_review=true` gives the revisions an examiner has signed on a case that requires a second review and
    that no peer has reviewed yet (a reviewer's queue), `false` every other one. A non-administrator sees the analyses of
    its cases only (C-2)."""
    from app.cases import review

    after = decode_cursor(cursor, 2)
    principal: Principal = request.state.principal
    query = select(Analysis)
    if (visible := access.visible_analyses(principal)) is not None:
        query = query.where(visible)
    for column, value in ((Analysis.case_id, case_id), (Analysis.kind, kind), (Analysis.status, status),
                          (Analysis.decision_code, decision), (Analysis.driving_hypothesis, hypothesis)):
        if value is not None:
            query = query.where(column == value)
    if awaiting_second_review is not None:  # N-19a (browser QA P1-12)
        waiting = review.awaits_second_review()
        query = query.where(waiting if awaiting_second_review else ~waiting)
    if after is not None:
        try:
            ts = datetime.fromisoformat(str(after[0]))
        except ValueError:
            raise ProblemError(400, "Invalid cursor", "The cursor does not belong to this collection.") from None
        query = query.where(or_(Analysis.created_at < ts, and_(Analysis.created_at == ts, Analysis.id < str(after[1]))))
    with get_session() as db:
        rows = db.scalars(query.order_by(Analysis.created_at.desc(), Analysis.id.desc()).limit(limit + 1)).all()
        review_states, positions = review.states(db, rows[:limit]), review.positions(db, rows[:limit])
        waiting_ids = review.awaiting_second(db, rows[:limit], review_states)
        belongs = _belongs(db, rows[:limit])
    page = rows[:limit]
    records = review.deciding_records(page)
    next_cursor = encode_cursor([page[-1].created_at.isoformat(), page[-1].id]) if len(rows) > limit else None
    return _json({"items": [_summary(a, review_states.get(a.id), positions.get(a.id), a.id in waiting_ids, belongs.get(a.id),
                                    records.get(a.id)) for a in page],
                  "next_cursor": next_cursor})


def _custodian_hash(db: Session, a: Analysis) -> dict[str, Any] | None:
    """The intake mismatch if any intake of this media into this case recorded one, else the newest intake's state (L-5)."""
    row = intake.custodian_check(db, a.case_id, a.media_id)
    if row is None:
        return None
    return {"state": row.custodian_hash_match, "stated_hash": row.custodian_stated_hash, "intake_at": _iso(row.intake_at)}


@router.get("/analyses/{aid}", response_model=AnalysisDetailV2, responses=problem_responses(401, 404, 500))
def get_analysis(aid: str):
    """Summary, decision, assessments, condition profile, coverage and manifest (no exhibit payloads)."""
    from app.cases import review

    with get_session() as db:
        a = _analysis(db, aid)
        result = _result(db, a)
        # one run per detector (N-8); a detector the result names whose run could not be stored counts as failed (N-10a)
        run_statuses = runs.statuses(db, aid, result if a.result_schema_version == 2 else None)
        on_server = runs.not_run_on_server(db, aid)
        brief = review.brief(db, a)
        belongs = _belongs(db, [a]).get(a.id)
        cancellation = _cancellation(db, a) if a.status == "cancelled" else None
        custodian = _custodian_hash(db, a)
        cancel_requested = a.status in ("queued", "running") and bool(db.scalar(select(Job.id).where(
            Job.subject_id == aid, Job.type.in_(("analysis", "refusion")), Job.status.in_(("queued", "leased")),
            Job.cancel_requested.is_(True)).limit(1)))
    statuses = [s.get("status", "ok") for s in (result or {}).get("signals") or []] if _legacy(a) else run_statuses
    body = {**_summary(a, brief["state"], awaits_second_review=brief["state"] == "reviewed" and bool(brief["requires_second_review"]),
                       belongs=belongs, deciding_record=review.deciding_records([a]).get(a.id)),
            "cancellation": cancellation, "result_hash": a.result_hash, "review": {**brief, "locked_at": _iso(brief["locked_at"])},
            "cancel_requested": cancel_requested, "custodian_hash": custodian,
            "coverage": {"detectors_total": len(statuses), "by_status": dict(sorted(Counter(statuses).items())),
                         "not_run_on_server": {} if _legacy(a) else on_server},
            "input": None, "manifest": None, "decision": None, "stored_record": None, "condition_profile": None,
            "assessments": [], "calibration": [], "caveats": [], "validation_ref": None, "not_validated": []}
    if result is not None:
        body.update({k: result.get(k) for k in ("input", "manifest", "decision", "condition_profile", "validation_ref")})
        body.update({k: result.get(k) or [] for k in ("assessments", "calibration", "caveats")})
        if body["decision_code"] is None and body["decision"]:
            body["decision_code"] = body["decision"]["code"]
        if body["decision"]:
            body["not_validated"] = disclosures.for_result({**result, "kind": a.kind})
            if not body["decision"].get("provenance_record"):
                # N-19l (code review R6-01): the rule for a result stored before N-19h is the server's. The detail holds
                # no signals, so the page cannot work the record out; it reads this field (`record.ts withRecord`)
                body["stored_record"] = report_view.stored_record(body["decision"], body["assessments"], list(result.get("signals") or []))
    from app.engine.customer_assessment import customer_assessment

    body["customer_assessment"] = customer_assessment({**(result or {}), **body})
    return _json(body)


@router.get("/analyses/{aid}/stages", response_model=Page[StageV2], responses=problem_responses(401, 404))
def list_stages(aid: str):
    """Stage rows in pipeline order (one page)."""
    from app.engine.pipelines import PIPELINES

    with get_session() as db:
        a = _analysis(db, aid)
        rows = db.scalars(select(AnalysisStage).where(AnalysisStage.analysis_id == aid)).all()
    order = {s.id: i for i, s in enumerate(PIPELINES.get(a.kind, ()))}
    rows = sorted(rows, key=lambda r: (order.get(r.stage, len(order)), r.stage))
    items = [{"stage": r.stage, "status": r.status, "attempt": r.attempt, "started_at": _iso(r.started_at),
              "finished_at": _iso(r.finished_at), "error": r.error} for r in rows]
    return _json({"items": items, "next_cursor": None})


def _run_names(db: Session, aid: str, detector_ids: list[str]) -> dict[str, str]:
    """N-19k (D2-05): each check as the ledger names it. A detector with one name among its signals has that name (a video's
    voice check is "Synthetic-voice detectors on the sound track" there); any other, and every detector of an analysis that
    is still running, has the registry's name."""
    by_detector: dict[str, set[str]] = {}
    for row in db.scalars(select(SignalRow).where(SignalRow.analysis_id == aid)):
        payload = row.payload or {}
        if payload.get("detector_id") and payload.get("name"):
            by_detector.setdefault(str(payload["detector_id"]), set()).add(str(payload["name"]))
    registry = get_registry()
    names = {}
    for detector_id in detector_ids:
        ledger = by_detector.get(detector_id) or set()
        name = next(iter(ledger)) if len(ledger) == 1 else registry.detector_name(detector_id)
        if name:
            names[detector_id] = name
    return names


@router.get("/analyses/{aid}/detector-runs", response_model=Page[DetectorRunV2], responses=problem_responses(400, 401, 404, 422))
def list_detector_runs(aid: str, cursor: Cursor = None, limit: Limit = DEFAULT_LIMIT, status: str | None = None):
    with get_session() as db:
        _analysis(db, aid)
        rows = runs.current(db, aid)  # one run per detector: an interrupted attempt's rows are not listed (N-8)
        names = _run_names(db, aid, [r.detector_id for r in rows])
    items = [{"id": r.id, "detector_id": r.detector_id, "name": names.get(r.detector_id), "detector_version": r.detector_version,
              "model_revisions": r.model_revisions or {}, "device": r.device, "dtype": r.dtype, "status": r.status,
              "reason": r.reason, "started_at": _iso(r.started_at), "duration_ms": r.duration_ms,
              "peak_memory_mb": r.peak_memory_mb, "raw_output_artefact_id": r.raw_output_artefact_id} for r in rows]
    started = {r.id: times.micros(r.started_at) for r in rows}  # the order of the times, not of their text

    def key(run: dict[str, Any]) -> tuple[int, str]:
        return started[run["id"]], run["id"]

    items = sorted(_filtered(items, status=status), key=key)
    page, next_cursor = paginate(items, key, cursor, limit)
    return _json({"items": page, "next_cursor": next_cursor})


@router.get("/analyses/{aid}/signals", response_model=Page[SignalV2], responses=problem_responses(400, 401, 403, 404, 422))
def list_signals(aid: str, cursor: Cursor = None, limit: Limit = DEFAULT_LIMIT, hypothesis: str | None = None,
                 lane: str | None = None, status: str | None = None):
    """The machine layer (read-only, RW-01), ordered by signal id."""
    with get_session() as db:
        a = _analysis(db, aid)
        if _legacy(a):
            items = (_result(db, a) or {}).get("signals") or []
        else:
            items = [r.payload for r in db.scalars(select(SignalRow).where(SignalRow.analysis_id == aid))]
    items = sorted(_filtered(items, hypothesis=hypothesis, lane=lane, status=status), key=lambda s: s["id"])
    page, next_cursor = paginate(items, lambda s: (s["id"],), cursor, limit)
    return _json({"items": page, "next_cursor": next_cursor})


@router.api_route("/analyses/{aid}/signals", methods=["POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
@router.api_route("/analyses/{aid}/signals/{signal_id}", methods=["PUT", "PATCH", "DELETE"], include_in_schema=False)
def refuse_signal_write(aid: str, signal_id: str | None = None):
    """The machine layer is immutable (RW-01): every attempt to write a signal is refused with 403 (C-4). An analysis that does
    not exist is not found, as one the caller cannot see is (`authorize`), so the refusal says nothing about another case."""
    with get_session() as db:
        _analysis(db, aid)
    raise ProblemError(403, "Machine layer is immutable",
                       "Signal values cannot be changed (RW-01). Record an examiner opinion with POST /api/v2/analyses/{id}/annotations.")


@router.get("/analyses/{aid}/findings", response_model=Page[Finding], responses=problem_responses(400, 401, 404, 422))
def list_findings(aid: str, cursor: Cursor = None, limit: Limit = DEFAULT_LIMIT, hypothesis: str | None = None,
                  family: str | None = None):
    """Findings by rank, each with the examiner layer joined at read time (RW-01, C-4)."""
    from app.cases.examiner import examiner_layers

    with get_session() as db:
        a = _analysis(db, aid)
        if _legacy(a):
            items = (_result(db, a) or {}).get("findings") or []
        else:
            items = [r.payload for r in db.scalars(select(FindingRow).where(FindingRow.analysis_id == aid))]
        layers = examiner_layers(db, aid)
    items = [{**f, "examiner": layers.get(f["id"], f.get("examiner"))} for f in items]
    items = sorted(_filtered(items, hypothesis=hypothesis, family=family), key=lambda f: (f["rank"], f["id"]))
    page, next_cursor = paginate(items, lambda f: (f["rank"], f["id"]), cursor, limit)
    return _json({"items": page, "next_cursor": next_cursor})


@router.get("/analyses/{aid}/assessments", response_model=Page[Assessment], responses=problem_responses(400, 401, 404, 422))
def list_assessments(aid: str, cursor: Cursor = None, limit: Limit = DEFAULT_LIMIT):
    """One assessment per hypothesis (H2 lane assessments are nested in `lanes`)."""
    with get_session() as db:
        a = _analysis(db, aid)
        if _legacy(a):
            items = (_result(db, a) or {}).get("assessments") or []
        else:
            items = [r.payload for r in db.scalars(select(AssessmentRow).where(AssessmentRow.analysis_id == aid,
                                                                               AssessmentRow.lane.is_(None)))]
    items = sorted(items, key=lambda x: x["hypothesis"])
    page, next_cursor = paginate(items, lambda x: (x["hypothesis"],), cursor, limit)
    return _json({"items": page, "next_cursor": next_cursor})


@router.get("/analyses/{aid}/exhibits", response_model=Page[ExhibitMeta], responses=problem_responses(400, 401, 404, 422))
def list_exhibits(aid: str, cursor: Cursor = None, limit: Limit = DEFAULT_LIMIT, kind: str | None = None):
    """Exhibit metadata by id, with the binary routes each exhibit has."""
    with get_session() as db:
        a = _analysis(db, aid)
        if _legacy(a):
            items = [_meta(aid, payload=x, legacy=True) for x in (_result(db, a) or {}).get("exhibits") or []]
        else:
            items = [_meta(aid, row=r) for r in db.scalars(select(ExhibitRow).where(ExhibitRow.analysis_id == aid))]
        # N-19c: the stored order of the published result, which is the same for every analysis of one file and the order in
        # which the reports number their figures. An exhibit id is derived from the analysis id, so the order by id differed
        # from one analysis of a file to the next, and with it the viewer's catalogue. Until a result is published: by id.
        stored = (_result(db, a) or {}).get("exhibits") or [] if _legacy(a) or (a.result_schema_version == 2 and a.result_hash) else []
    position = {str(x["id"]): n for n, x in enumerate(stored)}

    def key(x: dict[str, Any]) -> tuple[int, str]:
        return (position.get(x["id"], len(position)), x["id"])

    items = sorted(_filtered(items, kind=kind), key=key)
    page, next_cursor = paginate(items, key, cursor, limit)
    return _json({"items": page, "next_cursor": next_cursor})


# --- exhibits and binaries ---------------------------------------------------------------------------------

@router.get("/exhibits/{exhibit_id}", response_model=Exhibit, responses=problem_responses(401, 404, 500))
def get_exhibit(exhibit_id: str):
    with get_session() as db:
        return _json(_exhibit_payload(db, _exhibit_row(db, exhibit_id)))


def _exhibit_part(request: Request, exhibit_id: str, part: str) -> Response:
    with get_session() as db:
        row = _exhibit_row(db, exhibit_id)
        artefact_id = _binary_ids(_exhibit_payload(db, row)).get(part)
        if artefact_id is None:
            raise ProblemError(404, "Binary not available", f"This {row.kind} exhibit has no {part}.")
        artefact = _artefact(db, artefact_id, row.analysis_id)
    return _binary(request, artefact, exhibit=True)


@router.get("/exhibits/{exhibit_id}/payload", response_class=Response, responses=_binary_responses("image/png"))
def exhibit_payload(exhibit_id: str, request: Request):
    """Heatmap: 8-bit grayscale PNG over `value_range`. Image exhibits: the stored image."""
    return _exhibit_part(request, exhibit_id, "payload")


@router.get("/exhibits/{exhibit_id}/mask", response_class=Response, responses=_binary_responses("image/png"))
def exhibit_mask(exhibit_id: str, request: Request):
    """Heatmap reliability mask: 8-bit PNG, 255 = reliable."""
    return _exhibit_part(request, exhibit_id, "mask")


@router.get("/exhibits/{exhibit_id}/preview", response_class=Response, responses=_binary_responses("image/png"))
def exhibit_preview(exhibit_id: str, request: Request):
    """Heatmap preview: colormapped PNG for PDFs and thumbnails."""
    return _exhibit_part(request, exhibit_id, "preview")


@router.get("/exhibits/{exhibit_id}/tiles/{z}/{x}_{y}.png", response_class=Response, responses=_binary_responses("image/png"))
def exhibit_tile(request: Request, exhibit_id: str, z: int = Path(ge=0), x: int = Path(ge=0), y: int = Path(ge=0)):
    """One 512 px tile of the source image the heatmap is registered to (source images > 4096 px only)."""
    with get_session() as db:
        row = _exhibit_row(db, exhibit_id)
        manifest_id = _binary_ids(_exhibit_payload(db, row)).get("tiles")
        if manifest_id is None:
            raise ProblemError(404, "No tile pyramid",
                               "This exhibit has no tile pyramid: its source image is at most 4096 px on the long side.")
        manifest = json.loads(_read_verified(_artefact(db, manifest_id, row.analysis_id)))
        tile_id = manifest["tiles"].get(f"{z}/{x}_{y}")
        if tile_id is None:
            raise ProblemError(404, "Tile not found", f"The pyramid has {manifest['levels']} levels and no tile {z}/{x}_{y}.")
        artefact = _artefact(db, tile_id, row.analysis_id)
    return _binary(request, artefact, exhibit=True)


@router.get("/analyses/{aid}/stills/{pts}.jpg", response_class=Response, responses=_binary_responses("image/jpeg"))
def analysis_still(request: Request, aid: str, pts: int = Path(ge=0)):
    """The sampled video frame the detectors scored, at exhibit `frame_pts` (milliseconds until pts tables arrive, V-1)."""
    with get_session() as db:
        _analysis(db, aid)
        rows = db.scalars(select(ExhibitRow).where(ExhibitRow.analysis_id == aid, ExhibitRow.kind == "image")).all()
        still = next((r for r in rows if r.payload and r.payload.get("role") == "still" and r.payload.get("frame_pts") == pts),
                     None)
        if still is None:
            raise ProblemError(404, "Still not found", "No sampled-frame still exists at this pts.")
        artefact = _artefact(db, still.payload["payload_artefact_id"], aid)
    return _binary(request, artefact, exhibit=True)


def _stage_output(db: Session, analysis_id: str, stage: str) -> dict[str, Any] | None:
    """A finished stage's persisted JSON output, re-hashed before it is read; None when the stage has no stored output."""
    row = db.get(AnalysisStage, (analysis_id, stage))
    if row is None or row.status != "done" or row.output_artefact_id is None:
        return None
    return json.loads(_read_verified(_artefact(db, row.output_artefact_id, analysis_id)))


@router.get("/analyses/{aid}/frames", response_model=FrameTableV2, responses=problem_responses(401, 404, 409, 422, 500))
def analysis_frames(aid: str):
    """The pts table of a video analysis from its probe stage (V-1), planned sample frames and the shot cuts
    (ARCHITECTURE §7.5). 422 for other kinds; 409 while the probe has not finished; 404 when a finished analysis has no table
    (v1 pipeline, or no decodable video frames)."""
    with get_session() as db:
        a = _analysis(db, aid)
        if a.kind != "video":
            raise ProblemError(422, "Not a video analysis", f"Frame tables belong to video analyses; this is a {a.kind} analysis.")
        unfinished = a.status in ("queued", "running")
        probe = _stage_output(db, aid, "probe")
        shots, temporal, second = (_stage_output(db, aid, s) if probe is not None else None for s in ("shots", "temporal", "second_pass"))
    table = (probe or {}).get("frame_table")
    if not table:
        if unfinished and probe is None:
            raise ProblemError(409, "Frame table not ready", "The probe stage of this analysis has not finished.")
        raise ProblemError(404, "Frame table not available",
                           "This analysis has no frame table: it ran on the v1 pipeline, or the file had no decodable video frames.")
    plan = (probe or {}).get("plan") or {}
    tracking = (shots or {}).get("plan") or plan
    body = {k: table[k] for k in ("time_base", "pts", "key_frames", "pict_types", "width", "height", "rotation")}
    body.update({"truncated": bool(table.get("truncated")), "interpolated": list(table.get("interpolated") or []),
                 "sampled": list((plan.get("legacy") or {}).get("frames") or []),
                 "tracked": list((tracking.get("sparse") or {}).get("frames") or []),
                 "shot_cuts": [int(r[0]) for r in ((shots or {}).get("ranges") or [])[1:]], "shots_detected": shots is not None,
                 "lanes": [{"pass": which, **{k: lane.get(k) for k in ("lane", "track_id", "channel", "detectors", "series_exhibit_id", "segments_exhibit_id", "windows")}}
                           for which, output in (("first", temporal), ("second", second)) for lane in (output or {}).get("lanes") or []]})
    return _json(body)


# --- resource tokens ---------------------------------------------------------------------------------------

@router.post("/tokens", response_model=TokenGrant, responses=problem_responses(401, 404, 422))
def create_token(req: TokenRequest, request: Request):
    """A signed, resource-scoped token for GET URLs that cannot carry headers (media elements, EventSource). The token acts
    as the calling principal: disabling the principal invalidates it (C-1). A resource of a case the caller is not a member
    of is not found (C-2)."""
    kind, _, resource_id = req.resource.partition(":")
    model = {"analysis": Analysis, "events": Analysis, "exhibit": ExhibitRow, "media": MediaObject}.get(kind)
    params = {"analysis": {"aid": resource_id}, "events": {"aid": resource_id}, "exhibit": {"exhibit_id": resource_id},
              "media": {"media_id": resource_id}}.get(kind, {})
    with get_session() as db:
        exists = model is not None and db.get(model, resource_id) is not None
        if exists and access.denial(db, request.state.principal, params, "viewer") is not None:
            exists = False
    if not exists:
        raise ProblemError(404, "Resource not found", f"No {kind} exists with this id.")
    token, exp = tokens.make_token(req.resource, request.state.principal.id, req.ttl_s)
    return _json({"token": token, "resource": req.resource, "expires_at": times.iso_z(datetime.fromtimestamp(exp, UTC)),
                  "ttl_s": req.ttl_s})
