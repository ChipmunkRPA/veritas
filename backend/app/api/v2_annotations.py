"""API v2 examiner layer: annotations and notes (ARCHITECTURE §4.5 `examiner`, §4.7 `annotations`, §5.2; PRD RW-01, RW-02;
UX §5.13; BUILD_SPEC C-4).

- **Routes.**
  - `GET /analyses/{aid}/annotations` (viewer; `?target_type&target_id&include_deleted`), oldest first;
  - `POST /analyses/{aid}/annotations` (examiner);
  - `PATCH` and `DELETE /annotations/{id}` by the author or an administrator. Deletion is soft.
- **Targets** (`target_type`, `target_id`):
  - `finding` (F1…), `signal`, `exhibit`: ids of this analysis;
  - `region`: a drawn shape on the media (no id) or on an exhibit (its id); `interval`: the media timeline; `sentence`: the text;
  - `item`: the analysis as a whole; `case`: the analysis's case. These three take no id.
- **Shapes and geometry** in source-media coordinates:
  - `box` {x, y, w, h}, `polygon` {points ≥ 3}, `arrow` {x1, y1, x2, y2}, `text` {x, y} with a body; on video, optionally
    `frame_pts` and `time_base` for the key frame;
  - `interval` {start_pts, end_pts, time_base} on video and audio; `sentence` {start, end} character offsets or {sentence_idx}
    on documents; `none`: a note or a status without geometry.
- **Examiner status** (RW-01): accepted, disputed, needs_check, not_relevant, on finding and signal targets only. Disputed and
  not relevant need a note: 422 here, and a database CHECK backs it.
- **Bodies** are plain text (UX §5.13: "no rich formatting, since this is a record"). Line breaks are normalised, control and
  bidirectional-override characters removed, and the text trimmed.
- **Revisions.** Every state, the first included, is kept in `annotation_revisions` and returned as `history`.
- **Refusals.** 409 while the analysis is still running or once its revision is locked (C-5), and on a deleted annotation.
- **Custody** (C-3): `note_added` for a new annotation of shape `none`, else `annotation_created`; `annotation_edited` and
  `annotation_deleted`, with canonical hashes of the annotation state before and after.
- **Machine layer.** Findings carry the joined `examiner` layer (`app.cases.examiner`); signals refuse writes with 403
  (`app.api.v2_evidence`).
"""
from __future__ import annotations

import unicodedata
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import Principal
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import RequestModel
from app.api.v2 import MAX_SEQ, NO_STORE, ProblemError, problem_responses
from app.audit import chain as audit
from app.cases import access, review
from app.core import times
from app.core.canonical import canonical_hash
from app.core.logging import get_logger
from app.storage.db import (
    Analysis,
    AnnotationRevisionRow,
    AnnotationRow,
    ExhibitRow,
    FindingRow,
    SignalRow,
    get_session,
    utcnow,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
TargetType = Literal["finding", "signal", "exhibit", "region", "interval", "sentence", "item", "case"]
Shape = Literal["box", "polygon", "arrow", "text", "interval", "sentence", "none"]
ExaminerStatus = Literal["accepted", "disputed", "needs_check", "not_relevant"]
Visibility = Literal["internal", "in_report"]
# N-14b (S2-03, E-6): each of the two numbers has at most ten digits, so that a time of a geometry is a number a report can print
TIME_BASE = r"^[1-9]\d{0,9}/[1-9]\d{0,9}$"
MAX_COORDINATE = 1_000_000_000  # px: far beyond any image; a coordinate above it is a mistake or an attack on what prints it
MAX_INT = MAX_SEQ  # 2**63 - 1, as `after_seq` (SEC-16): what a database integer and a report's arithmetic hold
MAX_BODY = 10000
NOTE_REQUIRED = frozenset({"disputed", "not_relevant"})
DRAWN = frozenset({"box", "polygon", "arrow", "text"})
SHAPES: dict[str, frozenset[str]] = {
    "finding": frozenset({"none"}), "signal": frozenset({"none"}), "item": frozenset({"none"}), "case": frozenset({"none"}),
    "exhibit": DRAWN | {"none"}, "region": DRAWN, "interval": frozenset({"interval"}), "sentence": frozenset({"sentence"}),
}
STATUS_TARGETS = frozenset({"finding", "signal"})
UNIDENTIFIED_TARGETS = frozenset({"interval", "sentence", "item", "case"})
TERMINAL = frozenset({"done", "failed", "cancelled"})
_BIDI = frozenset(chr(c) for c in (*range(0x202A, 0x202F), *range(0x2066, 0x206A)))
# N-10 (X-6): characters that render as nothing and shape nothing, removed wherever they stand. The zero-width space, the word
# joiner and the invisible operators, the byte order mark, the soft hyphen, the Mongolian vowel separator, the Hangul filler
# of the compatibility blocks, the deprecated format characters, the interlinear and musical format characters, the language tag
_INVISIBLE = frozenset(chr(c) for c in (0x00AD, 0x180E, 0x200B, *range(0x2060, 0x2065), *range(0x206A, 0x2070), 0x3164, 0xFEFF, 0xFFA0,
                                        *range(0xFFF9, 0xFFFC), *range(0x1D173, 0x1D17B), 0xE0001))
# Kept, because text needs them between visible characters, but not text by themselves: the joiners of Persian, Indic and
# emoji sequences, the direction marks, variation selectors and tag characters are not letters, numbers, punctuation or
# symbols, so only the letters that are blank have to be named (the Hangul fillers of conjoining jamo, the blank braille cell)
_BLANK = frozenset("\u115f\u1160\u2800")


# --- geometry ------------------------------------------------------------------------------------------------------------

class _Geometry(BaseModel):
    # N-14b (S2-03): a number that is not finite (`1e999` reads as infinity) is refused here, in a polygon's points too; it
    # passed `ge=0` and ended in `canonical_json` as an unhandled error
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Coordinate = Annotated[float, Field(ge=0, le=MAX_COORDINATE)]
Extent = Annotated[float, Field(gt=0, le=MAX_COORDINATE)]
Count = Annotated[int, Field(ge=0, le=MAX_INT)]


class _OnFrame(_Geometry):
    frame_pts: Count | None = None
    time_base: str | None = Field(default=None, pattern=TIME_BASE)


class BoxGeometry(_OnFrame):
    x: Coordinate
    y: Coordinate
    w: Extent
    h: Extent


class PolygonGeometry(_OnFrame):
    points: list[tuple[Coordinate, Coordinate]] = Field(min_length=3, max_length=2000)


class ArrowGeometry(_OnFrame):
    x1: Coordinate
    y1: Coordinate
    x2: Coordinate
    y2: Coordinate


class TextGeometry(_OnFrame):
    x: Coordinate
    y: Coordinate


class IntervalGeometry(_Geometry):
    start_pts: Count
    end_pts: Count
    time_base: str = Field(pattern=TIME_BASE)

    @model_validator(mode="after")
    def _ordered(self) -> IntervalGeometry:
        if self.end_pts < self.start_pts:
            raise ValueError("an interval must not end before it starts")
        return self


class SentenceGeometry(_Geometry):
    start: Count | None = None
    end: Count | None = None
    sentence_idx: Count | None = None

    @model_validator(mode="after")
    def _span(self) -> SentenceGeometry:
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end go together")
        if self.start is None and self.sentence_idx is None:
            raise ValueError("a sentence annotation needs start and end offsets or sentence_idx")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("end must be after start")
        return self


GEOMETRIES: dict[str, type[_Geometry]] = {"box": BoxGeometry, "polygon": PolygonGeometry, "arrow": ArrowGeometry, "text": TextGeometry,
                                          "interval": IntervalGeometry, "sentence": SentenceGeometry}


# --- models --------------------------------------------------------------------------------------------------------------

class AnnotationCreate(RequestModel):
    model_config = ConfigDict(extra="forbid")
    target_type: TargetType
    target_id: str | None = Field(default=None, max_length=255)
    shape: Shape = "none"
    geometry: dict[str, Any] | None = Field(default=None, description="Source-media coordinates; the fields depend on the shape")
    examiner_status: ExaminerStatus | None = None
    body: str | None = Field(default=None, max_length=MAX_BODY, description="Plain text")
    visibility: Visibility = "internal"


class AnnotationPatch(RequestModel):
    model_config = ConfigDict(extra="forbid")
    geometry: dict[str, Any] | None = None
    examiner_status: ExaminerStatus | None = None
    body: str | None = Field(default=None, max_length=MAX_BODY)
    visibility: Visibility | None = None


class AnnotationRevisionOut(BaseModel):
    revision: int
    body: str | None
    examiner_status: ExaminerStatus | None
    geometry: dict[str, Any] | None
    visibility: Visibility
    edited_by: str
    edited_at: datetime


class AnnotationOut(BaseModel):
    id: str
    analysis_id: str
    case_id: str | None
    author_id: str
    target_type: TargetType
    target_id: str | None
    shape: Shape
    geometry: dict[str, Any] | None
    examiner_status: ExaminerStatus | None
    body: str | None
    visibility: Visibility
    created_at: datetime
    edited_at: datetime | None
    deleted_at: datetime | None
    revision: int
    history: list[AnnotationRevisionOut]


class AnnotationList(BaseModel):
    items: list[AnnotationOut]


# --- helpers -------------------------------------------------------------------------------------------------------------

def _json(model: BaseModel, status_code: int = 200) -> JSONResponse:
    return JSONResponse(jsonable_encoder(model), status_code=status_code, headers=NO_STORE)


_utc = times.to_utc  # one form on every engine: a time is printed in UTC (D-72)


def _invalid(detail: str) -> ProblemError:
    return ProblemError(422, "Invalid request", detail)


def clean_body(body: str | None) -> str | None:
    """Plain text (UX §5.13): line breaks normalised; control characters, bidirectional overrides and characters that render
    as nothing removed; trimmed. Text in which nothing can be seen (no letter, number, punctuation mark or symbol) is no text:
    a note made only of a zero-width space does not satisfy "a disputed finding needs a note" (N-10; X-6)."""
    if body is None:
        return None
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or (unicodedata.category(ch) != "Cc" and ch not in _BIDI and ch not in _INVISIBLE)).strip()
    return text if any(unicodedata.category(ch)[0] in "LNPS" and ch not in _BLANK for ch in text) else None


def _analysis(db: Session, aid: str) -> Analysis:
    analysis = db.get(Analysis, aid)
    if analysis is None:
        raise ProblemError(404, *access.NOT_FOUND["analysis"])
    return analysis


def _annotation(db: Session, annotation_id: str) -> AnnotationRow:
    row = db.get(AnnotationRow, annotation_id)
    if row is None:
        raise ProblemError(404, *access.NOT_FOUND["annotation"])
    return row


def _writable(analysis: Analysis) -> None:
    if analysis.status not in TERMINAL:
        raise ProblemError(409, "Analysis still running", f"The analysis is {analysis.status}; annotate it once it has finished.")
    if analysis.locked_at is not None:
        raise ProblemError(409, "Revision locked", "This analysis revision is signed off and locked; annotate a new revision instead.")


def _still_writable(db: Session, analysis_id: str) -> None:
    """Right before the write: the analysis row's lock is taken while unlocked, so a sign-off in between wins (D-47)."""
    if not review.hold_unlocked(db, analysis_id):
        if review.gone(db, analysis_id):  # N-9: purged or deleted while this request waited for the row
            db.rollback()
            raise ProblemError(404, *access.NOT_FOUND["analysis"])
        raise ProblemError(409, "Revision locked", "This analysis revision was signed off and locked; annotate a new revision instead.")


def _author_or_administrator(row: AnnotationRow, principal: Principal) -> None:
    if not principal.satisfies("administrator") and row.author_id != principal.id:
        raise ProblemError(403, "Forbidden", "Only the author of an annotation or an administrator can change it.")


def _target(db: Session, analysis: Analysis, target_type: str, target_id: str | None) -> None:
    if target_type in UNIDENTIFIED_TARGETS:
        if target_id is not None:
            raise _invalid(f"target_id is not used with target_type {target_type}.")
        if target_type == "case" and analysis.case_id is None:
            raise _invalid("This analysis is not in a case.")
        return
    if target_type == "region" and target_id is None:
        return  # a region on the media itself
    if target_id is None:
        raise _invalid(f"target_type {target_type} needs target_id.")
    if target_type == "finding":
        found = db.get(FindingRow, (analysis.id, target_id)) is not None
    elif target_type == "signal":
        found = db.get(SignalRow, (analysis.id, target_id)) is not None
    else:  # exhibit, or a region on an exhibit
        found = db.scalars(select(ExhibitRow.id).where(ExhibitRow.id == target_id, ExhibitRow.analysis_id == analysis.id)).first() is not None
    if not found:
        raise _invalid(f"No {'exhibit' if target_type == 'region' else target_type} {target_id} exists in this analysis.")


def _geometry(analysis: Analysis, shape: str, geometry: dict[str, Any] | None) -> dict[str, Any] | None:
    if shape == "none":
        if geometry is not None:
            raise _invalid("An annotation of shape none has no geometry.")
        return None
    if geometry is None:
        raise _invalid(f"A {shape} annotation needs geometry.")
    try:
        parsed = GEOMETRIES[shape].model_validate(geometry)
    except ValidationError as e:
        parts = [f"{'.'.join(str(p) for p in err['loc']) or 'geometry'}: {err['msg']}" for err in e.errors()[:3]]
        raise _invalid("geometry: " + "; ".join(parts)) from None
    if getattr(parsed, "frame_pts", None) is not None and analysis.kind != "video":
        raise _invalid("frame_pts applies to video analyses.")
    if shape == "interval" and analysis.kind not in ("video", "audio"):
        raise _invalid("Interval annotations are for video and audio analyses.")
    if shape == "sentence" and analysis.kind != "document":
        raise _invalid("Sentence annotations are for document analyses.")
    if shape in DRAWN and analysis.kind not in ("image", "video"):
        raise _invalid(f"A {shape} annotation is drawn on an image or a video frame.")
    return parsed.model_dump(mode="json", exclude_none=True)


def _validated(db: Session, analysis: Analysis, *, target_type: str, target_id: str | None, shape: str, geometry: dict[str, Any] | None,
               examiner_status: str | None, body: str | None) -> dict[str, Any] | None:
    """The normalised geometry of a valid annotation state; raises 422 with the rule that fails."""
    _target(db, analysis, target_type, target_id)
    if shape not in SHAPES[target_type]:
        raise _invalid(f"A {target_type} annotation takes shape {' or '.join(sorted(SHAPES[target_type]))}.")
    normalised = _geometry(analysis, shape, geometry)
    if examiner_status is not None and target_type not in STATUS_TARGETS:
        raise _invalid("examiner_status applies to findings and signals.")
    if examiner_status in NOTE_REQUIRED and body is None:
        raise _invalid(f"A {examiner_status.replace('_', ' ')} status needs a note (RW-01).")
    if shape == "text" and body is None:
        raise _invalid("A text annotation needs a body.")
    if shape == "none" and examiner_status is None and body is None:
        raise _invalid("A note needs a body.")
    return normalised


def _state(row: AnnotationRow) -> dict[str, Any]:
    return {"id": row.id, "target_type": row.target_type, "target_id": row.target_id, "shape": row.shape, "geometry": row.geometry,
            "examiner_status": row.examiner_status, "body": row.body, "visibility": row.visibility, "revision": row.revision,
            "deleted": row.deleted_at is not None}


def _revision(row: AnnotationRow, editor: str, at: datetime) -> AnnotationRevisionRow:
    return AnnotationRevisionRow(annotation_id=row.id, revision=row.revision, body=row.body, examiner_status=row.examiner_status,
                                 geometry=row.geometry, visibility=row.visibility, edited_by=editor, edited_at=at)


def _revision_out(r: AnnotationRevisionRow) -> AnnotationRevisionOut:
    return AnnotationRevisionOut(revision=r.revision, body=r.body, examiner_status=r.examiner_status,  # type: ignore[arg-type]
                                 geometry=r.geometry, visibility=r.visibility, edited_by=r.edited_by,  # type: ignore[arg-type]
                                 edited_at=_utc(r.edited_at))  # type: ignore[arg-type]


def _histories(db: Session, ids: list[str]) -> dict[str, list[AnnotationRevisionOut]]:
    histories: dict[str, list[AnnotationRevisionOut]] = defaultdict(list)
    if ids:
        for r in db.scalars(select(AnnotationRevisionRow).where(AnnotationRevisionRow.annotation_id.in_(ids))
                            .order_by(AnnotationRevisionRow.annotation_id, AnnotationRevisionRow.revision)):
            histories[r.annotation_id].append(_revision_out(r))
    return histories


def _out(row: AnnotationRow, history: list[AnnotationRevisionOut]) -> AnnotationOut:
    return AnnotationOut(id=row.id, analysis_id=row.analysis_id, case_id=row.case_id, author_id=row.author_id,
                         target_type=row.target_type, target_id=row.target_id, shape=row.shape,  # type: ignore[arg-type]
                         geometry=row.geometry, examiner_status=row.examiner_status, body=row.body,  # type: ignore[arg-type]
                         visibility=row.visibility, created_at=_utc(row.created_at), edited_at=_utc(row.edited_at),  # type: ignore[arg-type]
                         deleted_at=_utc(row.deleted_at), revision=row.revision, history=history)


# --- routes --------------------------------------------------------------------------------------------------------------

@router.get("/analyses/{aid}/annotations", response_model=AnnotationList, responses=problem_responses(401, 404, 422))
def list_annotations(aid: str, target_type: TargetType | None = None, target_id: str | None = Query(None, max_length=255),
                     include_deleted: bool = False) -> JSONResponse:
    with get_session() as db:
        _analysis(db, aid)
        query = select(AnnotationRow).where(AnnotationRow.analysis_id == aid)
        if target_type is not None:
            query = query.where(AnnotationRow.target_type == target_type)
        if target_id is not None:
            query = query.where(AnnotationRow.target_id == target_id)
        if not include_deleted:
            query = query.where(AnnotationRow.deleted_at.is_(None))
        rows = db.scalars(query.order_by(AnnotationRow.created_at, AnnotationRow.id)).all()
        histories = _histories(db, [r.id for r in rows])
        return _json(AnnotationList(items=[_out(r, histories[r.id]) for r in rows]))


@router.post("/analyses/{aid}/annotations", status_code=201, response_model=AnnotationOut, responses=problem_responses(401, 403, 404, 409, 422))
def create_annotation(aid: str, req: AnnotationCreate, request: Request) -> JSONResponse:
    principal: Principal = request.state.principal
    body = clean_body(req.body)

    def unit() -> AnnotationOut:
        with audit.writing() as db:  # N-9: a note or an examiner status is stored with its custody event or not at all
            analysis = _analysis(db, aid)
            _writable(analysis)
            geometry = _validated(db, analysis, target_type=req.target_type, target_id=req.target_id, shape=req.shape,
                                  geometry=req.geometry, examiner_status=req.examiner_status, body=body)
            now = utcnow()
            row = AnnotationRow(id=str(uuid.uuid4()), analysis_id=aid, case_id=analysis.case_id, author_id=principal.id,
                                target_type=req.target_type, target_id=req.target_id, shape=req.shape, geometry=geometry,
                                examiner_status=req.examiner_status, body=body, visibility=req.visibility, created_at=now,
                                revision=1)
            first = _revision(row, principal.id, now)
            _still_writable(db, aid)
            db.add(row)
            db.flush()  # the parent row first: see `storage/db.py` ("Insert order")
            db.add(first)
            out = _out(row, [_revision_out(first)])
            audit.commit_with(db, [{
                "action": "note_added" if req.shape == "none" else "annotation_created", "actor_id": principal.id,
                "analysis_id": aid, "object_ids": [out.id], "after_hash": canonical_hash(_state(row)),
                "details": {"target_type": out.target_type, "target_id": out.target_id, "shape": out.shape,
                            "examiner_status": out.examiner_status, "visibility": out.visibility}}])
            return out

    out = audit.repeated(unit)
    log.info("annotation.created", annotation_id=out.id, analysis_id=aid, target_type=out.target_type, by=principal.id)
    return _json(out, 201)


@router.patch("/annotations/{annotation_id}", response_model=AnnotationOut, responses=problem_responses(401, 403, 404, 409, 422))
def update_annotation(annotation_id: str, req: AnnotationPatch, request: Request) -> JSONResponse:
    """Change the body, status, geometry or visibility; the target and shape stay. The new state becomes the next revision."""
    principal: Principal = request.state.principal
    fields = sorted(req.model_fields_set)

    def unit() -> None:
        with audit.writing() as db:  # N-9: as `create_annotation`
            row = _annotation(db, annotation_id)
            _author_or_administrator(row, principal)
            if row.deleted_at is not None:
                raise ProblemError(409, "Annotation deleted", "A deleted annotation cannot be changed.")
            _writable(_analysis(db, row.analysis_id))
            if not fields:
                raise _invalid("Nothing to change: send body, examiner_status, geometry or visibility.")
            if "visibility" in fields and req.visibility is None:
                raise _invalid("visibility cannot be null.")
            body = clean_body(req.body) if "body" in fields else row.body
            status = req.examiner_status if "examiner_status" in fields else row.examiner_status
            geometry = _validated(db, _analysis(db, row.analysis_id), target_type=row.target_type, target_id=row.target_id,
                                  shape=row.shape, geometry=req.geometry if "geometry" in fields else row.geometry,
                                  examiner_status=status, body=body)
            before = canonical_hash(_state(row))
            _still_writable(db, row.analysis_id)
            now = utcnow()
            row.body, row.examiner_status, row.geometry = body, status, geometry
            row.visibility = req.visibility or row.visibility
            row.revision, row.edited_at = row.revision + 1, now
            db.add(_revision(row, principal.id, now))
            audit.commit_with(db, [{
                "action": "annotation_edited", "actor_id": principal.id, "analysis_id": row.analysis_id,
                "object_ids": [annotation_id], "before_hash": before, "after_hash": canonical_hash(_state(row)),
                "details": {"revision": row.revision, "fields": fields, "examiner_status": status}}])

    audit.repeated(unit)
    with get_session() as db:
        row = _annotation(db, annotation_id)
        return _json(_out(row, _histories(db, [annotation_id])[annotation_id]))


@router.delete("/annotations/{annotation_id}", status_code=204, responses=problem_responses(401, 403, 404, 409))
def delete_annotation(annotation_id: str, request: Request) -> Response:
    """Soft delete: the annotation leaves the examiner layer and its history stays (`?include_deleted=true`)."""
    principal: Principal = request.state.principal

    def unit() -> None:
        with audit.writing() as db:  # N-9: as `create_annotation`
            row = _annotation(db, annotation_id)
            _author_or_administrator(row, principal)
            if row.deleted_at is not None:
                raise ProblemError(409, "Annotation deleted", "The annotation is already deleted.")
            _writable(_analysis(db, row.analysis_id))
            before = canonical_hash(_state(row))
            _still_writable(db, row.analysis_id)
            row.deleted_at = utcnow()
            audit.commit_with(db, [{"action": "annotation_deleted", "actor_id": principal.id, "analysis_id": row.analysis_id,
                                    "object_ids": [annotation_id], "before_hash": before}])

    audit.repeated(unit)
    log.info("annotation.deleted", annotation_id=annotation_id, by=principal.id)
    return Response(status_code=204)
