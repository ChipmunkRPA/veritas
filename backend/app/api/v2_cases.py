"""API v2 cases, membership, intake and media (ARCHITECTURE §4.7, §5.2; PRD CM-01, CM-02, CM-04, CC-01; UX §5.1, §5.12;
BUILD_SPEC C-2).

- **Cases.**
  - `POST /cases`: the creator becomes owner. References are unique; without one, the next `VX-<year>-<nnn>` is assigned.
  - `GET /cases`: the caller's cases (administrators see every case), newest first.
  - `GET/PATCH /cases/{id}`. A quick analysis creates an implicit case (`implicit: true`, reference `UNFILED-…`).
- **Members.** `GET /cases/{id}/members`; `PUT/DELETE /cases/{id}/members/{principal_id}` by the owner or an administrator.
  Case roles are owner, examiner, reviewer and viewer (`app.cases.access`). The last owner cannot be removed or demoted.
  `GET /cases/{id}/members/candidates` (owner or administrator; D-67): the enabled principals that are not members yet, by
  id, name and instance role, so that an owner can add a member without the principals directory.
- **Closing** (D-70): `PATCH /cases/{id}` with `status` `closed` or `open`, by the owner or an administrator. A closed case
  takes no new items (`app.cases.intake`). The `case_updated` event records the fields and the new value of each rule.
- **Intake.** `POST /cases/{id}/items` (multipart with the acquisition fields; `?analyze=true&profile=…`) returns
  `{case_id, item, intake, analysis, new_media, new_item}` (`app.cases.intake`).
- **Items and media.**
  - `GET /cases/{id}/items`: items by exhibit label, each with its intake records and analyses.
  - `GET /media/{id}`: only the cases and intake records the caller can see (CM-04).
  - `GET /media/{id}/original` re-hashes the write-once original before serving it; a corrupt object is never served.
- **Access.** A case resource of a case the caller is not a member of is not found (404); a member whose case role is too
  low gets 403 (`security.authorize`).
"""
from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_, func, or_, select
from sqlalchemy import case as sql_case
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api import jobs
from app.api.auth import Principal
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import RequestModel, Text
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
from app.api.v2_evidence import AnalysisSummaryV2
from app.api.v2_evidence import _summary as analysis_summary
from app.audit import chain as audit
from app.cases import access, intake, review
from app.core import times
from app.core.logging import get_logger
from app.storage import files, locks
from app.storage.db import (
    Analysis,
    CaseMediaRow,
    CaseMemberRow,
    CaseRow,
    IntakeRecordRow,
    MediaObject,
    PrincipalRow,
    get_session,
    utcnow,
)
from app.storage.objects import default_store

log = get_logger(__name__)
router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
IMMUTABLE = "private, max-age=31536000, immutable"
CLIENT_SHA256 = r"^[0-9A-Fa-f]{64}$"
NON_NULL = frozenset({"title", "classification", "requires_second_review", "legal_hold", "status"})
CaseStatus = Literal["open", "in_review", "signed_off", "closed"]
Classification = Literal["public", "internal", "confidential", "restricted"]
CaseRole = Literal["owner", "examiner", "reviewer", "viewer"]
AcquisitionMethod = Literal["not_stated", "custodian_upload", "url_download", "screen_capture", "email_forward", "quick_upload", "other"]
HashMatch = Literal["match", "mismatch", "not_provided"]
MediaKind = Literal["image", "video", "audio", "document"]
Profile = Literal["quick", "full", "deep"]


# --- models ------------------------------------------------------------------------------------------------------------------

RETENTION_UNTIL = "With a UTC offset, for example 2031-03-01T00:00:00Z; between the years 1970 and 9000; kept in UTC"


class CaseCreate(RequestModel):
    model_config = ConfigDict(extra="forbid")
    reference: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,63}$",
                                  description="Unique case number; the next VX-<year>-<nnn> when omitted")
    title: str = Field(min_length=1, max_length=255)
    matter_reference: str | None = Field(default=None, max_length=255, description="Client or matter reference")
    description: str | None = Field(default=None, max_length=10000)
    classification: Classification = "internal"
    case_type: str | None = Field(default=None, max_length=64)
    requires_second_review: bool = False
    legal_hold: bool = False
    retention_until: datetime | None = Field(default=None, description=RETENTION_UNTIL)


class CasePatch(RequestModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=255)
    matter_reference: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=10000)
    classification: Classification | None = None
    case_type: str | None = Field(default=None, max_length=64)
    requires_second_review: bool | None = None
    legal_hold: bool | None = None
    retention_until: datetime | None = Field(default=None, description=RETENTION_UNTIL)
    status: Literal["open", "closed"] | None = Field(default=None, description="Close or reopen the case (owner or administrator)")


class CaseOut(BaseModel):
    id: str
    reference: str
    title: str
    matter_reference: str | None
    description: str | None
    classification: Classification
    case_type: str | None
    requires_second_review: bool
    status: CaseStatus
    legal_hold: bool
    retention_until: datetime | None
    implicit: bool = Field(description="Created by a quick analysis")
    created_by: str
    created_at: datetime
    updated_at: datetime
    my_role: CaseRole | None = Field(description="The caller's case role; null for an administrator who is not a member")
    item_count: int
    analysis_count: int
    review_counts: dict[str, int] = Field(default_factory=dict, description="Analysis revisions by their recorded review state")
    demo: bool = Field(default=False, description="Every item of the case was taken in by the guided demo: no work for an examiner")


class MemberOut(BaseModel):
    case_id: str
    principal_id: str
    username: str
    display_name: str
    principal_role: str
    case_role: CaseRole
    added_by: str
    added_at: datetime


class MemberList(BaseModel):
    items: list[MemberOut]


class MemberCandidate(BaseModel):
    principal_id: str
    username: str
    display_name: str
    principal_role: str


class MemberCandidates(BaseModel):
    items: list[MemberCandidate]


class MemberPut(RequestModel):
    model_config = ConfigDict(extra="forbid")
    role: CaseRole


class MediaOut(BaseModel):
    id: str
    sha256: str
    md5: str | None
    size_bytes: int
    mime_sniffed: str | None
    original_filename: str | None
    first_seen_at: datetime


class IntakeOut(BaseModel):
    id: str
    media_id: str
    case_id: str
    supplied_by: str | None
    acquisition_method: AcquisitionMethod
    source: str | None
    date_acquired: datetime | None
    custodian_stated_hash: str | None
    custodian_hash_match: HashMatch
    client_sha256: str | None = Field(description="The browser pre-hash, when the client sent one (it equalled sha256)")
    notes: str | None
    intake_principal_id: str
    intake_at: datetime
    audit_event_seq: int | None


class ItemOut(BaseModel):
    case_id: str
    label: str = Field(description="Exhibit label within the case (A, B, …)")
    media: MediaOut
    added_by: str
    added_at: datetime
    intakes: list[IntakeOut]
    analysis_ids: list[str]


class IntakeResult(BaseModel):
    case_id: str
    item: ItemOut
    intake: IntakeOut
    analysis: AnalysisSummaryV2 | None
    new_media: bool = Field(description="False when these bytes were taken in before and their media object is reused")
    new_item: bool = Field(description="False when the media object was already an item of this case")


class MediaCase(BaseModel):
    case_id: str
    reference: str
    title: str
    label: str


class MediaDetail(MediaOut):
    cases: list[MediaCase]
    intakes: list[IntakeOut]


# --- helpers ---------------------------------------------------------------------------------------------------------------

_utc = times.to_utc  # one form on every engine: a time is printed in UTC (D-72)


def _client_date(name: str, value: datetime | None) -> datetime | None:
    """A date of the request in UTC, or 422: it needs an offset and a year every engine can store and read back (D-72)."""
    try:
        return times.client_date(name, value)
    except ValueError as e:
        raise ProblemError(422, "Invalid request", str(e)) from e


def _json(model: BaseModel, status_code: int = 200) -> JSONResponse:
    return JSONResponse(jsonable_encoder(model), status_code=status_code, headers=NO_STORE)


def _text(value: str | None) -> str | None:
    return value.strip() or None if value is not None else None


def _case(db: Session, case_id: str) -> CaseRow:
    row = db.get(CaseRow, case_id)
    if row is None:
        raise ProblemError(404, *access.NOT_FOUND["case"])
    return row


def _case_outs(db: Session, rows: Sequence[CaseRow], principal: Principal) -> list[CaseOut]:
    ids = [r.id for r in rows]
    if not ids:
        return []
    items = {k: v for k, v in db.execute(select(CaseMediaRow.case_id, func.count()).where(CaseMediaRow.case_id.in_(ids))
                                         .group_by(CaseMediaRow.case_id)).tuples()}
    review_counts: dict[str, dict[str, int]] = defaultdict(dict)
    review_rows = select(Analysis.case_id, review.state_expression().label("review_state")).where(Analysis.case_id.in_(ids)).subquery()
    for case_id, review_state, count in db.execute(select(review_rows.c.case_id, review_rows.c.review_state, func.count())
            .group_by(review_rows.c.case_id, review_rows.c.review_state)).tuples():
        review_counts[case_id][review_state] = count
    # N-19d: a case that holds only what the guided demo took in (its intake records say so)
    by_demo = and_(IntakeRecordRow.supplied_by == intake.DEMO_SUPPLIED_BY, IntakeRecordRow.source == intake.DEMO_SOURCE)
    arrivals = {case_id: (total, of_demo or 0) for case_id, total, of_demo in db.execute(
        select(IntakeRecordRow.case_id, func.count(), func.sum(sql_case((by_demo, 1), else_=0)))
        .where(IntakeRecordRow.case_id.in_(ids)).group_by(IntakeRecordRow.case_id)).tuples()}
    demo = {case_id for case_id, (total, of_demo) in arrivals.items() if total > 0 and of_demo == total}
    roles = access.case_roles(db, principal.id, set(ids))
    return [CaseOut(id=r.id, reference=r.reference, title=r.title, matter_reference=r.matter_reference, description=r.description,
                    classification=r.classification, case_type=r.case_type,  # type: ignore[arg-type]
                    requires_second_review=r.requires_second_review, status=r.status, legal_hold=r.legal_hold,  # type: ignore[arg-type]
                    retention_until=_utc(r.retention_until), implicit=r.implicit, created_by=r.created_by,
                    created_at=_utc(r.created_at), updated_at=_utc(r.updated_at), my_role=roles.get(r.id),  # type: ignore[arg-type]
                    item_count=items.get(r.id, 0), analysis_count=sum(review_counts[r.id].values()),
                    review_counts=review_counts[r.id], demo=r.id in demo) for r in rows]


def _media_out(m: MediaObject) -> MediaOut:
    return MediaOut(id=m.id, sha256=m.sha256, md5=m.md5, size_bytes=int(m.size_bytes), mime_sniffed=m.mime_sniffed,
                    original_filename=m.original_filename, first_seen_at=_utc(m.first_seen_at))  # type: ignore[arg-type]


def _intake_out(r: IntakeRecordRow) -> IntakeOut:
    return IntakeOut(id=r.id, media_id=r.media_id, case_id=r.case_id, supplied_by=r.supplied_by,
                     acquisition_method=r.acquisition_method, source=r.source, date_acquired=_utc(r.date_acquired),  # type: ignore[arg-type]
                     custodian_stated_hash=r.custodian_stated_hash, custodian_hash_match=r.custodian_hash_match,  # type: ignore[arg-type]
                     client_sha256=r.client_sha256, notes=r.notes, intake_principal_id=r.intake_principal_id,
                     intake_at=_utc(r.intake_at), audit_event_seq=r.audit_event_seq)  # type: ignore[arg-type]


def _items(db: Session, case_id: str, media_ids: set[str] | None = None) -> list[ItemOut]:
    """The case's items by exhibit label (A … Z, AA …), each with its intake records and analyses, oldest first."""
    query = select(CaseMediaRow, MediaObject).join(MediaObject, MediaObject.id == CaseMediaRow.media_id).where(CaseMediaRow.case_id == case_id)
    if media_ids is not None:
        query = query.where(CaseMediaRow.media_id.in_(media_ids))
    intakes: dict[str, list[IntakeOut]] = defaultdict(list)
    for record in db.scalars(select(IntakeRecordRow).where(IntakeRecordRow.case_id == case_id)
                             .order_by(IntakeRecordRow.intake_at, IntakeRecordRow.id)):
        intakes[record.media_id].append(_intake_out(record))
    analyses: dict[str, list[str]] = defaultdict(list)
    names: dict[str, str] = {}
    for media_id, aid, filename in db.execute(select(Analysis.media_id, Analysis.id, Analysis.filename).where(Analysis.case_id == case_id)
                                              .order_by(Analysis.created_at, Analysis.id)).tuples():
        analyses[media_id or ""].append(aid)
        names.setdefault(media_id or "", filename)
    first_case: dict[str, str] = {}
    held = select(CaseMediaRow.media_id).where(CaseMediaRow.case_id == case_id)
    for media_id, holder in db.execute(select(IntakeRecordRow.media_id, IntakeRecordRow.case_id).where(IntakeRecordRow.media_id.in_(held))
                                       .order_by(IntakeRecordRow.intake_at, IntakeRecordRow.id)).tuples():
        first_case.setdefault(media_id, holder)
    items = []
    for item, media in db.execute(query).tuples():
        media_out = _media_out(media)
        if first_case.get(media.id) != case_id:
            # The bytes arrived in another case first, which the caller may not see: show this case's filename and first intake,
            # never the first uploader's (Q-2 F3; CM-04)
            media_out = media_out.model_copy(update={"original_filename": names.get(media.id), "first_seen_at": min(
                (i.intake_at for i in intakes[media.id]), default=_utc(item.added_at))})
        items.append(ItemOut(case_id=case_id, label=item.label, media=media_out, added_by=item.added_by, added_at=_utc(item.added_at),  # type: ignore[arg-type]
                             intakes=intakes[media.id], analysis_ids=analyses[media.id]))
    return sorted(items, key=lambda i: (len(i.label), i.label))


def _listed(words: Sequence[str]) -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} or {words[-1]}"


def _owner_or_administrator(db: Session, case_id: str, principal: Principal) -> None:
    if principal.satisfies("administrator"):
        return
    member = db.get(CaseMemberRow, (case_id, principal.id))
    if member is None or member.role != "owner":
        raise ProblemError(403, "Forbidden", "Only the case owner or an administrator can change the members of a case.")


def _keeps_an_owner(db: Session, case_id: str, principal_id: str) -> None:
    """Refuse (409) to take the owner role from the case's only owner. The caller holds the lock on the case's membership
    (`locks.serialise(db, "case-members", case_id)`), so the count is of what is committed and stays so until its commit."""
    others = db.scalar(select(func.count()).select_from(CaseMemberRow).where(
        CaseMemberRow.case_id == case_id, CaseMemberRow.role == "owner", CaseMemberRow.principal_id != principal_id))
    if not others:
        raise ProblemError(409, "Last owner", "A case keeps at least one owner: make another member owner first.")


def _member_out(member: CaseMemberRow, row: PrincipalRow) -> MemberOut:
    return MemberOut(case_id=member.case_id, principal_id=row.id, username=row.username, display_name=row.display_name,
                     principal_role=row.role, case_role=member.role, added_by=member.added_by,  # type: ignore[arg-type]
                     added_at=_utc(member.added_at))  # type: ignore[arg-type]


# --- cases -----------------------------------------------------------------------------------------------------------------

@router.post("/cases", status_code=201, response_model=CaseOut, responses=problem_responses(401, 403, 409, 422))
def create_case(req: CaseCreate, request: Request) -> JSONResponse:
    principal: Principal = request.state.principal
    if req.reference is not None and req.reference.upper().startswith(intake.IMPLICIT_PREFIX):
        raise ProblemError(422, "Invalid request", f"References starting with {intake.IMPLICIT_PREFIX} are kept for implicit cases.")
    fields = req.model_dump(exclude={"reference", "title"})
    fields["retention_until"] = _client_date("retention_until", req.retention_until)
    def unit() -> JSONResponse | None:
        with audit.writing() as db:  # N-9: the case and its `case_created` event are one transaction
            try:
                row = intake.new_case(db, principal.id, title=req.title, reference=req.reference, **fields)
                db.flush()
            except IntegrityError as e:
                db.rollback()
                if req.reference is not None:
                    raise ProblemError(409, "Reference taken", f"A case with reference {req.reference} already exists.") from e
                return None  # a concurrent request took the next VX reference
            audit.commit_with(db, [{"action": "case_created", "actor_id": principal.id, "case_id": row.id,
                                    "details": {"reference": row.reference, "title": row.title, "implicit": False}}])
            log.info("case.created", case_id=row.id, reference=row.reference, by=principal.id)
            return _json(_case_outs(db, [row], principal)[0], 201)

    for _ in range(3):
        if (created := audit.repeated(unit)) is not None:
            return created
    raise ProblemError(409, "Reference taken", "The next case reference was taken by concurrent requests; retry.")


@router.get("/cases", response_model=Page[CaseOut], responses=problem_responses(400, 401, 422))
def list_cases(request: Request, cursor: Cursor = None, limit: Limit = DEFAULT_LIMIT, status: CaseStatus | None = None,
               implicit: bool | None = None) -> JSONResponse:
    """The caller's cases, newest first; `implicit=false` hides the implicit cases of quick analyses."""
    principal: Principal = request.state.principal
    after = decode_cursor(cursor, 2)
    query = select(CaseRow)
    if (visible := access.visible_cases(principal)) is not None:
        query = query.where(visible)
    for column, value in ((CaseRow.status, status), (CaseRow.implicit, implicit)):
        if value is not None:
            query = query.where(column == value)
    if after is not None:
        try:
            ts = datetime.fromisoformat(str(after[0]))
        except ValueError:
            raise ProblemError(400, "Invalid cursor", "The cursor does not belong to this collection.") from None
        query = query.where(or_(CaseRow.created_at < ts, and_(CaseRow.created_at == ts, CaseRow.id < str(after[1]))))
    with get_session() as db:
        rows = db.scalars(query.order_by(CaseRow.created_at.desc(), CaseRow.id.desc()).limit(limit + 1)).all()
        page = rows[:limit]
        next_cursor = encode_cursor([page[-1].created_at.isoformat(), page[-1].id]) if len(rows) > limit else None
        return _json(Page[CaseOut](items=_case_outs(db, page, principal), next_cursor=next_cursor))


@router.get("/cases/{case_id}", response_model=CaseOut, responses=problem_responses(401, 404))
def get_case(case_id: str, request: Request) -> JSONResponse:
    with get_session() as db:
        return _json(_case_outs(db, [_case(db, case_id)], request.state.principal)[0])


RECEIVED_CLOCK_SKEW = timedelta(minutes=5)
RECEIVED_AFTER_INTAKE = ("The stated time of receipt is later than this intake. A file is received before it is taken in: correct "
                         "the time, or leave it empty.")

# Review and retention governance: a case member cannot switch off second review or a legal hold for their own case (Q-2 F4)
ADMINISTRATOR_FIELDS = frozenset({"requires_second_review", "legal_hold", "retention_until"})
RECORDED_VALUES = frozenset({"classification", "status"}) | ADMINISTRATOR_FIELDS


@router.patch("/cases/{case_id}", response_model=CaseOut, responses=problem_responses(401, 403, 404, 422))
def update_case(case_id: str, req: CasePatch, request: Request) -> JSONResponse:
    principal: Principal = request.state.principal

    def unit() -> JSONResponse:
        # N-9: a legal hold, a retention date or the second-review rule is changed with its `case_updated` event or not at all
        with audit.writing() as db:
            _case(db, case_id)
            # N-9 (Postgres QA PG-15): the row's lock, taken before anything is decided. A purge that has re-checked this
            # case's chains holds the row share-locked until it commits, so a hold either comes first and keeps them, or
            # waits here and applies to a case whose expired chains are gone; it is never stored before the purge deletes
            row = db.scalar(select(CaseRow).where(CaseRow.id == case_id).with_for_update().execution_options(populate_existing=True))
            if row is None:  # an implicit case that went with its last chain while this request waited
                raise ProblemError(404, *access.NOT_FOUND["case"])
            governed = sorted(ADMINISTRATOR_FIELDS & req.model_fields_set)
            if governed and not principal.satisfies("administrator"):
                raise ProblemError(403, "Forbidden", f"Only an administrator can change {', '.join(governed)}.")
            # D-70 (N-19j; browser QA Q2-10): every other field of a case is its owner's or an administrator's to change. The
            # route asked the owner only for `status`: an examiner of the case who is not its owner changed its title and
            # its classification, Confidential to Public
            if req.model_fields_set and not principal.satisfies("administrator"):
                member = db.get(CaseMemberRow, (case_id, principal.id))
                if member is None or member.role != "owner":
                    if req.model_fields_set == {"status"}:
                        raise ProblemError(403, "Forbidden", "Only the case owner or an administrator can close or reopen a case.")
                    raise ProblemError(403, "Forbidden", "Only the case owner or an administrator can change a case.")
            before = _case_hash(row)
            for field in sorted(req.model_fields_set):
                value = getattr(req, field)
                if field == "retention_until":
                    value = _client_date(field, value)
                if value is None and field in NON_NULL:
                    raise ProblemError(422, "Invalid request", f"{field} cannot be null.")
                setattr(row, field, value)
            row.updated_at = utcnow()
            fields = sorted(req.model_fields_set)
            # D-70: the custody log shows each change of a rule. The new value of a rule is recorded; a title, a reference
            # and a description are free text and are named only
            values = {f: _recorded(getattr(row, f)) for f in fields if f in RECORDED_VALUES}
            audit.commit_with(db, [{"action": "case_updated", "actor_id": principal.id, "case_id": case_id, "before_hash": before,
                                    "after_hash": _case_hash(row), "details": {"fields": fields, "values": values}}])
            log.info("case.updated", case_id=case_id, fields=fields, by=principal.id)
            return _json(_case_outs(db, [row], principal)[0])

    return audit.repeated(unit)


def _recorded(value: object) -> object:
    """A rule's new value as the `case_updated` event keeps it: a date in UTC, as it is printed."""
    return _utc(value).isoformat() if isinstance(value, datetime) else value


def _case_hash(row: CaseRow) -> str:
    """Canonical hash of a case's metadata: the before and after hashes of `case_updated`."""
    from app.core.canonical import canonical_hash

    return canonical_hash({"reference": row.reference, "title": row.title, "matter_reference": row.matter_reference,
                           "description": row.description, "classification": row.classification, "case_type": row.case_type,
                           "requires_second_review": row.requires_second_review, "legal_hold": row.legal_hold,
                           "retention_until": _utc(row.retention_until), "status": row.status})


# --- members ---------------------------------------------------------------------------------------------------------------

@router.get("/cases/{case_id}/members", response_model=MemberList, responses=problem_responses(401, 404))
def list_members(case_id: str) -> JSONResponse:
    with get_session() as db:
        _case(db, case_id)
        rows = db.execute(select(CaseMemberRow, PrincipalRow).join(PrincipalRow, PrincipalRow.id == CaseMemberRow.principal_id)
                          .where(CaseMemberRow.case_id == case_id).order_by(CaseMemberRow.added_at, PrincipalRow.username)).tuples()
        return _json(MemberList(items=[_member_out(m, p) for m, p in rows]))


@router.get("/cases/{case_id}/members/candidates", response_model=MemberCandidates, responses=problem_responses(401, 403, 404))
def list_member_candidates(case_id: str, request: Request) -> JSONResponse:
    """Who can be added to the case (owner or administrator; D-67): the enabled principals that are not members of it."""
    with get_session() as db:
        _case(db, case_id)
        _owner_or_administrator(db, case_id, request.state.principal)
        members = select(CaseMemberRow.principal_id).where(CaseMemberRow.case_id == case_id)
        # N-19e (Postgres QA PG2-04): ordered in code, the same on every engine. In SQL the order was the collation's:
        # "_Underscore, 10 Ten, …" on a database in en_US.UTF-8, "10 Ten, …, Zed Upper, _Underscore, adam lower" on SQLite
        rows = sorted(db.scalars(select(PrincipalRow).where(PrincipalRow.disabled_at.is_(None), PrincipalRow.id.not_in(members))),
                      key=_name_order)
        return _json(MemberCandidates(items=[MemberCandidate(principal_id=p.id, username=p.username, display_name=p.display_name,
                                                             principal_role=p.role) for p in rows]))


def _name_order(principal: PrincipalRow) -> tuple[str, str, str]:
    """The order of a list of people: by the display name without regard to case and accents (NFKD, the combining marks
    dropped, `casefold()`), then by the display name as stored, then by the username."""
    name = principal.display_name or ""
    plain = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)).casefold()
    return plain, name, principal.username


@router.put("/cases/{case_id}/members/{principal_id}", response_model=MemberOut, responses=problem_responses(401, 403, 404, 409, 422))
def put_member(case_id: str, principal_id: str, req: MemberPut, request: Request) -> JSONResponse:
    """Add a member or change its case role (owner or administrator)."""
    principal: Principal = request.state.principal

    def unit() -> JSONResponse:
        with audit.writing() as db:  # N-9: who may act on a case changes with its `permission_changed` event or not at all
            # N-9a (Postgres QA X-4): one change of a case's membership at a time, so "another owner is left" is still true
            # when this change commits. Two owners who demote each other at the same moment: the second finds itself no owner
            locks.serialise(db, "case-members", case_id)
            _case(db, case_id)
            _owner_or_administrator(db, case_id, principal)
            target = db.get(PrincipalRow, principal_id)
            if target is None:
                raise ProblemError(404, "Principal not found", "No principal exists with this id.")
            # D-67 (N-19j; browser QA Q2-10): a case role up to the member's instance role, also for a member that exists
            allowed = access.case_roles_for(target.role)
            if req.role not in allowed:
                raise ProblemError(422, "Invalid request", f"{target.display_name} has the instance role {target.role}, which can hold "
                                                           f"the case role{'s' if len(allowed) != 1 else ''} {_listed(allowed)}, not {req.role}.")
            member = db.get(CaseMemberRow, (case_id, principal_id))
            previous = member.role if member is not None else None
            if member is None:
                member = CaseMemberRow(case_id=case_id, principal_id=principal_id, role=req.role, added_by=principal.id,
                                       added_at=utcnow())
                db.add(member)
            elif member.role != req.role:
                if member.role == "owner":
                    _keeps_an_owner(db, case_id, principal_id)
                member.role = req.role
            audit.commit_with(db, [{"action": "permission_changed", "actor_id": principal.id, "case_id": case_id,
                                    "object_ids": [principal_id],
                                    "details": {"change": "case_member_set", "principal_id": principal_id, "case_role": req.role,
                                                "previous": previous, "principal_name": target.display_name}}])
            log.info("case.member_set", case_id=case_id, principal_id=principal_id, case_role=req.role, by=principal.id)
            return _json(_member_out(member, target))

    return audit.repeated(unit)


@router.delete("/cases/{case_id}/members/{principal_id}", status_code=204, responses=problem_responses(401, 403, 404, 409))
def delete_member(case_id: str, principal_id: str, request: Request) -> Response:
    principal: Principal = request.state.principal

    def unit() -> None:
        with audit.writing() as db:  # N-9: as `put_member`
            locks.serialise(db, "case-members", case_id)  # N-9a: the check and the change under one lock, as `put_member`
            _case(db, case_id)
            _owner_or_administrator(db, case_id, principal)
            member = db.get(CaseMemberRow, (case_id, principal_id))
            if member is None:
                raise ProblemError(404, "Member not found", "The principal is not a member of this case.")
            if member.role == "owner":
                _keeps_an_owner(db, case_id, principal_id)
            previous = member.role
            target = db.get(PrincipalRow, principal_id)
            db.delete(member)
            # N-19a: the name the member had, so that the log of the case names who was removed once the member list no longer does
            audit.commit_with(db, [{"action": "permission_changed", "actor_id": principal.id, "case_id": case_id,
                                    "object_ids": [principal_id],
                                    "details": {"change": "case_member_removed", "principal_id": principal_id, "case_role": None,
                                                "previous": previous,
                                                "principal_name": target.display_name if target is not None else None}}])

    audit.repeated(unit)
    log.info("case.member_removed", case_id=case_id, principal_id=principal_id, by=principal.id)
    return Response(status_code=204)


# --- items and media -------------------------------------------------------------------------------------------------------

@router.post("/cases/{case_id}/items", status_code=201, response_model=IntakeResult,
             responses=problem_responses(400, 401, 403, 404, 409, 413, 415, 422, 500, 503))
def add_item(case_id: str, request: Request, file: UploadFile = File(...), kind: MediaKind = Form(...),
             client_sha256: str | None = Form(None, pattern=CLIENT_SHA256,
                                              description="The SHA-256 the browser computed before upload (UX §5.1)"),
             acquisition_method: AcquisitionMethod = Form("not_stated"),
             supplied_by: Text | None = Form(None, max_length=255, description="Who supplied the item"),
             source: Text | None = Form(None, max_length=2048, description="Source URL or device"),
             date_acquired: datetime | None = Form(None, description="When the item was acquired, with a UTC offset"),
             custodian_stated_hash: Text | None = Form(None, max_length=80,
                                                      description="SHA-256 or MD5 hex stated by the custodian (optional sha256: / md5: prefix)"),
             notes: Text | None = Form(None, max_length=4000),
             analyze: bool = Query(False, description="Queue an analysis of the item"),
             profile: Profile | None = Query(None, description="Analysis profile (with analyze=true)")) -> JSONResponse:
    """Take an item into the case: SHA-256 and MD5, custodian hash comparison, write-once original, intake record (CC-01)."""
    principal: Principal = request.state.principal
    date_acquired = _client_date("date_acquired", date_acquired)
    # N-19j (browser QA Q2-06): a stated time of receipt later than the intake is refused. It was accepted up to a day
    # ahead: "acquired (as stated) 4 Oct 2026 15:05 UTC" stood on a file taken in at 12:25 UTC. A few minutes are left for
    # the difference between the clocks of the browser and of the server
    if date_acquired is not None and date_acquired > utcnow() + RECEIVED_CLOCK_SKEW:
        raise ProblemError(422, "Invalid request", RECEIVED_AFTER_INTAKE)
    if profile is not None and not analyze:
        raise ProblemError(422, "Invalid request", "profile applies only with analyze=true.")
    acquisition = intake.Acquisition(method=acquisition_method, supplied_by=_text(supplied_by), source=_text(source),
                                     date_acquired=date_acquired, custodian_stated_hash=_text(custodian_stated_hash),
                                     notes=_text(notes), client_sha256=client_sha256)
    try:
        with intake.upload_slot():
            outcome = intake.take_in(file.file, file.filename or f"upload.{kind}", kind, principal_id=principal.id, case_id=case_id,
                                     acquisition=acquisition, analyze=analyze, profile=profile,
                                     job_priority=jobs.PRIORITY_INTERACTIVE)  # N-9: queued with the row; `submit` wakes the worker
    except intake.IntakeError as e:
        raise ProblemError(e.status, e.title, e.detail) from e
    if outcome.analysis is not None:
        jobs.submit(outcome.analysis.id)
    with get_session() as db:
        [item] = _items(db, outcome.case.id, {outcome.media.id})
        # Reuse of bytes taken in by a case the caller cannot see is not disclosed (Q-2 F3; CM-04)
        elsewhere = select(CaseMediaRow.case_id).join(CaseRow, CaseRow.id == CaseMediaRow.case_id).where(
            CaseMediaRow.media_id == outcome.media.id, CaseMediaRow.case_id != outcome.case.id)
        if (visible := access.visible_cases(principal)) is not None:
            elsewhere = elsewhere.where(visible)
        seen_elsewhere = db.scalars(elsewhere.limit(1)).first() is not None
    analysis = AnalysisSummaryV2(**analysis_summary(outcome.analysis)) if outcome.analysis is not None else None
    return _json(IntakeResult(case_id=outcome.case.id, item=item, intake=_intake_out(outcome.intake), analysis=analysis,
                              new_media=outcome.new_media or (outcome.new_item and not seen_elsewhere), new_item=outcome.new_item), 201)


@router.get("/cases/{case_id}/items", response_model=Page[ItemOut], responses=problem_responses(400, 401, 404, 422))
def list_items(case_id: str, cursor: Cursor = None, limit: Limit = DEFAULT_LIMIT) -> JSONResponse:
    with get_session() as db:
        _case(db, case_id)
        items = _items(db, case_id)
    page, next_cursor = paginate(items, lambda i: (len(i.label), i.label), cursor, limit)
    return _json(Page[ItemOut](items=page, next_cursor=next_cursor))


@router.get("/media/{media_id}", response_model=MediaDetail, responses=problem_responses(401, 404))
def get_media(media_id: str, request: Request) -> JSONResponse:
    """A media object with the cases and intake records the caller can see (CM-04)."""
    principal: Principal = request.state.principal
    with get_session() as db:
        media = db.get(MediaObject, media_id)
        if media is None:
            raise ProblemError(404, *access.NOT_FOUND["media"])
        query = select(CaseMediaRow, CaseRow).join(CaseRow, CaseRow.id == CaseMediaRow.case_id).where(CaseMediaRow.media_id == media_id)
        if (visible := access.visible_cases(principal)) is not None:
            query = query.where(visible)
        cases = [MediaCase(case_id=c.id, reference=c.reference, title=c.title, label=item.label)
                 for item, c in db.execute(query.order_by(CaseMediaRow.added_at, CaseRow.id)).tuples()]
        records = db.scalars(select(IntakeRecordRow).where(IntakeRecordRow.media_id == media_id,
                                                           IntakeRecordRow.case_id.in_([c.case_id for c in cases]))
                             .order_by(IntakeRecordRow.intake_at, IntakeRecordRow.id)).all()
        out = _media_out(media)
        first_holder = db.scalars(select(IntakeRecordRow.case_id).where(IntakeRecordRow.media_id == media_id)
                                  .order_by(IntakeRecordRow.intake_at, IntakeRecordRow.id).limit(1)).first()
        if visible is not None and first_holder not in {c.case_id for c in cases}:
            # The bytes arrived first in a case the caller cannot see: not the first uploader's name or time (Q-2 F3; CM-04)
            name = db.scalars(select(Analysis.filename).where(Analysis.media_id == media_id, Analysis.case_id.in_([c.case_id for c in cases]))
                              .order_by(Analysis.created_at, Analysis.id).limit(1)).first()
            out = out.model_copy(update={"original_filename": name,
                                         "first_seen_at": _utc(records[0].intake_at) if records else out.first_seen_at})
        return _json(MediaDetail(**out.model_dump(), cases=cases, intakes=[_intake_out(r) for r in records]))


@router.get("/media/{media_id}/original", response_class=FileResponse,
            responses={200: {"content": {"application/octet-stream": {}}, "description": "The write-once original (ETag = sha256)"},
                       **problem_responses(401, 404, 500)})
def media_original(media_id: str, request: Request) -> Response:
    """The original bytes, re-hashed before they are served: `hash_verified` then `original_downloaded` (C-3). The events name
    the case when the caller sees the media object in exactly one case.

    N-19j (browser QA Q2-11): the file is served under the name of the exhibit through which the caller sees it: the name
    it was taken in under in that one case. Where the caller sees it in no case or in more than one, the name is made from
    its hash. It was the name of the first upload of the same bytes on the instance, whoever's case that was."""
    principal: Principal = request.state.principal
    with get_session() as db:
        media = db.get(MediaObject, media_id)
        query = select(CaseMediaRow.case_id).where(CaseMediaRow.media_id == media_id)
        if not principal.satisfies("administrator"):
            query = query.where(CaseMediaRow.case_id.in_(access.member_case_ids(principal.id)))
        case_ids = sorted(db.scalars(query))
        exhibit_name = _exhibit_name(db, media, case_ids[0]) if media is not None and len(case_ids) == 1 else None
    if media is None:
        raise ProblemError(404, *access.NOT_FOUND["media"])
    if media.storage_artefact_id is None:
        raise ProblemError(404, "Original not stored", "The original of this media object has not reached the evidence store yet.")
    case_id = case_ids[0] if len(case_ids) == 1 else None
    try:
        sha256 = audit.verify_original("download", actor_id=principal.id, media_id=media_id, case_id=case_id)
    except audit.OriginalIntegrityError as e:
        if e.missing:
            raise ProblemError(500, "Evidence object unavailable", "The stored original is missing.") from None
        raise ProblemError(500, "Evidence integrity failure",
                           "The stored original no longer matches its recorded hash, so it was not served.") from None
    # N-9: a read. The append comes before the bytes are served, so a failed append serves nothing
    audit.record("original_downloaded", actor_id=principal.id, media_id=media_id, case_id=case_id, after_hash=sha256,
                 details={"case_ids": case_ids} if len(case_ids) > 1 else {})
    headers = {"ETag": f'"{sha256}"', "Cache-Control": IMMUTABLE, "X-Content-Type-Options": "nosniff"}
    return FileResponse(default_store().path_for(sha256), media_type=media.mime_sniffed or "application/octet-stream",
                        filename=files.safe_filename(exhibit_name or _hash_name(sha256, media.mime_sniffed)), headers=headers)


def _exhibit_name(db: Session, media: MediaObject, case_id: str) -> str | None:
    """The file name of a media object as the exhibit of one case (as `_items` gives it): the name of the first analysis of
    it in that case; without one, the media object's own name where the bytes arrived in that case first. None otherwise:
    the name the object carries is then another case's."""
    name = db.scalars(select(Analysis.filename).where(Analysis.media_id == media.id, Analysis.case_id == case_id)
                      .order_by(Analysis.created_at, Analysis.id).limit(1)).first()
    if name:
        return name
    first_holder = db.scalars(select(IntakeRecordRow.case_id).where(IntakeRecordRow.media_id == media.id)
                              .order_by(IntakeRecordRow.intake_at, IntakeRecordRow.id).limit(1)).first()
    return media.original_filename if first_holder == case_id else None


def _hash_name(sha256: str, mime: str | None) -> str:
    """A file name that names no case: the first twelve characters of the hash, with the extension of the sniffed type."""
    import mimetypes

    return sha256[:12] + (mimetypes.guess_extension(mime or "") or "")
