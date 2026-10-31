"""API v2 audit log and custody verification (ARCHITECTURE §4.8, §5.2; PRD CC-01, CC-02; BUILD_SPEC C-3).

- `GET /audit?case_id&action&after_seq&limit`: events in seq order (reviewer or administrator). A reviewer sees the events of
  its own cases only; events outside any case (sign-ins, principal changes) are for administrators.
- `GET /audit/verify`: re-hashes the whole chain → `{valid, first_broken_seq, checked, reason}`.
- `GET /audit/export?format=jsonl|csv&case_id=`: streams the log, with the visibility of `/audit`. Only a whole-chain export
  verifies offline (`python -m app.cli audit verify --jsonl <file>`).
- `GET /cases/{case_id}/custody?after_seq&limit` (D-67): the events of one case in seq order, for its owner, for a reviewer who
  is a member of it and for an administrator. Another member gets 403; to anyone else the case is not found.
- `GET /cases/{case_id}/custody/verify` (case members): the chain, every original of the case (re-hashed and recorded as
  `hash_verified` / `hash_mismatch`) and every generated report against the sha256 its `report_generated` event recorded. A
  report of an analysis that was deleted or purged is listed as `removed` under that event and does not fail the case.
"""
from __future__ import annotations

import itertools
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.auth import Principal
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.v2 import DEFAULT_LIMIT, MAX_SEQ, NO_STORE, Limit, ProblemError, problem_responses
from app.audit import chain
from app.cases import access
from app.storage.db import CaseRow, get_session

router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
CheckStatus = Literal["verified", "mismatch", "missing"]
ReportStatus = Literal["verified", "mismatch", "missing", "removed"]


class AuditEventOut(BaseModel):
    seq: int
    ts: str
    actor_id: str
    # N-19j (browser QA Q2-10): the display name of the actor's account, so that the log names an actor who is no member
    # of the case (an administrator who recorded the second review stood there by id). Null where no account has the id
    actor_name: str | None = None
    action: str
    case_id: str | None
    media_id: str | None
    analysis_id: str | None
    report_id: str | None
    object_ids: list[str]
    before_hash: str | None
    after_hash: str | None
    details: dict[str, Any]
    prev_hash: str
    event_hash: str


class AuditPage(BaseModel):
    items: list[AuditEventOut]
    next_after_seq: int | None = None


class ChainVerification(BaseModel):
    valid: bool
    first_broken_seq: int | None
    checked: int
    reason: str | None = None


class OriginalCheck(BaseModel):
    media_id: str
    label: str
    sha256: str
    status: CheckStatus


class ReportCheck(BaseModel):
    analysis_id: str
    report_id: str | None = Field(default=None, description="The v2 report revision; null for the v1 report file of the analysis")
    file: str | None = Field(default=None, description="The v1 report file its event names (report.pdf, report-<hash>.pdf); null for a v2 report")
    sha256: str | None
    status: ReportStatus = Field(description="removed: the analysis of this report was deleted or purged; the case stays valid")
    removed_by: Literal["deletion_requested", "retention_purge"] | None = Field(
        default=None, description="The custody action that removed the analysis of this report")
    removed_seq: int | None = Field(default=None, description="The seq of that custody event")


class CustodyVerification(BaseModel):
    case_id: str
    valid: bool
    chain: ChainVerification
    originals: list[OriginalCheck]
    reports: list[ReportCheck]


def _json(model: BaseModel) -> JSONResponse:
    return JSONResponse(jsonable_encoder(model), headers=NO_STORE)


def _visible_cases(db: Session, principal: Principal, case_id: str | None) -> list[str] | None:
    """The cases whose events the caller may read (None: every event). A non-member asking for a case gets not found."""
    if principal.satisfies("administrator"):
        return None if case_id is None else [case_id]
    members = set(db.scalars(access.member_case_ids(principal.id)))
    if case_id is None:
        return sorted(members)
    if case_id not in members:
        raise ProblemError(404, *access.NOT_FOUND["case"])
    return [case_id]


def _action(action: str | None) -> str | None:
    if action is not None and action not in chain.ACTIONS:
        raise ProblemError(422, "Invalid request", f"Unknown audit action {action}.")
    return action


def _named(db_events: list[dict[str, Any]]) -> list[AuditEventOut]:
    """The events with the name of each actor that an account, a built-in principal or the engine gives."""
    from app.reports.snapshot import principal_names

    ids = {str(e["actor_id"]) for e in db_events}
    with get_session() as db:
        names = principal_names(db, ids)
    return [AuditEventOut(**e, actor_name=names[str(e["actor_id"])] if names.get(str(e["actor_id"])) != str(e["actor_id"]) else None)
            for e in db_events]


@router.get("/audit", response_model=AuditPage, responses=problem_responses(401, 403, 404, 422))
def list_audit(request: Request, case_id: str | None = Query(None, max_length=36), action: str | None = Query(None, max_length=32),
               after_seq: int = Query(0, ge=0, le=MAX_SEQ, description="Return events after this seq"), limit: Limit = DEFAULT_LIMIT) -> JSONResponse:
    with get_session() as db:
        cases = _visible_cases(db, request.state.principal, case_id)
    events = list(itertools.islice(chain.iter_events(case_ids=cases, action=_action(action), after_seq=after_seq, batch=limit + 1),
                                   limit + 1))
    page = events[:limit]
    return _json(AuditPage(items=_named(page), next_after_seq=page[-1]["seq"] if len(events) > limit else None))


@router.get("/audit/verify", response_model=ChainVerification, responses=problem_responses(401, 403))
def verify_audit() -> JSONResponse:
    result = chain.verify_chain()
    return _json(ChainVerification(valid=result.valid, first_broken_seq=result.first_broken_seq, checked=result.checked,
                                   reason=result.reason))


@router.get("/audit/export", response_class=StreamingResponse,
            responses={200: {"content": {"application/x-ndjson": {}, "text/csv": {}}, "description": "The audit log"},
                       **problem_responses(401, 403, 404, 422)})
def export_audit(request: Request, export_format: Literal["jsonl", "csv"] = Query("jsonl", alias="format"),
                 case_id: str | None = Query(None, max_length=36)) -> StreamingResponse:
    with get_session() as db:
        cases = _visible_cases(db, request.state.principal, case_id)
    events = chain.iter_events(case_ids=cases)
    if export_format == "jsonl":
        body, media_type = chain.jsonl_lines(events), "application/x-ndjson"
    else:
        body, media_type = chain.csv_lines(events), "text/csv; charset=utf-8"
    return StreamingResponse(body, media_type=media_type,
                             headers={**NO_STORE, "Content-Disposition": f'attachment; filename="veritas-audit.{export_format}"'})


@router.get("/cases/{case_id}/custody", response_model=AuditPage, responses=problem_responses(401, 403, 404, 422))
def list_custody(case_id: str, request: Request,
                 after_seq: int = Query(0, ge=0, le=MAX_SEQ, description="Return events after this seq"), limit: Limit = DEFAULT_LIMIT) -> JSONResponse:
    """The custody log of one case. `GET /audit` gives it to reviewers and administrators; the owner of a case reads the log
    of that case here, whatever the owner's instance role (D-67)."""
    principal: Principal = request.state.principal
    with get_session() as db:
        if db.get(CaseRow, case_id) is None:
            raise ProblemError(404, *access.NOT_FOUND["case"])
        if not principal.satisfies("reviewer") and access.case_roles(db, principal.id, {case_id}).get(case_id) != "owner":
            raise ProblemError(403, "Forbidden", "The custody log of a case is open to its owner, to reviewers and to administrators.")
    events = list(itertools.islice(chain.iter_events(case_ids=[case_id], after_seq=after_seq, batch=limit + 1), limit + 1))
    page = events[:limit]
    return _json(AuditPage(items=_named(page), next_after_seq=page[-1]["seq"] if len(events) > limit else None))


@router.get("/cases/{case_id}/custody/verify", response_model=CustodyVerification, responses=problem_responses(401, 404))
def verify_custody(case_id: str, request: Request) -> JSONResponse:
    with get_session() as db:
        if db.get(CaseRow, case_id) is None:
            raise ProblemError(404, *access.NOT_FOUND["case"])
    result = chain.verify_case(case_id, actor_id=request.state.principal.id)
    verification = result["chain"]
    return _json(CustodyVerification(
        case_id=case_id, valid=result["valid"],
        chain=ChainVerification(valid=verification.valid, first_broken_seq=verification.first_broken_seq,
                                checked=verification.checked, reason=verification.reason),
        originals=[OriginalCheck(**c) for c in result["originals"]], reports=[ReportCheck(**c) for c in result["reports"]]))
