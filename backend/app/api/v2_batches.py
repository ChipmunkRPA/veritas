"""API v2 batches (ARCHITECTURE §5.2; PRD API-01; BUILD_SPEC X-1).

- `POST /batches` (multipart): exactly one of `files` (one or more files), `archive` (one zip) or `local_path` (a file or directory
  inside `VERITAS_BATCH_LOCAL_ROOTS`), with optional `case_id` (else a new case) and `profile`. → 202 with the batch and its items;
  the `batch_expand` job takes the items in and queues their analyses at batch priority (`app.batches.service`).
- `GET /batches/{batch_id}`: the batch with per-item status. `GET /batches/{batch_id}/summary.json` and `/summary.csv`: the same
  items as a download.
- **Refusals.** 401 / 403 (examiner role, or examiner on the named case); 404 unknown batch or case, or a case the caller is not a
  member of (CM-04); 409 a closed case; 413 more than `batch_max_items` files or `batch_max_bytes`; 422 none or several sources, an
  unreadable archive, an empty batch, a `local_path` outside the allowed directories.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api import jobs
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import Text
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.batches import service as batches
from app.cases import access, intake
from app.core import times
from app.schemas.evidence import DecisionCode
from app.storage.db import BatchRow, CaseRow, get_session

router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
Profile = Literal["quick", "full", "deep"]
ItemStatus = Literal["staged", "rejected", "queued", "running", "done", "failed", "cancelled", "deleted"]


class BatchItemOut(BaseModel):
    seq: int = Field(description="0-based position in the request, archive or directory listing")
    source_name: str
    kind: str | None
    size_bytes: int | None
    sha256: str | None
    status: ItemStatus = Field(description="staged or rejected before intake; then the status of the item's analysis")
    analysis_id: str | None
    decision_code: DecisionCode | None
    driving_hypothesis: str | None = Field(default=None, description="The hypothesis that drives the decision; the provenance "
                                           "hypothesis where a provenance record decides it")
    probability_ai: float | None
    error: str | None = Field(description="Why intake refused the item, or why its analysis failed")
    deciding_record: Literal["capture_record", "declaration"] | None = Field(
        default=None, description="The trusted provenance record that decides the item's result (`capture_record`, "
                                  "`declaration`), as the stored result names it; null for a marker and where no record "
                                  "decides (N-19p, D6-01)")


class BatchOut(BaseModel):
    id: str
    case_id: str
    created_by: str
    profile: Profile | None
    source: Literal["upload", "archive", "local_path"]
    status: Literal["expanding", "running", "done", "failed"]
    counts: dict[str, int] = Field(description="Items per status")
    item_count: int
    error: str | None
    created_at: datetime
    expanded_at: datetime | None
    finished_at: datetime | None
    items: list[BatchItemOut]


_utc = times.to_utc  # one form on every engine: a time is printed in UTC (D-72)


def _out(db: Session, row: BatchRow) -> BatchOut:
    current = batches.view(db, row)
    return BatchOut(id=row.id, case_id=row.case_id, created_by=row.created_by, profile=row.profile, source=row.source,  # type: ignore[arg-type]
                    status=current.state, counts=current.counts, item_count=len(current.items), error=row.error,  # type: ignore[arg-type]
                    created_at=_utc(row.created_at), expanded_at=_utc(row.expanded_at), finished_at=_utc(row.finished_at),  # type: ignore[arg-type]
                    items=[BatchItemOut(**i.__dict__) for i in current.items])


def _batch(db: Session, batch_id: str) -> BatchRow:
    row = db.get(BatchRow, batch_id)
    if row is None:
        raise ProblemError(404, *access.NOT_FOUND["batch"])
    return row


@router.post("/batches", status_code=202, response_model=BatchOut, responses=problem_responses(401, 403, 404, 409, 413, 422))
def create_batch(request: Request, files: Annotated[list[UploadFile] | None, File(description="One or more files")] = None,
                 archive: Annotated[UploadFile | None, File(description="One zip archive")] = None,
                 local_path: Annotated[str | None, Form(max_length=4096, description="A path inside VERITAS_BATCH_LOCAL_ROOTS")] = None,
                 case_id: Annotated[Text | None, Form(max_length=36, description="The case to file the items into; default: a new case")] = None,
                 profile: Annotated[Profile | None, Form(description="Analysis profile of every item")] = None) -> JSONResponse:
    principal = request.state.principal
    if sum(x is not None for x in (files or None, archive, local_path)) != 1:
        raise ProblemError(422, "Invalid request", "Send exactly one of files, archive or local_path.")
    if case_id is not None:
        with get_session() as db:
            case = db.get(CaseRow, case_id)
            denial = access.denial(db, principal, {"case_id": case_id}, "examiner") if case is not None else None
            if case is None:
                raise ProblemError(404, *access.NOT_FOUND["case"])
            if denial is not None:
                raise ProblemError(denial.status, denial.title, denial.detail)
            if case.status in intake.CLOSED_STATUSES:
                raise ProblemError(409, "Case closed", f"The case is {case.status}; items are no longer added to it.")
    try:
        if files:
            batch_id = batches.create_from_uploads([(f.filename or f"file-{n + 1}", f.file) for n, f in enumerate(files)],
                                                   principal_id=principal.id, case_id=case_id, profile=profile)
        elif archive is not None:
            batch_id = batches.create_from_archive(archive.filename or "archive.zip", archive.file, principal_id=principal.id,
                                                   case_id=case_id, profile=profile)
        else:
            assert local_path is not None
            batch_id = batches.create_from_local_path(local_path, principal_id=principal.id, case_id=case_id, profile=profile)
    except batches.BatchRefused as e:
        raise ProblemError(e.status, e.title, e.detail) from None
    jobs.submit_batch(batch_id)
    with get_session() as db:
        return JSONResponse(jsonable_encoder(_out(db, _batch(db, batch_id))), status_code=202, headers=NO_STORE)


@router.get("/batches/{batch_id}", response_model=BatchOut, responses=problem_responses(401, 403, 404))
def get_batch(batch_id: str) -> JSONResponse:
    with get_session() as db:
        return JSONResponse(jsonable_encoder(_out(db, _batch(db, batch_id))), headers=NO_STORE)


@router.get("/batches/{batch_id}/summary.json", response_model=BatchOut, responses=problem_responses(401, 403, 404))
def batch_summary_json(batch_id: str) -> JSONResponse:
    with get_session() as db:
        out = _out(db, _batch(db, batch_id))
    return JSONResponse(jsonable_encoder(out), headers={**NO_STORE, "Content-Disposition": f'attachment; filename="veritas-batch-{batch_id[:8]}.json"'})


@router.get("/batches/{batch_id}/summary.csv", response_class=Response,
            responses={200: {"content": {"text/csv": {}}, "description": "One row per item"}, **problem_responses(401, 403, 404)})
def batch_summary_csv(batch_id: str) -> Response:
    with get_session() as db:
        out = _out(db, _batch(db, batch_id))
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(batches.CSV_FIELDS)
    for item in out.items:
        writer.writerow([batches.csv_cell(getattr(item, name)) for name in batches.CSV_FIELDS])
    return Response(buffer.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={**NO_STORE, "Content-Disposition": f'attachment; filename="veritas-batch-{batch_id[:8]}.csv"'})
