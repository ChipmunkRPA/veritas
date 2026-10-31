"""API v2 workspace policy thresholds (ARCHITECTURE §4.4 "Abstention"; PRD API-03; BUILD_SPEC X-2).

- `GET /policy/thresholds` (viewer): the version in force. `GET /policy/thresholds/versions` (viewer): every version, newest first.
  `GET /policy/thresholds/{policy_id}` (viewer): one version; analyses name it in `manifest.policy_thresholds_id`.
- `PUT /policy/thresholds` {alpha_strong, alpha_moderate, min_slice_auc, reason} (administrator): saves the next version and records
  `policy_changed` with the before and after hashes, the changed fields and the reason. New analyses use it; earlier analyses keep
  the version they recorded, and a re-run reuses that version (CC-03).
- **Refusals.** 401 / 403; 404 unknown version; 409 identical to the version in force, or a concurrent change; 422 invalid body (an
  α other than 0.01 or 0.05, α_strong above α_moderate, min_slice_auc outside 0.70–1, no reason, unknown fields).
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import RequestModel
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_annotations import clean_body
from app.policy import thresholds as policy
from app.storage.db import PolicyThresholdsRow, get_session

router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)


class PolicyThresholdsPut(policy.Thresholds, RequestModel):
    reason: str = Field(min_length=1, max_length=4000, description="Why the policy changes (recorded in the custody log)")


class PolicyThresholdsOut(BaseModel):
    id: str
    version: int
    alpha_strong: float
    alpha_moderate: float
    min_slice_auc: float
    sha256: str = Field(description="SHA-256 over the canonical JSON of the three thresholds")
    reason: str | None
    created_by: str
    created_at: datetime
    in_force: bool


class PolicyThresholdsList(BaseModel):
    items: list[PolicyThresholdsOut]


def _out(row: PolicyThresholdsRow, in_force_id: str) -> PolicyThresholdsOut:
    created = row.created_at if row.created_at.tzinfo is not None else row.created_at.replace(tzinfo=UTC)
    return PolicyThresholdsOut(id=row.id, version=row.version, **policy.Thresholds.model_validate(row.thresholds).model_dump(),
                               sha256=row.sha256, reason=row.reason, created_by=row.created_by, created_at=created,
                               in_force=row.id == in_force_id)


def _json(model: BaseModel) -> JSONResponse:
    return JSONResponse(jsonable_encoder(model), headers=NO_STORE)


@router.get("/policy/thresholds", response_model=PolicyThresholdsOut, responses=problem_responses(401, 403))
def current_thresholds() -> JSONResponse:
    """The policy thresholds version in force (version 1 holds the defaults)."""
    with get_session() as db:
        row = policy.current_row(db)
        return _json(_out(row, row.id))


@router.get("/policy/thresholds/versions", response_model=PolicyThresholdsList, responses=problem_responses(401, 403))
def list_threshold_versions() -> JSONResponse:
    with get_session() as db:
        rows = policy.versions(db)
        return _json(PolicyThresholdsList(items=[_out(r, rows[0].id) for r in rows]))


@router.get("/policy/thresholds/{policy_id}", response_model=PolicyThresholdsOut, responses=problem_responses(401, 403, 404))
def get_thresholds(policy_id: str) -> JSONResponse:
    with get_session() as db:
        row = db.get(PolicyThresholdsRow, policy_id)
        if row is None:
            raise ProblemError(404, "Policy version not found", "No policy thresholds version exists with this id.")
        return _json(_out(row, policy.current_row(db).id))


@router.put("/policy/thresholds", response_model=PolicyThresholdsOut, responses=problem_responses(401, 403, 409, 422))
def change_thresholds(req: PolicyThresholdsPut, request: Request) -> JSONResponse:
    """Save the next policy version and record `policy_changed` (API-03)."""
    reason = clean_body(req.reason)
    if reason is None:
        raise ProblemError(422, "Invalid request", "A policy change needs a reason.")
    try:
        _, saved = policy.change(policy.Thresholds.model_validate(req.model_dump(exclude={"reason"})),
                                 actor_id=request.state.principal.id, reason=reason)
    except policy.PolicyUnchanged as e:
        raise ProblemError(409, "Policy unchanged", str(e)) from None
    except policy.PolicyConflict as e:
        raise ProblemError(409, "Policy changed concurrently", str(e)) from None
    with get_session() as db:
        row = db.get(PolicyThresholdsRow, saved.id)
        assert row is not None
        return _json(_out(row, saved.id))
