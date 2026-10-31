"""Cooperative analysis cancellation; evidence is retained and a published result wins."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api import jobs
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import RequestModel
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_annotations import clean_body
from app.api.v2_evidence import AnalysisStatus

router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)


class CancelAnalysisRequest(RequestModel):
    model_config = ConfigDict(extra="forbid")
    note: str | None = Field(default=None, max_length=1000, description="Why the run is cancelled; recorded with the request "
                                                                       "in the custody log and shown on the analysis (N-19a)")


class CancelAnalysisResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    cancel_requested: bool = Field(description="An active job is waiting to acknowledge cooperative cancellation")


@router.post("/analyses/{aid}/cancel", status_code=202, response_model=CancelAnalysisResponse,
             responses={200: {"model": CancelAnalysisResponse, "description": "Already cancelled"},
                        **problem_responses(401, 403, 404, 409, 422, 500)})
def cancel_analysis(aid: str, request: Request, body: CancelAnalysisRequest | None = None) -> JSONResponse:
    """Cancel queued analysis/refusion work, or request cooperative cancellation. Retains all evidence."""
    try:
        result = jobs.request_analysis_cancel(aid, actor_id=request.state.principal.id,
                                              note=clean_body(body.note) if body is not None else None)
    except jobs.CancellationRefused as exc:
        raise ProblemError(exc.status_code, "Analysis not found" if exc.status_code == 404 else "Cancellation not possible", str(exc)) from None
    payload = CancelAnalysisResponse.model_validate(asdict(result)).model_dump()
    return JSONResponse(payload, status_code=200 if result.already_cancelled else 202, headers=NO_STORE)
