"""API v2 re-fusion (ARCHITECTURE §5.2 `POST /analyses/{id}/refusion`, §4.4 "Re-fusion"; BUILD_SPEC M-2).

Applies a fuser artefact to the stored raw scores of a published analysis and publishes the result as a new revision
(`revision + 1`, `supersedes_id`) without running detectors. The request answers 202 with the revision id; the `refusion`
job does the work, and progress is on the revision's `/progress` and `/events`.

- **Errors.** 404: no analysis with this id. 409: not a published v2 analysis, a newer revision exists, a stored stage
  output is unavailable, or no compatible artefact is available (or the pinned one is not compatible). 422: a malformed
  body.
- **Role.** administrator (§5.2), declared in `app/api/rbac.py` (C-1).
- **Custody.** The new revision is recorded as `rerun` (C-3), with the request `note` when one is given (J-3e).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from app.api import jobs
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import RequestModel
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_annotations import clean_body
from app.audit import chain as audit
from app.engine import refusion
from app.engine.registry import get_registry
from app.storage.objects import default_store

router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)


class RefusionRequest(RequestModel):
    model_config = ConfigDict(extra="forbid")

    fuser_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$",
                                     description="Pin one artefact by sha256; default: the newest compatible artefact")
    note: str | None = Field(default=None, max_length=4000, description="Why the analysis is re-fused (recorded in the custody log)")


class RefusionAccepted(BaseModel):
    analysis_id: str
    revision: int
    supersedes_id: str
    job_id: str
    fuser_sha256: str
    fuser_version: str | None = None


@router.post("/analyses/{aid}/refusion", status_code=202, response_model=RefusionAccepted,
             responses=problem_responses(401, 404, 409, 422))
def request_refusion(aid: str, request: Request, body: RefusionRequest | None = None):
    """Re-fuse the stored raw scores of a published analysis into a new revision; the job runs asynchronously."""
    wanted = body or RefusionRequest()
    note = clean_body(wanted.note)
    def unit() -> dict:
        # N-9 (code review 12): the revision, its `refusion` job and its `rerun` event are one transaction
        with audit.writing() as db:
            try:
                source = refusion.check_source(db, aid)
                selection = refusion.compatible_selection(db, get_registry(), source, wanted.fuser_sha256)
                revision = refusion.create_revision(db, default_store(), aid, fuser_sha256=selection.sha256)
                accepted = {"analysis_id": revision.id, "revision": revision.revision, "supersedes_id": aid,
                            "fuser_sha256": selection.sha256, "fuser_version": selection.version,
                            "job_id": jobs.submit_in(db, revision.id, "refusion")}
                audit.commit_with(db, [{
                    "action": "rerun", "actor_id": request.state.principal.id, "analysis_id": revision.id, "object_ids": [aid],
                    "details": {"mode": "refusion", "supersedes_id": aid, "revision": accepted["revision"],
                                "fuser_sha256": accepted["fuser_sha256"], **({"note": note} if note else {})}}])
                return accepted
            except LookupError:
                raise ProblemError(404, "Analysis not found", "No analysis exists with this id.") from None
            except refusion.RefusionError as e:
                from app.api.public_projection import neutral_text

                raise ProblemError(409, "Re-fusion not possible", neutral_text(str(e))) from None
            except IntegrityError:  # unique analyses.supersedes_id: a concurrent reanalyse or re-fusion committed first (D-47)
                db.rollback()
                raise ProblemError(409, "Re-fusion not possible", "a newer revision of this analysis was created at the same time") from None

    accepted = audit.repeated(unit)
    jobs.wake()
    return JSONResponse(status_code=202, headers=NO_STORE, content=RefusionAccepted(**accepted).model_dump())
