"""API v2 re-run with the pinned manifest, and revision deltas (ARCHITECTURE §5.2 `POST /analyses/{id}/rerun`, §6.5, §6.6;
PRD CC-03; BUILD_SPEC X-2).

- **Re-run** (`POST /analyses/{aid}/rerun` {note?}; examiner). The published analysis must be reproducible with its manifest
  (`app.engine.rerun.check`): every pinned detector version and weights still in the model manifest, installed, intact and
  enabled; the fuser artefact it used still loads (a fallback analysis stays on the fallback); its policy thresholds version exists.
  The original is re-hashed, then revision n+1 (`supersedes_id`) is queued on the same profile with those pins. 202 with the
  revision and its deltas URL. A revision cancelled or failed before it published supersedes nothing (K-2): re-running it
  queues a new revision of the revision it would have replaced, with its stored pins, and custody records `retry_of`.
- **Deltas** (`GET /analyses/{aid}/deltas` `?tolerance`; viewer). The revision against the one it supersedes: per signal status, raw
  score and calibrated probability; decision code and fused probability; a video's sampled and tracked frame indices; the manifest
  fields that changed. `pending` while the revision runs. Tolerance 0.001 on the same devices, 0.02 across devices (§6.5). Every
  revision has deltas (re-run, reanalyse, refusion); `identical` is the CC-03 check of a re-run.
- **Refusals.** 401 / 403; 404 unknown analysis, or a case the caller is not a member of; 409 not finished or not a published v2
  analysis, a newer revision exists, a pinned weight, detector, fuser artefact or policy version is unavailable (the detail names
  each), or the analysis supersedes nothing (deltas); 422 invalid body or tolerance; 500 the original fails its re-hash.
- **Custody** (C-3): `rerun` with mode `rerun` and the pins; `hash_verified` / `hash_mismatch` with purpose `rerun`.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from app.api import jobs
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import RequestModel
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_annotations import clean_body
from app.api.v2_evidence import _analysis, _result, _stage_output
from app.api.v2_review import signed_rerun_reason
from app.audit import chain as audit
from app.cases import review
from app.core.logging import get_logger
from app.engine import rerun
from app.engine.registry import get_registry
from app.storage.db import Analysis, get_session

log = get_logger(__name__)
router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
TERMINAL = frozenset({"done", "failed", "cancelled"})


class RerunRequest(RequestModel):
    model_config = ConfigDict(extra="forbid")
    note: str | None = Field(default=None, max_length=4000, description="Why the analysis is re-run (recorded in the custody log)")
    reason: str | None = Field(default=None, max_length=4000, description="Why a signed-off revision is re-run. Required when the "
                                                                          "revision is signed off; the note is taken when absent.")


class RerunAccepted(BaseModel):
    analysis_id: str
    revision: int
    supersedes_id: str
    status: Literal["queued"]
    profile: str | None
    fusion_mode: Literal["artefact", "fallback"] | None
    fuser_sha256: str | None
    policy_thresholds_id: str | None
    devices: list[str] = Field(description="The devices the pinned detectors ran on")
    deltas_url: str


class DeltaRow(BaseModel):
    item: str = Field(description="A signal id, `decision` or `frames`")
    field: str
    before: Any
    after: Any
    delta: float | None = Field(description="|after − before| of a numeric field")
    ok: bool = Field(description="Equal, or within the tolerance")


class ManifestChange(BaseModel):
    field: str
    before: Any
    after: Any


class RevisionDeltas(BaseModel):
    analysis_id: str
    compared_with: str = Field(description="The revision this one supersedes")
    revision: int
    mode: Literal["rerun", "reanalyse", "refusion"] | None
    status: Literal["pending", "ready", "failed", "unavailable"]
    detail: str | None
    tolerance: float
    cross_device: bool
    identical: bool | None = Field(description="Every signal, the decision and the frame indices agree within the tolerance")
    max_delta: float | None
    deltas: list[DeltaRow]
    manifest_changes: list[ManifestChange]


def _published(a: Analysis) -> bool:
    return a.status == "done" and a.result_schema_version == 2 and bool(a.result_hash)


@router.post("/analyses/{aid}/rerun", status_code=202, response_model=RerunAccepted,
             responses=problem_responses(401, 403, 404, 409, 422, 500))
def request_rerun(aid: str, request: Request, body: RerunRequest | None = None) -> JSONResponse:
    """Queue revision n+1 with the pinned manifest of this analysis (CC-03); refuses when a pin is unavailable. A signed-off
    revision is re-run only by its signer, its conclusion's author, a case owner or an administrator, with a reason."""
    principal = request.state.principal
    note = clean_body((body or RerunRequest()).note)
    stated = clean_body((body or RerunRequest()).reason) or note
    with get_session() as db:
        requested = _analysis(db, aid)
        if requested.status not in TERMINAL:
            raise ProblemError(409, "Analysis still running", f"The analysis is {requested.status}; re-run it once it has finished.")
        # K-2: re-running a revision that was cancelled or failed before it published re-runs the revision it would have
        # replaced, with the pins stored when it was queued
        source = review.revision_base(db, requested)
        stored = requested.rerun_pins if source.id != requested.id else None
        if review.superseded(db, source.id):
            raise ProblemError(409, "Revision superseded", "A newer revision exists; re-run the latest revision.")
        reason = signed_rerun_reason(db, principal, source, stated, "Re-running a signed revision", "re-run it")
        try:
            pins = rerun.check(db, get_registry(), source, _result(db, source) if _published(source) else None, stored)
        except rerun.RerunRefused as e:
            raise ProblemError(409, "Re-run not possible", str(e)) from None
        base_id = source.id
    try:
        sha256 = audit.verify_original("rerun", actor_id=principal.id, analysis_id=base_id)
    except audit.OriginalIntegrityError as e:
        raise ProblemError(500, "Evidence integrity failure", f"The original {e.problem}; the analysis was not re-run.") from None
    def unit() -> RerunAccepted:
        # N-9 (code review 12): the revision, its job and its `rerun` event are one transaction. A queued revision supersedes
        # the published one, so one left without a job or an event could be neither run, cancelled nor explained
        revision_id = rerun.prepare_revision(base_id, sha256)  # N-10a: an upload that is a copy, before the locks
        with audit.writing() as db:
            try:
                source = _analysis(db, base_id)
                revision = rerun.create_revision(db, source, rerun.original_path(db, source, sha256), principal_id=principal.id,
                                                 profile=pins["profile"], pins=pins, revision_id=revision_id)
                jobs.submit_in(db, revision.id)
                accepted = RerunAccepted(
                    analysis_id=revision.id, revision=revision.revision, supersedes_id=base_id, status="queued",
                    profile=revision.profile, fusion_mode=pins["fusion_mode"], fuser_sha256=pins["fuser_sha256"],
                    policy_thresholds_id=pins["policy_thresholds_id"], devices=pins["devices"],
                    deltas_url=f"/api/v2/analyses/{revision.id}/deltas")
                audit.commit_with(db, [_rerun_event(accepted)])
                return accepted
            except rerun.SourceGone:  # N-9: purged or deleted while this request waited for the base row
                db.rollback()
                rerun.discard_upload(revision_id)
                raise ProblemError(404, "Analysis not found", "No analysis exists with this id.") from None
            except IntegrityError:  # unique analyses.supersedes_id: a concurrent re-run of this revision committed first (D-47; N-8)
                db.rollback()
                rerun.discard_upload(revision_id)
                raise ProblemError(409, "Revision superseded", "A newer revision of this analysis was created at the same time; "
                                                               "re-run the latest revision.") from None
            except BaseException:  # nothing was committed (a failed append, a moved tail): the linked original goes too
                rerun.discard_upload(revision_id)
                raise

    def _rerun_event(accepted: RerunAccepted) -> dict[str, Any]:
        return {"action": "rerun", "actor_id": principal.id, "analysis_id": accepted.analysis_id, "object_ids": [base_id],
                "details": {"mode": "rerun", "supersedes_id": base_id, "revision": accepted.revision, "profile": accepted.profile,
                            "fusion_mode": accepted.fusion_mode, "fuser_sha256": accepted.fuser_sha256,
                            "policy_thresholds_id": accepted.policy_thresholds_id, **({"retry_of": aid} if base_id != aid else {}),
                            **({"note": note} if note else {}),
                            **({"reason": reason, "supersedes_signed": True} if reason is not None else {})}}

    accepted = audit.repeated(unit)
    jobs.wake()
    log.info("rerun.queued", analysis_id=aid, revision_id=accepted.analysis_id, by=principal.id)
    return JSONResponse(jsonable_encoder(accepted), status_code=202, headers=NO_STORE)


@router.get("/analyses/{aid}/deltas", response_model=RevisionDeltas, responses=problem_responses(401, 403, 404, 409, 422, 500))
def revision_deltas(aid: str, tolerance: Annotated[float | None, Query(
        ge=0.0, le=0.5, description="Default: 0.001 when both revisions ran on the same devices, else 0.02")] = None) -> JSONResponse:
    """This revision against the revision it supersedes (CC-03)."""
    with get_session() as db:
        revision = _analysis(db, aid)
        source = db.get(Analysis, revision.supersedes_id) if revision.supersedes_id else None
        if source is None:
            raise ProblemError(409, "Nothing to compare", f"Revision {revision.revision} of this analysis supersedes no earlier revision.")
        before = _result(db, source) if _published(source) else None
        after = _result(db, revision) if _published(revision) else None
        frames = [rerun.sampled_frames(_stage_output(db, a.id, "probe"), _stage_output(db, a.id, "shots"))
                  if a.kind == "video" and published else None for a, published in ((source, before), (revision, after))]
        body = rerun.deltas(revision, source, before=before, after=after, mode=rerun.mode_of(db, aid), before_frames=frames[0],
                            after_frames=frames[1], tolerance=tolerance)
    return JSONResponse(jsonable_encoder(RevisionDeltas(**body)), headers=NO_STORE)
