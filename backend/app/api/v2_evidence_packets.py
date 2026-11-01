"""Read-only customer evidence and hash-checked PNG/WAV/MP4 presentation assets."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_evidence import _analysis, _binary
from app.evidence.service import ASSET_TYPES, asset_refs, load_packet_set, unavailable_packet_set
from app.storage.db import Artefact, get_session
from app.storage.objects import ObjectStoreError, default_store

router = APIRouter(prefix="/api/v2", tags=["evidence"], dependencies=[Depends(rate_limit), Depends(authorize)],
                   route_class=CheckedRoute)


@router.get("/analyses/{aid}/evidence-packets", responses=problem_responses(404, 409, 500))
def image_evidence(aid: str, set_id: str | None = None):
    with get_session() as db:
        analysis = _analysis(db, aid)
        try:
            document = load_packet_set(db, analysis, default_store(), set_id=set_id)
        except (ObjectStoreError, ValueError, KeyError, TypeError):
            raise ProblemError(500, "Evidence unavailable", "The evidence failed an integrity check and was not served.") from None
        if document is None and set_id is not None:
            raise ProblemError(404, "Evidence set not found", "No evidence set with this id belongs to the analysis.")
        return JSONResponse(document or unavailable_packet_set(analysis), headers=NO_STORE)


@router.get("/analyses/{aid}/evidence-packets/{set_id}/images/{sha256}.png", responses=problem_responses(404, 500))
def image_evidence_plate(aid: str, set_id: str, sha256: str, request: Request):
    return evidence_asset(aid, set_id, sha256, "png", request)


@router.get("/analyses/{aid}/evidence-packets/{set_id}/assets/{sha256}.{extension}", responses=problem_responses(404, 500))
def evidence_asset(aid: str, set_id: str, sha256: str, extension: str, request: Request):
    with get_session() as db:
        analysis = _analysis(db, aid)
        try:
            document = load_packet_set(db, analysis, default_store(), set_id=set_id)
            allowed = asset_refs(document)
        except (ObjectStoreError, ValueError, KeyError, TypeError):
            raise ProblemError(500, "Evidence unavailable", "The evidence failed an integrity check and was not served.") from None
        media_type = ASSET_TYPES.get(extension)
        asset = allowed.get(sha256)
        if asset is None or media_type is None or asset["media_type"] != media_type:
            raise ProblemError(404, "Evidence asset not found", "The requested asset is not part of this evidence set.")
        row = db.scalars(select(Artefact).where(Artefact.analysis_id == aid, Artefact.sha256 == sha256,
                                               Artefact.role.in_(("preview", "working_copy")),
                                               Artefact.media_type == media_type)).first()
        if row is None:
            raise ProblemError(500, "Evidence unavailable", "The evidence asset is no longer indexed.")
        # Allowlisted media stays inline for the source-context players, while
        # old image URLs retain their existing behavior.
        return _binary(request, row, exhibit=media_type == "image/png", byte_ranges=media_type in {"audio/wav", "video/mp4"})
