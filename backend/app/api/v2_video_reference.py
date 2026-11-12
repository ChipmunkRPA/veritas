"""Authenticated local video comparison against a retained analysis."""

from __future__ import annotations

import base64
import hashlib
import shutil
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_evidence import _analysis
from app.cases import review
from app.core.config import settings
from app.core.logging import get_logger
from app.evidence.video_reference import MAX_BYTES
from app.storage.db import get_session
from app.storage.objects import CHUNK, ObjectNotFound, ObjectStoreError, default_store

SUFFIX = "/video-reference-comparison"
ROUTE = "/analyses/{aid}" + SUFFIX
_slot = threading.BoundedSemaphore()
log = get_logger(__name__)

router = APIRouter(prefix="/api/v2", tags=["evidence"], dependencies=[Depends(rate_limit), Depends(authorize)],
                   route_class=CheckedRoute)


class VideoReferenceAvailability(BaseModel):
    available: bool


class VideoReferenceResponse(BaseModel):
    comparison: dict
    assets: dict[str, str]
    pdf: str | None = None


def is_comparison_path(path: str) -> bool:
    return path.startswith("/api/v2/analyses/") and path.endswith(SUFFIX)


def enabled() -> bool:
    return settings.environment == "dev" and settings.video_reference_comparison_enabled


def upload_limit() -> int:
    return min(settings.max_video_bytes, MAX_BYTES)


def _authenticated(request: Request) -> bool:
    return request.state.principal.auth in {"session", "api_key", "bootstrap"}


def _original(aid: str) -> tuple[str, int] | None:
    with get_session() as db:
        item = _analysis(db, aid)
        if (enabled() and item.kind == "video" and item.status == "done" and item.result_hash
                and 0 < item.size_bytes <= upload_limit() and item.withdrawn_at is None and not review.superseded(db, aid)):
            return item.sha256, item.size_bytes
    return None


def _unavailable() -> ProblemError:
    return ProblemError(404, "Video comparison unavailable", "A retained, completed, active video analysis is required.")


@router.get(ROUTE, response_model=VideoReferenceAvailability, responses=problem_responses(401, 404))
def availability(aid: str, request: Request):
    original = _original(aid) if _authenticated(request) else None
    available = False
    if original is not None:
        sha, size = original
        try:
            source = default_store().path_for(sha)
            available = source.is_file() and source.stat().st_size == size and bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        except (OSError, ObjectStoreError):
            pass
    return JSONResponse({"available": available}, headers=NO_STORE)


def _stage(source: BinaryIO, path: Path, limit: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("xb") as output:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            size += len(chunk)
            if size > limit:
                raise ProblemError(413, "Video exceeds the limit", "Each video must fit the earlier-copy upload limit.")
            output.write(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _mismatch() -> ProblemError:
    return ProblemError(409, "Original integrity check failed", "The retained video no longer matches its recorded hash and size.")


def _base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _compare(aid: str, original: tuple[str, int], earlier: BinaryIO, include_pdf: bool) -> JSONResponse:
    from app.evidence.video_reference import compare

    try:
        with TemporaryDirectory(prefix="video-reference-") as directory:
            folder = Path(directory)
            submitted = folder / "submitted"
            try:
                with default_store().open(original[0]) as stream:
                    actual = _stage(stream, submitted, original[1])
            except ObjectNotFound:
                raise _unavailable() from None
            except (ObjectStoreError, ProblemError):
                raise _mismatch() from None
            if actual != original:
                raise _mismatch()
            reference = folder / "earlier"
            uploaded = _stage(earlier, reference, upload_limit())
            try:
                record, assets = compare(reference, submitted)
            except ValueError as error:
                raise ProblemError(422, "Could not compare videos", str(error)) from None
            if (record["submitted"]["sha256"], record["submitted"]["bytes"]) != original:
                raise _mismatch()
            if (record["earlier"]["sha256"], record["earlier"]["bytes"]) != uploaded:
                raise ProblemError(500, "Earlier copy integrity check failed", "The earlier copy changed during comparison.")
    except OSError:
        raise ProblemError(500, "Video comparison unavailable", "The local service could not read or stage the clips.") from None

    payload = {"comparison": record, "assets": {name: _base64(data) for name, data in assets.items()}}
    if include_pdf:
        from app.reports.video_reference_pdf import render

        try:
            payload["pdf"] = _base64(render(record, assets))
        except Exception as error:
            log.warning("video_reference.pdf_failed", exception=type(error).__name__)
    if _original(aid) != original:
        raise _unavailable()
    return JSONResponse(payload, headers=NO_STORE)


@router.post(
    ROUTE, response_model=VideoReferenceResponse, responses=problem_responses(401, 404, 409, 413, 422, 429, 500),
    openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["earlier"], "properties": {
            "earlier": {"type": "string", "format": "binary", "description": "Earlier video with the same native picture timeline."},
            "include_pdf": {"type": "string", "enum": ["true", "false"], "default": "false"},
        },
    }}}}},
)
async def comparison(aid: str, request: Request):
    if not _authenticated(request):
        raise ProblemError(401, "Sign in required", "Use an authenticated session or API key for earlier-video comparison.")
    original = await run_in_threadpool(_original, aid)
    if original is None:
        raise _unavailable()
    if not _slot.acquire(blocking=False):
        raise ProblemError(429, "Video comparison busy", "Try again after the active video comparison finishes.")
    try:
        async with request.form(max_files=1, max_fields=1, max_part_size=1024) as form:
            earlier = form.get("earlier")
            include_pdf = form.get("include_pdf", "false")
            if not isinstance(earlier, UploadFile) or include_pdf not in ("true", "false"):
                raise ProblemError(422, "Earlier video required", "Upload a single earlier clip to compare.")
            return await run_in_threadpool(_compare, aid, original, earlier.file, include_pdf == "true")
    finally:
        _slot.release()
