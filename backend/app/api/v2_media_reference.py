"""Authenticated comparisons against a retained audio or video analysis."""

from __future__ import annotations

import base64
import hashlib
import re
import shutil
import threading
from fractions import Fraction
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
from app.evidence import audio_reference, video_reference
from app.storage.db import get_session
from app.storage.objects import CHUNK, ObjectNotFound, ObjectStoreError, default_store

VIDEO_ROUTE = "/analyses/{aid}/video-reference-comparison"
AUDIO_ROUTE = "/analyses/{aid}/audio-reference-comparison"
_slot = threading.BoundedSemaphore()
log = get_logger(__name__)

router = APIRouter(prefix="/api/v2", tags=["evidence"], dependencies=[Depends(rate_limit), Depends(authorize)],
                   route_class=CheckedRoute)


class MediaReferenceAvailability(BaseModel):
    available: bool


class AudioReferenceAvailability(MediaReferenceAvailability):
    clean_speech_available: bool | None = None
    long_clean_speech_available: bool | None = None


class VideoReferenceAvailability(MediaReferenceAvailability):
    timing_available: bool | None = None
    picture_insertion_available: bool | None = None


class MediaReferenceResponse(BaseModel):
    comparison: dict
    assets: dict[str, str]
    pdf: str | None = None


def kind_for_path(path: str) -> str | None:
    if path.startswith("/api/v2/analyses/"):
        for kind in ("audio", "video"):
            if path.endswith(f"/{kind}-reference-comparison"):
                return kind
    return None


def enabled(kind: str) -> bool:
    return settings.environment == "dev" and (
        settings.audio_reference_comparison_enabled if kind == "audio" else settings.video_reference_comparison_enabled
    )


def upload_limit(kind: str) -> int:
    if kind == "audio":
        return min(settings.max_audio_bytes, audio_reference.MAX_BYTES)
    return min(settings.max_video_bytes, video_reference.MAX_BYTES)


def _authenticated(request: Request) -> bool:
    return request.state.principal.auth in {"session", "api_key", "bootstrap"}


def _original(aid: str, kind: str) -> tuple[str, int, str] | None:
    with get_session() as db:
        item = _analysis(db, aid)
        if (enabled(kind) and item.kind == kind and item.status == "done" and item.result_hash
                and 0 < item.size_bytes <= upload_limit(kind) and item.withdrawn_at is None and not review.superseded(db, aid)):
            return item.sha256, item.size_bytes, item.filename
    return None


def _unavailable(kind: str) -> ProblemError:
    return ProblemError(404, f"{kind.title()} comparison unavailable", f"A retained, completed, active {kind} analysis is required.")


@router.get(VIDEO_ROUTE, response_model=VideoReferenceAvailability, responses=problem_responses(401, 404))
@router.get(AUDIO_ROUTE, response_model=AudioReferenceAvailability, responses=problem_responses(401, 404))
def availability(aid: str, request: Request):
    kind = kind_for_path(request.url.path)
    assert kind is not None
    original = _original(aid, kind) if _authenticated(request) else None
    available = False
    if original is not None:
        sha, size, _ = original
        try:
            source = default_store().path_for(sha)
            available = source.is_file() and source.stat().st_size == size and bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        except (OSError, ObjectStoreError):
            pass
    result = {"available": available}
    if available and kind == "audio" and settings.audio_reference_clean_speech:
        result["clean_speech_available"] = True
        if settings.audio_reference_long_clean_speech:
            result["long_clean_speech_available"] = True
    if available and kind == "video" and settings.video_reference_picture_insertion:
        result["picture_insertion_available"] = True
    if available and kind == "video" and settings.video_reference_timing_model_path is not None:
        from app.evidence.video_reference_timing import available as timing_available

        if timing_available(settings.video_reference_timing_model_path):
            result["timing_available"] = True
    return JSONResponse(result, headers=NO_STORE)


def _stage(source: BinaryIO, path: Path, limit: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("xb") as output:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            size += len(chunk)
            if size > limit:
                raise ProblemError(413, "File exceeds the limit", "Each file must fit the earlier-copy upload limit.")
            output.write(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _mismatch() -> ProblemError:
    return ProblemError(409, "Original integrity check failed", "The retained file no longer matches its recorded hash and size.")


def _base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _compare(aid: str, kind: str, original: tuple[str, int, str], earlier: UploadFile, include_pdf: bool,
             timing_event_s: Fraction | None = None, earlier_has_clean_speech: bool = False) -> JSONResponse:
    expected = original[:2]

    try:
        with TemporaryDirectory(prefix=f"{kind}-reference-") as directory:
            folder = Path(directory)
            submitted = folder / "submitted"
            try:
                with default_store().open(original[0]) as stream:
                    actual = _stage(stream, submitted, expected[1])
            except ObjectNotFound:
                raise _unavailable(kind) from None
            except (ObjectStoreError, ProblemError):
                raise _mismatch() from None
            if actual != expected:
                raise _mismatch()
            reference = folder / "earlier"
            uploaded = _stage(earlier.file, reference, upload_limit(kind))
            try:
                if kind == "audio":
                    record, assets = audio_reference.compare(
                        reference, submitted, filenames=(earlier.filename or "", original[2]),
                        include_local_timing=settings.audio_reference_local_timing,
                        include_channel_timing=settings.audio_reference_local_timing,
                        include_aac_timing=settings.audio_reference_aac_timing,
                        include_clean_speech=settings.audio_reference_clean_speech and earlier_has_clean_speech,
                        allow_long_clean_speech=settings.audio_reference_long_clean_speech,
                        use_dense_clean_speech=settings.audio_reference_dense_clean_speech,
                    )
                elif timing_event_s is None:
                    record, assets = video_reference.compare(
                        reference, submitted, include_channel_timing=True, include_trimmed=True,
                        include_local_area=settings.video_reference_local_area,
                        include_measured_interval=settings.video_reference_measured_interval,
                        include_trim_reencoded=settings.video_reference_local_area,
                        include_soundtrack_reencode=settings.video_reference_soundtrack_reencode,
                        include_picture_insertion=settings.video_reference_picture_insertion,
                    )
                else:
                    record, assets = video_reference.compare(reference, submitted, timing_event_s=timing_event_s,
                                                              timing_model=settings.video_reference_timing_model_path,
                                                              timing_short_fallback=True, timing_local_sound=True,
                                                              include_channel_timing=True)
            except ValueError as error:
                raise ProblemError(422, f"Could not compare {kind} files", str(error)) from None
            if (record["submitted"]["sha256"], record["submitted"]["bytes"]) != expected:
                raise _mismatch()
            if (record["earlier"]["sha256"], record["earlier"]["bytes"]) != uploaded:
                raise ProblemError(500, "Earlier copy integrity check failed", "The earlier copy changed during comparison.")
    except OSError:
        raise ProblemError(500, f"{kind.title()} comparison unavailable", "The local service could not read or stage the files.") from None

    payload = {"comparison": record, "assets": {name: _base64(data) for name, data in assets.items()}}
    if include_pdf:
        from app.reports.media_reference_pdf import render

        try:
            payload["pdf"] = _base64(render(record, assets))
        except Exception as error:
            log.warning("media_reference.pdf_failed", kind=kind, exception=type(error).__name__)
    if _original(aid, kind) != original:
        raise _unavailable(kind)
    return JSONResponse(payload, headers=NO_STORE)


_UPLOAD_RESPONSES = problem_responses(401, 404, 409, 413, 422, 429, 500)
def _upload_body(kind):
    earlier = ("Earlier copy with the same timing or a supported small trim."
               if kind == "audio" else "Earlier copy with the same timing.")
    properties = {
        "earlier": {"type": "string", "format": "binary", "description": earlier},
        "include_pdf": {"type": "string", "enum": ["true", "false"], "default": "false"},
    }
    if kind == "video":
        properties["timing_event_s"] = {"type": "string", "description": "Optional marked mouth closure time in the earlier clip, in seconds."}
    else:
        properties["earlier_has_clean_speech"] = {
            "type": "string", "enum": ["true", "false"], "default": "false",
            "description": "Optional confirmation of an earlier clean copy of the same six- or twelve-second recording on the same clock.",
        }
    return {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["earlier"], "properties": properties,
    }}}}}


@router.post(VIDEO_ROUTE, response_model=MediaReferenceResponse, responses=_UPLOAD_RESPONSES, openapi_extra=_upload_body("video"))
@router.post(AUDIO_ROUTE, response_model=MediaReferenceResponse, responses=_UPLOAD_RESPONSES, openapi_extra=_upload_body("audio"))
async def comparison(aid: str, request: Request):
    kind = kind_for_path(request.url.path)
    assert kind is not None
    if not _authenticated(request):
        raise ProblemError(401, "Sign in required", "Use an authenticated session or API key for earlier-copy comparison.")
    original = await run_in_threadpool(_original, aid, kind)
    if original is None:
        raise _unavailable(kind)
    if not _slot.acquire(blocking=False):
        raise ProblemError(429, "Source comparison busy", "Try again after the active source comparison finishes.")
    try:
        async with request.form(max_files=1, max_fields=2, max_part_size=1024) as form:
            earlier = form.get("earlier")
            include_pdf = form.get("include_pdf", "false")
            if not isinstance(earlier, UploadFile) or include_pdf not in ("true", "false"):
                raise ProblemError(422, f"Earlier {kind} required", "Upload a single earlier copy to compare.")
            timing = form.getlist("timing_event_s")
            event = None
            if timing and (kind != "video" or len(timing) != 1 or not isinstance(timing[0], str)
                           or re.fullmatch(r"[0-9]{1,2}(?:\.[0-9]{1,3})?", timing[0]) is None
                           or (event := Fraction(timing[0])) > video_reference.MAX_SECONDS):
                raise ProblemError(422, "Invalid marked time", "Enter a time in seconds from the first picture in the earlier clip.")
            clean = form.getlist("earlier_has_clean_speech")
            if clean and (kind != "audio" or len(clean) != 1 or clean[0] not in ("true", "false")):
                raise ProblemError(422, "Invalid earlier recording option", "Confirm the optional earlier clean recording with true or false.")
            return await run_in_threadpool(_compare, aid, kind, original, earlier, include_pdf == "true", event, clean == ["true"])
    finally:
        _slot.release()
