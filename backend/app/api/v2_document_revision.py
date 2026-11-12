"""Local, authenticated comparison against a retained text or email original."""

from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_evidence import _analysis
from app.cases import review
from app.core.config import settings
from app.core.logging import get_logger
from app.evidence.document_revision import FORMATS, MAX_BYTES, SUPPORTED_DRAFTS
from app.storage.db import get_session
from app.storage.objects import ObjectNotFound, ObjectStoreError, default_store

SUFFIX = "/document-revision-comparison"
ROUTE = "/analyses/{aid}" + SUFFIX
_slot = threading.BoundedSemaphore()
log = get_logger(__name__)

router = APIRouter(prefix="/api/v2", tags=["evidence"], dependencies=[Depends(rate_limit), Depends(authorize)],
                   route_class=CheckedRoute)


class DocumentRevisionAvailability(BaseModel):
    available: bool
    extension: Literal[".eml", ".txt", ".docx"] | None = None


class DraftSource(BaseModel):
    sha256: str
    bytes: int
    decoded_text_sha256: str
    decoded_characters: int
    mime_part: str | None


class DraftParagraph(BaseModel):
    paragraph: int
    body_index: int
    start: int
    end: int


class DraftExcerpt(BaseModel):
    start: int
    end: int
    text: str
    changed_tokens: list[tuple[int, int]]
    paragraphs: list[DraftParagraph] | None = None


class DraftChange(BaseModel):
    id: str
    role: Literal["body", "quoted", "signature"]
    earlier: DraftExcerpt | None
    later: DraftExcerpt | None


class DraftComparison(BaseModel):
    schema_name: Literal["chipmunk.document-revision-comparison/1"] = Field(alias="schema")
    format: Literal["email", "plain_text", "docx"]
    coordinate_space: Literal["decoded_source_unicode_codepoint", "docx_body_unicode_codepoint"]
    earlier: DraftSource
    later: DraftSource
    changes: list[DraftChange]
    scope: str
    interpretation: str


class DocumentRevisionResponse(BaseModel):
    comparison: DraftComparison
    pdf: str | None = None


def is_comparison_path(path: str) -> bool:
    return path.startswith("/api/v2/analyses/") and path.endswith(SUFFIX)


def enabled() -> bool:
    return settings.environment == "dev" and settings.document_revision_comparison_enabled


def upload_limit() -> int:
    return min(settings.max_document_bytes, MAX_BYTES)


def _authenticated(request: Request) -> bool:
    return request.state.principal.auth in {"session", "api_key", "bootstrap"}


def _original(aid: str, *, reject_inactive: bool = False) -> tuple[str, int, str] | None:
    with get_session() as db:
        analysis = _analysis(db, aid)
        extension = Path(analysis.filename).suffix.lower()
        if (not enabled() or analysis.kind != "document" or analysis.status != "done" or not analysis.result_hash
                or extension not in FORMATS or not 0 < analysis.size_bytes <= upload_limit()):
            return None
        if analysis.withdrawn_at is not None:
            if reject_inactive:
                raise ProblemError(409, "Revision withdrawn", "This revision was withdrawn and should not be relied on.")
            return None
        if review.superseded(db, aid):
            if reject_inactive:
                raise ProblemError(409, "Revision superseded", "A newer revision exists. Compare against the latest analysis.")
            return None
        return analysis.sha256, analysis.size_bytes, extension


def _unavailable() -> ProblemError:
    return ProblemError(404, "Earlier-draft comparison unavailable", "A retained, completed, active .eml, .txt or .docx analysis is required.")


@router.get(ROUTE, response_model=DocumentRevisionAvailability, responses=problem_responses(401, 404))
def availability(aid: str, request: Request):
    original = _original(aid) if _authenticated(request) else None
    extension = None
    if original is not None:
        sha256, size, suffix = original
        try:
            path = default_store().path_for(sha256)
            if path.is_file() and path.stat().st_size == size:
                extension = suffix
        except (OSError, ObjectStoreError):
            pass
    return JSONResponse({"available": extension is not None, "extension": extension}, headers=NO_STORE)


def _compare(aid: str, original: tuple[str, int, str], earlier: bytes, include_pdf: bool) -> JSONResponse:
    from app.evidence.document_revision import compare

    sha256, size, extension = original
    try:
        with default_store().open(sha256) as stored:
            later = stored.read(size + 1)
    except ObjectNotFound:
        raise _unavailable() from None
    except (ObjectStoreError, OSError):
        raise ProblemError(409, "Original integrity check failed", "The retained document could not be read.") from None
    if len(later) != size or hashlib.sha256(later).hexdigest() != sha256:
        raise ProblemError(409, "Original integrity check failed", "The retained document no longer matches its recorded hash and size.")
    try:
        comparison = compare(earlier, later, extension)
    except (UnicodeError, LookupError, RecursionError):
        raise ProblemError(422, "Could not compare drafts", f"Use readable {SUPPORTED_DRAFTS}.") from None
    except ValueError as error:
        raise ProblemError(422, "Could not compare drafts", str(error)) from None
    except OSError:
        raise ProblemError(500, "Earlier-draft comparison unavailable", "The local service could not stage the comparison.") from None
    payload: dict = {"comparison": comparison}
    if include_pdf:
        from app.reports.document_revision_pdf import render

        try:
            payload["pdf"] = base64.b64encode(render(comparison)).decode("ascii")
        except Exception as error:
            log.warning("document_revision.pdf_failed", exception=type(error).__name__)
            payload["pdf"] = None
    if _original(aid, reject_inactive=True) != original:
        raise _unavailable()
    return JSONResponse(payload, headers=NO_STORE)


@router.post(
    ROUTE, response_model=DocumentRevisionResponse, responses=problem_responses(400, 401, 404, 409, 413, 422, 429, 500),
    openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["earlier"], "properties": {
            "earlier": {"type": "string", "format": "binary", "description": f"Earlier version. Use {SUPPORTED_DRAFTS}."},
            "include_pdf": {"type": "string", "enum": ["true", "false"], "default": "false"},
        },
    }}}}},
)
async def comparison(aid: str, request: Request):
    if not _authenticated(request):
        raise ProblemError(401, "Sign in required", "Use an authenticated session or API key for earlier-draft comparison.")
    original = await run_in_threadpool(_original, aid, reject_inactive=True)
    if original is None:
        raise _unavailable()
    if not _slot.acquire(blocking=False):
        raise ProblemError(429, "Draft comparison busy", "Try again after the active draft comparison finishes.")
    try:
        async with request.form(max_files=1, max_fields=1, max_part_size=1024) as form:
            earlier = form.get("earlier")
            include_pdf = form.get("include_pdf", "false")
            if not isinstance(earlier, UploadFile) or include_pdf not in ("true", "false"):
                raise ProblemError(422, "Earlier draft required", "Upload a single earlier draft to compare.")
            if Path(earlier.filename or "").suffix.lower() != original[2]:
                raise ProblemError(422, "Draft formats differ", f"Choose an earlier {original[2]} file to match the retained document.")
            limit = upload_limit()
            data = await earlier.read(limit + 1)
            if len(data) > limit:
                raise ProblemError(413, "Draft exceeds the limit", "The earlier draft is larger than the allowed upload size.")
        return await run_in_threadpool(_compare, aid, original, data, include_pdf == "true")
    finally:
        _slot.release()
