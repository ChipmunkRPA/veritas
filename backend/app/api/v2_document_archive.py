"""Optional source comparison on completed plain-text and Word analyses."""

from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_evidence import _analysis
from app.billing.entitlements import PaymentRequired, require_analysis_payment
from app.cases import review
from app.core.config import settings
from app.core.deadline import StepTimeout
from app.core.logging import get_logger
from app.detectors.text.extract import extract
from app.evidence import document_archive, document_archive_word, document_gao_archive
from app.evidence.document_revision import MAX_BYTES
from app.storage.db import get_session
from app.storage.objects import ObjectNotFound, ObjectStoreError, default_store

ROUTE = "/analyses/{aid}/document-archive-comparison"
FORMATS = {".txt", ".docx"}
_slot = threading.BoundedSemaphore()
log = get_logger(__name__)

router = APIRouter(prefix="/api/v2", tags=["evidence"], dependencies=[Depends(rate_limit), Depends(authorize)],
                   route_class=CheckedRoute)

ArchiveStatus = Literal["reference_bound", "no_qualified_reference", "ambiguous_reference", "out_of_scope"]
ArchiveAlignment = Literal["fingerprint_anchor", "bounded_pair", "bounded_insert", "bounded_delete"]


class DocumentArchiveAvailability(BaseModel):
    available: bool
    gao_available: bool | None = None


class DocumentArchiveLibrary(BaseModel):
    title: str
    publisher: str
    url: str
    license: str
    license_url: str
    revision: str
    sha256: str
    indexed_excerpts: int


class DocumentArchiveCoverage(BaseModel):
    body_paragraphs: int
    body_words: int
    matched_words: int


class DocumentArchiveReference(BaseModel):
    id: str
    url: str
    version_url: str
    volume: str
    document: str
    published_date: str | None = None
    source_record_sha256: str | None = None
    transcript_sha256: str | None = None
    submitted_range: tuple[int, int] | None = None
    transcript_range: tuple[int, int] | None = None


class DocumentArchiveWordRun(BaseModel):
    run: int
    range: tuple[int, int]


class DocumentArchiveWordToken(BaseModel):
    extracted_range: tuple[int, int]
    paragraph_range: tuple[int, int]
    runs: list[DocumentArchiveWordRun]


class DocumentArchiveWordLocation(BaseModel):
    paragraph: int
    body_index: int
    paragraph_range: tuple[int, int]
    changed_tokens: list[DocumentArchiveWordToken]
    runs_complete: bool


class DocumentArchiveParagraph(BaseModel):
    submitted_paragraph: int
    submitted_range: tuple[int, int]
    archive_paragraph: int | None
    alignment: ArchiveAlignment
    docx_paragraph: int | None = None
    docx_location: DocumentArchiveWordLocation | None = None


class DocumentArchiveExcerpt(BaseModel):
    start: int
    end: int
    text: str
    changed_tokens: list[tuple[int, int]]
    docx_location: DocumentArchiveWordLocation | None = None


class DocumentArchiveChange(BaseModel):
    submitted_paragraph: int
    archive_paragraph: int | None
    alignment: ArchiveAlignment
    submitted: DocumentArchiveExcerpt
    archive: DocumentArchiveExcerpt | None


class DocumentArchiveComparison(BaseModel):
    schema_name: Literal["chipmunk.document-archive-comparison/1", "chipmunk.document-archive-comparison/2"] = Field(alias="schema")
    status: ArchiveStatus
    coordinate_space: Literal["extracted_text_unicode_codepoint"]
    layout: Literal["submitted_paragraphs", "recovered_archive_boundaries"]
    scope: str
    interpretation: str
    authorship_confidence_contribution: Literal[0]
    submitted_sha256: str
    extracted_text_sha256: str
    library: DocumentArchiveLibrary
    coverage: DocumentArchiveCoverage
    reference: DocumentArchiveReference | None
    aligned_paragraphs: list[DocumentArchiveParagraph]
    changes: list[DocumentArchiveChange]
    submitted_sentence_ranges: list[tuple[int, int]] | None = None


class DocumentArchiveResponse(BaseModel):
    comparison: DocumentArchiveComparison
    pdf: str | None


def _enabled() -> bool:
    return settings.environment == "dev" and settings.document_archive_comparison_enabled


def _authenticated(request: Request) -> bool:
    return request.state.principal.auth in {"session", "api_key", "bootstrap"}


def _gao_available(extension: str) -> bool:
    return extension in FORMATS and settings.document_gao_archive_path is not None


def _original(aid: str, owner: str, *, reject_inactive: bool = False) -> tuple[str, int, str, str] | None:
    if not _enabled():
        return None
    with get_session() as db:
        analysis = _analysis(db, aid)
        extension = Path(analysis.filename).suffix.lower()
        if (analysis.submitted_by != owner or analysis.kind != "document"
                or analysis.status != "done" or not analysis.result_hash
                or extension not in FORMATS
                or not 0 < analysis.size_bytes <= min(settings.max_document_bytes, MAX_BYTES)):
            return None
        if analysis.withdrawn_at is not None or review.superseded(db, aid):
            if reject_inactive:
                raise ProblemError(409, "Revision inactive", "Use an active analysis revision for the archive comparison.")
            return None
        try:
            require_analysis_payment(db, analysis)
        except PaymentRequired:
            if reject_inactive:
                raise ProblemError(402, "Payment receipt required", "This analysis has no confirmed payment receipt.") from None
            return None
        return analysis.sha256, analysis.size_bytes, analysis.result_hash, extension


def _unavailable() -> ProblemError:
    return ProblemError(404, "Archive comparison unavailable",
                        "A completed, active analysis with a retained .txt or .docx file is required.")


@router.get(ROUTE, response_model=DocumentArchiveAvailability, responses=problem_responses(401, 404))
def document_archive_availability(aid: str, request: Request):
    original = _original(aid, request.state.principal.id) if _authenticated(request) else None
    available = False
    if original is not None:
        sha, size, _, _ = original
        try:
            path = default_store().path_for(sha)
            available = path.is_file() and path.stat().st_size == size
        except (OSError, ObjectStoreError):
            pass
    body = {"available": available}
    if available and original is not None and _gao_available(original[3]):
        body["gao_available"] = True
    return JSONResponse(body, headers=NO_STORE)


@router.post(ROUTE, response_model=DocumentArchiveResponse, responses=problem_responses(401, 402, 404, 409, 422, 429, 503))
def compare_document_archive(aid: str, request: Request, include_pdf: bool = False,
                             collection: Literal["frus", "gao"] = "frus"):
    if not _authenticated(request):
        raise ProblemError(401, "Sign in required", "Use an authenticated session or API key for the archive comparison.")
    owner = request.state.principal.id
    original = _original(aid, owner, reject_inactive=True)
    if original is None:
        raise _unavailable()
    if collection == "gao" and not _gao_available(original[3]):
        raise ProblemError(404, "GAO highlights comparison unavailable", "A completed TXT or Word analysis and local GAO archive are required.")
    if not _slot.acquire(blocking=False):
        raise ProblemError(429, "Archive comparison busy", "Try again after the active comparison finishes.")
    try:
        return _compare(aid, owner, original, include_pdf, collection)
    finally:
        _slot.release()


def _compare(aid: str, owner: str, original: tuple[str, int, str, str], include_pdf: bool,
             collection: Literal["frus", "gao"]) -> JSONResponse:
    if _original(aid, owner, reject_inactive=True) != original:
        raise _unavailable()
    sha, size, _, extension = original
    try:
        with default_store().open(sha) as stored:
            data = stored.read(size + 1)
    except ObjectNotFound:
        raise _unavailable() from None
    except (ObjectStoreError, OSError):
        raise ProblemError(409, "Original integrity check failed", "The retained document could not be read.") from None
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
        raise ProblemError(409, "Original integrity check failed", "The retained document differs from its recorded hash and size.")
    try:
        if extension == ".docx":
            if collection == "gao":
                pack_sha256 = settings.document_gao_archive_sha256
                catalog, index = document_gao_archive.library(settings.document_gao_archive_path, pack_sha256)
                comparison = document_archive_word.compare_gao(data, sha, pack_sha256, catalog, index)
            else:
                comparison = document_archive_word.compare(data, sha)
        else:
            with TemporaryDirectory(prefix="chipmunk-archive-") as directory:
                path = Path(directory) / "retained.txt"
                path.write_bytes(data)
                doc = extract(path, ".txt")
                if "\ufffd" in doc.text or "\x00" in doc.text:
                    raise ValueError("The file contains undecodable text.")
                if collection == "gao":
                    pack_sha256 = settings.document_gao_archive_sha256
                    catalog, index = document_gao_archive.library(settings.document_gao_archive_path, pack_sha256)
                    comparison = document_gao_archive.compare(doc, sha, pack_sha256, catalog, index)
                else:
                    comparison = document_archive.compare(doc, sha)
    except document_archive.ArchiveUnavailable:
        raise ProblemError(503, "Archive library unavailable", "The local archive library could not be verified.") from None
    except (UnicodeError, LookupError, ValueError, StepTimeout):
        raise ProblemError(422, "Could not read document", "Use a readable plain-text or standard Word document.") from None
    except OSError:
        raise ProblemError(503, "Archive comparison unavailable", "The local service could not read the document.") from None
    pdf = None
    if include_pdf:
        from app.reports.document_archive_pdf import render

        try:
            pdf = base64.b64encode(render(comparison)).decode("ascii")
        except Exception as error:
            log.warning("document_archive.pdf_failed", exception=type(error).__name__)
    if _original(aid, owner, reject_inactive=True) != original:
        raise _unavailable()
    payload = DocumentArchiveResponse.model_validate({"comparison": comparison, "pdf": pdf})
    return JSONResponse(payload.model_dump(mode="json", by_alias=True, exclude_unset=True), headers=NO_STORE)
