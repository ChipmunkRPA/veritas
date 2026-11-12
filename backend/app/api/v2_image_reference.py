"""Optional local comparison of a retained image and a reviewer-supplied earlier copy."""
from __future__ import annotations

import base64
import hashlib
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Any, BinaryIO

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_evidence import _analysis
from app.core.config import settings
from app.core.logging import get_logger
from app.storage.db import get_session
from app.storage.objects import CHUNK, ObjectNotFound, ObjectStoreError, default_store

SUFFIX = "/image-reference-comparison"
ROUTE = "/analyses/{aid}" + SUFFIX
MAX_PIXELS = 4_000_000
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_slot = threading.BoundedSemaphore()
_image_errors = (Image.DecompressionBombError, OSError, SyntaxError, ValueError)
log = get_logger(__name__)

router = APIRouter(prefix="/api/v2", tags=["evidence"], dependencies=[Depends(rate_limit), Depends(authorize)],
                   route_class=CheckedRoute)


class Availability(BaseModel):
    available: bool


class Comparison(BaseModel):
    comparison: dict[str, Any]
    pngs: dict[str, str]
    pdf: str | None = None


def is_comparison_path(path: str) -> bool:
    return path.startswith("/api/v2/analyses/") and path.endswith(SUFFIX)


def enabled() -> bool:
    return settings.environment == "dev" and settings.image_reference_comparison_enabled


def upload_limit() -> int:
    return min(settings.max_image_bytes, 60 * 1024 * 1024)


def _original(aid: str) -> tuple[str, int] | None:
    with get_session() as db:
        analysis = _analysis(db, aid)
        if (not enabled() or analysis.kind != "image" or analysis.status != "done"
                or not 0 < analysis.size_bytes <= upload_limit()):
            return None
        return analysis.sha256, analysis.size_bytes


def _check_image(path: Path) -> None:
    with Image.open(path, formats=["PNG", "JPEG"]) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ProblemError(413, "Image dimensions exceed the limit", "Each image must be 4 megapixels or smaller.")
        if getattr(image, "n_frames", 1) != 1:
            raise ProblemError(422, "Unsupported image", "Comparison requires a single-frame JPEG or PNG image.")


def _unavailable() -> ProblemError:
    return ProblemError(404, "Image comparison unavailable", "This analysis has no retained image available for comparison.")


@router.get(ROUTE, response_model=Availability, responses=problem_responses(404))
def image_reference_availability(aid: str):
    original = _original(aid)
    available = False
    if original is not None:
        sha256, size = original
        try:
            path = default_store().path_for(sha256)
            if path.is_file() and path.stat().st_size == size:
                _check_image(path)
                available = True
        except (*_image_errors, ObjectStoreError, ProblemError):
            pass
    return JSONResponse({"available": available}, headers=NO_STORE)


def _copy_image(source: BinaryIO, destination: Path, byte_limit: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            size += len(chunk)
            if size > byte_limit:
                raise ProblemError(413, "Image file exceeds the limit", "The image is larger than the allowed upload size.")
            digest.update(chunk)
            output.write(chunk)
    return digest.hexdigest(), size


def _integrity_failure() -> ProblemError:
    return ProblemError(409, "Original integrity check failed", "The retained image no longer matches its recorded hash and size.")


def _base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _comparison_pdf(comparison: dict, assets: dict[str, bytes]) -> str | None:
    try:
        from app.reports.image_reference_pdf import render

        pdf = render(comparison, assets)
    except Exception as error:  # the rendering dependency may reject a comparison exhibit
        log.warning("image_reference.pdf_failed", exception=type(error).__name__)
        return None
    if len(pdf) > MAX_ARTIFACT_BYTES:
        log.warning("image_reference.pdf_too_large", bytes=len(pdf))
        return None
    return _base64(pdf)


def _compare(original: tuple[str, int], reference: UploadFile, include_pdf: bool, workspace: Path) -> JSONResponse:
    from cv2 import error as OpenCVError

    from app.evidence.image_reference import inspect

    sha256, size = original
    submitted_path = workspace / "submitted"
    try:
        with default_store().open(sha256) as stored:
            received = _copy_image(stored, submitted_path, size)
    except ObjectNotFound:
        raise _unavailable() from None
    except (ObjectStoreError, ProblemError):
        raise _integrity_failure() from None
    if received != original:
        raise _integrity_failure()
    try:
        _check_image(submitted_path)
    except (*_image_errors, ProblemError):
        raise _unavailable() from None

    reference_path = workspace / "reference"
    reference_sha, _ = _copy_image(reference.file, reference_path, upload_limit())
    try:
        _check_image(reference_path)
    except _image_errors:
        raise ProblemError(422, "Unsupported earlier copy", "Upload a readable, single-frame JPEG or PNG image.") from None

    artifacts = workspace / "evidence"
    try:
        comparison = inspect(reference_path, submitted_path, evidence_dir=artifacts, allow_proportional_resize=True)
    except (*_image_errors, OpenCVError):
        raise ProblemError(422, "Image comparison could not finish", "The comparison could not read or compare the two images.") from None
    if comparison["submitted"]["sha256"] != sha256:
        raise _integrity_failure()
    if comparison["reference"]["sha256"] != reference_sha:
        raise ProblemError(500, "Earlier copy integrity check failed", "The staged earlier copy changed during the comparison.")
    png_paths = sorted(artifacts.glob("*.png"))
    if sum(path.stat().st_size for path in png_paths) > MAX_ARTIFACT_BYTES:
        raise ProblemError(422, "Comparison images exceed the limit", "This pair produced too much visual evidence for the local comparison.")
    assets = {path.name: path.read_bytes() for path in png_paths}
    payload = {"comparison": comparison}
    if include_pdf:
        payload["pdf"] = _comparison_pdf(comparison, assets)
    payload["pngs"] = {name: _base64(data) for name, data in assets.items()}
    return JSONResponse(payload, headers=NO_STORE)


@router.post(ROUTE, response_model=Comparison, responses=problem_responses(404, 409, 413, 422, 429, 500))
def image_reference_comparison(aid: str, reference: Annotated[UploadFile, File()],
                               include_pdf: Annotated[bool, Form()] = False):
    original = _original(aid)
    if original is None:
        raise _unavailable()
    if not _slot.acquire(blocking=False):
        raise ProblemError(429, "Image comparison busy", "Try again after the active image comparison finishes.")
    try:
        with tempfile.TemporaryDirectory(prefix="image-reference-") as workspace:
            return _compare(original, reference, include_pdf, Path(workspace))
    except OSError:
        raise ProblemError(500, "Image comparison unavailable", "The local service could not read or stage the comparison files.") from None
    finally:
        _slot.release()
