"""API v2 reports, signatures and evidence packages (ARCHITECTURE §5.2, §6; PRD RP-03, RP-05, RP-06; D-10, D-11, D-25, D-27, D-30;
BUILD_SPEC R-1, R-4).

- `POST /analyses/{aid}/reports` {profile, redact?, revision_reason?} → 202 with the queued report revision. Needs the examiner
  role on the case. The snapshot is frozen now; the `report` job renders it.
- `GET /analyses/{aid}/reports`: every revision of every profile, newest first (RP-06: earlier revisions stay downloadable).
- `GET /reports/{report_id}`: one report. `GET /reports/{report_id}/file`: the PDF, re-hashed before it is served (ETag =
  SHA-256, immutable). `GET /reports/{report_id}/preview.html`: the same document as HTML from the stored snapshot, without font
  files and under a CSP that loads nothing, for visual checks (§6.1).
- **Refusals.** 404: unknown analysis or report, or a case the caller is not a member of (CM-04). 409: the analysis is not
  finished, withdrawn, superseded or has no result; the report is not done (file), or its preview template identity differs
  from this renderer. 422: invalid body; `redact` outside the
  newsroom profile. 503: this build has no template for the profile, or WeasyPrint cannot load. 500: a stored object is missing
  or altered.
- `GET /instance/public-key`: the Ed25519 instance public key that signs reports and packages (D-11).
- `POST /analyses/{aid}/packages` {report_id, kind?, include_original?} → 202 with the queued package (examiner). `GET
  /packages/{package_id}` and `/packages/{package_id}/file` (the ZIP, re-hashed, immutable). 404: the report is not a report of
  this analysis. 409: the report is not done or unsigned; `issue` for a report not requested as reviewed or approved (D-27), or
  for a withdrawn or superseded revision (D-47; the package job checks again).
  422: invalid body; a redacted (newsroom) report, since a package carries the full result.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api import jobs
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import RequestModel
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_annotations import clean_body
from app.api.v2_evidence import _analysis, _artefact, _binary, _binary_responses, _read_verified, _result
from app.audit import chain as audit
from app.cases import access, review
from app.core import times
from app.reports import engine, package, service, signing, snapshot
from app.reports.public_policy import (
    downloadable_template,
    legacy_face_scope_safe,
    needs_legacy_face_scope_check,
    package_template,
)
from app.storage.db import Analysis, PackageRow, ReportRow, get_session

router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
Profile = Literal["summary", "standard", "forensic", "newsroom"]
PREVIEW_HEADERS = {**NO_STORE, "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; frame-ancestors 'none'",
                   "X-Content-Type-Options": "nosniff"}


class ReportRequest(RequestModel):
    model_config = ConfigDict(extra="forbid")
    profile: Profile
    redact: bool = Field(default=False, description="Newsroom only: hide filename, source fields and metadata (RP-03)")
    revision_reason: str | None = Field(default=None, max_length=2000, description="Why this revision supersedes the previous one (RP-06)")
    reuse: bool = Field(default=False, description="Answer with the newest report of this profile whose snapshot holds the record "
                                                    "as it stands now, when there is one (200), and make a report only otherwise "
                                                    "(202). An evidence bundle asks this way, so that a bundle request mints no "
                                                    "further report.")


class ReportOut(BaseModel):
    id: str = Field(description="VR-<year>-<6 digits>")
    display_id: str = Field(description="R-<analysis id8>-DRAFT-<revision> until the report is issuable, then R-<analysis id8>-<revision>")
    revision: int
    supersedes_report_id: str | None
    revision_reason: str | None
    analysis_id: str
    analysis_revision: int
    case_id: str | None
    profile: Profile
    redacted: bool
    review_state: str = Field(description="Displayed review state when the report was requested")
    issuable: bool = Field(description="The report was requested in an issuable state: approved, or reviewed on a case that does not "
                                       "require a second review. Only such a report is labelled issued and can be issued in a package.")
    template_set: str
    engine: str
    engine_version: str | None
    status: Literal["queued", "running", "done", "failed"]
    downloadable: bool = Field(description="Whether this immutable report uses an approved public template and has a file")
    error: str | None
    snapshot_hash: str
    sha256: str | None = Field(description="SHA-256 of the PDF once done")
    page_count: int | None
    duration_ms: int | None
    signature: str | None = Field(description="Ed25519 detached signature over the PDF bytes, base64 (RP-06, D-11)")
    signing_key_id: str | None = Field(description="ed25519:<first 16 hex of SHA-256 over the raw public key>")
    requested_by: str
    requested_at: datetime
    finished_at: datetime | None


class ReportList(BaseModel):
    items: list[ReportOut]


_utc = times.to_utc  # one form on every engine: a time is printed in UTC (D-72)


def _public_snapshot(db: Session, row: ReportRow):
    return snapshot.load(_read_verified(_artefact(db, row.snapshot_artefact_id, row.analysis_id)))


def _public_downloadable(db: Session, row: ReportRow, analysis_kind: str) -> bool:
    if not downloadable_template(row.template_set, analysis_kind):
        return False
    if not needs_legacy_face_scope_check(row.template_set, analysis_kind):
        return True
    try:
        frozen = _public_snapshot(db, row)
    except (ProblemError, ValueError):
        return False
    return (frozen.report.id == row.id and frozen.report.template_set == row.template_set
            and frozen.analysis.id == row.analysis_id and frozen.analysis.kind == analysis_kind
            and legacy_face_scope_safe(frozen))


def _out(db: Session, row: ReportRow, analysis_kind: str) -> ReportOut:
    return ReportOut(id=row.id, display_id=row.display_id, revision=row.revision, supersedes_report_id=row.supersedes_report_id,
                     revision_reason=row.revision_reason, analysis_id=row.analysis_id, analysis_revision=row.analysis_revision,
                     case_id=row.case_id, profile=row.profile, redacted=row.redacted, review_state=row.review_state,  # type: ignore[arg-type]
                     issuable=bool(row.issuable),
                     template_set=row.template_set, engine=row.engine, engine_version=row.engine_version, status=row.status,  # type: ignore[arg-type]
                     downloadable=row.status == 'done' and row.artefact_id is not None and _public_downloadable(db, row, analysis_kind),
                     error=row.error, snapshot_hash=row.snapshot_hash, sha256=row.sha256, page_count=row.page_count,
                     duration_ms=row.duration_ms, signature=row.signature, signing_key_id=row.signing_key_id, requested_by=row.requested_by,
                     requested_at=_utc(row.requested_at), finished_at=_utc(row.finished_at))  # type: ignore[arg-type]


def _json(model: BaseModel, status_code: int = 200) -> JSONResponse:
    return JSONResponse(jsonable_encoder(model), status_code=status_code, headers=NO_STORE)


def _report(db: Session, report_id: str) -> ReportRow:
    row = db.get(ReportRow, report_id)
    if row is None:
        raise ProblemError(404, *access.NOT_FOUND["report"])
    return row


@router.post("/analyses/{aid}/reports", status_code=202, response_model=ReportOut, responses=problem_responses(401, 403, 404, 409, 422, 503))
def request_report(aid: str, req: ReportRequest, request: Request) -> JSONResponse:
    principal = request.state.principal
    if req.redact and req.profile != "newsroom":
        raise ProblemError(422, "Invalid request", "redact applies to the newsroom profile only (RP-03).")

    # N-10a (code re-verification R2-02): what takes time comes before the unit, outside the revision's lock and the append
    # lock, on a session that is closed before the unit opens its own (one connection at a time, N-9a). First the refusals,
    # so a request that is refused walks no chain; then, for the two full profiles, the whole custody chain. The unit
    # checks again and hashes only the events appended since
    with get_session() as db:
        _reportable(db, aid, req)
        verified = audit.verify_chain(db=db) if req.profile in snapshot.FULL_PROFILES else None

    def unit() -> tuple[ReportOut, bool]:
        # N-9 (code review 12, 14): the report row, its job and the `redaction` event of a redacted report are one transaction
        with audit.writing() as db:
            return _queue_report(db, aid, req, principal.id, verified)

    try:
        out, made = audit.repeated(unit)
    except IntegrityError:  # an id or a revision number taken by a writer the allocation's lock did not order: nothing stored
        raise ProblemError(503, "Report not queued", "Another request took the same report number first. Send the request "
                                                     "again.") from None
    if not made:  # N-19: `reuse` found a report of the record as it stands; nothing was written
        return _json(out, 200)
    jobs.wake()
    return _json(out, 202)


def _reportable(db: Session, aid: str, req: ReportRequest) -> tuple[Analysis, dict[str, Any], str]:
    """The refusals of a report request, on the revision as it is stored now: the revision, its result document and the
    version of the report engine. Asked before the request does its slow work and again inside its unit."""
    analysis = db.scalar(select(Analysis).where(Analysis.id == aid).execution_options(populate_existing=True))
    if analysis is None:
        raise ProblemError(404, *access.NOT_FOUND["analysis"])
    if analysis.status != "done":
        raise ProblemError(409, "Analysis not finished", f"The analysis is {analysis.status}; only a finished analysis is reported.")
    if analysis.withdrawn_at is not None:
        raise ProblemError(409, "Revision withdrawn", "This revision was withdrawn and should not be relied on.")
    if review.superseded(db, aid):
        raise ProblemError(409, "Revision superseded", "A newer revision of this analysis exists; report on the latest revision.")
    try:
        engine.require_profile(req.profile)
        version = engine.engine_version()
    except engine.ProfileUnavailable as e:
        raise ProblemError(503, "Report profile unavailable", str(e)) from None
    except engine.EngineUnavailable as e:
        raise ProblemError(503, "Report engine unavailable", str(e)) from None
    result = _result(db, analysis)
    if not result or not result.get("decision"):
        raise ProblemError(409, "No result", "This analysis has no result document to report on.")
    return analysis, result, version


def _queue_report(db: Session, aid: str, req: ReportRequest, principal_id: str,
                  verified: audit.Verification | None = None) -> tuple[ReportOut, bool]:
    """The checks of a report request and its row, job and events, in the caller's transaction (committed here).
    `verified` is the walk of the custody chain the request made before this unit (`snapshot.custody_verification`).
    Returns the report and whether it was made here: a request with `reuse` gets the report that already holds the record as
    it stands (N-19), and then nothing is written."""
    _analysis(db, aid)
    # N-9 (Postgres QA X-7): the revision's row lock first, as a note and a conclusion take it. A purge or a delete that
    # holds the row finishes before the snapshot is frozen, and the request then finds no revision (404) where its
    # snapshot row failed on a foreign key (500)
    if not review.hold_revision(db, aid):
        db.rollback()
        raise ProblemError(404, *access.NOT_FOUND["analysis"])
    analysis, result, version = _reportable(db, aid, req)
    if req.reuse:
        existing = service.current(db, analysis, result, profile=req.profile, redacted=req.profile == "newsroom", verified=verified)
        if existing is not None:
            out = _out(db, existing, analysis.kind)
            db.rollback()
            return out, False
    # D-10: the newsroom explainer is always redacted, with a `redaction` custody event
    row, events = service.prepare(db, analysis, result, profile=req.profile, redacted=req.profile == "newsroom",
                                  revision_reason=clean_body(req.revision_reason), requested_by=principal_id, engine_version=version,
                                  verified=verified)
    out = _out(db, row, analysis.kind)
    jobs.submit_report_in(db, row.id)
    audit.commit_with(db, events)
    return out, True


@router.get("/analyses/{aid}/reports", response_model=ReportList, responses=problem_responses(401, 404))
def list_reports(aid: str) -> JSONResponse:
    with get_session() as db:
        analysis = _analysis(db, aid)
        rows = db.scalars(select(ReportRow).where(ReportRow.analysis_id == aid).order_by(ReportRow.requested_at.desc(), ReportRow.id.desc()))
        return _json(ReportList(items=[_out(db, r, analysis.kind) for r in rows]))


@router.get("/reports/{report_id}", response_model=ReportOut, responses=problem_responses(401, 404))
def get_report(report_id: str) -> JSONResponse:
    with get_session() as db:
        row = _report(db, report_id)
        return _json(_out(db, row, _analysis(db, row.analysis_id).kind))


@router.get("/reports/{report_id}/file", response_class=Response,
            responses={**_binary_responses("application/pdf"), **problem_responses(409)})
def report_file(report_id: str, request: Request) -> Response:
    with get_session() as db:
        row = _report(db, report_id)
        if row.status != "done" or row.artefact_id is None:
            raise ProblemError(409, "Report not ready", f"The report is {row.status}; its file exists once it is done.")
        if not _public_downloadable(db, row, _analysis(db, row.analysis_id).kind):
            raise ProblemError(409, "Earlier report unavailable", "This earlier report may contain model names. Generate a new report to download a version with neutral check labels. The original remains stored.")
        response = _binary(request, _artefact(db, row.artefact_id, row.analysis_id))
        filename = f"chipmunk-verify-{row.display_id}.pdf"
    if response.status_code == 200:
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@router.get("/reports/{report_id}/preview.html", response_class=HTMLResponse,
            responses={200: {"content": {"text/html": {}}, "description": "The report document as HTML"}, **problem_responses(401, 404, 409, 500, 503)})
def report_preview(report_id: str) -> HTMLResponse:
    with get_session() as db:
        row = _report(db, report_id)
        if row.template_set != engine.TEMPLATE_SET:
            if _public_downloadable(db, row, _analysis(db, row.analysis_id).kind):
                raise ProblemError(409, "Earlier preview unavailable",
                                   "Generate a new report for an HTML preview. The completed report's saved PDF is still available.")
            raise ProblemError(409, "Earlier report unavailable", "This earlier report may contain model names. Generate a new report to preview a version with neutral check labels. The original remains stored.")
        data = _read_verified(_artefact(db, row.snapshot_artefact_id, row.analysis_id))
        requested_template = row.template_set
    frozen = snapshot.load(data)
    if not requested_template == frozen.report.template_set == engine.TEMPLATE_SET:
        guidance = "Request a new report revision; earlier reports may contain model names."
        raise ProblemError(409, "Report template mismatch",
                           "The requested report template does not match the stored snapshot and current HTML renderer. " + guidance)
    try:
        html = engine.render_html(frozen, embed_fonts=False)
    except engine.ProfileUnavailable as e:
        raise ProblemError(503, "Report profile unavailable", str(e)) from None
    return HTMLResponse(html, headers=PREVIEW_HEADERS)


# --- signatures and evidence packages (R-4) ------------------------------------------------------------------------------------

class PublicKeyOut(BaseModel):
    key_id: str = Field(description="ed25519:<first 16 hex of SHA-256 over the raw public key>; pass it to "
                                    "`python -m app.cli verify --expect-key-id`")
    algorithm: Literal["Ed25519"]
    public_key_pem: str = Field(description="SubjectPublicKeyInfo PEM; pass it to `python -m app.cli verify --public-key`")


class PackageRequest(RequestModel):
    model_config = ConfigDict(extra="forbid")
    report_id: str = Field(min_length=1, max_length=32, description="A finished, signed report of this analysis (report.pdf)")
    kind: Literal["draft", "issue"] = Field(default="draft", description="D-27: issue needs a report requested as reviewed or approved")
    include_original: bool = Field(default=False, description="Include unredacted user source bytes; public packages do not support inference reproduction")


class PackageOut(BaseModel):
    id: str
    analysis_id: str
    report_id: str
    case_id: str | None
    kind: Literal["draft", "issue"]
    schema_id: str
    include_original: bool
    status: Literal["queued", "running", "done", "failed"]
    error: str | None
    filename: str | None = Field(description="Download filename; new public packages use chipmunk-evidence-<analysis id8>-r<revision>[-draft].zip")
    sha256: str | None = Field(description="SHA-256 of the ZIP once done, recorded as package_exported")
    manifest_sha256: str | None
    signature: str | None = Field(description="Ed25519 signature over manifest.sha256, base64")
    signing_key_id: str | None
    file_count: int | None
    size_bytes: int | None
    created_by: str
    created_at: datetime
    finished_at: datetime | None


def _package_out(db: Session, row: PackageRow) -> PackageOut:
    report = db.get(ReportRow, row.report_id)
    return PackageOut(id=row.id, analysis_id=row.analysis_id, report_id=row.report_id, case_id=row.case_id, kind=row.kind,  # type: ignore[arg-type]
                      schema_id=row.schema_id,
                      include_original=row.include_original, status=row.status, error=row.error,  # type: ignore[arg-type]
                      filename=package.filename(row, report) if report is not None else None, sha256=row.sha256,
                      manifest_sha256=row.manifest_sha256, signature=row.signature, signing_key_id=row.signing_key_id,
                      file_count=row.file_count, size_bytes=row.size_bytes, created_by=row.created_by,
                      created_at=_utc(row.created_at), finished_at=_utc(row.finished_at))  # type: ignore[arg-type]


def _package(db: Session, package_id: str) -> PackageRow:
    row = db.get(PackageRow, package_id)
    if row is None:
        raise ProblemError(404, *access.NOT_FOUND["package"])
    return row


@router.get("/instance/public-key", response_model=PublicKeyOut, responses=problem_responses(401, 503))
def instance_public_key() -> JSONResponse:
    try:
        key = signing.instance_key()
        signing.ensure_registered(key)
    except signing.SigningKeyError as e:
        raise ProblemError(503, "Signing key unavailable", str(e)) from None
    return _json(PublicKeyOut(key_id=key.key_id, algorithm="Ed25519", public_key_pem=key.public_pem().decode("ascii")))


def _issue_refusal(report: ReportRow, brief: dict) -> str | None:
    """Why a finished report of a current revision is not issued, else None. The report's frozen `issuable` flag decides, and
    the revision must still be issuable now: a peer review that does not concur, or a second review the case has since come to
    require, stops an earlier report too."""
    if not brief["signatures_valid"]:
        return ("A review record of this revision does not verify: the signature of its conclusion or of a sign-off does not match "
                "its content, so nothing is issued. Export a draft and have an administrator examine the record.")
    if brief["state"] == "disputed":
        return ("The peer reviewer does not concur with this revision's conclusion, so it is not issued. Export a draft; the way "
                "forward is a new revision.")
    if not report.issuable:
        if report.review_state == "reviewed" and brief["requires_second_review"]:
            return ("This case requires a second review, so only a report requested after a concurring peer review is issued. "
                    "Export a draft, or complete the peer review and request a new report.")
        if report.review_state == "reviewed":
            return ("This report was requested while it could not be issued: a required second review was outstanding, or a review "
                    "record did not verify. Request a new report and issue that.")
        if report.review_state in ("approved", "disputed"):  # a report of a disputed revision, whatever an earlier release called it
            return ("This report was requested after a peer review that does not concur, so it is not issued. Export a draft; the "
                    "way forward is a new revision.")
        return (f"The report was requested in the review state '{report.review_state}'; only reports of a signed-off revision "
                "are issued. Export a draft, or sign off and request a new report.")
    if not brief["issuable"]:
        return ("This case requires a second review, so only a report requested after a concurring peer review is issued. "
                "Export a draft, or complete the peer review and request a new report.")
    return None


@router.post("/analyses/{aid}/packages", status_code=202, response_model=PackageOut, responses=problem_responses(401, 403, 404, 409, 422))
def request_package(aid: str, req: PackageRequest, request: Request) -> JSONResponse:
    principal = request.state.principal
    with get_session() as db:
        analysis = _analysis(db, aid)
        report = db.get(ReportRow, req.report_id)
        if report is None or report.analysis_id != aid:
            raise ProblemError(404, *access.NOT_FOUND["report"])
        if report.redacted:
            raise ProblemError(422, "Invalid request", "Evidence packages include source excerpts and crops. Choose an unredacted report.")
        if not package_template(report.template_set, analysis.kind):
            raise ProblemError(409, "Earlier report unavailable", "Generate a new report with the current public evidence format. The original remains stored.")
        if report.status != 'done' or report.artefact_id is None or not report.signature:
            raise ProblemError(409, "Report not ready", "A public package needs a completed, signed report.")
        if req.kind == 'issue':
            if analysis.withdrawn_at is not None or review.superseded(db, aid):
                raise ProblemError(409, "Revision unavailable", "A withdrawn or superseded revision cannot be issued.")
            if refusal := _issue_refusal(report, review.brief(db, analysis)):
                raise ProblemError(409, "Report not issuable", refusal)
        # Check availability now, then verify it again against the frozen report in the job.
        from app.reports.public_package import _evidence
        try:
            frozen = _public_snapshot(db, report)
            if not legacy_face_scope_safe(frozen):
                raise ProblemError(409, "Earlier report unavailable",
                                   "Generate a new report and public evidence package. The original remains stored.")
            _evidence(frozen)
        except (package.PackageError, ValueError) as error:
            raise ProblemError(409, "Evidence unavailable", str(error)) from None
        row = package.prepare(db, analysis, report, kind=req.kind, include_original=req.include_original,
                              created_by=principal.id, schema_id=package.PUBLIC_SCHEMA)
        out = _package_out(db, row)
        jobs.submit_package_in(db, row.id)
        db.commit()
    jobs.wake()
    return _json(out, 202)


@router.get("/packages/{package_id}", response_model=PackageOut, responses=problem_responses(401, 403, 404))
def get_package(package_id: str) -> JSONResponse:
    with get_session() as db:
        return _json(_package_out(db, _package(db, package_id)))


@router.get("/packages/{package_id}/file", response_class=Response,
            responses={**_binary_responses("application/zip"), **problem_responses(403, 409)})
def package_file(package_id: str, request: Request) -> Response:
    with get_session() as db:
        row = _package(db, package_id)
        if row.schema_id != package.PUBLIC_SCHEMA:
            raise ProblemError(409, "Earlier package unavailable", "This internal package may contain model names. Export a new public evidence package. The original remains stored.")
        if row.status != 'done' or row.artefact_id is None:
            raise ProblemError(409, "Package not ready", f"The package is {row.status}; its file exists once it is done.")
        report = db.get(ReportRow, row.report_id)
        if report is None or not _public_downloadable(db, report, _analysis(db, row.analysis_id).kind):
            raise ProblemError(409, "Earlier package unavailable", "Generate a new report and public evidence package. The original remains stored.")
        response = _binary(request, _artefact(db, row.artefact_id, row.analysis_id))
        name = package.filename(row, report) if report is not None else f'chipmunk-evidence-{row.id}.zip'
    if response.status_code == 200:
        response.headers['Content-Disposition'] = f'attachment; filename="{name}"'
    return response
