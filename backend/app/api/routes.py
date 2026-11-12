from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import BinaryIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

from app.api import jobs
from app.api.schemas import AnalysisDetail, AnalysisSummary, Health, TextRequest
from app.api.security import CheckedRoute, authorize, make_download_token, rate_limit
from app.audit import chain as audit
from app.cases import access, intake, review
from app.cases.intake import KIND_LIMITS
from app.core.config import settings
from app.core.device import DEVICE
from app.storage import files
from app.storage.db import Analysis, get_session

router = APIRouter(prefix="/api", deprecated=True, dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)  # v1: through v2.1 (API-03)
public = APIRouter(prefix="/api")  # liveness/readiness probes: no auth, no rate limit
CLIENT_SHA256 = r"^[0-9A-Fa-f]{64}$"
PROFILE = r"^(quick|full|deep)$"


def _quick(request: Request, src: BinaryIO, filename: str, kind: str, client_sha256: str | None = None,
           profile: str | None = None) -> Analysis:
    """A quick analysis (CM-01): the upload becomes the one item of a new implicit case (C-2), then the analysis is queued."""
    try:
        with intake.upload_slot():
            outcome = intake.take_in(src, filename, kind, principal_id=request.state.principal.id, analyze=True, profile=profile,
                                     acquisition=intake.Acquisition(method="quick_upload", client_sha256=client_sha256),
                                     job_priority=jobs.PRIORITY_INTERACTIVE)  # N-9: queued with the row; `submit` wakes the worker
    except intake.IntakeError as e:
        raise HTTPException(e.status, e.detail) from e
    assert outcome.analysis is not None
    jobs.submit(outcome.analysis.id)
    return outcome.analysis


@router.post("/analyze/{kind}", response_model=AnalysisSummary, status_code=202)
def analyze(kind: str, request: Request, file: UploadFile = File(...),
            client_sha256: str | None = Form(None, pattern=CLIENT_SHA256, description="SHA-256 the browser computed before upload"),
            profile: str | None = Form(None, pattern=PROFILE, description="Analysis profile; the instance default when absent")):
    """Sync handler (runs in the threadpool). Starlette spools the body; intake hashes it and streams it into the evidence store,
    so the upload is never held in process memory. The analysis gets an implicit case (C-2)."""
    if kind not in KIND_LIMITS:
        raise HTTPException(404, "kind must be image, video, audio or document")
    return _quick(request, file.file, file.filename or f"upload.{kind}", kind, client_sha256, profile).to_summary()


@router.post("/analyze-text", response_model=AnalysisSummary, status_code=202)
def analyze_text(req: TextRequest, request: Request):
    import re

    if len(re.findall(r"\w+", req.text)) + len(re.findall(r"[\u3400-\u9fff]", req.text)) // 2 < 15:
        raise HTTPException(422, "Text is too short to analyse (at least 15 words are required)")
    name = req.title if req.title.lower().endswith((".txt", ".md")) else f"{req.title}.txt"
    return _quick(request, io.BytesIO(req.text.encode("utf-8")), name, "document", profile=req.profile).to_summary()


@router.get("/analyses", response_model=list[AnalysisSummary])
def list_analyses(request: Request, limit: int = Query(50, ge=1, le=500), kind: str | None = None):
    """Newest first. A non-administrator sees the analyses of its cases only (C-2)."""
    from app.api.public_projection import project

    with get_session() as db:
        q = db.query(Analysis).order_by(Analysis.created_at.desc())
        if (visible := access.visible_analyses(request.state.principal)) is not None:
            q = q.filter(visible)
        if kind:
            q = q.filter(Analysis.kind == kind)
        return JSONResponse(project([a.to_summary() for a in q.limit(limit).all()]), headers={'X-Chipmunk-View': 'customer-projection'})


def _get(aid: str) -> Analysis:
    with get_session() as db:
        a = db.get(Analysis, aid)
    if a is None:
        raise HTTPException(404, "Analysis not found")
    return a


@router.get("/analyses/{aid}", response_model=AnalysisDetail)
def get_analysis(aid: str):
    from app.api.public_projection import project

    a = _get(aid)
    return JSONResponse(project({**a.to_summary(), "result": a.result}), headers={'X-Chipmunk-View': 'customer-projection'})


def _verified(aid: str, purpose: str, actor_id: str) -> None:
    """Re-hash the original before a report or an export (CC-01, C-3); a mismatch blocks the action."""
    try:
        audit.verify_original(purpose, actor_id=actor_id, analysis_id=aid)
    except audit.OriginalIntegrityError as e:
        raise HTTPException(500, f"The original of this analysis {e.problem}; the {purpose} was not produced.") from None


@router.delete("/analyses/{aid}", status_code=204)
def delete_analysis(aid: str, request: Request):
    """Delete one analysis with its evidence, review, report and package rows and the stored bytes nothing else references.
    Refused (409) while it is queued or running, when the revision is signed off, under a legal hold and before the case's
    retention date."""
    # D-55: the guard and the removal are the retention rule's; RW-04 keeps a signed revision
    from app import retention

    _get(aid)
    # N-9: decided inside the transaction that deletes, with `deletion_requested` committed in it
    outcome = retention.delete_requested(aid, actor_id=request.state.principal.id)
    if outcome is None:
        raise HTTPException(404, "Analysis not found")
    if isinstance(outcome, str):
        raise HTTPException(409, outcome)
    return Response(status_code=204)


@router.get("/analyses/{aid}/artifacts/{name}")
def artifact(aid: str, name: str):
    from app.api.public_projection import artifact_alias

    _get(aid)
    directory = settings.artifacts_dir / aid
    p = directory / files.safe_filename(name)
    if not p.is_file() and directory.is_dir():
        p = next((entry for entry in directory.iterdir() if entry.is_file() and artifact_alias(entry.name) == name), p)
    if not p.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(p)


VIEW_WINDOW_S = 60.0  # the requests of one view of an original: one `original_viewed` event


@router.get("/analyses/{aid}/source")
def source(aid: str, request: Request):
    """The original served inline for viewing, recorded as `original_viewed` (C-3). Views are not re-hashed; ARCHITECTURE §4.8
    requires the full re-hash before reports, exports and downloads."""
    a = _get(aid)
    p = settings.uploads_dir / aid / files.safe_filename(a.filename)  # N-19: the stored name is the name as received
    if not p.is_file():
        raise HTTPException(404, "Source file not found")
    # N-9: a read. The append comes before the bytes are served, so a failed append serves nothing
    # N-19a (browser QA, "Workflow"): a player reads one file in several requests (the head, the index at the end, a seek),
    # and opening the video workspace wrote three events in one second. The requests of one principal for one original
    # within `VIEW_WINDOW_S` are one view
    actor = request.state.principal.id
    if not audit.recorded_since("original_viewed", actor_id=actor, analysis_id=aid, seconds=VIEW_WINDOW_S):
        audit.record("original_viewed", actor_id=actor, analysis_id=aid, before_hash=a.sha256)
    return FileResponse(p, media_type=a.content_type, filename=files.safe_filename(a.filename))


@router.get("/analyses/{aid}/download-token")
def download_token(aid: str, request: Request):
    """Short-lived token for browser downloads (artifacts, source, report, export) — keeps the API key out of URLs. The token
    acts as the calling principal (C-1). A resource token cannot mint another one, so its 600 s expiry holds (Q-2 F1)."""
    if request.state.principal.auth == "token":
        raise HTTPException(403, "A resource token cannot issue another token")
    _get(aid)
    return {"token": make_download_token(aid, request.state.principal.id), "ttl_seconds": min(settings.download_token_ttl_s, 600)}


def _current_revision(a: Analysis) -> str:
    """Refuse (409) a withdrawn or superseded revision, as the v2 report and package routes do (N-3; SEC-09), and return the
    review status line of the revision for the v1 PDF."""
    from app.reports import snapshot, view

    if a.withdrawn_at is not None:
        raise HTTPException(409, "This revision was withdrawn and should not be relied on; no report or export is produced from it.")
    with get_session() as db:
        if review.superseded(db, a.id):
            raise HTTPException(409, "A newer revision of this analysis exists; take the report or export of the latest revision.")
        return view.review_status_of(snapshot.review_of(db, a)).line


def _v1_report_file(aid: str, line: str) -> Path:
    """One v1 PDF per review status line, so a PDF made before a sign-off is never served after it."""
    from app.reports import view

    base = files.report_path(aid)
    return base if line == view.UNREVIEWED else base.with_name(f"report-{hashlib.sha256(line.encode()).hexdigest()[:12]}.pdf")


def _v1_made_with(aid: str, name: str, number: str) -> bool:
    """The newest `report_generated` event of the v1 file `name` says it was made with `number` (`_v1_number`). A file an
    earlier build made, which printed a fallback score as "% p(AI)", has no such event: it is made again in place, and its
    new event is the one custody verification reads (the newest event of a file counts)."""
    from sqlalchemy import select

    from app.storage.db import AuditEvent

    with get_session() as db:
        events = db.scalars(select(AuditEvent).where(AuditEvent.analysis_id == aid, AuditEvent.action == "report_generated",
                                                     AuditEvent.report_id.is_(None)).order_by(AuditEvent.seq.desc()))
        for e in events:
            named = next(iter(e.object_ids or []), None)
            if (Path(str(named)).name if named else "report.pdf") == name:
                return (e.details or {}).get("number") == number
    return False


def _v1_number(a: Analysis) -> str | None:
    """What the v1 PDF prints for the number of a v2 result that is not a calibrated estimate, as the result page and the v2
    reports say it (D-68; N-10b, code review 3: the v1 PDF of a fallback result printed "77 % p(AI)"). None for a v1 result
    and for a calibrated estimate, which keep the verdict's number."""
    if a.result_schema_version != 2 or not a.result_hash:
        return None
    from app.api.v2_evidence import _result
    from app.reports import view

    with get_session() as db:
        result = _result(db, db.get(Analysis, a.id)) or {}
    n = view.result_number(view.decision_of(result), list(result.get("findings") or []), result.get("condition_profile"))  # N-19h: the stored rules
    if n.kind == "calibrated":
        return None
    if n.kind == "capped":
        return view.NO_ESTIMATE_OUTSIDE
    if n.kind == "none" or not n.text:
        return "No estimate"
    return n.text[0].upper() + n.text[1:]


@router.get("/analyses/{aid}/report.pdf")
def report(aid: str, request: Request):
    """The PDF report, with the review status line of the revision. Generating it re-hashes the original first and records
    `report_generated` (C-3). A withdrawn or superseded revision answers 409."""
    _get(aid)
    raise HTTPException(410, "This older report format is unavailable in the customer view. Generate a new report from the analysis Export menu.")


@router.get("/analyses/{aid}/export.json")
def export_json(aid: str, request: Request):
    """The analysis as JSON. The original is re-hashed first; the export is recorded as `package_exported` (C-3). A withdrawn
    or superseded revision answers 409."""
    a = _get(aid)
    _current_revision(a)
    actor = request.state.principal.id
    _verified(aid, "export", actor)
    from app.api.public_projection import project
    from app.engine.customer_assessment import customer_assessment
    from app.reports import view as report_view

    # Analysis.result is the compatibility/v1 result. The confidence rule must
    # read the frozen v2 decision rather than infer one from those legacy scores.
    stored = a.result or {}
    if getattr(a, "result_schema_version", 1) == 2:
        from app.api.v2_evidence import _result

        with get_session() as db:
            stored = _result(db, db.get(Analysis, a.id)) or {}
    evidence = {}
    if getattr(a, "kind", None) in {"image", "video", "audio", "document"}:
        from app.evidence.service import load_packet_set, unavailable_packet_set
        from app.storage.objects import ObjectStoreError, default_store

        with get_session() as db:
            try:
                packet_set = load_packet_set(db, a, default_store())
            except (ObjectStoreError, ValueError, KeyError, TypeError):
                raise HTTPException(status_code=500, detail="Evidence failed integrity checks; export was not served.") from None
            evidence = {"image_evidence" if a.kind == "image" else "evidence": packet_set or unavailable_packet_set(a)}
    if getattr(a, "kind", None) == "image":
        from app.evidence import image_thumbnail_evidence, image_whole_review

        thumbnail_evidence = image_thumbnail_evidence.from_result(stored)
        if thumbnail_evidence is not None:
            evidence["thumbnail_evidence"] = thumbnail_evidence
        whole_image_review = image_whole_review.from_result(stored)
        if whole_image_review is not None and whole_image_review["source_sha256"] == a.sha256:
            evidence["whole_image_review"] = whole_image_review
    decision = stored.get("decision") or {}
    display_result = {**stored, "kind": getattr(a, "kind", stored.get("kind")), "status": a.status}
    if decision and not decision.get("provenance_record"):
        display_result["stored_record"] = report_view.stored_record(
            decision, list(stored.get("assessments") or []), list(stored.get("signals") or []))
    summary = a.to_summary()
    summary.setdefault('result_hash', getattr(a, 'result_hash', None))
    response = JSONResponse({**project({**summary, "result": a.result}), **evidence,
                             "customer_assessment": customer_assessment(display_result),
                             "projection": {"view": "customer", "result_hash": "hashes the original stored result"}},
                            headers={"Content-Disposition": f'attachment; filename="chipmunk-verify-{aid[:8]}.json"',
                                     'X-Chipmunk-View': 'customer-projection'})
    # N-9: a read. The append comes before the response leaves, so a failed append exports nothing
    audit.record("package_exported", actor_id=actor, analysis_id=aid, after_hash=hashlib.sha256(response.body).hexdigest(),
                 details={"format": "v1-export-json"})
    return response


@public.get("/health", response_model=Health)
async def health():
    """Liveness: this process answers. It asks nothing else: not the database, not the data volume, not a worker (N-13)."""
    return {"status": "ok"}


@public.get("/ready")
async def ready(request: Request, profile: str | None = Query(None, pattern="^(quick|full|deep)$")):
    """Readiness: 200 with `status: ready` when the database can be written, the data directory can be written and its volume
    is above the free-space floor and a worker serves the queue (its heartbeat, when this process runs none); else 503 with
    `not_ready`.

    A caller without credentials is told the `status` only. With credentials the body adds `checks` (each with `ok`, and
    why it failed), `version`, `build_commit`, `queue_depth` and what the serving worker runs on: `device`, `profile`,
    `loaded` models, and per analysis kind `fusion` (`artefact` or `fallback`), `fusers` (version, the start of the sha256,
    review state) and `text_classifier`; with `profile`, what serves under that profile."""
    from app.api import probes
    from app.api.public_projection import project

    code, body = await probes.ready_body(request, profile)
    return JSONResponse(status_code=code, content=project(body),
                        headers={"Cache-Control": "no-store", 'X-Chipmunk-View': 'customer-projection'})


@router.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """The Prometheus text exposition, read from the database at scrape time, so every process of an instance gives the same
    answer (N-13): the queue, the analyses, the worker's heartbeat, the data volume, version and build commit."""
    from app.api import metrics as exposition
    from app.storage.db import get_engine

    return exposition.render(get_engine())


@router.get("/models")
def models():
    from app.engine.registry import get_registry

    available = sum(1 for item in get_registry().decisions() if item.enabled)
    return {
        "device": str(DEVICE),
        "available_checks": available,
    }
