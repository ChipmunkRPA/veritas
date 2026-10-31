"""Repairs of report and package rows at start (BUILD_SPEC N-9; code review 12 and 14).

Since N-9 a report or package row is committed with its job, and its final write with its custody event. Rows written by an
earlier build, or by a process that died at the wrong moment, can lack either:

- `orphans`: a `queued` or `running` row with no active job. A row that never had a job is queued; a row whose job has ended
  (the poison guard, a cancellation) takes that end and is failed.
- `custody`: a `done` row with `audit_event_seq` NULL. When the chain holds its event, the row takes that seq; else the event
  is appended now from the row, marked `recorded_late` (the lint figures and a package's custody anchor are not in the row
  and are left out). A package built from a report without its `report_generated` event fails offline verification.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.audit import chain as audit
from app.core import times
from app.core.logging import get_logger
from app.engine.queue import PRIORITY_INTERACTIVE, JobQueue
from app.reports import package, service
from app.storage.db import AuditEvent, PackageRow, ReportRow, get_session

log = get_logger(__name__)
OPEN = ("queued", "running")


def orphans(queue: JobQueue) -> tuple[int, int]:
    """Queue or fail the report and package rows that are open without an active job. Returns (queued, failed)."""
    queued = failed = 0
    for model, job_type, fail in ((ReportRow, "report", service.mark_failed), (PackageRow, "package", package.mark_failed)):
        active = queue.active_subjects(job_type)
        with get_session() as db:
            pending = [i for i in db.scalars(select(model.id).where(model.status.in_(OPEN)).order_by(model.id)) if i not in active]
        for subject_id in pending:
            ended = queue.latest(subject_id, job_type)
            if ended is None:
                queue.enqueue(job_type, subject_id, priority=PRIORITY_INTERACTIVE)
                queued += 1
            else:
                fail(subject_id, ended.error)
                failed += 1
    if queued or failed:
        log.warning("reports.orphans_repaired", queued=queued, failed=failed)
    return queued, failed


def _report_event(row: ReportRow) -> dict[str, Any]:
    return {"action": "report_generated", "actor_id": row.requested_by, "analysis_id": row.analysis_id, "report_id": row.id,
            "object_ids": [row.artefact_id, row.snapshot_artefact_id], "after_hash": row.sha256,
            "details": {"format": "v2-pdf", "profile": row.profile, "revision": row.revision, "display_id": row.display_id,
                        "supersedes_report_id": row.supersedes_report_id, "redacted": row.redacted,
                        "snapshot_hash": row.snapshot_hash, "template_set": row.template_set,
                        "engine_version": row.engine_version, "page_count": row.page_count, "signature": row.signature,
                        "signing_key_id": row.signing_key_id, "recorded_late": True,
                        "finished_at": times.iso_z(row.finished_at) if row.finished_at is not None else None}}


def _package_event(row: PackageRow) -> dict[str, Any]:
    return {"action": "package_exported", "actor_id": row.created_by, "analysis_id": row.analysis_id, "report_id": row.report_id,
            "object_ids": [row.artefact_id], "after_hash": row.sha256,
            "details": {"package_id": row.id, "kind": row.kind, "include_original": row.include_original, "files": row.file_count,
                        "manifest_sha256": row.manifest_sha256, "signature": row.signature,
                        "signing_key_id": row.signing_key_id, "recorded_late": True,
                        "finished_at": times.iso_z(row.finished_at) if row.finished_at is not None else None}}


def custody() -> tuple[int, int]:
    """Give every done report and package the custody event it lacks. Returns (linked, appended)."""
    linked = appended = 0
    for model, action, event_of in ((ReportRow, "report_generated", _report_event), (PackageRow, "package_exported", _package_event)):
        with get_session() as db:
            pending = list(db.scalars(select(model.id).where(model.status == "done", model.audit_event_seq.is_(None))
                                      .order_by(model.finished_at, model.id)))
        for row_id in pending:
            def unit(model: Any = model, action: str = action, event_of: Any = event_of, row_id: str = row_id) -> str | None:
                with audit.writing() as db:
                    row = db.get(model, row_id)
                    if row is None or row.status != "done" or row.audit_event_seq is not None or row.sha256 is None:
                        return None
                    recorded = db.scalars(select(AuditEvent.seq).where(
                        AuditEvent.action == action, AuditEvent.analysis_id == row.analysis_id,
                        AuditEvent.report_id == (row.id if model is ReportRow else row.report_id),
                        AuditEvent.after_hash == row.sha256).order_by(AuditEvent.seq).limit(1)).first()
                    if recorded is not None:
                        row.audit_event_seq = int(recorded)
                        db.commit()
                        return "linked"
                    audit.commit_with(db, [event_of(row)], before_commit=lambda seqs: setattr(row, "audit_event_seq", seqs[0]))
                    return "appended"

            outcome = audit.repeated(unit)
            linked, appended = linked + (outcome == "linked"), appended + (outcome == "appended")
    if linked or appended:
        log.warning("reports.custody_repaired", linked=linked, appended=appended)
    return linked, appended
