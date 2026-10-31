"""Analysis jobs on the DB job queue (ARCHITECTURE §3.4).

`submit`, `start`, `stop` and `queue_depth` keep their v0.3 signatures, so `routes.py` and `demo.py` do not change.
- `start()` adopts analyses that are queued or running but have no active job (v0.3 rows, pre-queue databases). It
  applies the v0.3 poison guard, then starts the in-process worker unless `VERITAS_RUN_WORKER=false`.
- `cancel()` cancels a queued analysis at once, and a running one cooperatively, via `deadline.check()` between stages.
"""
from __future__ import annotations

import json
import shutil
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import structlog
from sqlalchemy import Connection, insert, select, update
from sqlalchemy.orm import Session

from app.audit import chain as audit
from app.core import deadline
from app.core.config import settings
from app.core.errors import (  # noqa: F401 (re-exported for v1 callers and tests)
    FRIENDLY,
    USER_FACING,
    friendly_error,
)
from app.core.logging import get_logger
from app.engine.boundary import Cancelled
from app.engine.events import finish_analysis
from app.engine.heartbeat import Publisher
from app.engine.queue import PRIORITY_BATCH, PRIORITY_INTERACTIVE, JobQueue, JobView
from app.engine.worker import JobContext, Worker
from app.storage import db, files, space
from app.storage.db import Analysis, AnalysisEvent, get_session, utcnow

log = get_logger(__name__)

CANCELLED_MESSAGE = "Cancelled on request"

_queue: JobQueue | None = None
_worker: Worker | None = None
_heartbeat: Publisher | None = None  # the heartbeat of `_worker` (N-13)
_lock = threading.Lock()


def _sanitize(obj):
    """Replace NaN/Inf with None so the stored JSON is strictly valid for browsers."""
    import math

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _mark(analysis_id: str, status: str, error: str | None) -> None:
    with get_session() as s:
        a = s.get(Analysis, analysis_id)
        if a is not None and a.status != "done":
            a.status, a.finished_at, a.error = status, utcnow(), error
            s.commit()


def _locked(s: Session, analysis_id: str) -> Analysis | None:
    """The analysis row, locked against a concurrent cancellation or recovery projection until the session commits."""
    conn = s.connection()
    if conn.dialect.name == "sqlite":
        conn.exec_driver_sql("BEGIN IMMEDIATE")
    return s.scalar(select(Analysis).where(Analysis.id == analysis_id).with_for_update())


def _run(analysis_id: str, ctx: JobContext | None = None) -> str:
    """Run the v1 pipeline for one analysis. Returns the job outcome: done, failed or cancelled."""
    structlog.contextvars.bind_contextvars(analysis_id=analysis_id)
    kind = "unknown"
    try:
        with get_session() as s:
            a = _locked(s, analysis_id)
            if a is None:
                return "done"
            if a.status == "cancelled":
                return "cancelled"
            a.status, a.started_at = "running", utcnow()
            a.attempts = (a.attempts or 0) + 1
            s.commit()
            kind, path = a.kind, Path(settings.uploads_dir) / a.id / files.safe_filename(a.filename)
        deadline.set_deadline(settings.job_timeout_s)
        if ctx is not None:
            deadline.set_cancel_event(ctx.cancel_event)
        art = files.artifact_dir(analysis_id)
        if kind == "image":
            from app.detectors.image.pipeline import analyze_image

            result = analyze_image(path, art)
        elif kind == "video":
            from app.detectors.video.pipeline import analyze_video

            result = analyze_video(path, art)
        elif kind == "document":
            from app.detectors.text.pipeline import analyze_document

            result = analyze_document(path, path.suffix, art)
        else:
            raise ValueError(f"unknown kind {kind}")
        result = _sanitize(result)
        with get_session() as s:
            a = s.get(Analysis, analysis_id)
            if a is None:  # deleted while running
                log.info("analysis.deleted_while_running")
                shutil.rmtree(art, ignore_errors=True)
                return "done"
            summary = a.to_summary()
        report_path = None
        try:  # build the report BEFORE publishing `done` so clients never see a half-written PDF
            from app.reports.builder import build_report
            from app.reports.view import UNREVIEWED

            audit.verify_original("report", actor_id=audit.SYSTEM_ACTOR, analysis_id=analysis_id)  # CC-01: a mismatch blocks it
            # a finished analysis has no review yet (N-3: the v1 PDF prints the status line)
            report_path = str(build_report(summary, result, art, files.report_path(analysis_id), review_line=UNREVIEWED))
            audit.record_report(analysis_id, Path(report_path), actor_id=audit.SYSTEM_ACTOR)
        except Exception as e:  # report failure must not fail the analysis; it is rebuilt on demand
            log.error("report.failed", error=str(e), tb=traceback.format_exc()[-800:])
        with get_session() as s:
            a = _locked(s, analysis_id)
            if a is None:
                return "done"
            if a.status != "running":  # recovery finalized it (cancelled or failed) while a model or report call stalled
                log.info("analysis.late_result_discarded", status=a.status)
                return a.status
            if _lease_lost(ctx):  # N-8: the new lease owner publishes; this run writes no result
                log.warning("analysis.lease_lost")
                return LEASE_LOST
            a.result_json = json.dumps(result, allow_nan=False)
            a.probability_ai = result["verdict"]["probability_ai"]
            a.band = result["verdict"]["band"]
            a.report_path = report_path
            a.status, a.finished_at, a.error = "done", utcnow(), None
            s.commit()
        log.info("analysis.done", kind=kind, p=result["verdict"]["probability_ai"], seconds=result.get("timings", {}).get("total"))
        return "done"
    except deadline.JobCancelled:
        if _lease_lost(ctx):  # N-8 (K-3): a lost lease is not a cancel request; the new owner decides the outcome
            log.warning("analysis.lease_lost")
            return LEASE_LOST
        log.info("analysis.cancelled")
        _mark(analysis_id, "cancelled", CANCELLED_MESSAGE)
        return "cancelled"
    except Exception as e:
        if _lease_lost(ctx):
            log.warning("analysis.lease_lost", error=str(e))
            return LEASE_LOST
        log.error("analysis.failed", error=str(e), tb=traceback.format_exc()[-1200:])  # full detail stays in logs only
        _mark(analysis_id, "failed", friendly_error(e))
        return "failed"
    finally:
        deadline.clear()
        structlog.contextvars.clear_contextvars()


LEASE_LOST = "lease_lost"  # the outcome of a run whose job lease was lost (`RunOutcome.status`); the worker records nothing


def _lease_lost(ctx: JobContext | None) -> bool:
    return bool(getattr(ctx, "lease_lost", False))


def _leased(ctx: JobContext, run: Callable[[], str]) -> Callable[[], str]:
    """The run of a job handler: an error raised after the lease was lost is the lease loss, not a failure of the analysis."""
    def wrapped() -> str:
        try:
            return run()
        except Exception as e:
            if not _lease_lost(ctx):
                raise
            log.warning("jobs.error_after_lease_loss", analysis_id=ctx.job.subject_id, error=f"{type(e).__name__}: {e}"[:300])
            return LEASE_LOST
    return wrapped


def _audited(analysis_id: str, refusion: bool, run: Callable[[], str]) -> str:
    """`analysis_started` and `analysis_finished` around one run (ARCHITECTURE §4.8). The actor is the submitter, else
    `system`. An analysis that is cancelled or gone before it starts records nothing. A run that lost its job lease records no
    `analysis_finished`: it did not finish the analysis, and the new lease owner's run records its own pair (N-8)."""
    with get_session() as s:
        a = s.get(Analysis, analysis_id)
        submitted_by, status = (a.submitted_by, a.status) if a is not None else (None, None)
    if status in (None, "cancelled"):
        return run()
    actor, details = submitted_by or audit.SYSTEM_ACTOR, {"refusion": True} if refusion else {}
    # N-9, N-10a (code re-verification R2-04): `analysis_started` and `analysis_finished` are the worker's own record around
    # a run, appended outside any transaction; the result's record is `manifest_recorded`, which is committed with the
    # result (`stages/common.publish`).
    # - `analysis_started` that cannot be appended raises before anything ran: the job fails and is not retried, and the
    #   handler ends the analysis as failed (`_fail_unfinished`), so it does not stay queued.
    # - `analysis_finished` that cannot be appended is logged and lost: the run has ended, the job keeps its outcome, and
    #   the completion events of the handler are still sent.
    audit.record("analysis_started", actor_id=actor, analysis_id=analysis_id, details=details)
    outcome = "failed"
    try:
        outcome = run()
        return outcome
    finally:
        if outcome != LEASE_LOST:
            try:
                with get_session() as s:
                    a = s.get(Analysis, analysis_id)
                    final = a.status if a is not None else "deleted"
                audit.record("analysis_finished", actor_id=actor, analysis_id=analysis_id, details={**details, "status": final})
            except Exception as e:
                log.error("jobs.analysis_finished_not_recorded", analysis_id=analysis_id, error=f"{type(e).__name__}: {e}"[:300])


def _fail_unfinished(analysis_id: str, error: BaseException) -> None:
    """A handler that raised outside the runner (N-10a; code re-verification R2-04): the append of `analysis_started`, stored
    pins that do not resolve, a kind without a pipeline. The worker fails the job, which is final, so the analysis ends
    failed here if it is still queued or running: the projection of a job the queue failed (`_on_terminal`). A revision
    that ends failed no longer supersedes the published one (N-8). A published result stays done (`finish_analysis`).
    A problem here is logged; `adopt_orphans` fails the analysis at the next start."""
    try:
        with get_session() as s:
            a = s.get(Analysis, analysis_id)
            unfinished = a is not None and a.status in ("queued", "running")
        if unfinished:
            finish_analysis(get_session, analysis_id, "failed", {"reason": friendly_error(error)})
            log.warning("jobs.failed_outside_runner", analysis_id=analysis_id, error=f"{type(error).__name__}: {error}"[:300])
    except Exception as e:
        log.error("jobs.fail_projection_failed", analysis_id=analysis_id, error=f"{type(e).__name__}: {e}"[:300])


def _finished_or_locked(analysis_id: str) -> bool:
    """D-47: a job requeued after a crash between publish and completion never re-runs a done or signed-off analysis."""
    with get_session() as s:
        a = s.get(Analysis, analysis_id)
        skip = a is not None and (a.status == "done" or a.locked_at is not None)
    if skip:
        log.info("jobs.skipped_finished", analysis_id=analysis_id)
    return skip


def _sweep_tmp(analysis_id: str | None = None) -> None:
    """D-48: copies of originals under `data/tmp`: the job's own when it ends, or at worker start what a crashed process left."""
    try:
        from app.engine.stages.common import sweep_tmp

        sweep_tmp(only=analysis_id)
    except Exception as e:  # noqa: BLE001 - hygiene never fails a job
        log.warning("jobs.tmp_sweep_failed", analysis_id=analysis_id, error=str(e))


def _handle_analysis(ctx: JobContext) -> str:
    if _finished_or_locked(ctx.job.subject_id):
        # N-8: the earlier owner of this job published and then lost its lease (or died) before the completion events, which
        # it does not emit without the lease: this owner emits them. `batch.completed` is stamped once; `analysis.completed`
        # is delivered again when the earlier owner died after emitting it (at least once, with a new event id)
        _finished(ctx.job.subject_id)
        return "done"
    try:
        outcome = _audited(ctx.job.subject_id, False, _leased(ctx, lambda: _run_analysis_job(ctx)))
    except (deadline.JobCancelled, Cancelled):
        raise
    except Exception as e:
        if not _lease_lost(ctx):
            _fail_unfinished(ctx.job.subject_id, e)
        raise
    finally:
        if not _lease_lost(ctx):  # K-3: the new lease owner may be using the temporary files
            _sweep_tmp(ctx.job.subject_id)
            # N-10a: in the `finally`, so a handler that raised still sends `analysis.failed` and completes its batch
            _finished(ctx.job.subject_id)
    return outcome


def _finished(analysis_id: str) -> None:
    """The webhook events of a finished analysis and of its batch (API-02). A notification problem never fails the analysis job."""
    try:
        with get_session() as s:
            a = s.get(Analysis, analysis_id)
            status, case_id, kind, code = (a.status, a.case_id, a.kind, a.decision_code) if a is not None else (None, None, None, None)
        if status in ("done", "failed"):
            emit("analysis.completed" if status == "done" else "analysis.failed",
                 {"analysis_id": analysis_id, "case_id": case_id, "kind": kind, "status": status, "decision_code": code})
        from app.batches import service as batches

        if (batch_id := batches.batch_of(analysis_id)) is not None and (data := batches.completed_event(batch_id)) is not None:
            emit("batch.completed", data)
    except Exception as e:  # noqa: BLE001
        log.error("jobs.notify_failed", analysis_id=analysis_id, error=str(e))


def emit(event: str, data: dict) -> None:
    """Queue the webhook deliveries of an event (API-02); nothing while webhooks are off."""
    from app.webhooks import service as webhooks

    for delivery_id in webhooks.record_event(event, data):
        get_queue().enqueue("webhook_delivery", delivery_id, priority=PRIORITY_BATCH)
    if _worker is not None:
        _worker.wake()


def _run_analysis_job(ctx: JobContext) -> str:
    analysis_id = ctx.job.subject_id
    if settings.pipeline == "v2":
        with get_session() as s:
            a = s.get(Analysis, analysis_id)
            kind, status, profile, pins = (a.kind, a.status, a.profile, dict(a.rerun_pins or {})) if a is not None else (None, None, None, {})
        if status == "cancelled":
            return "cancelled"
        from app.engine.pipelines import PIPELINES

        if kind in PIPELINES:
            from app.engine.models import get_model_manager
            from app.engine.pipelines import run_analysis

            # the profile requested at intake (C-2 `?profile=`), else the instance default; a re-run revision (CC-03) reuses the
            # fuser artefact or the fallback, and the policy thresholds version, of the analysis it re-runs
            return run_analysis(analysis_id, deadline=ctx.deadline.nested(settings.job_timeout_s), profile=profile,
                                models=get_model_manager(), fuser_pin=pins.get("fuser_sha256"),
                                force_fallback=pins.get("fusion_mode") == "fallback",
                                policy_thresholds_id=pins.get("policy_thresholds_id")).status
    return _run(analysis_id, ctx)


def _handle_refusion(ctx: JobContext) -> str:
    """Run a revision created by `app.engine.refusion.create_revision` with its pinned artefact (ARCHITECTURE §4.4)."""
    if _finished_or_locked(ctx.job.subject_id):
        return "done"
    try:
        return _audited(ctx.job.subject_id, True, _leased(ctx, lambda: _run_refusion_job(ctx)))
    except (deadline.JobCancelled, Cancelled):
        raise
    except Exception as e:
        if not _lease_lost(ctx):
            _fail_unfinished(ctx.job.subject_id, e)  # N-10a: the revision does not stay queued over the published one
        raise
    finally:
        if not _lease_lost(ctx):  # K-3: the new lease owner may be using the temporary files
            _sweep_tmp(ctx.job.subject_id)


def _run_refusion_job(ctx: JobContext) -> str:
    analysis_id = ctx.job.subject_id
    with get_session() as s:
        a = s.get(Analysis, analysis_id)
        if a is None:
            return "done"
        if a.status == "cancelled":
            return "cancelled"
        profile, pin = a.profile, a.fuser_sha256
    from app.engine.pipelines import run_analysis

    return run_analysis(analysis_id, deadline=ctx.deadline.nested(settings.job_timeout_s), profile=profile, refusion=True,
                        fuser_pin=pin).status


def _handle_report(ctx: JobContext) -> str:
    """Render a requested report from its frozen snapshot (ARCHITECTURE §6.3; R-1)."""
    from app.reports import service

    return service.run(ctx.job.subject_id, timeout_s=settings.job_timeout_s)  # N-9: a hung render does not hold the worker


def _handle_package(ctx: JobContext) -> str:
    """Build a requested evidence package (ARCHITECTURE §6.5; R-4)."""
    from app.reports import package

    return package.run(ctx.job.subject_id, timeout_s=settings.job_timeout_s)


def _handle_batch_expand(ctx: JobContext) -> str:
    """Take in the staged items of a batch and queue their analyses at batch priority (API-01; X-1)."""
    from app.batches import service as batches

    try:
        outcome = batches.expand(ctx.job.subject_id, submit=submit_batch_item)
    except Exception as e:  # the job ends here and is not retried: the batch must not read `expanding` for ever
        try:
            batches.mark_failed(ctx.job.subject_id, f"The batch could not be expanded: an internal error stopped it ({type(e).__name__}). "
                                                    "The server log has the detail under this batch id.")
        except Exception as marking:  # noqa: BLE001 - the database is away: the first error is the one to report
            log.error("batches.mark_failed_failed", batch_id=ctx.job.subject_id, error=str(marking)[:500])
        raise
    if (data := batches.completed_event(ctx.job.subject_id)) is not None:  # every item was rejected
        emit("batch.completed", data)
    return outcome


def _handle_webhook_delivery(ctx: JobContext) -> str:
    """One webhook delivery attempt; a failed attempt queues the next one after its backoff (API-02; X-1)."""
    from app.webhooks import service as webhooks

    return webhooks.deliver(ctx.job.subject_id, retry=lambda delivery_id, at: get_queue().enqueue(
        "webhook_delivery", delivery_id, priority=PRIORITY_BATCH, not_before=at))


def _on_terminal(job: JobView) -> None:
    """A lease expired and the job was failed by the poison guard, or cancelled: reflect it on the analysis, report, package or
    batch."""
    if job.type in ("analysis", "refusion"):
        finish_analysis(get_session, job.subject_id, job.status, {"reason": job.error})
        if job.type == "analysis":
            _finished(job.subject_id)
    elif job.type == "batch_expand":
        from app.batches import service as batches

        batches.mark_failed(job.subject_id, job.error)
    elif job.type == "report":
        from app.reports import service

        service.mark_failed(job.subject_id, job.error)
    elif job.type == "package":
        from app.reports import package

        package.mark_failed(job.subject_id, job.error)


def _on_released(job: JobView) -> None:
    """A worker that was stopped gave the job back to the queue (N-17a; ops re-review O2-05): its analysis waits again, and
    says so. It read `running`, with the stage it had reached, until the next worker claimed the job, which after a stop
    for an upgrade may be minutes. Only a row that still reads `running` and has no result is touched; the stage rows stay
    for the run that follows, as after a lost lease."""
    if job.type not in ("analysis", "refusion"):
        return
    with get_session() as s:
        s.execute(update(Analysis).where(Analysis.id == job.subject_id, Analysis.status == "running",
                                         Analysis.locked_at.is_(None), Analysis.result_hash.is_(None))
                  .values(status="queued").execution_options(synchronize_session=False))
        s.commit()


def get_queue() -> JobQueue:
    global _queue
    engine = db.get_engine()
    with _lock:
        if _queue is None or _queue.engine is not engine:
            _queue = JobQueue(engine, lease_s=settings.job_lease_s, max_attempts=settings.max_requeue_attempts)
        return _queue


def _storage_hold() -> str | None:
    """N-13 (OPS-19): below the free-space floor the worker claims nothing, so a queued job is not failed by a full volume;
    it runs once space is freed."""
    shortfall = space.low()
    return None if shortfall is None else f"storage: {shortfall}"


def build_worker() -> Worker:
    return Worker(get_queue(), {"analysis": _handle_analysis, "refusion": _handle_refusion, "report": _handle_report,
                                "package": _handle_package, "batch_expand": _handle_batch_expand,
                                "webhook_delivery": _handle_webhook_delivery},
                  threads=settings.worker_threads,
                  poll_s=settings.job_poll_s,
                  heartbeat_s=settings.job_heartbeat_s, on_terminal=_on_terminal, hold=_storage_hold,
                  on_released=_on_released)


def _cancelled_job(queue: JobQueue, analysis_id: str) -> bool:
    """The latest analysis/refusion job of an analysis was cancelled, or had a cancel requested (its terminal projection may have
    failed, e.g. in the expiry callback)."""
    latest = max((j for t in ("analysis", "refusion") if (j := queue.latest(analysis_id, t)) is not None),
                 key=lambda j: j.created_at, default=None)
    return latest is not None and (latest.status == "cancelled" or latest.cancel_requested)


def _failed_job(queue: JobQueue, analysis_id: str) -> JobView | None:
    """The latest analysis/refusion job of an analysis, when it failed: nothing will run the analysis again, and its
    projection on the analysis did not happen (the handler raised before N-10a, or `_fail_unfinished` could not write)."""
    latest = max((j for t in ("analysis", "refusion") if (j := queue.latest(analysis_id, t)) is not None),
                 key=lambda j: j.created_at, default=None)
    return latest if latest is not None and latest.status == "failed" else None


def _no_job(queue: JobQueue, analysis_id: str) -> bool:
    """The analysis has no analysis or refusion job row, in any state."""
    return all(queue.latest(analysis_id, t) is None for t in ("analysis", "refusion"))


def adopt_orphans(queue: JobQueue) -> int:
    """Queue analyses that are queued or running without an active job, with the v0.3 poison guard."""
    active = queue.active_subjects("analysis") | queue.active_subjects("refusion")
    adopt: list[tuple[str, str]] = []
    cancelled: list[str] = []
    failed: list[tuple[str, str | None]] = []
    with get_session() as s:
        for a in s.query(Analysis).filter(Analysis.status.in_(["queued", "running"])).all():
            if a.id in active:
                continue
            if _cancelled_job(queue, a.id):
                # a cancellation whose terminal projection failed (e.g. the expiry callback) is honoured, never re-run
                cancelled.append(a.id)
            elif (job := _failed_job(queue, a.id)) is not None:
                # N-10a (code re-verification R2-04): the job failed outside the runner and nothing told the analysis. It is
                # not queued again: `attempts` counts runs, so a job that fails before its run would be adopted at every start
                failed.append((a.id, job.error))
            elif (a.attempts or 0) >= settings.max_requeue_attempts:
                a.status, a.finished_at = "failed", utcnow()
                a.error = f"Processing was interrupted {a.attempts} times; the input is not re-queued again."
            else:
                a.status = "queued"
                # a refusion revision (pinned artefact, superseding a published analysis) must never run detectors
                adopt.append(("refusion" if a.supersedes_id and a.fuser_sha256 else "analysis", a.id))
        s.commit()
    for aid in cancelled:
        finish_analysis(get_session, aid, "cancelled", {"reason": CANCELLED_MESSAGE})
        _finished(aid)
    for aid, error in failed:
        finish_analysis(get_session, aid, "failed", {"reason": error})
        _finished(aid)
    for job_type, aid in adopt:
        queue.enqueue(job_type, aid, priority=PRIORITY_INTERACTIVE)
    if adopt:
        log.info("jobs.requeued", n=len(adopt))
    return len(adopt)


def publish_heartbeat(worker: Worker, *, in_process: bool) -> Publisher:
    """Start the heartbeat of a worker that serves the queue (N-13; `app.engine.heartbeat`): the API reads its row for the
    metrics and for readiness. The caller stops what is returned before the worker."""
    from app.engine import capacity

    capacity.at_start()  # N-14a: which detectors of the default profile the model memory budget cannot run; the beats carry it
    publisher = Publisher(worker, db.get_engine(), in_process=in_process)
    publisher.start()
    return publisher


def start() -> None:
    global _worker, _heartbeat
    stop()
    queue = get_queue()
    adopt_orphans(queue)
    _repair_exports(queue)
    _repair_custody()
    _sweep_tmp()
    if settings.run_worker:
        _worker = build_worker()
        _worker.start()
        _heartbeat = publish_heartbeat(_worker, in_process=True)


def _repair_custody() -> None:
    """N-9: sign-offs stored before their custody event was part of their transaction get the event they lack. A problem here
    is logged and does not stop the start: the rows stay as they are and the next start tries again."""
    try:
        from app.cases import review

        linked, appended = review.repair_signoff_events()
        if linked or appended:
            log.warning("jobs.signoff_events_repaired", linked=linked, appended=appended)
    except Exception as e:  # noqa: BLE001
        log.error("jobs.custody_repair_failed", error=f"{type(e).__name__}: {e}"[:300])
    try:  # done reports and packages without their event (code review 14)
        from app.reports import repair

        repair.custody()
    except Exception as e:  # noqa: BLE001
        log.error("jobs.export_custody_repair_failed", error=f"{type(e).__name__}: {e}"[:300])


def _repair_exports(queue: JobQueue) -> None:
    """N-9: report and package rows that are queued or running without an active job are queued or failed. A problem here is
    logged and does not stop the start."""
    try:
        from app.reports import repair

        repair.orphans(queue)
    except Exception as e:  # noqa: BLE001
        log.error("jobs.export_repair_failed", error=f"{type(e).__name__}: {e}"[:300])


def stop() -> None:
    global _worker, _heartbeat
    if _heartbeat is not None:  # first: a worker that stops claiming is no worker, and no beat follows the database's close
        _heartbeat.stop()
        _heartbeat = None
    if _worker is not None:
        _worker.stop()
        _worker = None


def submit(analysis_id: str) -> None:
    get_queue().enqueue("analysis", analysis_id, priority=PRIORITY_INTERACTIVE)
    if _worker is not None:
        _worker.wake()


def submit_in(session: Session, analysis_id: str, type: str = "analysis") -> str:
    """Queue the job of a revision in the transaction that creates the revision (N-9; code review 12); returns the job id.
    The caller calls `wake` after its commit."""
    return get_queue().enqueue_in(session.connection(), type, analysis_id, priority=PRIORITY_INTERACTIVE)


def wake() -> None:
    if _worker is not None:
        _worker.wake()


def submit_refusion(analysis_id: str) -> str:
    """Queue the `refusion` job for a revision created by `app.engine.refusion.create_revision`; returns the job id."""
    job = get_queue().enqueue("refusion", analysis_id, priority=PRIORITY_INTERACTIVE)
    if _worker is not None:
        _worker.wake()
    return job.id


def submit_report_in(session: Session, report_id: str) -> str:
    """Queue the `report` job in the transaction that adds the report row (`app.reports.service.prepare`; N-9, code review
    14): no queued report is left without a job. Returns the job id; the caller calls `wake` after its commit."""
    return get_queue().enqueue_in(session.connection(), "report", report_id, priority=PRIORITY_INTERACTIVE)


def submit_package_in(session: Session, package_id: str) -> str:
    """Queue the `package` job in the transaction that adds the package row (`app.reports.package.prepare`)."""
    return get_queue().enqueue_in(session.connection(), "package", package_id, priority=PRIORITY_INTERACTIVE)


def submit_batch(batch_id: str) -> str:
    """Queue the `batch_expand` job of a batch created by `app.batches.service`; returns the job id."""
    job = get_queue().enqueue("batch_expand", batch_id, priority=PRIORITY_INTERACTIVE)
    if _worker is not None:
        _worker.wake()
    return job.id


def submit_batch_item(analysis_id: str) -> None:
    """Queue a batch item's analysis behind interactive work (§3.4: batch priority)."""
    get_queue().enqueue("analysis", analysis_id, priority=PRIORITY_BATCH)
    if _worker is not None:
        _worker.wake()


def cancel(analysis_id: str) -> str | None:
    """Request cancellation. Returns "cancelled" (it was queued), "leased" (pending cooperative cancel) or None."""
    job = get_queue().request_cancel(subject_id=analysis_id, type="analysis")
    if job is None:
        return None
    if job.status == "cancelled":
        _mark(analysis_id, "cancelled", CANCELLED_MESSAGE)
    return job.status


class CancellationRefused(RuntimeError):
    def __init__(self, detail: str, status_code: int = 409):
        self.status_code = status_code
        super().__init__(detail)


@dataclass(frozen=True)
class AnalysisCancellation:
    analysis_id: str
    status: str
    cancel_requested: bool
    already_cancelled: bool = False


def request_analysis_cancel(analysis_id: str, *, actor_id: str, note: str | None = None) -> AnalysisCancellation:
    """Retain evidence; cancel queued work atomically or request cooperative cancellation.

    Publication wins once it has stored a result, even before the runner marks the analysis done. Queued work
    has no worker to emit its terminal event; that event and the status share the queue transaction. The named
    custody request is recorded before mutation; an audit failure leaves the work unchanged. An accepted
    request can still lose a race to publication, so its audit entry records intent only.
    """
    def current() -> AnalysisCancellation:
        with get_session() as session:
            a = session.get(Analysis, analysis_id)
            if a is None:
                raise CancellationRefused("No analysis exists with this id.", 404)
            if a.status == "cancelled":
                return AnalysisCancellation(analysis_id, "cancelled", False, True)
            if a.status not in ("queued", "running") or a.locked_at is not None or a.result_hash:
                raise CancellationRefused("This analysis is finished, published or locked and cannot be cancelled.")
            return AnalysisCancellation(analysis_id, a.status, False)

    initial = current()
    if initial.already_cancelled:
        return initial
    # This append holds no queue/analysis lock: audit.record owns its own transaction. Recording intent first
    # is fail-closed without lock inversion; concurrent/repeated attempts may legitimately leave multiple requests.
    audit.record("analysis_cancel_requested", actor_id=actor_id, analysis_id=analysis_id,
                 details={"intent": "request_cooperative_cancellation", "observed_status": initial.status,
                          **({"note": note} if note else {})})  # N-19a: the note of who cancels, shown on the analysis
    status = initial.status

    def project(conn: Connection, job: JobView) -> None:
        nonlocal status
        a = conn.execute(select(Analysis.__table__).where(Analysis.id == analysis_id).with_for_update()).mappings().first()
        if a is None:
            raise CancellationRefused("No analysis exists with this id.", 404)
        if a["status"] not in ("queued", "running") or a["locked_at"] is not None or a["result_hash"]:
            raise CancellationRefused("This analysis is finished, published or locked and cannot be cancelled.")
        status = a["status"]
        if job.status == "queued":
            now = utcnow()
            conn.execute(insert(AnalysisEvent).values(analysis_id=analysis_id, ts=now, type="cancelled",
                                                     payload={"reason": CANCELLED_MESSAGE}))
            conn.execute(update(Analysis).where(Analysis.id == analysis_id).values(
                status="cancelled", finished_at=now, error=CANCELLED_MESSAGE, current_stage=None))
            status = "cancelled"

    try:
        job = get_queue().request_cancel(subject_id=analysis_id, types=("analysis", "refusion"), on_request=project)
    except ValueError as exc:
        raise CancellationRefused(str(exc)) from exc
    if job is None:
        fresh = current()  # another request may have cancelled it while this one waited for the queue lock
        if fresh.already_cancelled:
            return fresh
        if _cancelled_job(get_queue(), analysis_id) or (fresh.status == "queued" and _no_job(get_queue(), analysis_id)):
            # K-3: no job is active and the latest one was cancelled (expiry cancellation whose projection failed): finish the
            # analysis as cancelled, the `adopt_orphans` rule, instead of refusing until the next API start. A result committed
            # meanwhile still wins.
            # N-9 (code review 12): the same for a queued analysis that has no job row at all (its submission stopped between
            # the row and the job): nothing will run it before the next start adopts it, so the request finishes it. A job
            # queued a moment later finds the analysis cancelled and does not run.
            if finish_analysis(get_session, analysis_id, "cancelled", {"reason": CANCELLED_MESSAGE}) == "cancelled":
                _finished(analysis_id)
                return AnalysisCancellation(analysis_id, "cancelled", False, True)
            raise CancellationRefused("This analysis is finished, published or locked and cannot be cancelled.")
        raise CancellationRefused("No active analysis job is available to cancel; refresh its status and try again.")
    if job.status == "cancelled":
        _finished(analysis_id)  # queued work never reaches the worker's batch completion callback
    return AnalysisCancellation(analysis_id, status, job.status == "leased")


def queue_depth() -> int:
    return get_queue().depth()
