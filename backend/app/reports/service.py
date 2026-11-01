"""Report requests and the `report` job (ARCHITECTURE §4.7 `reports`, §6.3–6.4; PRD RP-03, RP-06; D-10, D-25; BUILD_SPEC R-1).

- **Request** (`prepare`). Numbers the revision per (analysis, profile), citing the latest earlier revision as superseded;
  allocates `VR-<year>-<6 digits>`; freezes the snapshot as a `stage_output` artefact; adds the row as `queued`. The caller
  queues the job in the same transaction and commits it with the `redaction` event of a redacted report (N-9): the row, its
  job and its event are stored together or not at all. Ids come from a counter per year that a delete does not lower
  (`next_report_id`, N-9a), one request at a time: the caller holds the revision's row lock, which is the database's write
  lock on SQLite, and Postgres takes an advisory lock.
- **Job** (`run`). Re-hashes the original first (CC-01: a mismatch fails the report and records `hash_mismatch`), reads the
  stored snapshot back (re-hashed), renders it within the job's time limit, signs the PDF bytes with the Ed25519 instance
  key (R-4, D-11: detached, stored in `reports.signature`), and then, in one transaction, stores the PDF as a `report`
  artefact, marks the row `done` and records `report_generated` with the PDF's SHA-256, the signature and key id, the
  snapshot hash and the engine version. That final write is conditional (N-9; code review 14): a row that is already done,
  that the queue failed, or that is gone is left as it is, so a second worker that finishes the same job writes nothing.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.audit import chain as audit
from app.cases import review
from app.core import deadline
from app.core.canonical import canonical_hash
from app.core.config import settings
from app.core.errors import friendly_error
from app.core.logging import get_logger
from app.reports import engine, signing, snapshot
from app.storage.artefacts import ArtefactNotFound, get_artefact, register_artefact
from app.storage.db import Analysis, AuditEvent, ReportCounterRow, ReportRow, get_session, utcnow
from app.storage.objects import ObjectNotFound, ObjectStore, default_store

log = get_logger(__name__)
OPEN = ("queued", "running")  # the states the final write of a job may change
ID_LOCK = int.from_bytes(hashlib.sha256(b"veritas.reports.id").digest()[:8], "big", signed=True)  # Postgres advisory lock
WITHHELD_FIELDS = ("filename", "case_reference", "source_fields", "metadata_findings")  # newsroom (D-10, RP-03 minimum)


def _number(report_id: object, prefix: str) -> int:
    text = str(report_id or "")
    return int(text.removeprefix(prefix)) if text.startswith(prefix) and text.removeprefix(prefix).isdigit() else 0


def _highest_named(db: Session, prefix: str) -> int:
    """The highest report number of a year that a report row or the custody log names: where a year's counter starts on an
    instance that gave ids before the counter existed (N-9a). Read once per year."""
    like = f"{prefix}%"
    named = [db.scalar(select(func.max(ReportRow.id)).where(ReportRow.id.like(like))),
             db.scalar(select(func.max(AuditEvent.report_id)).where(AuditEvent.report_id.like(like)))]
    for details in db.scalars(select(AuditEvent.details).where(AuditEvent.action == "retention_purge")):
        named.extend(report.get("id") for report in (details or {}).get("reports") or [] if isinstance(report, dict))
    return max((_number(report_id, prefix) for report_id in named), default=0)


def next_report_id(db: Session, year: int) -> str:
    """The next `VR-<year>-<6 digits>`, from a counter that a delete does not lower, so an id is given once (N-9a; Postgres
    QA X-3). Allocated under a lock that lasts until the caller's transaction ends: an advisory lock on Postgres; on SQLite
    the counter's update takes the database's write lock, which a report request already holds (`review.hold_revision`).
    A transaction that is rolled back gives its number back."""
    prefix = f"VR-{year}-"
    if db.get_bind().dialect.name == "postgresql":
        db.execute(select(func.pg_advisory_xact_lock(ID_LOCK)))
    counter = db.scalar(select(ReportCounterRow).where(ReportCounterRow.year == year).with_for_update()
                        .execution_options(populate_existing=True))
    if counter is None:
        counter = ReportCounterRow(year=year, last=_highest_named(db, prefix))
        db.add(counter)
    counter.last += 1
    db.flush()
    return f"{prefix}{counter.last:06d}"


def latest(db: Session, analysis_id: str, profile: str) -> ReportRow | None:
    return db.scalars(select(ReportRow).where(ReportRow.analysis_id == analysis_id, ReportRow.profile == profile)
                      .order_by(ReportRow.revision.desc()).limit(1)).first()


# N-19 (browser QA P1-8): what a report's snapshot holds of the report itself and of the custody log, which grows with every
# report made. Two snapshots that agree in everything else describe the same state of the record
OWN_FIELDS = frozenset({"report", "custody", "custody_verification", "revisions", "actor_names"})
REUSE_CANDIDATES = 5


def record_state(frozen: snapshot.ReportSnapshot) -> str:
    """The hash of what a snapshot says of the record: the analysis, the file and its intake, the case, the result, the review
    with its conclusion and sign-offs, the annotations and what is printed of the exhibits."""
    return canonical_hash({k: v for k, v in frozen.model_dump(mode="json").items() if k not in OWN_FIELDS})


def _chain_valid(frozen: snapshot.ReportSnapshot) -> bool:
    """Whether the walk a snapshot records found the whole custody chain intact."""
    return bool((frozen.custody_verification or {}).get("valid"))


def current(db: Session, analysis: Analysis, result: dict[str, Any], *, profile: str, redacted: bool,
            store: ObjectStore | None = None, verified: audit.Verification | None = None) -> ReportRow | None:
    """The newest report of this profile that is not failed and whose snapshot holds the record as it stands now (N-19;
    browser QA P1-8: every bundle request minted a further forensic report, R-…-05 for a draft bundle and -06 for the issued
    one). None when the record changed since: a note, a status, the conclusion, a sign-off, the case, the template set."""
    store = store or default_store()
    review_now = snapshot.review_of(db, analysis)
    rows = list(db.scalars(select(ReportRow).where(
        ReportRow.analysis_id == analysis.id, ReportRow.profile == profile, ReportRow.redacted.is_(redacted),
        ReportRow.status != "failed", ReportRow.template_set == engine.TEMPLATE_SET, ReportRow.review_state == review_now.state,
        ReportRow.issuable.is_(bool(review_now.issuable))).order_by(ReportRow.revision.desc()).limit(REUSE_CANDIDATES)))
    if not rows:
        return None
    now = snapshot.build(db, analysis, result, report_id="", revision=0, profile=profile, redacted=redacted, supersedes_report_id=None,
                         revision_reason=None, requested_at=utcnow(), requested_by=rows[0].requested_by,
                         template_set=engine.TEMPLATE_SET, veritas_version=settings.version, verified=verified, store=store)
    # N-19f (code review R4-02): the walk of the custody log is left out of `record_state` (the log grows with every report),
    # so a request got an earlier report that says "intact" after its own walk had found the chain broken, and a report that
    # says "broken" once the chain verified again. A report is reused only when this request's walk is valid and the report's
    # own snapshot says the same; otherwise the request makes a new report, which says what the walk found. The profiles
    # that record no walk (summary, newsroom) are reused as before.
    walked = now.custody_verification is not None
    if walked and not _chain_valid(now):
        return None
    wanted = record_state(now)
    for row in rows:
        try:
            art = get_artefact(db, row.snapshot_artefact_id, analysis_id=analysis.id)
            data = store.read_bytes(art.sha256)
        except (ArtefactNotFound, ObjectNotFound):
            continue
        if hashlib.sha256(data).hexdigest() != art.sha256:
            continue
        earlier = snapshot.load(data)
        if record_state(earlier) == wanted and (not walked or _chain_valid(earlier)):
            return row
    return None


def prepare(db: Session, analysis: Analysis, result: dict[str, Any], *, profile: str, redacted: bool, revision_reason: str | None,
            requested_by: str, engine_version: str, store: ObjectStore | None = None,
            verified: audit.Verification | None = None) -> tuple[ReportRow, list[dict[str, Any]]]:
    """Freeze the snapshot and add the report row as `queued`, flushed and not committed. Returns the row and its custody
    events (the `redaction` of a redacted report: RP-03, D-10), which the caller commits with the row and its job
    (`audit.commit_with`). The caller holds the revision's row lock (`review.hold_revision`). `verified` is the caller's
    walk of the custody chain from before that lock (`snapshot.custody_verification`); without it a standard or forensic
    snapshot walks the whole chain here."""
    store = store or default_store()
    requested_at = utcnow()
    previous = latest(db, analysis.id, profile)  # the revision this one supersedes: the same profile
    # one revision counter per analysis, so display ids are unique across profiles (DR P1-9; D-25 format kept)
    newest = db.scalars(select(ReportRow.revision).where(ReportRow.analysis_id == analysis.id)
                        .order_by(ReportRow.revision.desc()).limit(1)).first()
    revision = (newest or 0) + 1
    report_id = next_report_id(db, requested_at.year)
    frozen = snapshot.build(db, analysis, result, report_id=report_id, revision=revision, profile=profile, redacted=redacted,
                            supersedes_report_id=previous.id if previous is not None else None, revision_reason=revision_reason,
                            requested_at=requested_at, requested_by=requested_by, template_set=engine.TEMPLATE_SET,
                            veritas_version=settings.version, verified=verified, store=store)
    stored = register_artefact(db, store, snapshot.dump(frozen), media_type="application/json", role="stage_output",
                               created_by="report", analysis_id=analysis.id)
    row = ReportRow(id=report_id, display_id=frozen.report.display_id, revision=revision, supersedes_report_id=frozen.report.supersedes_report_id,
                    revision_reason=revision_reason, analysis_id=analysis.id, analysis_revision=analysis.revision or 1,
                    case_id=analysis.case_id, profile=profile, redacted=redacted, review_state=frozen.review.state,
                    issuable=frozen.review.issuable,
                    template_set=engine.TEMPLATE_SET, engine="weasyprint", engine_version=engine_version, status="queued",
                    snapshot_hash=stored.sha256, snapshot_artefact_id=stored.id, requested_by=requested_by, requested_at=requested_at)
    db.add(row)
    db.flush()
    events: list[dict[str, Any]] = []
    if redacted:  # the redaction is logged when the redacted snapshot is frozen
        events.append({"action": "redaction", "actor_id": requested_by, "analysis_id": analysis.id, "report_id": report_id,
                       "object_ids": [stored.id], "after_hash": stored.sha256,
                       "details": {"profile": profile, "withheld": list(WITHHELD_FIELDS)}})
    log.info("reports.requested", report_id=report_id, analysis_id=analysis.id, profile=profile, revision=revision)
    return row, events


def create(db: Session, analysis: Analysis, result: dict[str, Any], *, profile: str, redacted: bool, revision_reason: str | None,
           requested_by: str, engine_version: str, store: ObjectStore | None = None) -> ReportRow:
    """`prepare`, committed with its events and without a job: for a caller that runs the job itself (tools, tests). The API
    queues the job in the transaction (`v2_reports.request_report`)."""
    row, events = prepare(db, analysis, result, profile=profile, redacted=redacted, revision_reason=revision_reason,
                          requested_by=requested_by, engine_version=engine_version, store=store)
    audit.commit_with(db, events)
    return row


def _fail(report_id: str, error: str) -> str:
    with get_session() as db:
        row = db.get(ReportRow, report_id)
        if row is not None and row.status != "done":
            row.status, row.error, row.finished_at = "failed", error, utcnow()
            db.commit()
    log.warning("reports.failed", report_id=report_id, error=error)
    return "failed"


def mark_failed(report_id: str, error: str | None) -> None:
    """A lease expired and the poison guard failed or cancelled the job: reflect it on the report."""
    _fail(report_id, error or "The report job did not finish.")


def _outcome(report_id: str) -> str:
    """What a job that wrote nothing returns: the state another worker, or the queue, left on the row."""
    with get_session() as db:
        row = db.get(ReportRow, report_id)
        return "failed" if row is not None and row.status == "failed" else "done"


def run(report_id: str, *, store: ObjectStore | None = None, timeout_s: float | None = None) -> str:
    """The `report` job. Returns the job outcome (done or failed). `timeout_s` bounds the render (None: no limit)."""
    store = store or default_store()
    with get_session() as db:
        # conditional, as the final write: a report another worker finished meanwhile is not set back to running
        started_now = db.execute(update(ReportRow).where(ReportRow.id == report_id, ReportRow.status.in_(OPEN)).values(status="running")
                                 .execution_options(synchronize_session=False))
        db.commit()
        row = db.get(ReportRow, report_id) if started_now.rowcount == 1 else None
        if row is None or row.status != "running":  # done, failed or gone
            return "done"
        analysis_id, actor, snapshot_artefact_id = row.analysis_id, row.requested_by, row.snapshot_artefact_id
        requested_template = row.template_set
        analysis = db.get(Analysis, analysis_id)
        withdrawn = analysis is not None and analysis.withdrawn_at is not None
        superseded = review.superseded(db, analysis_id)
    if withdrawn or superseded:  # D-47: checked at the request and again here
        return _fail(report_id, "The revision was withdrawn or superseded after the request, so the report was not produced.")
    started = time.perf_counter()
    try:
        audit.verify_original("report", actor_id=actor, analysis_id=analysis_id)
    except audit.OriginalIntegrityError as e:
        return _fail(report_id, f"The original {e.problem}; the report was not produced.")
    except LookupError:
        return _fail(report_id, "The analysis no longer exists; the report was not produced.")
    try:
        with get_session() as db:
            sha256 = get_artefact(db, snapshot_artefact_id, analysis_id=analysis_id).sha256
        data = store.read_bytes(sha256)
        if hashlib.sha256(data).hexdigest() != sha256:
            return _fail(report_id, "The stored report snapshot no longer matches its hash; the report was not produced.")
        frozen = snapshot.load(data)
        if not requested_template == frozen.report.template_set == engine.TEMPLATE_SET:
            return _fail(report_id, "The requested report template does not match the stored snapshot and current renderer; "
                         "the report was not produced. Request a new report revision.")
        rendered = deadline.run_within(timeout_s, lambda: engine.render_pdf(frozen), what="The report")
        key = signing.instance_key()
        signature = key.sign(rendered.pdf)
        signing.ensure_registered(key)
    except (ArtefactNotFound, ObjectNotFound):
        return _fail(report_id, "The stored report snapshot or a frozen evidence artifact is missing; the report was not produced.")
    except (engine.EngineUnavailable, signing.SigningKeyError) as e:
        return _fail(report_id, str(e))
    except deadline.JobTimeout as e:  # N-9: the render is left to end by itself; nothing it returns is stored
        log.error("reports.render_timeout", report_id=report_id, timeout_s=timeout_s)
        return _fail(report_id, f"{e}; the report was not produced. Request it again.")
    except Exception as e:
        log.error("reports.render_failed", report_id=report_id, error=str(e))
        return _fail(report_id, friendly_error(e))
    duration_ms = round((time.perf_counter() - started) * 1000)

    def unit() -> str | None:
        """The final write and its event, in one transaction (N-9). None when the row was not this job's to finish."""
        with audit.writing() as db:
            # the transaction's first statement claims the row: done, failed (the queue's poison guard) or deleted, it is left
            claimed = db.execute(update(ReportRow).where(ReportRow.id == report_id, ReportRow.status.in_(OPEN))
                                 .values(status="done", error=None, finished_at=utcnow()).execution_options(synchronize_session=False))
            if claimed.rowcount != 1:
                db.rollback()
                return None
            pdf = register_artefact(db, store, rendered.pdf, media_type="application/pdf", role="report", created_by="report",
                                    analysis_id=analysis_id)
            row = db.scalar(select(ReportRow).where(ReportRow.id == report_id).execution_options(populate_existing=True))
            row.artefact_id, row.sha256 = pdf.id, pdf.sha256
            row.page_count, row.engine_version = rendered.page_count, rendered.engine_version
            row.signature, row.signing_key_id, row.duration_ms = signature, key.key_id, duration_ms
            details = {"format": "v2-pdf", "profile": row.profile, "revision": row.revision, "display_id": row.display_id,
                       "supersedes_report_id": row.supersedes_report_id, "redacted": row.redacted, "snapshot_hash": row.snapshot_hash,
                       "template_set": row.template_set, "engine_version": row.engine_version, "page_count": row.page_count,
                       "signature": signature, "signing_key_id": key.key_id, "lint": rendered.lint}
            audit.commit_with(db, [{"action": "report_generated", "actor_id": actor, "analysis_id": analysis_id, "report_id": report_id,
                                    "object_ids": [pdf.id, snapshot_artefact_id], "after_hash": pdf.sha256, "details": details}],
                              before_commit=lambda seqs: setattr(row, "audit_event_seq", seqs[0]))
            return pdf.sha256

    try:
        sha = audit.repeated(unit)
    except Exception as e:  # the row and its event are stored together: neither is, and the report is failed
        log.error("reports.record_failed", report_id=report_id, error=f"{type(e).__name__}: {e}"[:300])
        return _fail(report_id, "The finished report could not be recorded, so it was not issued. Request the report again.")
    if sha is None:
        log.warning("reports.not_recorded", report_id=report_id, reason="the report was finished, failed or deleted meanwhile")
        return _outcome(report_id)
    log.info("reports.generated", report_id=report_id, pages=rendered.page_count, sha256=sha)
    return "done"
