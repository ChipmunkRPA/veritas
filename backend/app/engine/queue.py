"""DB-backed job queue with leases (ARCHITECTURE §3.4).

- **Claim.** SQLite takes the write lock up front (`BEGIN IMMEDIATE`), then selects and updates by id, so two workers
  can never lease the same job. Postgres uses a single `UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED LIMIT 1)
  RETURNING *`. The order is priority DESC, then created_at, then id.
- **Leases.** A claim increments `attempts` and leases the job for `lease_s`. Heartbeats extend the lease and report
  `cancel_requested`. Completing or heartbeating a lease that another worker now holds is refused.
- **Crash recovery and poison guard.** `requeue_expired()` returns expired leases to the queue. Once `attempts` has
  reached `max_attempts` the job fails instead (the same policy as the v0.3 executor). A job with a pending
  cancellation becomes `cancelled`.
- **Cancellation.** A queued job is cancelled at once. A leased job gets `cancel_requested`, which the worker turns
  into a cooperative cancel.

All timestamps come from the injected clock, so tests can advance time.
"""
from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Connection, Engine, Update, func, insert, select, update

from app.core.logging import get_logger
from app.storage.db import Job, utcnow

log = get_logger(__name__)

PRIORITY_INTERACTIVE = 100
PRIORITY_BATCH = 10
ACTIVE_STATUSES = ("queued", "leased")
TERMINAL_STATUSES = ("done", "failed", "cancelled")
JOB_TYPES = frozenset({"analysis", "report", "package", "refusion", "webhook_delivery", "batch_expand"})
_JOBS = Job.__table__


def _aware(dt: datetime | None) -> datetime | None:
    return dt if dt is None or dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _enqueue_lock_key(type: str, subject_id: str) -> int:
    """Stable transaction-lock identity; length prefixes keep the pair unambiguous."""
    parts = (type.encode("utf-8"), subject_id.encode("utf-8"))
    payload = b"veritas.queue.enqueue\0" + b"".join(len(part).to_bytes(4, "big") + part for part in parts)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=True)


@dataclass(frozen=True)
class JobView:
    id: str
    type: str
    subject_id: str
    priority: int
    status: str
    attempts: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    cancel_requested: bool
    not_before: datetime
    error: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: Any) -> JobView:
        m = row._mapping
        return cls(id=m["id"], type=m["type"], subject_id=m["subject_id"], priority=m["priority"], status=m["status"],
                   attempts=m["attempts"], lease_owner=m["lease_owner"], lease_expires_at=_aware(m["lease_expires_at"]),
                   cancel_requested=bool(m["cancel_requested"]), not_before=_aware(m["not_before"]), error=m["error"],
                   created_at=_aware(m["created_at"]), updated_at=_aware(m["updated_at"]))


@dataclass(frozen=True)
class Heartbeat:
    held: bool  # False: the lease was lost (expired and re-queued, or completed elsewhere)
    cancel_requested: bool


class JobQueue:
    def __init__(self, engine: Engine, *, lease_s: float = 60.0, max_attempts: int = 3, retry_backoff_s: float = 0.0,
                 clock: Callable[[], datetime] = utcnow):
        if lease_s <= 0 or max_attempts < 1:
            raise ValueError("lease_s must be positive and max_attempts at least 1")
        self.engine = engine
        self.lease_s = lease_s
        self.max_attempts = max_attempts
        self.retry_backoff_s = retry_backoff_s
        self.clock = clock

    @contextmanager
    def _write(self) -> Iterator[Connection]:
        """A write transaction. On SQLite the RESERVED lock is taken before the first read."""
        with self.engine.connect() as conn:
            if self.engine.dialect.name == "sqlite":
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def _now(self) -> datetime:
        return self.clock()

    # --- submission ------------------------------------------------------------------------------------------

    def enqueue(self, type: str, subject_id: str, *, priority: int = PRIORITY_INTERACTIVE,
                not_before: datetime | None = None) -> JobView:
        """Queue a job, returning an active job for the same (type, subject) when present.

        PostgreSQL uses READ COMMITTED: after the transaction lock is acquired, the subsequent SELECT sees
        the previous submitter's commit. This coordinates callers of enqueue; it does not repair historical
        duplicates or constrain arbitrary SQL writers. SQLite retains its BEGIN IMMEDIATE write lock.
        """
        if type not in JOB_TYPES:
            raise ValueError(f"unknown job type {type!r}")
        if not subject_id or len(subject_id) > 64:
            raise ValueError("subject_id must be 1-64 characters")
        now = self._now()
        with self._write() as conn:
            if self.engine.dialect.name == "postgresql":
                conn.execute(select(func.pg_advisory_xact_lock(_enqueue_lock_key(type, subject_id))))
            existing = conn.execute(select(_JOBS).where(
                Job.type == type, Job.subject_id == subject_id, Job.status.in_(ACTIVE_STATUSES))).first()
            if existing is not None:
                return JobView.from_row(existing)
            job_id = str(uuid.uuid4())
            conn.execute(insert(_JOBS).values(
                id=job_id, type=type, subject_id=subject_id, priority=priority, status="queued", attempts=0,
                cancel_requested=False, not_before=not_before or now, created_at=now, updated_at=now))
            return JobView.from_row(conn.execute(select(_JOBS).where(Job.id == job_id)).one())

    def enqueue_in(self, conn: Connection, type: str, subject_id: str, *, priority: int = PRIORITY_INTERACTIVE) -> str:
        """Queue a job inside the caller's transaction and return its id (N-9): a row and its job are committed together or
        not at all, so no queued revision is left without a job when what follows its commit fails. For a subject that the
        same transaction creates: no active job of it can exist, so nothing is looked up."""
        if type not in JOB_TYPES:
            raise ValueError(f"unknown job type {type!r}")
        if not subject_id or len(subject_id) > 64:
            raise ValueError("subject_id must be 1-64 characters")
        now, job_id = self._now(), str(uuid.uuid4())
        conn.execute(insert(_JOBS).values(
            id=job_id, type=type, subject_id=subject_id, priority=priority, status="queued", attempts=0,
            cancel_requested=False, not_before=now, created_at=now, updated_at=now))
        return job_id

    # --- claiming --------------------------------------------------------------------------------------------

    def _claim_values(self, worker_id: str, now: datetime) -> dict[str, Any]:
        return {"status": "leased", "lease_owner": worker_id, "lease_expires_at": now + timedelta(seconds=self.lease_s),
                "attempts": Job.attempts + 1, "updated_at": now}

    def _pick(self, now: datetime, types: Sequence[str] | None):
        conditions = [Job.status == "queued", Job.not_before <= now, Job.cancel_requested.is_(False)]
        if types:
            conditions.append(Job.type.in_(list(types)))
        return select(Job.id).where(*conditions).order_by(Job.priority.desc(), Job.created_at, Job.id).limit(1)

    def claim_statement_postgres(self, worker_id: str, now: datetime, types: Sequence[str] | None = None) -> Update:
        """The single-statement Postgres claim (exposed so the SQL can be checked without a server)."""
        subquery = self._pick(now, types).with_for_update(skip_locked=True).scalar_subquery()
        return (update(Job).where(Job.id == subquery, Job.status == "queued")
                .values(**self._claim_values(worker_id, now)).returning(*_JOBS.c))

    def claim(self, worker_id: str, types: Sequence[str] | None = None) -> JobView | None:
        now = self._now()
        with self._write() as conn:
            if self.engine.dialect.name == "postgresql":
                row = conn.execute(self.claim_statement_postgres(worker_id, now, types)).first()
                if row is None:
                    return None
                view = JobView.from_row(row)
            else:
                job_id = conn.execute(self._pick(now, types)).scalar()
                if job_id is None:
                    return None
                conn.execute(update(Job).where(Job.id == job_id, Job.status == "queued").values(**self._claim_values(worker_id, now)))
                view = JobView.from_row(conn.execute(select(_JOBS).where(Job.id == job_id)).one())
        log.info("queue.claimed", job_id=view.id, type=view.type, subject_id=view.subject_id, attempt=view.attempts)
        return view

    # --- running jobs ----------------------------------------------------------------------------------------

    def heartbeat(self, job_id: str, worker_id: str) -> Heartbeat:
        now = self._now()
        with self._write() as conn:
            held = conn.execute(update(Job).where(Job.id == job_id, Job.status == "leased", Job.lease_owner == worker_id)
                                .values(lease_expires_at=now + timedelta(seconds=self.lease_s), updated_at=now)).rowcount
            if not held:
                return Heartbeat(held=False, cancel_requested=False)
            cancel = conn.execute(select(Job.cancel_requested).where(Job.id == job_id)).scalar_one()
        return Heartbeat(held=True, cancel_requested=bool(cancel))

    def complete(self, job_id: str, worker_id: str, status: str, error: str | None = None) -> bool:
        """Finish a leased job. Returns False when this worker no longer holds the lease."""
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"a job completes as one of {TERMINAL_STATUSES}, not {status!r}")
        now = self._now()
        with self._write() as conn:
            done = conn.execute(update(Job).where(Job.id == job_id, Job.status == "leased", Job.lease_owner == worker_id)
                                .values(status=status, error=error, lease_owner=None, lease_expires_at=None,
                                        updated_at=now)).rowcount
        if not done:
            log.warning("queue.complete_without_lease", job_id=job_id, worker_id=worker_id, status=status)
        return bool(done)

    def release(self, job_id: str, worker_id: str) -> JobView | None:
        """Give back a job this worker holds, before it ended, without counting the attempt (BUILD_SPEC N-14; ops review
        OPS-15, OPS-16).

        A worker that is told to stop while a job outlasts its stop timeout used to exit with the lease held. The job
        then waited for the lease to expire, and the claim that followed counted as one more attempt: three stops during
        one long job failed it ("Processing was interrupted 3 times"). A released job is `queued` again at once, and its
        `attempts` is what it was before this worker claimed it. A job whose cancellation was requested is `cancelled`,
        as the sweep of an expired lease would make it (a queued job with that flag would never be claimed).

        Returns the job as it is afterwards; None when this worker does not hold it (it ended, or the lease was lost)."""
        now = self._now()
        held = (Job.id == job_id, Job.status == "leased", Job.lease_owner == worker_id)
        with self._write() as conn:
            row = conn.execute(select(_JOBS).where(*held).with_for_update()).first()
            if row is None:
                return None
            job = JobView.from_row(row)
            values: dict[str, Any] = {"lease_owner": None, "lease_expires_at": None, "updated_at": now}
            if job.cancel_requested:
                values.update(status="cancelled", error="Cancelled on request")
            else:
                values.update(status="queued", not_before=now, attempts=max(0, job.attempts - 1))
            if conn.execute(update(Job).where(*held).values(**values)).rowcount != 1:
                return None
            view = JobView.from_row(conn.execute(select(_JOBS).where(Job.id == job_id)).one())
        log.warning("queue.released", job_id=view.id, type=view.type, subject_id=view.subject_id, outcome=view.status,
                    attempts=view.attempts)
        return view

    def request_cancel(self, *, job_id: str | None = None, subject_id: str | None = None,
                       type: str = "analysis", types: Sequence[str] | None = None,
                       on_request: Callable[[Connection, JobView], None] | None = None) -> JobView | None:
        """Lock against concurrent claim/complete and cancel an active job, or return None.

        `on_request` validates/projects the subject in this transaction; exceptions roll it back. The callback
        must use this connection, never open another write transaction. Ambiguous subject/type sets are refused.
        """
        if (job_id is None) == (subject_id is None):
            raise ValueError("pass exactly one of job_id or subject_id")
        now = self._now()
        target = Job.id == job_id if job_id is not None else (Job.subject_id == subject_id) & (Job.type.in_(types or (type,)))
        with self._write() as conn:
            rows = conn.execute(select(_JOBS).where(target, Job.status.in_(ACTIVE_STATUSES)).order_by(Job.id).with_for_update()).all()
            if not rows:
                return None
            if len(rows) != 1:
                raise ValueError("More than one active job exists for this subject")
            view = JobView.from_row(rows[0])
            if on_request is not None:
                on_request(conn, view)
            if view.status == "queued":
                conn.execute(update(Job).where(Job.id == view.id, Job.status == "queued").values(
                    status="cancelled", cancel_requested=True, error="Cancelled on request", updated_at=now))
            else:
                conn.execute(update(Job).where(Job.id == view.id, Job.status == "leased").values(cancel_requested=True, updated_at=now))
            return JobView.from_row(conn.execute(select(_JOBS).where(Job.id == view.id)).one())

    def requeue_expired(self) -> list[JobView]:
        """Return expired leases to the queue, or fail them (poison guard), or cancel them (pending cancellation)."""
        now = self._now()
        expired = Job.status == "leased", Job.lease_expires_at < now
        with self.engine.connect() as conn:  # cheap check before taking the write lock
            if conn.execute(select(func.count()).select_from(_JOBS).where(*expired)).scalar_one() == 0:
                return []
        changed: list[JobView] = []
        with self._write() as conn:
            for row in conn.execute(select(_JOBS).where(*expired)).all():
                job = JobView.from_row(row)
                base = {"lease_owner": None, "lease_expires_at": None, "updated_at": now}
                if job.cancel_requested:
                    values = {**base, "status": "cancelled", "error": "Cancelled on request"}
                elif job.attempts >= self.max_attempts:
                    values = {**base, "status": "failed",
                              "error": f"Processing was interrupted {job.attempts} times; the input is not re-queued again."}
                else:
                    values = {**base, "status": "queued", "not_before": now + timedelta(seconds=self.retry_backoff_s)}
                # A heartbeat, completion, cancellation flag or competing sweep may have committed since
                # selection. Only transition the exact lease state whose expiry/policy we evaluated.
                transitioned = conn.execute(update(Job).where(
                    Job.id == job.id, Job.status == job.status, Job.lease_owner == job.lease_owner,
                    Job.lease_expires_at == job.lease_expires_at, Job.attempts == job.attempts,
                    Job.cancel_requested == job.cancel_requested, Job.updated_at == job.updated_at,
                ).values(**values))
                if transitioned.rowcount != 1:
                    continue
                changed.append(JobView.from_row(conn.execute(select(_JOBS).where(Job.id == job.id)).one()))
        for job in changed:
            log.warning("queue.lease_expired", job_id=job.id, subject_id=job.subject_id, outcome=job.status, attempts=job.attempts)
        return changed

    # --- inspection ------------------------------------------------------------------------------------------

    def get(self, job_id: str) -> JobView | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(_JOBS).where(Job.id == job_id)).first()
        return JobView.from_row(row) if row is not None else None

    def latest(self, subject_id: str, type: str = "analysis") -> JobView | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(_JOBS).where(Job.subject_id == subject_id, Job.type == type)
                               .order_by(Job.created_at.desc(), Job.id.desc()).limit(1)).first()
        return JobView.from_row(row) if row is not None else None

    def active_subjects(self, type: str = "analysis") -> set[str]:
        with self.engine.connect() as conn:
            return set(conn.execute(select(Job.subject_id).where(Job.type == type, Job.status.in_(ACTIVE_STATUSES))).scalars())

    def depth(self, types: Sequence[str] | None = None) -> int:
        conditions = [Job.status.in_(ACTIVE_STATUSES)]
        if types:
            conditions.append(Job.type.in_(list(types)))
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(_JOBS).where(*conditions)).scalar_one()
