"""Retention (D-55; PRD CC-04): what may be deleted, what a deletion removes, and which bytes it frees.

- **Rule.** `chain_verdict` decides for a revision chain (an analysis and the revisions linked to it through `supersedes_id`),
  which is purged whole or kept whole:
  - never purged: a chain with a queued or running revision, or with a report or package still being produced; a chain in a case
    under legal hold; a chain in a case whose `retention_until` lies in the future;
  - a case whose retention date has passed: every chain of the case goes, signed or not;
  - no retention date: a named case is kept. A quick analysis (implicit case, or no case) goes when its newest revision is older
    than `VERITAS_RETENTION_DAYS` and no revision is signed.
  `delete_refusal` is the same guard for `DELETE /api/analyses/{id}`; a request never deletes a signed revision.
- **Removal.** `remove` deletes the analyses with their evidence, review, report and package rows, then the case link, the intake
  records and the media record of a media item that has no analysis left in the case, and an implicit case that has no item left.
  A named case keeps its row. The custody log is never rewritten.
- **Tombstone.** `purge` appends one `retention_purge` event per purged revision by the system actor: ids and hashes, the rule
  that applied, no file name and no text. The events are written in the transaction that deletes the chain
  (`audit.commit_with`), so a purge that is stopped at any point leaves a chain either whole or gone with its events. What a
  stopped purge leaves behind is directories and bytes: the next purge removes the directories (`_leftovers`), `store gc` the
  bytes.
- **One write transaction per chain.** The purge decides again inside the transaction that deletes (`_still_purgeable`), and
  nothing may commit to the chain between that re-check and the delete. On Postgres the re-check locks the chain's analysis
  rows (`FOR UPDATE`). On SQLite a read takes no lock and the driver opens no transaction before the first write, so
  `_purge_chain` takes the database's write lock first (`BEGIN IMMEDIATE`, `_write_lock`): the re-check, the delete and the
  events are one write transaction, and a sign-off that arrives meanwhile waits and then finds no row.
- **Bytes.** `release` unlinks the store objects that no remaining row references (`REFERENCE_COLUMNS`). It does so only while
  no worker holds a job of the queue (`held`: a lease that has not run out), because a job writes its objects before it indexes
  them; otherwise the objects wait for `python -m app.cli store gc`, which is the same sweep over the whole store. Work that
  waits in the queue does not block (N-17a).
"""
from __future__ import annotations

import re
import shutil
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import Select, delete, func, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.audit import chain as audit
from app.core import times
from app.core.config import settings
from app.core.logging import get_logger
from app.storage.artefacts import delete_evidence
from app.storage.db import (
    Analysis,
    Artefact,
    AuditEvent,
    CaseMediaRow,
    CaseMemberRow,
    CaseRow,
    ExhibitRow,
    IntakeRecordRow,
    Job,
    MediaObject,
    PackageRow,
    ReportRow,
    get_session,
    utcnow,
)
from app.storage.objects import ObjectStore, default_store

log = get_logger(__name__)

IN_PROGRESS = ("queued", "running")
# every column that names a store object by its sha256 (tests/test_retention.py checks the list against the schema)
REFERENCE_COLUMNS = (Artefact.sha256, MediaObject.sha256, Analysis.sha256, ReportRow.sha256, ReportRow.snapshot_hash, PackageRow.sha256,
                     ExhibitRow.sha256)
_SHA256 = re.compile(r"[0-9a-f]{64}")
GC_GRACE_S = 900.0  # `store gc` leaves an object written in the last 15 minutes alone: its row may not be committed yet
_CHUNK = 500  # bound parameters per IN list
_TAIL_TRIES = 20  # a chain's delete is repeated this often when another process extends the custody log during its append
_WHOLE_TABLE = 5000  # above this many names, `referenced` reads the reference columns whole


_aware = times.to_utc  # compared and printed in UTC (D-72): "retained until" names the UTC date on every engine


# --- the rule ------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    purge: bool
    rule: str  # in_progress | legal_hold | retained | named_case | signed | recent | case_retention_date_passed | quick_analysis_expired
    reason: str


def _held(case: CaseRow | None, now: datetime) -> Verdict | None:
    """The refusals a case imposes on any deletion: a legal hold, or a retention date that lies in the future."""
    if case is None:
        return None
    if case.legal_hold:
        return Verdict(False, "legal_hold", "the case is under legal hold")
    until = _aware(case.retention_until)
    if until is not None and until > now:
        return Verdict(False, "retained", f"the case is retained until {until:%Y-%m-%d}")
    return None


def chain_verdict(revisions: Sequence[Revision | Analysis], case: CaseRow | None, *, now: datetime | None = None,
                  retention_days: int | None = None, producing: Iterable[str] = ()) -> Verdict:
    """Whether a revision chain may be purged, and why or why not (D-55). `producing` names the revisions of the chain that have
    a report or a package queued or running."""
    now = now or utcnow()
    days = settings.retention_days if retention_days is None else retention_days
    ordered = sorted(revisions, key=lambda r: (r.revision or 1, _aware(r.created_at) or now))
    for r in ordered:
        if r.status in IN_PROGRESS:
            return Verdict(False, "in_progress", f"revision {r.revision or 1} is {r.status}")
    busy = set(producing)
    for r in ordered:
        if r.id in busy:
            return Verdict(False, "in_progress", f"a report or package of revision {r.revision or 1} is being produced")
    if (held := _held(case, now)) is not None:
        return held
    until = _aware(case.retention_until) if case is not None else None
    if until is not None:
        return Verdict(True, "case_retention_date_passed", f"the retention date of the case, {until:%Y-%m-%d}, has passed")
    if case is not None and not case.implicit:
        return Verdict(False, "named_case", "a named case without a retention date is kept; its retention date schedules the purge")
    for r in ordered:
        if r.locked_at is not None:
            return Verdict(False, "signed", f"revision {r.revision or 1} is signed off; a signed record goes only through a case retention date")
    newest = max((_aware(r.created_at) or now) for r in ordered)
    age = (now - newest).days
    if newest >= now - timedelta(days=days):
        return Verdict(False, "recent", f"the newest revision is {age} day(s) old; quick analyses are kept for {days} day(s)")
    return Verdict(True, "quick_analysis_expired", f"an unsigned quick analysis, {age} day(s) old; quick analyses are kept for {days} day(s)")


def delete_refusal(analysis: Analysis, case: CaseRow | None, *, now: datetime | None = None) -> str | None:
    """Why `DELETE /api/analyses/{id}` is refused (the detail of its 409), or None. The status, legal-hold and retention
    refusals are those of the purge; a request also never deletes a signed revision (RW-04)."""
    now = now or utcnow()
    if analysis.status in IN_PROGRESS:
        return f"Analysis is {analysis.status}; wait for it to finish before deleting"
    if analysis.locked_at is not None:
        return "The analysis revision is signed off and read-only"
    held = _held(case, now)
    if held is None:
        return None
    if held.rule == "legal_hold":
        return "The case is under legal hold; its analyses cannot be deleted"
    return f"The case is retained until {_aware(case.retention_until):%Y-%m-%d}; its analyses cannot be deleted before then"


# --- chains --------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Revision:
    """The columns of an analysis the rule reads (the result body is not loaded)."""
    id: str
    revision: int
    status: str
    locked_at: datetime | None
    created_at: datetime
    case_id: str | None
    supersedes_id: str | None


_REVISION = (Analysis.id, Analysis.revision, Analysis.status, Analysis.locked_at, Analysis.created_at, Analysis.case_id,
             Analysis.supersedes_id)


@dataclass(frozen=True)
class Chain:
    root_id: str
    revisions: tuple[Revision, ...]  # oldest first
    case: CaseRow | None
    verdict: Verdict

    @property
    def ids(self) -> list[str]:
        return [r.id for r in self.revisions]


def _producing(db: Session) -> set[str]:
    return set(db.scalars(select(ReportRow.analysis_id).where(ReportRow.status.in_(IN_PROGRESS)))) | set(
        db.scalars(select(PackageRow.analysis_id).where(PackageRow.status.in_(IN_PROGRESS))))


def _chain(db: Session, root_id: str, members: list[Revision], producing: set[str], now: datetime, retention_days: int | None) -> Chain:
    members = sorted(members, key=lambda r: (r.revision, _aware(r.created_at) or now, r.id))
    cases = {case_id: db.get(CaseRow, case_id) for case_id in dict.fromkeys(r.case_id for r in members) if case_id}
    # one case per chain by construction; if revisions ever disagree, any case that keeps the chain keeps it
    verdicts = [chain_verdict(members, cases.get(case_id) if case_id else None, now=now, retention_days=retention_days, producing=producing)
                for case_id in dict.fromkeys(r.case_id for r in members)]
    verdict = next((v for v in verdicts if not v.purge), verdicts[0])
    return Chain(root_id, tuple(members), cases.get(members[-1].case_id) if members[-1].case_id else None, verdict)


def _chains(db: Session, now: datetime, retention_days: int | None) -> list[Chain]:
    rows = {r.id: r for r in (Revision(row.id, row.revision or 1, row.status, row.locked_at, row.created_at, row.case_id, row.supersedes_id)
                              for row in db.execute(select(*_REVISION)))}

    def root(r: Revision) -> str:
        seen = {r.id}
        while r.supersedes_id is not None and r.supersedes_id in rows and r.supersedes_id not in seen:
            r = rows[r.supersedes_id]
            seen.add(r.id)
        return r.id

    grouped: dict[str, list[Revision]] = {}
    for r in rows.values():
        grouped.setdefault(root(r), []).append(r)
    producing = _producing(db)
    chains = [_chain(db, root_id, members, producing, now, retention_days) for root_id, members in grouped.items()]
    chains.sort(key=lambda c: (_aware(c.revisions[0].created_at) or now, c.root_id))
    return chains


def _locked_revisions(ids: Sequence[str]) -> Select:
    """The chain's analysis rows and any revision made of them since, locked until the transaction ends (`FOR UPDATE` on
    Postgres): a sign-off or a status change of one of them waits for the delete and then finds no row. On SQLite the clause
    is not emitted and a read locks nothing: the caller holds the database's write lock instead (`_write_lock`)."""
    return select(*_REVISION).where(or_(Analysis.id.in_(ids), Analysis.supersedes_id.in_(ids))).order_by(Analysis.id).with_for_update()


def _locked_cases(db: Session, case_ids: Iterable[str | None]) -> dict[str, CaseRow]:
    """The case rows of a chain, read now and share-locked until the transaction ends (`FOR SHARE` on Postgres; N-9, Postgres
    QA PG-15): a legal hold or a retention date either committed first and is seen here, or its update waits for this
    transaction (`v2_cases.update_case` takes the row's lock) and then applies to a case whose chain is gone. On SQLite the
    clause is not emitted and the caller holds the write lock."""
    wanted = sorted({c for c in case_ids if c})
    if not wanted:
        return {}
    rows = db.scalars(select(CaseRow).where(CaseRow.id.in_(wanted)).order_by(CaseRow.id).with_for_update(read=True)
                      .execution_options(populate_existing=True))
    return {row.id: row for row in rows}


def _still_purgeable(db: Session, chain: Chain, now: datetime, retention_days: int | None) -> bool:
    """Decided again inside the transaction that deletes, on the locked rows: a sign-off, a hold, a new revision or a report
    requested since the listing keeps the chain."""
    ids = chain.ids
    rows = [Revision(row.id, row.revision or 1, row.status, row.locked_at, row.created_at, row.case_id, row.supersedes_id)
            for row in db.execute(_locked_revisions(ids))]
    if sorted(r.id for r in rows) != sorted(ids):
        return False
    _locked_cases(db, [r.case_id for r in rows])
    fresh = _chain(db, chain.root_id, rows, _producing(db), now, retention_days)
    return fresh.verdict.purge and fresh.verdict.rule == chain.verdict.rule


# --- removal -------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tombstone:
    """What a `retention_purge` event says about one revision: ids and hashes only (PRD CC-04)."""
    analysis_id: str
    case_id: str | None
    media_id: str | None
    sha256: str
    revision: int
    result_hash: str | None
    reports: tuple[tuple[str, str | None], ...]
    packages: tuple[tuple[str, str | None], ...]


@dataclass
class Released:
    unlinked: int = 0
    bytes: int = 0
    young: int = 0  # `store gc` only: objects written within `GC_GRACE_S`, not examined
    deferred: int = 0  # unreferenced, left in place because a worker held a job
    busy: int = 0  # the jobs of the queue under a live lease (`held`)
    sha256s: list[str] = field(default_factory=list)  # the unreferenced objects found (unlinked, or left in place)


@dataclass
class Removal:
    tombstones: list[Tombstone] = field(default_factory=list)
    candidates: set[str] = field(default_factory=set)  # sha256 of objects the removed rows named
    cases_removed: list[str] = field(default_factory=list)
    media_removed: list[str] = field(default_factory=list)
    released: Released = field(default_factory=Released)


def _chunks(values: Iterable[str]) -> Iterable[list[str]]:
    batch = sorted(set(values))
    for start in range(0, len(batch), _CHUNK):
        yield batch[start:start + _CHUNK]


def _delete_rows(db: Session, analysis_ids: Sequence[str]) -> Removal:
    """Delete the analyses and what hangs on them, in the caller's transaction. Returns what the rows named."""
    out = Removal()
    analyses = [a for ids in _chunks(analysis_ids) for a in db.scalars(select(Analysis).where(Analysis.id.in_(ids)))]
    analyses.sort(key=lambda a: (a.revision or 1, a.id))
    pairs = {(a.case_id, a.media_id) for a in analyses}
    job_subjects: set[str] = set()
    for a in analyses:
        reports = db.execute(select(ReportRow.id, ReportRow.sha256, ReportRow.snapshot_hash).where(ReportRow.analysis_id == a.id)
                             .order_by(ReportRow.id)).all()
        packages = db.execute(select(PackageRow.id, PackageRow.sha256).where(PackageRow.analysis_id == a.id).order_by(PackageRow.id)).all()
        out.tombstones.append(Tombstone(analysis_id=a.id, case_id=a.case_id, media_id=a.media_id, sha256=a.sha256, revision=a.revision or 1,
                                        result_hash=a.result_hash, reports=tuple((r.id, r.sha256) for r in reports),
                                        packages=tuple((p.id, p.sha256) for p in packages)))
        out.candidates.update(db.scalars(select(Artefact.sha256).where(Artefact.analysis_id == a.id)))
        out.candidates.update(sha for r in reports for sha in (r.sha256, r.snapshot_hash) if sha)
        out.candidates.update(p.sha256 for p in packages if p.sha256)
        out.candidates.update(sha for sha in db.scalars(select(ExhibitRow.sha256).where(ExhibitRow.analysis_id == a.id)) if sha)
        out.candidates.add(a.sha256)
        job_subjects.update([a.id, *(r.id for r in reports), *(p.id for p in packages)])
        delete_evidence(db, a.id)
        db.delete(a)
    for subjects in _chunks(job_subjects):
        db.execute(delete(Job).where(Job.subject_id.in_(subjects)))
    db.flush()

    # a media item with no analysis left in the case leaves the case; media that nothing holds any more goes with its record
    for case_id, media_id in sorted(pairs, key=lambda p: (p[0] or "", p[1] or "")):
        if media_id is None:
            continue
        if case_id is not None and db.scalar(select(func.count()).select_from(Analysis).where(
                Analysis.case_id == case_id, Analysis.media_id == media_id)) == 0:
            db.execute(delete(IntakeRecordRow).where(IntakeRecordRow.case_id == case_id, IntakeRecordRow.media_id == media_id))
            db.execute(delete(CaseMediaRow).where(CaseMediaRow.case_id == case_id, CaseMediaRow.media_id == media_id))
    db.flush()
    for media_id in sorted({m for _, m in pairs if m is not None}):
        held = (db.scalar(select(func.count()).select_from(Analysis).where(Analysis.media_id == media_id))
                + db.scalar(select(func.count()).select_from(CaseMediaRow).where(CaseMediaRow.media_id == media_id))
                + db.scalar(select(func.count()).select_from(IntakeRecordRow).where(IntakeRecordRow.media_id == media_id)))
        media = db.get(MediaObject, media_id)
        if held or media is None:
            continue
        original = media.storage_artefact_id
        out.candidates.add(media.sha256)
        db.delete(media)
        db.flush()
        if original is not None and db.scalar(select(func.count()).select_from(MediaObject).where(
                MediaObject.storage_artefact_id == original)) == 0:
            db.execute(delete(Artefact).where(Artefact.id == original))
        out.media_removed.append(media_id)
    for case_id in sorted({c for c, _ in pairs if c is not None}):
        case = db.get(CaseRow, case_id)
        if case is None or not case.implicit:
            continue  # a named case keeps its row
        left = (db.scalar(select(func.count()).select_from(CaseMediaRow).where(CaseMediaRow.case_id == case_id))
                + db.scalar(select(func.count()).select_from(Analysis).where(Analysis.case_id == case_id)))
        if left == 0:
            db.execute(delete(IntakeRecordRow).where(IntakeRecordRow.case_id == case_id))
            db.execute(delete(CaseMemberRow).where(CaseMemberRow.case_id == case_id))
            db.delete(case)
            out.cases_removed.append(case_id)
    db.flush()
    return out


def remove(analysis_ids: Sequence[str], *, store: ObjectStore | None = None) -> Removal:
    """Delete analyses with their rows, their directories and the bytes nothing else references. The caller has applied the
    guard (`chain_verdict` or `delete_refusal`) and writes the custody event."""
    with get_session() as db:
        removal = _delete_rows(db, analysis_ids)
        db.commit()
    for t in removal.tombstones:
        for directory in _directories(t.analysis_id):
            shutil.rmtree(directory, ignore_errors=True)
    removal.released = release(removal.candidates, store=store)
    return removal


def delete_requested(analysis_id: str, *, actor_id: str, store: ObjectStore | None = None) -> Removal | str | None:
    """`DELETE /api/analyses/{id}` (N-9): the guard is decided inside the transaction that deletes, on the locked revision
    and its share-locked case, and the `deletion_requested` event is committed with the delete (`audit.commit_with`). A hold
    or a sign-off committed since the request arrived refuses it, and a refused or failed delete leaves no event.

    Returns the removal, the refusal (the detail of the 409), or None when the analysis no longer exists."""
    def unit() -> Removal | str | None:
        with audit.writing() as db:
            _write_lock(db)
            db.execute(_locked_revisions([analysis_id])).all()
            analysis = db.scalar(select(Analysis).where(Analysis.id == analysis_id).execution_options(populate_existing=True))
            if analysis is None:
                return None
            case = _locked_cases(db, [analysis.case_id]).get(analysis.case_id) if analysis.case_id else None
            if (refusal := delete_refusal(analysis, case)) is not None:
                return refusal
            event = {"action": "deletion_requested", "actor_id": actor_id, "analysis_id": analysis_id,
                     "case_id": analysis.case_id, "media_id": analysis.media_id, "before_hash": analysis.sha256,
                     "details": {"kind": analysis.kind, "filename": analysis.filename, "status": analysis.status}}
            removal = _delete_rows(db, [analysis_id])
            audit.commit_with(db, [event])
            return removal

    outcome = audit.repeated(unit)
    if isinstance(outcome, Removal):
        for t in outcome.tombstones:
            for directory in _directories(t.analysis_id):
                shutil.rmtree(directory, ignore_errors=True)
        outcome.released = release(outcome.candidates, store=store)
    return outcome


# --- bytes ---------------------------------------------------------------------------------------------------------------------

def held(db: Session, now: datetime | None = None) -> int:
    """The jobs of the queue that a worker holds: under a lease that has not run out. Such a job may have written store objects
    that no row names yet, so nothing is unlinked while there is one.

    N-17a (operations re-review O2-02): work that waits does not count. Before, every queued or running row blocked the
    release of bytes: below the free-space floor the worker claims nothing, so a backlog stayed queued and `retention purge`
    and `store gc` could free nothing in the one state the runbook names them for. A lease that has run out is no longer
    held: its worker has stopped renewing it.

    N-18a (operations re-check O3-01): an interrupted analysis that waits (given back at a stop, or its worker dead) may
    have written objects that no row names. N-17a's text said it had not, and the sweep then took the maps of a finished
    `detect` that only `render` would have indexed. What makes the rule safe is in the runner: a stage's objects are
    indexed with the stage's `done`, a stage is reused only with every object it names, and nothing is published while one
    is missing (`engine.runner`, `stages.common.publish`). What can still go while a run waits is what it wrote for the
    stage it was stopped in, which runs again."""
    now = now or utcnow()
    return int(db.scalar(select(func.count()).select_from(Job).where(Job.status == "leased", Job.lease_expires_at > now)) or 0)


def referenced(db: Session, sha256s: Iterable[str]) -> set[str]:
    """The subset of these objects that a remaining row still names."""
    wanted = set(sha256s)
    found: set[str] = set()
    if len(wanted) > _WHOLE_TABLE:  # a sweep of the store: read each column once instead of one lookup per batch
        for column in REFERENCE_COLUMNS:
            found.update(sha for sha in db.scalars(select(column).distinct()) if sha in wanted)
        return found
    for batch in _chunks(wanted):
        for column in REFERENCE_COLUMNS:
            found.update(db.scalars(select(column).where(column.in_(batch)).distinct()))
    return found


def _may_go(moved: set[str]) -> set[str]:
    """Asked again after the objects were moved aside: none of them while a worker holds a job, else those still unreferenced."""
    with get_session() as db:
        return set() if held(db) else moved - referenced(db, moved)


def release(sha256s: Iterable[str], *, store: ObjectStore | None = None, dry_run: bool = False) -> Released:
    """Unlink the objects among `sha256s` that no row references. Nothing is unlinked while a worker holds a job (`held`);
    `busy` is the number of such jobs."""
    store = store or default_store()
    wanted = {sha for sha in sha256s if sha and _SHA256.fullmatch(sha)}
    with get_session() as db:
        busy = held(db)
        free = sorted(sha for sha in wanted - referenced(db, wanted) if store.exists(sha))
    out = Released(busy=busy, sha256s=free)
    if busy:
        out.deferred = len(free)
    elif dry_run:
        out.unlinked, out.bytes = len(free), sum(store.path_for(sha).stat().st_size for sha in free)
    else:
        freed = store.delete(free, confirm=_may_go)
        out.unlinked, out.bytes, out.deferred = len(freed), sum(freed.values()), len(free) - len(freed)
    if out.unlinked or out.deferred:
        log.info("retention.released", unlinked=out.unlinked, bytes=out.bytes, deferred=out.deferred, busy=busy, dry_run=dry_run)
    return out


def store_gc(*, store: ObjectStore | None = None, dry_run: bool = False) -> Released:
    """Sweep the whole store: objects no row references (the orphans of deletes that freed nothing, and what `release` deferred)
    are unlinked; objects a crash left set aside are put back first. An object written within `GC_GRACE_S` is not examined: a
    report or an intake writes its object a moment before the row that names it is committed."""
    store = store or default_store()
    if not dry_run:
        store.recover()
    newest = time.time() - GC_GRACE_S
    settled, young = [], 0
    for sha in store.iter_sha256():
        try:
            recent = store.path_for(sha).stat().st_mtime > newest
        except FileNotFoundError:
            continue
        if recent:
            young += 1
        else:
            settled.append(sha)
    out = release(settled, store=store, dry_run=dry_run)
    out.young = young
    return out


# --- the purge -----------------------------------------------------------------------------------------------------------------

@dataclass
class PurgeReport:
    chains: list[Chain]
    changed: list[str] = field(default_factory=list)  # chains that stopped being purgeable between the listing and the delete
    failed: dict[str, str] = field(default_factory=dict)  # chains whose delete the database refused: root id -> error class
    purged_chains: int = 0
    purged_revisions: int = 0
    cases_removed: int = 0
    media_removed: int = 0
    leftovers: int = 0  # revisions an earlier, interrupted run deleted whose directories this run removed
    unrecorded_media: int = 0  # N-19e: media records of uploads that were never recorded, removed (or, dry, found)
    released: Released = field(default_factory=Released)


def _directories(analysis_id: str) -> list[Path]:
    """The v0.3 directories of an analysis; `uploads/<id>/<file>` is a hard link to the store object."""
    return [root / analysis_id for root in (settings.uploads_dir, settings.artifacts_dir, settings.reports_dir)]


def _tombstone_event(t: Tombstone, rule: str) -> dict:
    return {"action": "retention_purge", "actor_id": audit.SYSTEM_ACTOR, "case_id": t.case_id, "media_id": t.media_id,
            "analysis_id": t.analysis_id, "before_hash": t.sha256, "details": {
                "rule": rule, "revision": t.revision, "result_hash": t.result_hash,
                "reports": [{"id": rid, "sha256": sha} for rid, sha in t.reports],
                "packages": [{"id": pid, "sha256": sha} for pid, sha in t.packages]}}


def _write_lock(db: Session) -> None:
    """On SQLite, take the database's write lock now and hold it until the session commits or closes (`BEGIN IMMEDIATE`, as
    `engine/queue.py` and `api/jobs.py` do). Without it the transaction would begin at the first DELETE, after the re-check
    has read. Postgres locks rows instead (`_locked_revisions`)."""
    conn = db.connection()
    if conn.dialect.name == "sqlite":
        conn.exec_driver_sql("BEGIN IMMEDIATE")


def _purge_chain(chain: Chain, now: datetime, retention_days: int | None) -> Removal | None:
    """Delete one chain and append its `retention_purge` events in one transaction, which on SQLite holds the write lock from
    before the re-check. None when the chain is no longer purgeable. Raises `audit.TailMoved` (nothing was changed) when
    another process extended the custody log first. The custody chain's append lock comes before the database's write lock
    (`audit.writing`), the order of every action that appends in its own transaction (N-9), so a purge beside API traffic
    in one process waits for neither in the opposite order."""
    with audit.writing() as db:
        _write_lock(db)
        if not _still_purgeable(db, chain, now, retention_days):
            return None
        removal = _delete_rows(db, chain.ids)
        audit.commit_with(db, [_tombstone_event(t, chain.verdict.rule) for t in removal.tombstones])
    return removal


def _leftovers() -> list[str]:
    """Revisions that are gone and recorded as deleted in the custody log, whose directories are still there: a purge or a
    delete was stopped between its commit and the unlink."""
    with get_session() as db:
        recorded = set(db.scalars(select(AuditEvent.analysis_id).where(
            AuditEvent.action.in_(("retention_purge", "deletion_requested")), AuditEvent.analysis_id.is_not(None)).distinct()))
        found = {aid for aid in recorded if Path(aid).name == aid and any(d.exists() for d in _directories(aid))}
        live = {aid for ids in _chunks(found) for aid in db.scalars(select(Analysis.id).where(Analysis.id.in_(ids)))}
    return sorted(found - live)


def _unnamed() -> tuple:
    """The conditions of a media record that nothing names: no analysis (by its media id or by its hash), no case item
    and no intake record."""
    return (~select(Analysis.id).where(or_(Analysis.media_id == MediaObject.id, Analysis.sha256 == MediaObject.sha256)).exists(),
            ~select(CaseMediaRow.media_id).where(CaseMediaRow.media_id == MediaObject.id).exists(),
            ~select(IntakeRecordRow.id).where(IntakeRecordRow.media_id == MediaObject.id).exists())


def unrecorded_media(db: Session, now: datetime, only: str | None = None) -> list[str]:
    """N-19e (Postgres QA X2-01): the media records that a failed or interrupted intake left: nothing names them, and they
    were first seen longer ago than the sweep's grace period (`GC_GRACE_S`: an intake makes the record a moment before
    the unit that names it). No command removed such a record: the purge works by revision chains, and `store gc` leaves
    an object that a media record names. `now`: the clock's time, not the date a purge decides its chains for: the grace
    period is of an intake under way. `only`: ask of one record."""
    found = select(MediaObject.id).where(MediaObject.first_seen_at < now - timedelta(seconds=GC_GRACE_S), *_unnamed())
    if only is not None:
        found = found.where(MediaObject.id == only)
    return list(db.scalars(found.order_by(MediaObject.id)))


def _remove_unrecorded(media_id: str, now: datetime) -> str | None:
    """Remove one such record with the artefact row of its original, in a transaction of its own; returns the hash of its
    object, or None when the record stays. It is asked again inside the transaction. Where the foreign key refuses the
    delete, an intake has named the record meanwhile (on PostgreSQL the delete waits for that intake's unit and is then
    refused): the record stays, and that is no failure of the purge. No custody event names such a record, so none is
    written."""
    with get_session() as db:
        _write_lock(db)
        if media_id not in unrecorded_media(db, now, media_id):
            return None
        media = db.get(MediaObject, media_id)
        if media is None:
            return None
        sha256, original = media.sha256, media.storage_artefact_id
        try:
            db.delete(media)
            db.flush()
            if original is not None and db.scalar(select(func.count()).select_from(MediaObject).where(
                    MediaObject.storage_artefact_id == original)) == 0:
                db.execute(delete(Artefact).where(Artefact.id == original))
            db.commit()
        except IntegrityError:
            db.rollback()
            log.info("retention.unrecorded_media_named", media_id=media_id)
            return None
    return sha256


def purge(*, dry_run: bool = False, now: datetime | None = None, retention_days: int | None = None,
          store: ObjectStore | None = None) -> PurgeReport:
    """Apply the D-55 rule to every revision chain. Each purged revision leaves one `retention_purge` event, committed with
    the delete; with `dry_run` nothing changes and the report holds the decisions."""
    now = now or utcnow()
    with get_session() as db:
        chains = _chains(db, now, retention_days)
    report = PurgeReport(chains=chains)
    candidates: set[str] = set()
    if not dry_run:
        for analysis_id in _leftovers():
            for directory in _directories(analysis_id):
                shutil.rmtree(directory, ignore_errors=True)
            report.leftovers += 1
        if report.leftovers:
            log.warning("retention.leftovers_removed", revisions=report.leftovers)
    for chain in chains:
        if not chain.verdict.purge:
            continue
        if dry_run:
            report.purged_chains += 1
            report.purged_revisions += len(chain.revisions)
            continue
        removal = None
        try:
            for attempt in range(_TAIL_TRIES):
                try:
                    removal = _purge_chain(chain, now, retention_days)
                    break
                except audit.TailMoved:
                    log.info("retention.tail_moved", root_id=chain.root_id, attempt=attempt + 1)
            else:
                report.failed[chain.root_id] = audit.TailMoved.__name__
                log.error("retention.chain_failed", root_id=chain.root_id, error="the custody log kept being extended by another process")
                continue
        except SQLAlchemyError as e:  # this chain stays whole (the transaction is rolled back); the others are still decided
            report.failed[chain.root_id] = type(e).__name__
            log.error("retention.chain_failed", root_id=chain.root_id, error=str(e))
            continue
        if removal is None:
            report.changed.append(chain.root_id)
            log.warning("retention.chain_changed", root_id=chain.root_id)
            continue
        report.purged_chains += 1
        report.purged_revisions += len(chain.revisions)
        for t in removal.tombstones:
            for directory in _directories(t.analysis_id):
                shutil.rmtree(directory, ignore_errors=True)
        candidates |= removal.candidates
        report.cases_removed += len(removal.cases_removed)
        report.media_removed += len(removal.media_removed)
        log.info("retention.purged", root_id=chain.root_id, revisions=len(chain.revisions), rule=chain.verdict.rule)
    # N-19e (X2-01): what a failed or interrupted intake left, each record in a transaction of its own
    clock = utcnow()
    with get_session() as db:
        unrecorded = unrecorded_media(db, clock)
    if dry_run:
        report.unrecorded_media = len(unrecorded)
    for media_id in unrecorded if not dry_run else ():
        sha256 = _remove_unrecorded(media_id, clock)
        if sha256 is not None:
            candidates.add(sha256)
            report.unrecorded_media += 1
    if report.unrecorded_media and not dry_run:
        log.info("retention.unrecorded_media_removed", records=report.unrecorded_media)
    if not dry_run:
        report.released = release(candidates, store=store)
    return report
