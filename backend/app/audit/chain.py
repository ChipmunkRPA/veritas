"""Hash-chained, append-only audit log (ARCHITECTURE §4.8; PRD CC-01, CC-02; BUILD_SPEC C-3).

- **One global chain.** Each event hashes its predecessor:
  `event_hash = sha256(prev_hash || canonical_json({seq, ts, actor_id, action, case_id, media_id, analysis_id, report_id,
  object_ids, before_hash, after_hash, details}))`, with 64 zeros before the first event. Per-case views are filtered
  projections. Verification always walks the whole chain, so an edited, deleted or reordered row is found at its seq.
- **Append** (`record`) runs in its own short transaction, right after the action it describes has committed. `seq` is the
  tail seq + 1 and is part of the hashed body, so two appenders cannot both extend the same tail: the loser hits the
  primary key and retries on the new tail. This holds across processes, on SQLite and on Postgres. Callers append only
  when they hold no open transaction (SQLite would block the append behind their own read lock).
- **Append with the action** (`commit_with`) adds the events to the caller's transaction and commits it, under the same
  lock and the same sequence check: the action and its events are stored together or not at all. A deletion uses it,
  because a row that is gone cannot be given its event later (the retention purge, N-1a). So does every action whose
  record is the point and cannot be re-created afterwards (N-9): a conclusion, a sign-off, a withdrawal, a case update, a
  member change, a note or annotation, a new revision. Those run in `writing()`, which takes the append lock before the
  transaction's first statement (the order `record` uses), and are repeated on a moved tail (`repeated`).
- **Append-only.** There is no update or delete code path, and migration 0007 adds triggers that abort UPDATE and DELETE.
- **Re-verification** (`verify_original`, CC-01): analysis start, report generation, export and original downloads re-hash
  the original first and record `hash_verified` or `hash_mismatch`; a mismatch blocks the action.
- **Unauthenticated instance** (D-09): events by `local-examiner` carry `details.unauthenticated_instance = true`.
- **Limits.** For `record`, a crash between an action's commit and its event loses the event. Removing the newest events leaves a valid,
  shorter chain; an exported chain head (evidence packages, RP-05) anchors it.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.canonical import canonical_json
from app.core.config import settings
from app.core.logging import get_logger
from app.storage.db import Analysis, AuditEvent, CaseMediaRow, MediaObject, get_session, utcnow
from app.storage.files import safe_filename
from app.storage.objects import (
    CHUNK,
    InvalidObjectId,
    ObjectIntegrityError,
    ObjectNotFound,
    ObjectStore,
    default_store,
)

log = get_logger(__name__)

GENESIS = "0" * 64
ACTIONS = frozenset({
    "intake", "original_viewed", "original_downloaded", "hash_verified", "hash_mismatch", "analysis_started",
    "analysis_finished", "analysis_cancel_requested", "rerun", "manifest_recorded", "annotation_created", "annotation_edited", "annotation_deleted",
    "note_added", "status_changed", "conclusion_recorded", "signoff", "report_generated", "package_exported", "redaction",
    "retention_purge", "deletion_requested", "permission_changed", "login", "login_failed", "policy_changed",
    "fuser_loaded", "webhook_configured",
    "image_evidence_generated", "video_evidence_generated", "audio_evidence_generated", "document_evidence_generated",
    "fuser_review_recorded", "fuser_review_withdrawn",  # N-9, D-57: the review of a served fuser artefact (`fusion.review_log`)
    "case_created", "case_updated",  # PRD CM-01 "case CRUD with audit events"; not in the §4.8 list
    "signing_key_adopted",  # N-17a: the key an earlier build derived from the API key, written to its file (`keyguard.adopt_derived`)
})
SYSTEM_ACTOR, ANONYMOUS_ACTOR = "system", "anonymous"
UNAUTHENTICATED_ACTORS = frozenset({"local-examiner"})  # app.api.auth.LOCAL_ID (D-09)
EXPORT_FIELDS = ("seq", "ts", "actor_id", "action", "case_id", "media_id", "analysis_id", "report_id", "object_ids",
                 "before_hash", "after_hash", "details", "prev_hash", "event_hash")
BODY_FIELDS = EXPORT_FIELDS[:-2]
TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"  # the canonical form (app.core.canonical)
_append_lock = threading.RLock()  # re-entered by `commit_with` inside `writing`
_writing = threading.local()  # depth of `writing` in this thread

Sessions = Callable[[], Session]


class OriginalIntegrityError(RuntimeError):
    """The original no longer hashes to its recorded sha256, or its file is missing. The action it guarded is blocked."""

    def __init__(self, sha256: str, *, missing: bool):
        self.sha256, self.missing = sha256, missing
        self.problem = "is missing" if missing else "no longer matches its recorded hash"
        super().__init__(f"the original {sha256[:12]}… {self.problem}")


@dataclass(frozen=True)
class Verification:
    """`seq` and `head` name the last event that verified, the event before a break included: where a later walk continues
    (`verify_chain(after=…)`) and what a report snapshot records. Zero and the genesis hash when no event verified."""
    valid: bool
    first_broken_seq: int | None
    checked: int
    reason: str | None = None
    seq: int = field(default=0, compare=False)
    head: str = field(default=GENESIS, compare=False)


def event_hash(prev_hash: str, body: Mapping[str, Any]) -> str:
    return hashlib.sha256(prev_hash.encode("ascii") + canonical_json({name: body[name] for name in BODY_FIELDS})).hexdigest()


def exported(row: Any) -> dict[str, Any]:
    """The export form of an event: JSON types only, `ts` in the canonical form (hashes identically to the stored datetime)."""
    ts = row.ts if row.ts.tzinfo is not None else row.ts.replace(tzinfo=UTC)
    event = {name: getattr(row, name) for name in EXPORT_FIELDS}
    event.update(seq=int(row.seq), ts=ts.astimezone(UTC).strftime(TS_FORMAT), object_ids=list(row.object_ids or []),
                 details=dict(row.details or {}))
    return event


# --- append ------------------------------------------------------------------------------------------------------------

class TailMoved(RuntimeError):
    """Another process extended the chain between the tail read and the insert of `commit_with`. The caller's transaction
    has been rolled back; the caller repeats its unit."""


def _add(db: Session, action: str, *, actor_id: str, case_id: str | None = None, media_id: str | None = None,
         analysis_id: str | None = None, report_id: str | None = None, object_ids: Sequence[str] = (),
         before_hash: str | None = None, after_hash: str | None = None, details: Mapping[str, Any] | None = None) -> int:
    """Add one event on the present tail to the session, not committed; returns its seq. The caller holds `_append_lock`."""
    if action not in ACTIONS:
        raise ValueError(f"unknown audit action {action!r}")
    info = dict(details or {})
    if actor_id in UNAUTHENTICATED_ACTORS:
        info.setdefault("unauthenticated_instance", True)
    if analysis_id is not None and (case_id is None or media_id is None):
        analysis = db.get(Analysis, analysis_id)
        if analysis is not None:
            case_id, media_id = case_id or analysis.case_id, media_id or analysis.media_id
    seq, prev = _tail(db)
    body = {"seq": seq + 1, "ts": utcnow(), "actor_id": actor_id, "action": action, "case_id": case_id,
            "media_id": media_id, "analysis_id": analysis_id, "report_id": report_id,
            "object_ids": list(object_ids), "before_hash": before_hash, "after_hash": after_hash, "details": info}
    db.add(AuditEvent(**body, prev_hash=prev, event_hash=event_hash(prev, body)))
    return seq + 1


def _tail(db: Session) -> tuple[int, str]:
    """The seq and hash of the newest event, as this session sees it (its own unflushed events included)."""
    tail = db.execute(select(AuditEvent.seq, AuditEvent.event_hash).order_by(AuditEvent.seq.desc()).limit(1)).first()
    return (int(tail[0]), str(tail[1])) if tail is not None else (0, GENESIS)


def record(action: str, *, actor_id: str, case_id: str | None = None, media_id: str | None = None,
           analysis_id: str | None = None, report_id: str | None = None, object_ids: Sequence[str] = (),
           before_hash: str | None = None, after_hash: str | None = None, details: Mapping[str, Any] | None = None,
           sessions: Sessions | None = None) -> int:
    """Append one event; returns its seq. With `analysis_id`, a missing case or media id is taken from the analysis."""
    if action not in ACTIONS:
        raise ValueError(f"unknown audit action {action!r}")
    if getattr(_writing, "depth", 0):  # its own transaction would wait for the write lock this thread holds (SQLite)
        raise RuntimeError("audit.record inside audit.writing: pass the event to commit_with")
    sessions = sessions or get_session
    with _append_lock:
        for _ in range(50):
            with sessions() as db:
                seq = _add(db, action, actor_id=actor_id, case_id=case_id, media_id=media_id, analysis_id=analysis_id,
                           report_id=report_id, object_ids=object_ids, before_hash=before_hash, after_hash=after_hash, details=details)
                try:
                    db.commit()
                except IntegrityError:  # another process appended to the same tail first
                    db.rollback()
                    continue
            return seq
    raise RuntimeError("The audit log could not be appended to: the chain tail kept moving")


def recorded_since(action: str, *, actor_id: str, analysis_id: str, seconds: float, sessions: Sessions | None = None) -> bool:
    """Whether an event of this action by this actor on this analysis was appended within the last `seconds` (N-19a: one
    view of an original is read in several requests, and is one event)."""
    cutoff = utcnow() - timedelta(seconds=seconds)
    with (sessions or get_session)() as db:
        return db.scalar(select(AuditEvent.seq).where(AuditEvent.ts >= cutoff, AuditEvent.action == action, AuditEvent.actor_id == actor_id,
                                                      AuditEvent.analysis_id == analysis_id).limit(1)) is not None


def commit_with(db: Session, events: Sequence[Mapping[str, Any]], *,
                before_commit: Callable[[list[int]], None] | None = None) -> list[int]:
    """Append events in the caller's transaction and commit it: what the transaction did and its events are stored together
    or not at all. Each event is the arguments of `record` (`action`, `actor_id`, …). Returns the seqs. `before_commit` gets
    the seqs inside the transaction (a row that stores the seq of its event, N-9).

    The chain's lock is held from the tail read to the commit, and the seq primary key is the check across processes: when
    another process extended the tail first, the transaction is rolled back and `TailMoved` is raised. Any other error leaves
    the transaction to the caller, who closes the session without committing."""
    db.flush()  # the caller's own statements fail as themselves, not as a moved tail
    with _append_lock:
        seqs: list[int] = []
        try:
            for event in events:
                seqs.append(_add(db, **event))
                db.flush()
        except IntegrityError as e:
            db.rollback()
            raise TailMoved("the audit chain was extended by another process during the append") from e
        if before_commit is not None:
            before_commit(seqs)
        db.commit()
    return seqs


@contextmanager
def writing(sessions: Sessions | None = None) -> Iterator[Session]:
    """A session for one write transaction that ends in `commit_with` (N-9).

    The chain's append lock is taken before the transaction's first statement, which is the order of `record` (lock, then
    write): inside one process an action that appends in its transaction and one that appends after its commit never wait
    for each other in opposite orders. The transaction itself is an ordinary one (no write lock before its first write), so
    the guards against a concurrent writer stay the conditional updates the callers already make (`review.hold_unlocked`,
    `lock_revision`, the unique indexes). Leaving the block without `commit_with` rolls back. `record` must not be called
    inside."""
    with _append_lock:
        _writing.depth = getattr(_writing, "depth", 0) + 1
        try:
            with (sessions or get_session)() as db:
                yield db
        finally:
            _writing.depth -= 1


def repeated[T](unit: Callable[[], T], attempts: int = 5) -> T:
    """Run a unit that ends in `commit_with`, again when another process extended the chain first (`TailMoved`: the unit's
    transaction was rolled back, so it starts from what is stored now)."""
    for _ in range(attempts - 1):
        try:
            return unit()
        except TailMoved:
            continue
    return unit()


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _object_status(store: ObjectStore, sha256: str | None) -> str:
    """verified, missing or mismatch for a store object named by its recorded sha256 (re-hashed)."""
    try:
        store.verify(sha256 or "")
    except ObjectNotFound:
        return "missing"
    except (ObjectIntegrityError, InvalidObjectId):
        return "missing" if not sha256 else "mismatch"
    return "verified"


CUSTODY_VERIFY = "custody_verify"


def _mismatch_recorded(media_id: str | None, case_id: str | None, found: str | None, sessions: Sessions) -> bool:
    """Whether the newest integrity event of this original in this case already is the mismatch a custody verification
    found now (the same bytes, or the same absence)."""
    with sessions() as db:
        last = db.scalars(select(AuditEvent).where(AuditEvent.media_id == media_id, AuditEvent.case_id == case_id,
                                                   AuditEvent.action.in_(("hash_verified", "hash_mismatch")))
                          .order_by(AuditEvent.seq.desc()).limit(1)).first()
        return (last is not None and last.action == "hash_mismatch" and last.after_hash == found
                and (last.details or {}).get("purpose") == CUSTODY_VERIFY)


def verify_original(purpose: str, *, actor_id: str, analysis_id: str | None = None, media_id: str | None = None,
                    case_id: str | None = None, sessions: Sessions | None = None, store: ObjectStore | None = None) -> str:
    """Re-hash the original of an analysis or a media object before an action (CC-01) and record `hash_verified` or
    `hash_mismatch`. The original is the store object once intake or ingest put it there, else the v0.3 upload file.

    Returns the sha256. Raises LookupError for unknown ids and OriginalIntegrityError (after recording the mismatch)."""
    sessions, store = sessions or get_session, store or default_store()
    with sessions() as db:
        analysis = db.get(Analysis, analysis_id) if analysis_id is not None else None
        wanted = media_id or (analysis.media_id if analysis is not None else None)
        media = db.get(MediaObject, wanted) if wanted is not None else None
        if media is not None and media.storage_artefact_id is not None:
            expected, path = media.sha256, store.path_for(media.sha256)
        elif analysis is not None:
            expected, path = analysis.sha256, settings.uploads_dir / analysis.id / safe_filename(analysis.filename)
        elif media is not None:
            expected, path = media.sha256, None
        else:
            raise LookupError("no analysis or media object with this id")
        media_id = media.id if media is not None else None
    found = _sha256_file(path)
    ok = found == expected
    if purpose == CUSTODY_VERIFY:
        # N-10 (SEC-13): the custody verification is a read that every viewer of the case can repeat. An original that
        # verifies appends nothing; one that does not is recorded once, not on every call
        if ok:
            return expected
        if _mismatch_recorded(media_id, case_id, found, sessions):
            log.error("audit.original_mismatch", purpose=purpose, analysis_id=analysis_id, media_id=media_id, expected=expected,
                      found=found)
            raise OriginalIntegrityError(expected, missing=found is None)
    record("hash_verified" if ok else "hash_mismatch", actor_id=actor_id, analysis_id=analysis_id, media_id=media_id,
           case_id=case_id, before_hash=expected, after_hash=found,
           details={"purpose": purpose, **({"missing": True} if found is None else {})}, sessions=sessions)
    if not ok:
        log.error("audit.original_mismatch", purpose=purpose, analysis_id=analysis_id, media_id=media_id, expected=expected,
                  found=found)
        raise OriginalIntegrityError(expected, missing=found is None)
    return expected


def record_report(analysis_id: str, path: Path, *, actor_id: str, sessions: Sessions | None = None, number: str | None = None) -> int:
    """`report_generated` with the sha256 of the report file, which custody verification re-hashes later. `number` is what the
    file prints in place of the verdict's own number (N-10b: the score of a fallback result, or that none is shown)."""
    return record("report_generated", actor_id=actor_id, analysis_id=analysis_id, object_ids=[f"reports/{analysis_id}/{path.name}"],
                  after_hash=_sha256_file(path), details={"format": "v1-pdf", **({"number": number} if number else {})}, sessions=sessions)


# --- read, verify, export ----------------------------------------------------------------------------------------------

def iter_events(sessions: Sessions | None = None, *, case_ids: Iterable[str] | None = None, action: str | None = None,
                after_seq: int = 0, batch: int = 500) -> Iterator[dict[str, Any]]:
    """Events in seq order (export form), read in batches. `case_ids=None` is the whole chain."""
    sessions = sessions or get_session
    cases = None if case_ids is None else sorted(set(case_ids))
    last = after_seq
    while True:
        with sessions() as db:
            query = select(AuditEvent).where(AuditEvent.seq > last)
            if cases is not None:
                query = query.where(AuditEvent.case_id.in_(cases))
            if action is not None:
                query = query.where(AuditEvent.action == action)
            rows = [exported(row) for row in db.scalars(query.order_by(AuditEvent.seq).limit(batch))]
        if not rows:
            return
        yield from rows
        last = rows[-1]["seq"]


def verify_events(events: Iterable[Mapping[str, Any]], *, after: Verification | None = None) -> Verification:
    """Walk events in seq order from the first: a gap, a wrong `prev_hash` or a wrong `event_hash` is reported at that seq.

    With `after`, an earlier walk that found the chain intact, the walk continues it: `events` begin with the event that
    walk ended on, which must still be that event, and only what follows it is hashed. A walk that found a break is
    returned as it is."""
    if after is not None and not after.valid:
        return after
    resumed = after is None or after.seq == 0
    prev, expected, checked = (GENESIS, 1, 0) if after is None else (after.head, after.seq + 1, after.checked)
    for event in events:
        seq = int(event["seq"])
        if not resumed:
            if seq != expected - 1:
                break
            if event["event_hash"] != prev:
                return Verification(False, seq, checked, "the event the verification had reached has changed", expected - 1, prev)
            resumed = True
            continue
        checked += 1
        if seq != expected:
            return Verification(False, seq, checked, f"sequence gap: seq {expected} is missing", expected - 1, prev)
        if event["prev_hash"] != prev:
            return Verification(False, seq, checked, "prev_hash does not name the previous event", expected - 1, prev)
        if event_hash(prev, event) != event["event_hash"]:
            return Verification(False, seq, checked, "event_hash does not match the event's fields", expected - 1, prev)
        prev, expected = str(event["event_hash"]), seq + 1
    if not resumed:
        return Verification(False, expected - 1, checked, "the event the verification had reached is no longer in the log",
                            expected - 1, prev)
    return Verification(True, None, checked, None, expected - 1, prev)


def verify_chain(sessions: Sessions | None = None, *, db: Session | None = None, after: Verification | None = None) -> Verification:
    """Re-hash the whole chain. With `db`, through that session and its connection (a report request: N-9a, Postgres QA X-2);
    else in short sessions of its own. With `after`, only the events since that walk (`verify_events`): a report request
    walks the chain before it takes its locks and, inside them, what was appended meanwhile (N-10a)."""
    start = max(after.seq - 1, 0) if after is not None else 0
    return verify_events(_events_in(db, after_seq=start) if db is not None else iter_events(sessions, after_seq=start), after=after)


def _events_in(db: Session, batch: int = 500, *, after_seq: int = 0) -> Iterator[dict[str, Any]]:
    """Every event after `after_seq` in seq order (export form), read as plain rows through the caller's session: no second
    connection is checked out and nothing is added to the session."""
    columns, last = [getattr(AuditEvent, name) for name in EXPORT_FIELDS], after_seq
    while True:
        rows = [exported(row) for row in db.execute(select(*columns).where(AuditEvent.seq > last).order_by(AuditEvent.seq).limit(batch))]
        if not rows:
            return
        yield from rows
        last = rows[-1]["seq"]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def jsonl_lines(events: Iterable[Mapping[str, Any]]) -> Iterator[str]:
    for event in events:
        yield json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def csv_lines(events: Iterable[Mapping[str, Any]]) -> Iterator[str]:
    """CSV with a header row; `object_ids` and `details` as canonical JSON, absent values empty."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for row in ((EXPORT_FIELDS,), ([canonical_json(e[f]).decode() if f in ("object_ids", "details") else ("" if e[f] is None else e[f])
                                    for f in EXPORT_FIELDS] for e in events)):
        for values in row:
            writer.writerow(values)
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)


REMOVALS = ("deletion_requested", "retention_purge")


def verify_case(case_id: str, *, actor_id: str, sessions: Sessions | None = None, store: ObjectStore | None = None) -> dict[str, Any]:
    """Custody verification of one case: the whole chain; every original of the case, re-hashed; every generated report
    against the sha256 of its `report_generated` event. An original that verifies leaves no event (SEC-13: the route is a
    read, open to every viewer of the case); one that does not is recorded as `hash_mismatch`, once for each finding.

    Each report is verified against what its own event names (N-9a; Postgres QA X-3): a v1 report against the file of its
    event (`reports/<analysis>/report.pdf`, or `report-<hash>.pdf` for a revision with a review status line; the newest event
    of a file counts), a v2 report revision against its stored PDF (R-1). A report whose analysis has a `deletion_requested`
    or `retention_purge` event is listed as `removed`, under that event, and does not make the case invalid: the delete took
    the file, and the log says so."""
    sessions = sessions or get_session
    chain = verify_chain(sessions)
    with sessions() as db:
        items = sorted(db.execute(select(CaseMediaRow.media_id, CaseMediaRow.label).where(CaseMediaRow.case_id == case_id)).tuples(),
                       key=lambda item: (len(item[1]), item[1]))
        events = list(db.scalars(select(AuditEvent).where(AuditEvent.case_id == case_id,
                                                          AuditEvent.action.in_(("report_generated", *REMOVALS)))
                                 .order_by(AuditEvent.seq)))
    removed = {e.analysis_id: (e.action, int(e.seq)) for e in events if e.action in REMOVALS}
    files: dict[tuple[str, str], AuditEvent] = {}  # the newest event of each v1 report file
    revisions = []
    for e in events:
        if e.action != "report_generated":
            continue
        if e.report_id is None:
            named = next(iter(e.object_ids or []), None)
            files[(e.analysis_id, Path(str(named)).name if named else "report.pdf")] = e
        else:
            revisions.append(e)
    originals = []
    for media_id, label in items:
        try:
            sha256 = verify_original(CUSTODY_VERIFY, actor_id=actor_id, media_id=media_id, case_id=case_id, sessions=sessions,
                                     store=store)
            originals.append({"media_id": media_id, "label": label, "sha256": sha256, "status": "verified"})
        except OriginalIntegrityError as e:
            originals.append({"media_id": media_id, "label": label, "sha256": e.sha256, "status": "missing" if e.missing else "mismatch"})

    def check(e: AuditEvent, file: str | None, status: Callable[[], str]) -> dict[str, Any]:
        action, seq = removed.get(e.analysis_id, (None, None))
        return {"analysis_id": e.analysis_id, "report_id": e.report_id, "file": file, "sha256": e.after_hash,
                "status": "removed" if action is not None else status(), "removed_by": action, "removed_seq": seq}

    def file_status(e: AuditEvent, name: str) -> str:
        found = _sha256_file(settings.reports_dir / str(e.analysis_id) / name)
        return "missing" if found is None else "verified" if found == e.after_hash else "mismatch"

    reports = [check(e, name, lambda e=e, name=name: file_status(e, name))
               for (_, name), e in sorted(files.items(), key=lambda item: int(item[1].seq))]
    reports += [check(e, None, lambda e=e: _object_status(store or default_store(), e.after_hash)) for e in revisions]
    valid = chain.valid and all(c["status"] == "verified" for c in originals) and all(c["status"] in ("verified", "removed") for c in reports)
    return {"case_id": case_id, "valid": valid, "chain": chain, "originals": originals, "reports": reports}
