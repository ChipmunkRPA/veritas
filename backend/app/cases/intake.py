"""Evidence intake (ARCHITECTURE §4.7 `media_objects`, `intake_records`, §4.8; PRD CC-01, CM-01; UX §5.1; BUILD_SPEC C-2).

One code path for every arrival of media: `POST /api/v2/cases/{id}/items`, the v1 quick analysis (`POST /api/analyze/{kind}`,
`/api/analyze-text`) and the guided demo.
1. **Hash** the spooled upload once (SHA-256 and MD5, streaming) and check its first bytes against the declared kind.
2. **Browser pre-hash.** A `client_sha256` that differs from the server's hash fails with "The file changed in transit
   (hash mismatch)", and nothing is stored.
3. **Custodian-stated hash** (SHA-256 or MD5 hex) → `match`, `mismatch` or `not_provided`. A mismatch is recorded, not blocked.
4. **Original, written once.** The bytes stream into the content-addressed store (0444), which hashes them again; a
   difference from step 1 fails the intake. An object that already exists is re-hashed first, and a corrupt one is
   quarantined and replaced by these verified bytes. There is one media object per SHA-256: the same bytes can arrive
   twice, in one case or in several, and each arrival gets its own intake record.
5. **Case item.** A media object joins a case once, under the next exhibit label (A … Z, AA …). Without a case id a new
   implicit case is created (the quick-analysis flow, CM-01), with reference `UNFILED-<12 hex>`.
6. **Analysis** (optional): a queued analysis in the case. Its upload directory links the stored original, where the v1
   pipeline and the v1 source route read it. The caller queues the job.

Custody audit events (`intake`, `hash_mismatch`) arrive with C-3; `intake_records.audit_event_seq` stays null until then.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.audit import chain as audit
from app.cases.access import NOT_FOUND
from app.core.config import settings
from app.core.logging import get_logger
from app.storage import files, locks, space
from app.storage.artefacts import index_object
from app.storage.db import (
    Analysis,
    CaseMediaRow,
    CaseMemberRow,
    CaseRow,
    IntakeRecordRow,
    MediaObject,
    PrincipalRow,
    get_session,
    utcnow,
)
from app.storage.objects import (
    CHUNK,
    ObjectIntegrityError,
    ObjectNotFound,
    ObjectStore,
    StoredObject,
    default_store,
)

log = get_logger(__name__)

KIND_LIMITS = {"image": settings.max_image_bytes, "video": settings.max_video_bytes, "audio": settings.max_audio_bytes,
               "document": settings.max_document_bytes}
ACQUISITION_METHODS = ("not_stated", "custodian_upload", "url_download", "screen_capture", "email_forward", "quick_upload",
                       "other")
# N-19d (browser QA "Demo tour"): what the guided demo writes into the intake record of a sample it takes in. A repeat demo
# reuses only analyses of such arrivals, and a case that holds nothing else is no work for an examiner.
DEMO_SUPPLIED_BY, DEMO_SOURCE = "Veritas demo", "bundled demo sample"
HASH_MISMATCH = "The file changed in transit (hash mismatch)"
REFERENCE_PREFIX, IMPLICIT_PREFIX = "VX", "UNFILED-"
CLOSED_STATUSES = frozenset({"signed_off", "closed"})
HEAD_BYTES = 8192
_STATED = re.compile(r"(?:(sha256|md5):)?([0-9a-f]{64}|[0-9a-f]{32})")
_upload_slots = threading.BoundedSemaphore(settings.max_upload_concurrency)


class IntakeError(Exception):
    """An intake that cannot proceed. `status` is the HTTP status of the answer."""

    def __init__(self, status: int, title: str, detail: str):
        super().__init__(detail)
        self.status, self.title, self.detail = status, title, detail


@dataclass(frozen=True)
class Acquisition:
    """The acquisition fields of an intake record (CC-01). All optional: case fields never block submission (UX §5.1)."""

    method: str = "not_stated"
    supplied_by: str | None = None
    source: str | None = None  # URL or device
    date_acquired: datetime | None = None
    custodian_stated_hash: str | None = None
    notes: str | None = None
    client_sha256: str | None = None


@dataclass(frozen=True)
class IntakeOutcome:
    case: CaseRow
    media: MediaObject
    item: CaseMediaRow
    intake: IntakeRecordRow
    analysis: Analysis | None
    new_media: bool  # False: these bytes were taken in before and their media object is reused
    new_item: bool  # False: the media object was already an item of this case


@contextmanager
def upload_slot() -> Iterator[None]:
    """At most `max_upload_concurrency` uploads are taken in at once; 503 after waiting 30 s for a slot."""
    if not _upload_slots.acquire(timeout=30):
        raise IntakeError(503, "Too many uploads", "Too many concurrent uploads; retry shortly")
    try:
        yield
    finally:
        _upload_slots.release()


def exhibit_label(index: int) -> str:
    """0 → A, 25 → Z, 26 → AA."""
    label, n = "", index + 1
    while n:
        n, r = divmod(n - 1, 26)
        label = chr(ord("A") + r) + label
    return label


def custodian_check(db: Session, case_id: str | None, media_id: str | None) -> IntakeRecordRow | None:
    """The intake that recorded a custodian hash mismatch if any intake of this media into this case did, else the newest intake
    (L-5; N-3). None when the file was not taken in through a case intake."""
    if case_id is None or media_id is None:
        return None
    rows = list(db.scalars(select(IntakeRecordRow).where(IntakeRecordRow.case_id == case_id, IntakeRecordRow.media_id == media_id)
                           .order_by(IntakeRecordRow.intake_at, IntakeRecordRow.id)))
    if not rows:
        return None
    return next((r for r in reversed(rows) if r.custodian_hash_match == "mismatch"), rows[-1])


def stated_hash(value: str | None) -> tuple[str, str] | None:
    """(algorithm, hex digest) of a custodian-stated hash, or None when none was stated. Raises IntakeError(422)."""
    if value is None or not value.strip():
        return None
    match = _STATED.fullmatch(value.strip().lower())
    algorithm = "md5" if match is None or len(match.group(2)) == 32 else "sha256"
    if match is None or match.group(1) not in (None, algorithm):
        raise IntakeError(422, "Invalid request",
                          "custodian_stated_hash must be a SHA-256 (64 hex characters) or MD5 (32 hex characters) digest.")
    return algorithm, match.group(2)


def next_reference(db: Session, now: datetime) -> str:
    """The next free `VX-<year>-<nnn>` reference."""
    prefix = f"{REFERENCE_PREFIX}-{now.year}-"
    taken = set(db.scalars(select(CaseRow.reference).where(CaseRow.reference.like(f"{prefix}%"))))
    n = len(taken) + 1
    while f"{prefix}{n:03d}" in taken:
        n += 1
    return f"{prefix}{n:03d}"


def next_label(db: Session, case_id: str) -> str:
    taken = set(db.scalars(select(CaseMediaRow.label).where(CaseMediaRow.case_id == case_id)))
    index = len(taken)
    while exhibit_label(index) in taken:
        index += 1
    return exhibit_label(index)


def new_case(db: Session, principal_id: str, *, title: str, implicit: bool = False, reference: str | None = None,
             **fields: object) -> CaseRow:
    """A new open case. Its creator becomes owner when the creator is a principal row (`bootstrap-admin` and `local-examiner`
    are administrators and need no membership). Flushes; the caller commits and handles IntegrityError on the reference."""
    now, case_id = utcnow(), str(uuid.uuid4())
    if reference is None and not implicit:
        # N-9a (Postgres QA PG-12): one allocation of a year's next reference at a time, instead of three tries on the
        # unique index (about half of twenty concurrent creations were refused on Postgres)
        locks.serialise(db, "case-reference", now.year)
    if reference is None:
        reference = f"{IMPLICIT_PREFIX}{case_id.replace('-', '')[:12].upper()}" if implicit else next_reference(db, now)
    fields.setdefault("classification", "internal")
    row = CaseRow(id=case_id, reference=reference, title=title, status="open", implicit=implicit, created_by=principal_id,
                  created_at=now, updated_at=now, **fields)
    db.add(row)
    db.flush()  # the case row before its first member: see `storage/db.py` ("Insert order")
    if db.get(PrincipalRow, principal_id) is not None:
        db.add(CaseMemberRow(case_id=case_id, principal_id=principal_id, role="owner", added_by=principal_id, added_at=now))
    db.flush()
    return row


def _open_case(db: Session, case_id: str) -> CaseRow:
    row = db.get(CaseRow, case_id)
    if row is None:
        raise IntakeError(404, *NOT_FOUND["case"])
    if row.status in CLOSED_STATUSES:
        raise IntakeError(409, "Case closed", f"The case is {row.status.replace('_', ' ')}; items cannot be added to it.")
    return row


def _digest(src: BinaryIO, limit: int) -> tuple[bytes, str, str, int]:
    """(first bytes, SHA-256, MD5, size) of the upload, streaming. Raises IntakeError(413) past `limit`."""
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    head, size = b"", 0
    src.seek(0)
    while chunk := src.read(CHUNK):
        size += len(chunk)
        if size > limit:
            raise IntakeError(413, "Upload too large", f"File exceeds the {limit // 1048576} MB limit")
        sha.update(chunk)
        md5.update(chunk)
        if len(head) < HEAD_BYTES:
            head += chunk[:HEAD_BYTES - len(head)]
    return head, sha.hexdigest(), md5.hexdigest(), size


ATTEMPTS = 3  # how often the unit of an intake runs, and how often an original that goes away under a write is written


def _not_recorded() -> IntakeError:
    return IntakeError(500, "Intake failed", "The case item could not be recorded; retry the upload.")


def _write_original(store: ObjectStore, src: BinaryIO) -> StoredObject:
    """Write the upload to the store, read from its start, and verify an object that was there already. N-19e (X2-01): a
    sweep can set the object aside between the write that found it there and its verification (`ObjectNotFound`, which
    nothing handled); it is then written again, a bounded number of times."""
    for _ in range(ATTEMPTS):
        src.seek(0)
        try:
            obj = store.put_stream(src)
            if not obj.created:
                store.verify(obj.sha256)
            return obj
        except ObjectNotFound:
            continue
    raise _not_recorded()


def _store_original(store: ObjectStore, src: BinaryIO, sha256: str, size: int, *, actor_id: str, case_id: str | None) -> StoredObject:
    try:
        obj = _write_original(store, src)
    except ObjectIntegrityError:
        quarantined = store.quarantine(sha256)
        # N-9: a record of a check that failed, appended at once; no row of this intake exists yet
        audit.record("hash_mismatch", actor_id=actor_id, case_id=case_id, before_hash=sha256,
                     details={"purpose": "intake_store", "stored_object": "quarantined" if quarantined is not None else "unverifiable"})
        if quarantined is None:
            raise IntakeError(500, "Evidence integrity failure", "The evidence store could not verify the stored original.") from None
        obj = _write_original(store, src)  # the verified bytes of this upload replace the quarantined object
    if obj.sha256 != sha256 or obj.size_bytes != size:
        raise IntakeError(500, "Evidence integrity failure", "The upload changed while it was stored; nothing was recorded.")
    return obj


def _media(sha256: str, md5: str, size: int, mime: str, filename: str) -> tuple[MediaObject, bool]:
    """The media object of these bytes, created on their first arrival, with the original indexed. (row, created)."""
    for _ in range(3):
        with get_session() as db:
            row = db.scalars(select(MediaObject).where(MediaObject.sha256 == sha256)).first()
            created = row is None
            if row is None:
                row = MediaObject(id=str(uuid.uuid4()), sha256=sha256, md5=md5, size_bytes=size, mime_sniffed=mime,
                                  original_filename=filename, first_seen_at=utcnow())
            if row.storage_artefact_id is None:  # a new object, or a v0.3 media row whose original never reached the store
                row.storage_artefact_id = index_object(db, sha256, size, media_type=row.mime_sniffed or mime, role="original",
                                                       created_by="intake").id
            row.md5 = row.md5 or md5
            db.add(row)
            try:
                db.commit()
            except IntegrityError:  # a concurrent intake of the same bytes created the media object first
                db.rollback()
                continue
            return row, created
    raise IntakeError(500, "Intake failed", "The media object could not be recorded; retry the upload.")


def _original_missing(error: FileNotFoundError, original: Path) -> bool:
    """Whether the file that a link or a copy of an upload did not find is the stored original (N-19e)."""
    return error.filename is not None and Path(os.fsdecode(error.filename)) == original


def linkable(original: Path) -> bool:
    """The uploads directory and the stored original are on one file system, so the upload of an analysis is a hard link."""
    try:
        return os.stat(original).st_dev == os.stat(settings.uploads_dir).st_dev
    except OSError:
        return True  # the link itself says what is wrong


def prepare_upload(analysis_id: str, filename: str, original: Path) -> Path | None:
    """Before the unit that makes an analysis row (N-10a; code re-verification, the reviewer's notes): when uploads and objects
    are on different file systems the upload is a copy of the original, and the copy is made here, outside the unit's locks.
    Returns the directory it made, which the caller removes when its unit stores nothing; None when the upload will be a
    hard link, which the unit makes itself (`_link_upload`)."""
    if linkable(original):
        return None
    directory = settings.uploads_dir / analysis_id
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, directory / filename)
    return directory


def _link_upload(analysis_id: str, filename: str, original: Path) -> Path:
    directory = settings.uploads_dir / analysis_id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.is_file():  # copied before the unit (`prepare_upload`)
        return directory
    try:
        os.link(original, target)  # read-only inode shared with the store object
    except OSError:  # the upload directory is on another file system
        shutil.copyfile(original, target)
    return directory


def take_in(src: BinaryIO, filename: str, kind: str, *, principal_id: str, case_id: str | None = None,
            acquisition: Acquisition | None = None, analyze: bool = False, profile: str | None = None,
            store: ObjectStore | None = None, job_priority: int | None = None) -> IntakeOutcome:
    """Take one seekable upload in (module docstring). `case_id=None` creates the implicit case of a quick analysis.

    The case item, the intake record, the analysis row and the `intake` (and `case_created`) custody events are one
    transaction (N-9): an intake exists with its events or not at all. With `job_priority` the analysis job is queued in
    that transaction too, so no queued analysis is left without a job; the caller's later submit finds that job.

    Raises IntakeError: 400 empty, 404 unknown case, 409 closed case, 413 too large, 415 unsupported type, 422 hash mismatch
    or an invalid stated hash, 500 store integrity, 507 the data volume is below its free-space floor (`storage.space`)."""
    acquisition = acquisition or Acquisition()
    if kind not in KIND_LIMITS:
        raise ValueError(f"unknown kind {kind!r}")
    if acquisition.method not in ACQUISITION_METHODS:
        raise ValueError(f"unknown acquisition method {acquisition.method!r}")
    if space.low() is not None:  # N-13 (OPS-19): said before the upload is read, in words; nothing is stored
        raise IntakeError(507, "Insufficient storage", space.FULL)
    store = store or default_store()
    stated = stated_hash(acquisition.custodian_stated_hash)
    if case_id is not None:
        with get_session() as db:
            _open_case(db, case_id)
    head, sha256, md5, size = _digest(src, KIND_LIMITS[kind])
    if size == 0:
        raise IntakeError(400, "Empty upload", "Empty upload")
    try:
        mime, ext = files.validate_kind(kind, head, filename, size=size)
        if ext == ".docx":
            files.check_docx_safe(src)
    except ValueError as e:
        raise IntakeError(415, "Unsupported media type", str(e)) from e
    client = acquisition.client_sha256.lower() if acquisition.client_sha256 else None
    if client is not None and client != sha256:
        log.warning("intake.client_hash_mismatch", client_sha256=client, sha256=sha256, case_id=case_id)
        # N-9: appended before the refusal and outside a transaction on purpose: nothing is stored, and a failed append
        # fails the request, which is refused either way
        audit.record("hash_mismatch", actor_id=principal_id, case_id=case_id, before_hash=client, after_hash=sha256,
                     details={"purpose": "intake", "filename": files.received_filename(filename)})
        raise IntakeError(422, "Hash mismatch", HASH_MISMATCH)
    custody = "not_provided" if stated is None else "match" if stated[1] == (sha256 if stated[0] == "sha256" else md5) else "mismatch"
    # N-19 (browser QA P1-9; design review DS-15): the record holds the name as it was received (`shown`); the sanitised
    # name (`name`) is the name of the upload on disk and nothing else. Every reader of the upload derives the path the
    # same way, from the stored name (`files.safe_filename(analysis.filename)`)
    shown = files.received_filename(filename)
    if kind == "document" and not shown.lower().endswith(ext):
        shown = files.received_filename(f"{Path(shown).stem}{ext}")  # a document is read by its extension
    name = files.safe_filename(shown)

    obj = _store_original(store, src, sha256, size, actor_id=principal_id, case_id=case_id)
    media, new_media = _media(sha256, md5, size, mime, shown)
    made = {media.id} if new_media else set()  # the media records this intake made
    if not store.exists(sha256):
        # a retention sweep unlinked the object between the write above and the media record (D-55): a record this intake
        # made keeps it from here on, so it is written once more. A record it found does not: see the unit below
        obj = _store_original(store, src, sha256, size, actor_id=principal_id, case_id=case_id)
    now = utcnow()
    acquired = acquisition.date_acquired.astimezone(UTC) if acquisition.date_acquired is not None else None
    for attempt in range(ATTEMPTS):
        analysis_id = str(uuid.uuid4()) if analyze else None
        # the directory of this attempt's upload, which only this attempt makes (`prepare_upload`, `_link_upload`): an
        # attempt that stores nothing leaves none (N-19e: the one `_link_upload` made before a link that failed stayed)
        upload_dir: Path | None = settings.uploads_dir / analysis_id if analysis_id is not None else None
        try:
            if analysis_id is not None:  # N-10a: an upload that has to be a copy is copied before the unit takes its locks
                prepare_upload(analysis_id, name, obj.path)
            with audit.writing() as db:
                if case_id is not None:  # N-9a (PG-12): one allocation of a case's next exhibit label at a time
                    locks.serialise(db, "case-label", case_id)
                case = _open_case(db, case_id) if case_id is not None else new_case(db, principal_id, title=shown, implicit=True)
                item = db.get(CaseMediaRow, (case.id, media.id))
                new_item = item is None
                if item is None:
                    item = CaseMediaRow(case_id=case.id, media_id=media.id, label=next_label(db, case.id), added_by=principal_id,
                                        added_at=now)
                    db.add(item)
                record = IntakeRecordRow(id=str(uuid.uuid4()), media_id=media.id, case_id=case.id, supplied_by=acquisition.supplied_by,
                                         acquisition_method=acquisition.method, source=acquisition.source, date_acquired=acquired,
                                         custodian_stated_hash=stated[1] if stated is not None else None,
                                         custodian_hash_match=custody, client_sha256=client, notes=acquisition.notes,
                                         intake_principal_id=principal_id, intake_at=now)
                db.add(record)
                analysis = None
                if analyze:
                    analysis = Analysis(id=analysis_id, kind=kind, filename=shown, content_type=mime, size_bytes=size,
                                        sha256=sha256, status="queued", case_id=case.id, media_id=media.id,
                                        submitted_by=principal_id, profile=profile)
                    db.add(analysis)
                    _link_upload(analysis.id, name, obj.path)
                case.updated_at = now
                db.flush()  # a taken label or reference fails here, as itself
                if analysis is not None and job_priority is not None:
                    from app.api import jobs

                    jobs.get_queue().enqueue_in(db.connection(), "analysis", analysis.id, priority=job_priority)
                events = [{"action": "case_created", "actor_id": principal_id, "case_id": case.id,
                           "details": {"reference": case.reference, "title": case.title, "implicit": True}}] if case_id is None else []
                events.append({
                    "action": "intake", "actor_id": principal_id, "case_id": case.id, "media_id": media.id,
                    "analysis_id": analysis.id if analysis is not None else None, "object_ids": [record.id], "after_hash": sha256,
                    "details": {"label": item.label, "filename": shown, "acquisition_method": acquisition.method,
                                "custodian_hash_match": custody, "new_media": new_media, "new_item": new_item}})
                audit.commit_with(db, events, before_commit=lambda seqs, record=record: setattr(record, "audit_event_seq", seqs[-1]))
        except (IntegrityError, audit.TailMoved, FileNotFoundError) as e:
            # a concurrent intake took the next label, the implicit reference collided, or another process extended the
            # custody chain first: nothing was stored, so the unit runs again.
            # N-19e (X2-01): or a purge removed the last analysis of the same bytes between the check above and this unit.
            # The media record this intake found is then gone (the foreign key of the item and of the intake record
            # refuses the unit), or the release that followed set the original aside before the upload was linked or
            # copied (the missing file; nothing handled it). Nothing of the attempt stays, and the intake makes sure of the
            # media record and then of the stored original, as its first pass does. In that order: a release asks again
            # after it has set an object aside, and then finds the record
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            if isinstance(e, FileNotFoundError) and not _original_missing(e, obj.path):
                raise  # another file is missing: as before
            if attempt == ATTEMPTS - 1:
                raise _not_recorded() from None
            media, again = _media(sha256, md5, size, mime, shown)
            if again:
                made.add(media.id)
            new_media = media.id in made  # true when this intake made the record it ends with
            if not store.exists(sha256):
                obj = _store_original(store, src, sha256, size, actor_id=principal_id, case_id=case_id)
            continue
        except BaseException:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            raise
        log.info("intake.recorded", case_id=case.id, media_id=media.id, label=item.label, new_media=new_media, new_item=new_item,
                 custodian_hash_match=custody, analysis_id=analysis.id if analysis is not None else None, by=principal_id)
        return IntakeOutcome(case=case, media=media, item=item, intake=record, analysis=analysis, new_media=new_media,
                             new_item=new_item)
    raise AssertionError("unreachable")
