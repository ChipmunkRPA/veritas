"""Batch submission (ARCHITECTURE §4.7 `batches`, §5.2; PRD API-01; BUILD_SPEC X-1).

- **Request.** One of: the files of a multipart request (copied into `data/batches/<id>/incoming/`), one zip archive (kept as
  `data/batches/<id>/archive.zip`, its members listed), or a `local_path` inside `VERITAS_BATCH_LOCAL_ROOTS` (listed, nothing
  copied). The batch row and one `batch_items` row per file are committed, then the `batch_expand` job takes the items in.
- **Expansion** (`expand`): each item goes through the same intake as `POST /cases/{id}/items?analyze=true` (SHA-256 and MD5,
  write-once original, intake record, custody event) and its analysis is queued at batch priority. An item that cannot be taken in
  (unrecognised type, empty, too large, refused by the zip guard) is `rejected` with the reason and the batch continues. Staged
  copies are deleted once taken in.
- **Zip guard.** Before zipfile reads the archive, the End Of Central Directory record (ZIP64 when present) and the central
  directory headers are counted: an archive listing more than `ZIP_ENTRY_FACTOR × batch_max_items` entries, directories and
  metadata included, is refused. Members are read into bounded spools, never extracted to paths. Directories, `__MACOSX/` and
  dot-files are skipped; at most `batch_max_items` members and `batch_max_bytes` declared uncompressed bytes per archive; an
  encrypted member refuses the batch; a member larger than the largest kind limit, with a compression ratio above
  `batch_zip_max_ratio`, or a nested archive is rejected; a member that inflates past its declared size fails its read (zipfile
  stops at the declared size and checks the CRC), and any member that cannot be read (corrupt data, unsupported method) is rejected.
- **local_path.** Resolved with realpath and accepted only inside an allow-listed root; regular files below it, sorted, dot-files and
  symlinks skipped; each file is resolved again when it is opened.
- **Status** is computed on read: an item is `staged`, `rejected`, or its analysis status (queued, running, done, failed,
  cancelled; `deleted` once the analysis is gone). The batch is `expanding` while items are staged, `running` while an analysis is
  queued or running, else `done`; the first check that finds it done stamps `finished_at` and returns the `batch.completed` event.
"""
from __future__ import annotations

import shutil
import tempfile
import traceback
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.audit import chain as audit
from app.cases import intake, review
from app.core.config import settings
from app.core.logging import get_logger
from app.engine.queue import PRIORITY_BATCH
from app.storage import files, space
from app.storage.db import Analysis, BatchItemRow, BatchRow, get_session, utcnow

log = get_logger(__name__)
CHUNK = 1024 * 1024
SOURCES = ("upload", "archive", "local_path")
ARCHIVE_SUFFIXES = (".zip", ".7z", ".rar", ".tar", ".tgz", ".gz", ".bz2", ".xz")
CSV_FIELDS = ("seq", "source_name", "kind", "size_bytes", "sha256", "status", "analysis_id", "decision_code", "probability_ai", "error")
ZIP_ENTRY_FACTOR = 4  # central-directory entries allowed per batch item: room for the directories and metadata that are skipped
EOCD, ZIP64_LOCATOR, ZIP64_EOCD, CENTRAL_HEADER = b"PK\x05\x06", b"PK\x06\x07", b"PK\x06\x06", b"PK\x01\x02"
EOCD_SEARCH = 22 + 65535  # the fixed record plus the longest archive comment


class BatchRefused(ValueError):
    """The whole batch is refused; `status` is the HTTP status of the answer."""

    def __init__(self, status: int, title: str, detail: str):
        super().__init__(detail)
        self.status, self.title, self.detail = status, title, detail


class ItemRefused(ValueError):
    """One item cannot be taken in; the batch continues."""


@dataclass(frozen=True)
class ItemView:
    seq: int
    source_name: str
    kind: str | None
    size_bytes: int | None
    sha256: str | None
    status: str
    analysis_id: str | None
    decision_code: str | None
    driving_hypothesis: str | None  # N-19m (D3-01): with the code it tells a row that a provenance record decides
    probability_ai: float | None
    error: str | None
    deciding_record: str | None = None  # N-19p (D6-01): the trusted record that decides, which the code and hypothesis do not tell


@dataclass
class BatchView:
    batch: BatchRow
    items: list[ItemView]
    counts: dict[str, int] = field(default_factory=dict)
    state: str = "expanding"


def batch_dir(batch_id: str) -> Path:
    return settings.data_dir / "batches" / batch_id


def max_item_bytes() -> int:
    return max(intake.KIND_LIMITS.values())


def detect_kind(head: bytes, filename: str) -> str | None:
    """image, audio, video or document from the first bytes (and the extension for text documents), else None."""
    mime, ext = files.sniff(head, filename)
    if mime in files.IMAGE_MIMES:
        return "image"
    if files.sniff_audio(head, filename) is not None:
        return "audio"
    if mime in files.VIDEO_MIMES:
        return "video"
    if ext in files.DOC_EXTS and (mime is None or mime in ("application/pdf", "application/zip")
                                  or mime.startswith("application/vnd.openxmlformats")):
        return "document"
    return None


def _skipped(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return not parts or parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts)


def _check_count(n: int) -> None:
    if n == 0:
        raise BatchRefused(422, "Empty batch", "The batch holds no files.")
    if n > settings.batch_max_items:
        raise BatchRefused(413, "Batch too large", f"A batch holds at most {settings.batch_max_items} files; this one holds {n}.")


def _bounded_copy(src: BinaryIO, dst: Path, limit: int) -> tuple[int, bool]:
    """Copy at most `limit` bytes; (bytes written, True when the source is longer)."""
    written = 0
    with open(dst, "wb") as out:
        while chunk := src.read(CHUNK):
            if written + len(chunk) > limit:
                return written, True
            out.write(chunk)
            written += len(chunk)
    return written, False


def _item(batch_id: str, seq: int, name: str, *, status: str = "staged", size: int | None = None, error: str | None = None) -> BatchItemRow:
    return BatchItemRow(id=str(uuid.uuid4()), batch_id=batch_id, seq=seq, source_name=name.replace("\x00", "")[:1024], size_bytes=size, status=status,
                        error=error, created_at=utcnow())


def _commit(batch_id: str, principal_id: str, case_id: str | None, profile: str | None, source: str, source_ref: str | None,
            items: Sequence[BatchItemRow]) -> str:
    for attempt in range(3):
        created = None
        try:
            with audit.writing() as db:  # N-9: the batch, its new case and the `case_created` event are one transaction
                if case_id is None:
                    created = intake.new_case(db, principal_id, title=f"Batch of {len(items)} files, {utcnow():%d %b %Y %H:%M} UTC")
                db.add(BatchRow(id=batch_id, case_id=case_id or created.id, created_by=principal_id, profile=profile, source=source,
                                source_ref=source_ref, status="expanding", created_at=utcnow()))
                db.flush()  # the batch row before its items: see `storage/db.py` ("Insert order")
                db.add_all(items)
                audit.commit_with(db, [{"action": "case_created", "actor_id": principal_id, "case_id": created.id,
                                        "details": {"reference": created.reference, "title": created.title, "implicit": False,
                                                    "batch_id": batch_id}}] if created is not None else [])
        except (IntegrityError, audit.TailMoved):  # the next case reference was taken, or the custody chain moved: again
            if attempt == 2:
                raise
            continue
        log.info("batches.created", batch_id=batch_id, source=source, items=len(items))
        return batch_id
    raise RuntimeError("unreachable")


def _check_space() -> None:
    """N-13 (OPS-19): a batch is refused in words before its files are copied when the data volume is below its floor."""
    if space.low() is not None:
        raise BatchRefused(507, "Insufficient storage", space.FULL)


def create_from_uploads(uploads: Sequence[tuple[str, BinaryIO]], *, principal_id: str, case_id: str | None, profile: str | None) -> str:
    _check_count(len(uploads))
    _check_space()
    batch_id = str(uuid.uuid4())
    incoming = batch_dir(batch_id) / "incoming"
    incoming.mkdir(parents=True)
    items, total, limit = [], 0, max_item_bytes()
    try:
        for seq, (name, src) in enumerate(uploads):
            size, over = _bounded_copy(src, incoming / f"{seq:05d}", limit)
            total += size
            if total > settings.batch_max_bytes:
                raise BatchRefused(413, "Batch too large", f"A batch holds at most {settings.batch_max_bytes // 1048576} MB.")
            if over:
                (incoming / f"{seq:05d}").unlink(missing_ok=True)
                items.append(_item(batch_id, seq, name, status="rejected", error=f"The file exceeds the largest upload limit ({limit // 1048576} MB)."))
            else:
                items.append(_item(batch_id, seq, name, size=size))
        return _commit(batch_id, principal_id, case_id, profile, "upload", None, items)
    except BaseException:
        shutil.rmtree(batch_dir(batch_id), ignore_errors=True)
        raise


def _members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [info for info in archive.infolist() if not info.is_dir() and not _skipped(info.filename)]


def _u(data: bytes, at: int, size: int) -> int:
    return int.from_bytes(data[at:at + size], "little")


def _zip_entries(path: Path, cap: int) -> int | None:
    """Entries of the zip at `path`, counted before zipfile builds one object per entry; None without an EOCD record.

    The larger of the total the End Of Central Directory record declares (the ZIP64 record's when a locator precedes it) and the
    central-directory headers walked from where zipfile reads them; the walk stops past `cap`, so a false total does not hide entries."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        start = max(0, size - EOCD_SEARCH)
        f.seek(start)
        tail = f.read()
        at = tail.rfind(EOCD)  # the last signature, as zipfile takes it
        if at < 0 or len(tail) - at < 22:
            return None
        eocd = start + at
        declared, cd_size, cd_end = _u(tail, at + 10, 2), _u(tail, at + 12, 4), eocd
        if eocd >= 76:
            f.seek(eocd - 76)
            record = f.read(76)
            if record[56:60] == ZIP64_LOCATOR and record[:4] == ZIP64_EOCD:
                declared, cd_size, cd_end = _u(record, 32, 8), _u(record, 40, 8), eocd - 76
        count, pos = 0, max(0, cd_end - cd_size)
        while pos < cd_end and count <= cap:
            f.seek(pos)
            header = f.read(46)
            if len(header) < 46 or header[:4] != CENTRAL_HEADER:
                break
            pos += 46 + _u(header, 28, 2) + _u(header, 30, 2) + _u(header, 32, 2)  # name, extra field, comment
            count += 1
    return max(declared, count)


def create_from_archive(name: str, src: BinaryIO, *, principal_id: str, case_id: str | None, profile: str | None) -> str:
    _check_space()
    batch_id = str(uuid.uuid4())
    directory = batch_dir(batch_id)
    directory.mkdir(parents=True)
    try:
        _, over = _bounded_copy(src, directory / "archive.zip", settings.batch_max_bytes)
        if over:
            raise BatchRefused(413, "Batch too large", f"The archive exceeds {settings.batch_max_bytes // 1048576} MB.")
        cap = ZIP_ENTRY_FACTOR * settings.batch_max_items
        try:
            entries = _zip_entries(directory / "archive.zip", cap)
            if entries is None:
                raise zipfile.BadZipFile("no end of central directory record")
            if entries > cap:
                raise BatchRefused(413, "Batch too large", f"The archive lists more than {cap} entries (directories and metadata included).")
            with zipfile.ZipFile(directory / "archive.zip") as archive:
                members = _members(archive)
        except (zipfile.BadZipFile, OSError):
            raise BatchRefused(422, "Invalid archive", "The archive is not a readable zip file.") from None
        _check_count(len(members))
        if encrypted := sum(1 for m in members if m.flag_bits & 0x1):
            raise BatchRefused(422, "Invalid archive", f"The archive holds {encrypted} encrypted member(s), which cannot be read.")
        if sum(m.file_size for m in members) > settings.batch_max_bytes:
            raise BatchRefused(413, "Batch too large", f"The archive expands beyond {settings.batch_max_bytes // 1048576} MB (zip-bomb guard).")
        items = [_item(batch_id, seq, m.filename, size=m.file_size) for seq, m in enumerate(members)]
        return _commit(batch_id, principal_id, case_id, profile, "archive", files.received_filename(name), items)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def allowed_root(path: Path) -> Path | None:
    """The allow-listed root that `path` (already resolved) lies inside, else None."""
    for raw in settings.batch_local_roots:
        root = Path(raw).resolve()
        if path == root or root in path.parents:
            return root
    return None


def create_from_local_path(local_path: str, *, principal_id: str, case_id: str | None, profile: str | None) -> str:
    if not settings.batch_local_roots:
        raise BatchRefused(422, "Invalid request", "local_path batches are off: VERITAS_BATCH_LOCAL_ROOTS names no directory.")
    try:
        path = Path(local_path).resolve()
    except (ValueError, OSError, RuntimeError):  # a NUL byte (ValueError), a symlink loop
        raise BatchRefused(422, "Invalid request", "local_path is not a valid path.") from None
    if allowed_root(path) is None:
        raise BatchRefused(422, "Invalid request", "local_path is not inside an allowed directory (VERITAS_BATCH_LOCAL_ROOTS).")
    if path.is_file():
        base, found = path.parent, [path]
    elif path.is_dir():
        base = path
        found = sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink() and not _skipped(p.relative_to(path).as_posix()))
    else:
        raise BatchRefused(422, "Invalid request", "local_path names no file or directory.")
    _check_count(len(found))
    _check_space()
    batch_id = str(uuid.uuid4())
    items = [_item(batch_id, seq, p.relative_to(base).as_posix(), size=p.stat().st_size) for seq, p in enumerate(found)]
    return _commit(batch_id, principal_id, case_id, profile, "local_path", str(base), items)


# --- expansion -----------------------------------------------------------------------------------------------------------------

@contextmanager
def _source(batch: dict[str, Any], seq: int, name: str, archive: zipfile.ZipFile | None,
            members: list[zipfile.ZipInfo]) -> Iterator[BinaryIO]:
    limit = max_item_bytes()
    if batch["source"] == "upload":
        path = batch_dir(batch["id"]) / "incoming" / f"{seq:05d}"
        if not path.is_file():
            raise ItemRefused("The staged copy of this file is missing.")
        with open(path, "rb") as f:
            yield f
    elif batch["source"] == "archive":
        if archive is None or seq >= len(members) or members[seq].filename != name:
            raise ItemRefused("The archive no longer holds this member.")
        member = members[seq]
        if name.lower().endswith(ARCHIVE_SUFFIXES):
            raise ItemRefused("Nested archives are not expanded.")
        if member.file_size > limit:
            raise ItemRefused(f"The member exceeds the largest upload limit ({limit // 1048576} MB).")
        if member.file_size > 64 * 1024 and member.file_size > settings.batch_zip_max_ratio * max(member.compress_size, 1):
            raise ItemRefused(f"The member expands more than {settings.batch_zip_max_ratio} times its compressed size (zip-bomb guard).")
        with tempfile.SpooledTemporaryFile(max_size=8 * CHUNK, dir=batch_dir(batch["id"])) as spool:
            try:
                with archive.open(member) as f:
                    shutil.copyfileobj(f, spool, CHUNK)
            except Exception as e:  # a CRC or zlib / bz2 / lzma error, an encrypted member (RuntimeError), an unsupported method
                log.info("batches.member_unreadable", batch_id=batch["id"], seq=seq, error=type(e).__name__)
                raise ItemRefused("The member cannot be read from the archive (corrupt, encrypted or an unsupported compression method).") from None
            spool.seek(0)
            yield spool  # type: ignore[misc]
    else:
        base = Path(batch["source_ref"] or "")
        path = (base / name).resolve()
        if allowed_root(path) is None or not path.is_file():
            raise ItemRefused("The file is gone or lies outside the allowed directory.")
        with open(path, "rb") as f:
            yield f


def _mark(item_id: str, **values: Any) -> None:
    with get_session() as db:
        db.execute(update(BatchItemRow).where(BatchItemRow.id == item_id).values(**values, updated_at=utcnow()))
        db.commit()


def expand(batch_id: str, *, submit: Callable[[str], None]) -> str:
    """The `batch_expand` job: take in every staged item. Returns the job outcome.

    An item that cannot be taken in, for whatever reason, is rejected with the reason and the others go on. An error outside the
    items (the archive cannot be opened, the database is away) ends the job, and `jobs._handle_batch_expand` marks the batch
    failed."""
    with get_session() as db:
        row = db.get(BatchRow, batch_id)
        if row is None or row.status == "failed":
            return "done"
        batch = {"id": row.id, "source": row.source, "source_ref": row.source_ref, "case_id": row.case_id, "created_by": row.created_by,
                 "profile": row.profile}
        staged = [(i.id, i.seq, i.source_name) for i in db.scalars(select(BatchItemRow).where(
            BatchItemRow.batch_id == batch_id, BatchItemRow.status == "staged").order_by(BatchItemRow.seq))]
    archive: zipfile.ZipFile | None = None
    try:
        if batch["source"] == "archive" and staged:
            archive = zipfile.ZipFile(batch_dir(batch_id) / "archive.zip")
        members = _members(archive) if archive is not None else []
        for item_id, seq, name in staged:
            try:
                with _source(batch, seq, name, archive, members) as src:
                    head = src.read(intake.HEAD_BYTES)
                    src.seek(0)
                    kind = detect_kind(head, name)
                    if kind is None:
                        raise ItemRefused("Unrecognised file type: accepted are images, videos, audio and documents.")
                    outcome = intake.take_in(src, PurePosixPath(name).name, kind, principal_id=batch["created_by"], case_id=batch["case_id"],
                                             acquisition=intake.Acquisition(method="other", notes=f"Batch {batch_id}, item {seq + 1}: {name}"[:4000]),
                                             analyze=True, profile=batch["profile"],
                                             job_priority=PRIORITY_BATCH)  # N-9: queued with the row; `submit` finds that job
            except ItemRefused as e:
                _mark(item_id, status="rejected", error=str(e))
                continue
            except intake.IntakeError as e:
                _mark(item_id, status="rejected", error=e.detail)
                continue
            except Exception as e:  # noqa: BLE001 - one file that cannot be taken in does not stop the others
                # e.g. a value the database refuses. Left to propagate, it ended the job with this item and every later one
                # `staged`, so the batch read `expanding` for ever.
                log.error("batches.item_failed", batch_id=batch_id, seq=seq, error=str(e)[:500], tb=traceback.format_exc()[-1200:])
                _mark(item_id, status="rejected", error=(
                    f"This file could not be taken in: an internal error stopped it ({type(e).__name__}). The other files of the "
                    f"batch were not affected; the server log has the detail under batch {batch_id}, item {seq + 1}."))
                continue
            finally:
                if batch["source"] == "upload":
                    (batch_dir(batch_id) / "incoming" / f"{seq:05d}").unlink(missing_ok=True)
            assert outcome.analysis is not None
            _mark(item_id, status="submitted", kind=kind, analysis_id=outcome.analysis.id, sha256=outcome.media.sha256,
                  size_bytes=outcome.media.size_bytes)
            submit(outcome.analysis.id)
    finally:  # the staged archive and copies go on every exit, including an error that ends the job
        if archive is not None:
            archive.close()
        shutil.rmtree(batch_dir(batch_id), ignore_errors=True)
    with get_session() as db:
        db.execute(update(BatchRow).where(BatchRow.id == batch_id).values(expanded_at=utcnow()))
        db.commit()
    log.info("batches.expanded", batch_id=batch_id, items=len(staged))
    return "done"


def mark_failed(batch_id: str, error: str | None) -> None:
    with get_session() as db:
        db.execute(update(BatchRow).where(BatchRow.id == batch_id, BatchRow.finished_at.is_(None))
                   .values(status="failed", error=error or "The batch job did not finish.", finished_at=utcnow()))
        db.commit()


# --- reads ---------------------------------------------------------------------------------------------------------------------

def view(db: Any, batch: BatchRow) -> BatchView:
    rows = list(db.scalars(select(BatchItemRow).where(BatchItemRow.batch_id == batch.id).order_by(BatchItemRow.seq)))
    ids = [r.analysis_id for r in rows if r.analysis_id]
    analyses = {a.id: a for a in db.scalars(select(Analysis).where(Analysis.id.in_(ids)))} if ids else {}
    records = review.deciding_records(list(analyses.values()))
    items = []
    for r in rows:
        a = analyses.get(r.analysis_id or "")
        status = r.status if r.status in ("staged", "rejected") else (a.status if a is not None else "deleted")
        items.append(ItemView(seq=r.seq, source_name=r.source_name, kind=r.kind, size_bytes=r.size_bytes, sha256=r.sha256, status=status,
                              analysis_id=r.analysis_id, decision_code=a.decision_code if a is not None else None,
                              driving_hypothesis=a.driving_hypothesis if a is not None else None,
                              probability_ai=a.probability_ai if a is not None else None, error=r.error or (a.error if a is not None else None),
                              deciding_record=records.get(r.analysis_id or "")))
    counts = dict(Counter(i.status for i in items))
    state = ("failed" if batch.status == "failed" else "expanding" if "staged" in counts
             else "running" if counts.get("queued") or counts.get("running") else "done")
    return BatchView(batch=batch, items=items, counts=counts, state=state)


def completed_event(batch_id: str) -> dict[str, Any] | None:
    """Stamp `finished_at` the first time the batch is found done; return the `batch.completed` event data then, else None."""
    with get_session() as db:
        batch = db.get(BatchRow, batch_id)
        if batch is None or batch.finished_at is not None or batch.expanded_at is None:
            return None
        current = view(db, batch)
        if current.state != "done":
            return None
        stamped = db.execute(update(BatchRow).where(BatchRow.id == batch_id, BatchRow.finished_at.is_(None))
                             .values(status="done", finished_at=utcnow())).rowcount
        db.commit()
        if stamped != 1:
            return None
        return {"batch_id": batch_id, "case_id": batch.case_id, "items": len(current.items), "counts": current.counts}


def batch_of(analysis_id: str) -> str | None:
    with get_session() as db:
        return db.scalars(select(BatchItemRow.batch_id).where(BatchItemRow.analysis_id == analysis_id)).first()


def csv_cell(value: Any) -> str:
    """A CSV cell that a spreadsheet does not evaluate (a leading = + - @ tab or CR is quoted with an apostrophe)."""
    text = "" if value is None else str(value)
    return f"'{text}" if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text
