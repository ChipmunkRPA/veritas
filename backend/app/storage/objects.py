"""Content-addressed object store: `data/objects/sha256/aa/bb/<sha>`, written once, read-only (ARCHITECTURE §4.7).

- Writes stream to a temp file in the store, hashing on the way, then `os.link` it to its final path. A link never
  replaces an existing file, so an object is written exactly once. A concurrent writer of the same bytes loses the
  race harmlessly.
- Objects are chmod 0444. Callers never receive a writable handle.
- `verify()` re-hashes the file (streaming). A mismatch raises `ObjectIntegrityError` and never "repairs" silently.
- `delete()` unlinks objects the caller found unreferenced (`app.retention`, D-55). The store itself never decides that.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from app.core.logging import get_logger

log = get_logger(__name__)

CHUNK = 1 << 20
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ObjectStoreError(RuntimeError):
    """Base class for object store failures."""


class InvalidObjectId(ObjectStoreError, ValueError):
    """The object id is not 64 lower-case hex characters (also refuses path traversal)."""


class ObjectNotFound(ObjectStoreError, LookupError):
    """No object with this sha256 exists in the store."""


class ObjectIntegrityError(ObjectStoreError):
    """The stored bytes no longer hash to their name (tampering or disk corruption)."""


class ObjectTooLarge(ObjectStoreError, ValueError):
    """A streamed write exceeded its byte limit (the partial temp file is removed)."""


@dataclass(frozen=True)
class StoredObject:
    sha256: str
    size_bytes: int
    path: Path
    created: bool  # False when the bytes were already in the store (deduplicated)


def check_sha256(sha256: str) -> str:
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise InvalidObjectId("object id must be 64 lower-case hex characters")
    return sha256


LINK_TRIES = 3  # how often `_put` links an object that a sweep sets aside under it (N-19e)


class ObjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def _tmp_dir(self) -> Path:
        return self.root / "tmp"

    def path_for(self, sha256: str) -> Path:
        sha = check_sha256(sha256)
        return self.root / "sha256" / sha[:2] / sha[2:4] / sha

    def exists(self, sha256: str) -> bool:
        return self.path_for(sha256).is_file()

    def put_bytes(self, data: bytes) -> StoredObject:
        return self._put(iter((data,)), limit=None)

    def put_stream(self, src: BinaryIO, limit: int | None = None) -> StoredObject:
        """Stream a file-like object into the store without holding it in memory."""
        return self._put(iter(lambda: src.read(CHUNK), b""), limit=limit)

    def put_file(self, path: Path) -> StoredObject:
        with open(path, "rb") as f:
            return self.put_stream(f)

    def _put(self, chunks: Iterator[bytes], limit: int | None) -> StoredObject:
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        size = 0
        fd, tmp_name = tempfile.mkstemp(dir=self._tmp_dir, prefix="put-", suffix=".part")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                for chunk in chunks:
                    size += len(chunk)
                    if limit is not None and size > limit:
                        raise ObjectTooLarge(f"object exceeds the {limit} byte limit")
                    h.update(chunk)
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            sha = h.hexdigest()
            final = self.path_for(sha)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(tmp, 0o444)
            for _ in range(LINK_TRIES):
                try:
                    os.link(tmp, final)
                    return StoredObject(sha256=sha, size_bytes=size, path=final, created=True)
                except FileExistsError:
                    try:
                        existing = final.stat().st_size
                    except FileNotFoundError:
                        # N-19e (X2-01): a sweep set the object aside (`delete`) between the link that was refused because
                        # it was there and this look at it: the object is linked again
                        continue
                    if existing != size:  # cheap check on dedupe; a full re-hash is `verify()`
                        raise ObjectIntegrityError(
                            f"stored object {sha[:12]}… has {existing} bytes, expected {size}: the store is corrupt") from None
                    return StoredObject(sha256=sha, size_bytes=size, path=final, created=False)
            raise ObjectNotFound(f"object {sha[:12]}… was removed each time it was written")
        finally:
            tmp.unlink(missing_ok=True)

    def open(self, sha256: str) -> BinaryIO:
        path = self.path_for(sha256)
        try:
            return open(path, "rb")
        except FileNotFoundError:
            raise ObjectNotFound(f"object {sha256[:12]}… is not in the store") from None

    def read_bytes(self, sha256: str) -> bytes:
        with self.open(sha256) as f:
            return f.read()

    def verify(self, sha256: str) -> int:
        """Re-hash an object; returns its size. Raises ObjectNotFound or ObjectIntegrityError."""
        h = hashlib.sha256()
        size = 0
        with self.open(sha256) as f:
            while chunk := f.read(CHUNK):
                size += len(chunk)
                h.update(chunk)
        if h.hexdigest() != sha256:
            log.error("objects.integrity_failure", sha256=sha256)
            raise ObjectIntegrityError(f"object {sha256[:12]}… does not match its hash")
        return size


    def quarantine(self, sha256: str) -> Path | None:
        """Move an object that fails re-hashing out of the addressable store, so correct bytes can be written again.

        The corrupt file is kept under `quarantine/<sha>-<utc stamp>` for inspection and is never deleted. Intact or
        missing objects are left alone (returns None).
        """
        path = self.path_for(sha256)
        try:
            self.verify(sha256)
            return None
        except ObjectNotFound:
            return None
        except ObjectIntegrityError:
            pass
        target_dir = self.root / "quarantine"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{sha256}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        try:
            os.replace(path, target)
        except FileNotFoundError:  # another thread quarantined it first
            return None
        log.error("objects.quarantined", sha256=sha256, target=str(target))
        return target

    # --- deletion (D-55) ---------------------------------------------------------------------------------------------------

    def iter_sha256(self) -> Iterator[str]:
        """The name of every object in the store (unordered)."""
        for path in (self.root / "sha256").glob("??/??/*"):
            if _SHA256.fullmatch(path.name) and path.is_file():
                yield path.name

    def delete(self, sha256s: Iterable[str], confirm: Callable[[set[str]], set[str]] | None = None) -> dict[str, int]:
        """Unlink objects; returns `{sha256: bytes}` of those removed. A name that is not in the store is skipped.

        The store does not know what references an object: the caller decides. A writer may store the same bytes again between
        that decision and the unlink, so deletion has two steps. Each object is first moved aside (`trash/<sha256>`); from then
        on a writer of the same bytes creates the object afresh. Then `confirm(moved)` says which of them may still go, and
        the others are put back. An object written again meanwhile is left as written."""
        names = [check_sha256(sha) for sha in dict.fromkeys(sha256s)]
        trash = self.root / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        moved: dict[str, tuple[Path, int]] = {}
        for sha in names:
            path, aside = self.path_for(sha), trash / sha
            try:
                size = path.stat().st_size
                os.replace(path, aside)
            except FileNotFoundError:
                continue
            moved[sha] = (aside, size)
        try:
            allowed = set(moved) if confirm is None else set(confirm(set(moved)))
        except BaseException:
            for sha, (aside, _) in moved.items():
                self._put_back(sha, aside)
            raise
        freed: dict[str, int] = {}
        for sha, (aside, size) in moved.items():
            if sha in allowed:
                aside.unlink(missing_ok=True)
                freed[sha] = size
            else:
                self._put_back(sha, aside)
        return freed

    def _put_back(self, sha256: str, aside: Path) -> None:
        final = self.path_for(sha256)
        final.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError):  # the same bytes were written again while the object was set aside
            os.link(aside, final)
        aside.unlink(missing_ok=True)

    def recover(self) -> int:
        """Put back the objects a crash left set aside between the two steps of `delete`; returns how many. Whether they
        may go is decided again by the next sweep."""
        trash = self.root / "trash"
        found = [p for p in trash.iterdir() if p.is_file() and _SHA256.fullmatch(p.name)] if trash.is_dir() else []
        for aside in found:
            self._put_back(aside.name, aside)
        return len(found)


def default_store() -> ObjectStore:
    from app.core.config import settings

    return ObjectStore(settings.objects_dir)
