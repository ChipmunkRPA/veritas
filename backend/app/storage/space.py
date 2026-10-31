"""Free space of the data volume (BUILD_SPEC N-13; ops review OPS-19).

A full evidence volume gave bare 500s to uploads and a worker that failed at every poll. Now:

- **The floor.** `VERITAS_MIN_FREE_BYTES` (1 GiB; 0 switches the checks off). Below it intake answers 507 before it reads
  the upload (`require`), readiness is 503 (`app.api.probes`), and the worker claims no job (`app.api.jobs.build_worker`):
  what is queued stays queued and runs once space is freed.
- **A write that still meets a full volume** (an upload larger than what is left, a report, the database's own file) is
  recognised by `is_full` and answered 507 instead of 500 (`app.api.v2.install_problem_handlers`).
"""
from __future__ import annotations

import errno
import shutil
import tempfile
from pathlib import Path

from app.core.config import settings

FULL = "The server's evidence storage is full, so nothing was stored. Ask the administrator to free space, then try again."
FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})


class StorageLow(RuntimeError):
    """The data volume is below the floor. `free` and `floor` are bytes."""

    def __init__(self, free: int, floor: int):
        super().__init__(f"{free} bytes free on the data volume, below the floor of {floor}")
        self.free, self.floor = free, floor


def free_bytes(path: Path | None = None) -> int:
    return shutil.disk_usage(path or settings.data_dir).free


def low(path: Path | None = None) -> StorageLow | None:
    """The shortfall when the volume of the data directory is below the floor; None above it, with the floor at 0, and when
    the volume cannot be asked (the write that follows then says what is wrong)."""
    floor = settings.min_free_bytes
    if floor <= 0:
        return None
    try:
        free = free_bytes(path)
    except OSError:
        return None
    return StorageLow(free, floor) if free < floor else None


def require(path: Path | None = None) -> None:
    """Raise StorageLow below the floor."""
    shortfall = low(path)
    if shortfall is not None:
        raise shortfall


def writable(path: Path | None = None) -> None:
    """Write and remove one small file in the data directory; raises OSError when that fails."""
    with tempfile.TemporaryFile(dir=path or settings.data_dir, prefix=".ready-") as probe:
        probe.write(b"veritas")
        probe.flush()


def is_full(error: BaseException) -> bool:
    """The error, or one it was raised from, says that a volume or the database has no room left: the operating system's
    "no space left" and "quota exceeded", SQLite's "database or disk is full", Postgres' class 53 `disk_full`."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in FULL_ERRNOS:
            return True
        if getattr(current, "sqlstate", None) == "53100" or "database or disk is full" in str(current):
            return True
        current = getattr(current, "orig", None) or current.__cause__ or current.__context__
    return False
