"""Whether an instance runs on a data directory (BUILD_SPEC N-12a; code re-check R3-01).

The API and the worker hold a shared lock on `data/instance.lock` while they run (`hold`; the kernel drops it when the
process stops or dies, so a killed instance leaves no stale lock). `backup restore` takes the lock exclusively, without
waiting (`exclusive`): it never puts a database under a running instance, and no instance starts on a directory while a
restore writes into it. `flock` is advisory and local; like the database (D-73), the data directory is on a local disk.
"""
from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from app.core.config import make_data_dir, settings

LOCK_FILE = "instance.lock"


class InstanceRunning(RuntimeError):
    """The API or a worker runs on this data directory."""


class RestoreInProgress(RuntimeError):
    """A restore is writing into this data directory."""


def _open(data_dir: Path) -> int:
    make_data_dir(data_dir)
    return os.open(data_dir / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)


def hold(data_dir: Path | None = None) -> int:
    """Mark this process as running on the data directory until `release` (or its end). Raises `RestoreInProgress`."""
    data_dir = Path(data_dir or settings.data_dir)
    fd = _open(data_dir)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RestoreInProgress(f"A restore is writing into {data_dir}: start once `backup restore` has finished.") from None
    return fd


def release(fd: int | None) -> None:
    if fd is not None:
        with suppress(OSError):
            os.close(fd)


@contextmanager
def exclusive(data_dir: Path) -> Iterator[None]:
    """Hold the directory against every instance for the time of a restore. Raises `InstanceRunning` at once when the API or
    a worker runs on it."""
    fd = _open(Path(data_dir))
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise InstanceRunning(f"An instance runs on {data_dir} (the API or a worker holds {LOCK_FILE}): stop it, then "
                                  "restore.") from None
        yield
    finally:
        os.close(fd)
