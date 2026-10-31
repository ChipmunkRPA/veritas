"""Small standalone worker readiness probe: standard library and psutil only, no DB or model imports."""
from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

import psutil

SCHEMA = "veritas/worker-health/1"
MAX_BYTES = 65_536


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _read_record(path: Path) -> tuple[dict, tuple[int, int]]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("health path is not a regular file")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("health file is too large")
    record = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(record, dict):
        raise ValueError("health file is not an object")
    return record, (info.st_dev, info.st_ino)


def _owner(record: dict) -> tuple[int, float]:
    pid, created = record.get("pid"), record.get("process_create_time")
    if (record.get("schema") != SCHEMA or type(pid) is not int or pid <= 0
            or not _finite(created) or created <= 0):
        raise ValueError("unrecognized health file owner")
    return pid, created


class HealthFile:
    """Atomic snapshots owned by the standalone main loop; no background thread."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.pid = os.getpid()
        self.process_create_time = psutil.Process(self.pid).create_time()
        self._owned_identity: tuple[int, int] | None = None

    def prepare(self) -> None:
        """Remove a recognized dead process marker; refuse unrelated files or another live publisher."""
        try:
            record, identity = _read_record(self.path)
        except FileNotFoundError:
            return
        pid, created = _owner(record)
        try:
            process = psutil.Process(pid)
            if (process.create_time() == created and process.is_running()
                    and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)):
                raise ValueError("health file belongs to a live worker")
        except psutil.NoSuchProcess:
            pass
        current = self.path.lstat()
        if (current.st_dev, current.st_ino) != identity:
            raise ValueError("health file changed during startup")
        self.path.unlink()

    def _owns_current(self) -> bool:
        try:
            record, identity = _read_record(self.path)
            return identity == self._owned_identity and _owner(record) == (self.pid, self.process_create_time)
        except (OSError, ValueError, TypeError, OverflowError, RecursionError):
            return False

    def clear(self) -> None:
        if self._owned_identity is not None and self._owns_current():
            self.path.unlink(missing_ok=True)
        self._owned_identity = None

    def write(self, snapshot: dict) -> None:
        payload = {**snapshot, "schema": SCHEMA, "pid": self.pid,
                   "process_create_time": self.process_create_time, "generated_at": time.time()}
        encoded = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(encoded) > MAX_BYTES:
            raise ValueError("worker health snapshot is too large")
        if self._owned_identity is not None and not self._owns_current():
            raise ValueError("worker health file ownership changed")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=f".{self.path.name}.", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                info = os.fstat(stream.fileno())
            if self._owned_identity is None:
                os.link(temporary, self.path)
            else:
                os.replace(temporary, self.path)
            self._owned_identity = (info.st_dev, info.st_ino)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def check_health_file(path: str | Path, *, max_age_s: float = 45.0) -> tuple[bool, str]:
    """Fail closed unless a fresh snapshot reports all slots ready and its original process still lives."""
    if not _finite(max_age_s) or max_age_s <= 0:
        return False, "max-age must be a positive finite number"
    try:
        record, _ = _read_record(Path(path))
        if record.get("schema") != SCHEMA:
            return False, "unknown health schema"
        if record.get("ready") is not True or record.get("stopping") is not False:
            return False, "worker is not ready"
        if (record.get("device_startup_checked") is not True or not isinstance(record.get("device"), str)
                or not record["device"].strip()):
            return False, "configured device startup has not passed"
        pid, created, generated = (record.get(key) for key in ("pid", "process_create_time", "generated_at"))
        if type(pid) is not int or pid <= 0:
            return False, "invalid worker PID"
        if not _finite(created) or not _finite(generated) or not 0 < created <= generated:
            return False, "invalid process or snapshot timestamp"
        age = time.time() - generated
        if not _finite(age) or not 0 <= age <= max_age_s:
            return False, "health snapshot is stale or future-dated"
        configured, alive = record.get("configured_threads"), record.get("alive_threads")
        if type(configured) is not int or not 1 <= configured <= 1024 or type(alive) is not int or alive != configured:
            return False, "worker thread counts are not ready"
        slots = record.get("slots")
        if not isinstance(slots, list) or len(slots) != configured:
            return False, "invalid worker slot count"
        seen = set()
        for slot in slots:
            if not isinstance(slot, dict) or type(slot.get("slot")) is not int:
                return False, "invalid worker slot"
            index, last_age = slot["slot"], slot.get("last_db_success_age_s")
            if index in seen or not 0 <= index < configured or slot.get("alive") is not True:
                return False, "worker slot is missing, duplicated or dead"
            seen.add(index)
            if not _finite(last_age) or last_age < 0 or last_age + age > max_age_s:
                return False, "worker slot has no recent database success"
        process = psutil.Process(pid)
        if process.create_time() != created:
            return False, "worker PID was reused"
        if not process.is_running() or process.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            return False, "worker process is not running"
        return True, "worker is ready"
    except (OSError, ValueError, TypeError, OverflowError, RecursionError, psutil.Error):
        return False, "health file or worker process is unavailable or malformed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.engine.worker_health", description=__doc__)
    parser.add_argument("--file", required=True, help="standalone worker health snapshot")
    parser.add_argument("--max-age", type=float, default=45.0, help="maximum snapshot and DB-success age in seconds")
    args = parser.parse_args(argv)
    healthy, reason = check_health_file(args.file, max_age_s=args.max_age)
    print(reason)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
