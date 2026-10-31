"""One transaction at a time for a check or an allocation that reads before it writes (BUILD_SPEC N-9a; Postgres QA X-4, PG-12).

The last owner of a case, the next case reference of a year and the next exhibit label of a case are each decided from what
the transaction reads: "is another owner left", "the highest number so far". Two transactions that read at the same moment
both decide from the same rows. Retrying on the unique index refused work under real concurrency (three tries: about half of
twenty case creations answered 409 on Postgres), and nothing at all caught two owners removing each other.

`serialise(db, name, key)` orders them: the transaction that calls it second waits until the first has ended, and then reads
what the first wrote.

- **Postgres:** an advisory transaction lock on (`name`, `key`), as `engine/queue.py` takes for a job's enqueue. It is released
  when the transaction commits or rolls back. Call it before the reads it protects; under READ COMMITTED each statement
  after it sees what the earlier holder committed.
- **SQLite:** the database's write lock (`BEGIN IMMEDIATE`, as `retention._write_lock`), whatever the name and key: SQLite has
  one writer. It must be taken before the transaction's first write, and is best taken before its first read.
"""
from __future__ import annotations

import hashlib

from sqlalchemy import func, select
from sqlalchemy.orm import Session


def lock_key(name: str, key: str) -> int:
    """A stable 64-bit identity for an advisory lock; the length prefixes keep the pair unambiguous."""
    parts = (name.encode("utf-8"), key.encode("utf-8"))
    payload = b"veritas.serialise\0" + b"".join(len(part).to_bytes(4, "big") + part for part in parts)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=True)


def serialise(db: Session, name: str, key: object) -> None:
    """Hold the lock named (`name`, `key`) until the session's transaction ends."""
    conn = db.connection()
    if conn.dialect.name == "postgresql":
        db.execute(select(func.pg_advisory_xact_lock(lock_key(name, str(key)))))
    elif conn.dialect.name == "sqlite":
        conn.exec_driver_sql("BEGIN IMMEDIATE")
