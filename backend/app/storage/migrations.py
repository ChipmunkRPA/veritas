"""Schema migrations (Alembic), applied by `init_db()` (ARCHITECTURE §4.7, §4.9).

- An unversioned database that already has the v0.3 `analyses` table is stamped at the baseline, then upgraded.
- An empty database is upgraded to head.
- **One process migrates at a time** (BUILD_SPEC N-14; ops review OPS-08, Postgres QA PG-11, code re-check R3-02). The API,
  the worker and the command line all start through `init_db()`, and two of them starting together both read "nothing is
  there" and both created it: one died ("table analyses already exists", "index … already exists", on Postgres a duplicate
  key in `pg_type`). `migration` now takes an exclusive lock as the first statement of its transaction, before anything
  in it reads the schema: `BEGIN IMMEDIATE` on SQLite, with a busy timeout of `MIGRATION_BUSY_TIMEOUT_MS` instead of the
  30 seconds of ordinary work (the process that waits must outlast a migration); on Postgres an advisory transaction lock
  whose key holds the schema name. `upgrade_to_head` reads the revision first without the lock (a database at the head
  needs none, and a start does not wait behind another process's write), and again under it, so the process that waited
  finds the head and changes nothing. On SQLite the schema change is thereby one transaction: a migration that fails
  leaves the database as it found it.
- **With two Postgres roles the migration is the owner's step** (N-14; Postgres QA PG-13, SEC-17). The service connects as
  a role that owns nothing, so that it cannot drop the triggers of the custody log; such a role cannot change the schema
  either. A start as that role on a database that is not at the head stopped with the first statement that Postgres
  refused ("must be owner of table analyses", or "permission denied for schema public" on an empty database).
  `upgrade_to_head` now raises `MigrationOwnerError`, whose one sentence names the step: `python -m app.cli db setup` as
  the owner (`pg_grants.setup`). Postgres rolls the refused migration back whole.
- SQLite enforces foreign keys on every connection (`db._enforce_sqlite_foreign_keys`, D-72) except while a migration runs:
  a table rebuild drops the old table, and with enforcement on that drop would delete or refuse the rows that reference it.
  `migration` switches the enforcement off before its transaction and on again after it, as SQLite prescribes, and
  `upgrade_to_head` ends with `PRAGMA foreign_key_check`.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine, inspect
from sqlalchemy.exc import DBAPIError

from app.core.config import BACKEND_DIR
from app.core.logging import get_logger

log = get_logger(__name__)

ALEMBIC_DIR = BACKEND_DIR / "alembic"
BASELINE_REVISION = "0001"
# the v0.3 `analyses` columns mirrored by 0001_baseline
BASELINE_COLUMNS = frozenset({
    "id", "kind", "filename", "content_type", "size_bytes", "sha256", "status", "attempts", "created_at",
    "started_at", "finished_at", "error", "probability_ai", "band", "result_json", "report_path",
})


class LegacySchemaError(RuntimeError):
    """An unversioned database whose `analyses` table does not match the 0001 baseline."""


class MigrationOwnerError(RuntimeError):
    """The database is not at the head, and the role this process connects as may not change its schema."""


INSUFFICIENT_PRIVILEGE = "42501"  # Postgres: "must be owner of …", "permission denied for …"


def alembic_config(connection: Connection | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(ALEMBIC_DIR))
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    if head is None:
        raise RuntimeError(f"no Alembic revisions found in {ALEMBIC_DIR}")
    return head


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def _foreign_keys(conn: Connection, *, on: bool) -> None:
    """Switch SQLite's enforcement for this connection. The pragma does nothing inside a transaction, so the result is read back."""
    conn.exec_driver_sql(f"PRAGMA foreign_keys={'ON' if on else 'OFF'}")
    enforced = bool(conn.exec_driver_sql("PRAGMA foreign_keys").scalar())
    conn.commit()
    if enforced is not on:
        raise RuntimeError("SQLite did not change its foreign key enforcement: the connection has a transaction open")


MIGRATION_BUSY_TIMEOUT_MS = 300_000  # how long a start waits for another process that is migrating the SQLite database
# the key of the advisory lock is made on the server from this prefix and the schema the migration runs in
PG_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended('veritas.migrations:' || current_schema(), 0))"


def _lock(conn: Connection) -> None:
    """The first statement of a schema change: no other process is inside one until this transaction ends."""
    if conn.dialect.name == "sqlite":
        conn.exec_driver_sql(f"PRAGMA busy_timeout={int(MIGRATION_BUSY_TIMEOUT_MS)}")
        conn.exec_driver_sql("BEGIN IMMEDIATE")
    elif conn.dialect.name == "postgresql":
        conn.exec_driver_sql(PG_LOCK)


@contextmanager
def migration(engine: Engine) -> Iterator[Connection]:
    """A connection inside the one transaction of a schema change, which holds the migration lock from its first statement
    (module docstring); on SQLite its foreign keys are not enforced meanwhile."""
    from app.storage.db import SQLITE_BUSY_TIMEOUT_MS

    with engine.connect() as conn:
        sqlite = conn.dialect.name == "sqlite"
        if sqlite:
            _foreign_keys(conn, on=False)
        try:
            with conn.begin():
                _lock(conn)
                yield conn
        finally:
            if sqlite and not conn.closed and not conn.invalidated:
                conn.exec_driver_sql(f"PRAGMA busy_timeout={int(SQLITE_BUSY_TIMEOUT_MS)}")  # the connection goes back to the pool
                conn.commit()
                _foreign_keys(conn, on=True)


def foreign_key_violations(conn: Connection) -> dict[str, int]:
    """SQLite: the rows whose foreign key names no parent row, counted by table (`PRAGMA foreign_key_check`)."""
    counts = Counter(row[0] for row in conn.exec_driver_sql("PRAGMA foreign_key_check"))
    return dict(sorted(counts.items()))


def _revision(conn: Connection) -> str | None:
    """The revision the database is at; None without a version table (an empty or an unversioned database)."""
    if "alembic_version" not in set(inspect(conn).get_table_names()):
        return None
    return MigrationContext.configure(conn).get_current_revision()


def _warn_violations(conn: Connection) -> None:
    if conn.dialect.name == "sqlite":
        violations = foreign_key_violations(conn)
        if violations:
            log.warning("db.foreign_key_violations", rows=sum(violations.values()), tables=violations)


def _owner_step(engine: Engine, head: str, error: DBAPIError) -> str:
    """The sentence of `MigrationOwnerError`: what was found, what Postgres refused, and the step that cures it."""
    reason = (str(getattr(error, "orig", None) or error).strip().splitlines() or ["not permitted"])[0]
    role, found = "this role", None
    try:
        with engine.connect() as conn:
            role = str(conn.exec_driver_sql("SELECT current_user").scalar_one())
            found = _revision(conn)
    except Exception:  # noqa: BLE001 (the sentence is still worth more than this second error)
        pass
    named = role if role != "this role" else "<the application role>"
    return (f"the database is at revision {found or 'none'}, this build needs {head}, and the role {role} may not change the "
            f"schema ({reason}). With two roles the migration is the owner's step: run `python -m app.cli db setup --app-role "
            f"{named} --database-url <the owner's URL>` as the owner (it migrates, then renews the grants of the application "
            "role), then start the service again. With docker-compose.postgres.yml, `docker compose … up` runs that step "
            "before the API starts. Runbook: \"Postgres: two roles\".")


def upgrade_to_head(engine: Engine) -> str | None:
    """Bring the database to the head revision; returns the resulting revision. A role that Postgres does not let change
    the schema gets `MigrationOwnerError` (module docstring)."""
    head = head_revision()
    try:
        return _upgrade_to_head(engine, head)
    except DBAPIError as e:
        if getattr(e.orig, "sqlstate", None) != INSUFFICIENT_PRIVILEGE:
            raise
        raise MigrationOwnerError(_owner_step(engine, head, e)) from None


def _upgrade_to_head(engine: Engine, head: str) -> str | None:
    """`upgrade_to_head` without the translation of a refusal.

    A start that finds the database at the head changes nothing and takes no lock: it does not wait behind another
    process's long write. One that finds anything else takes the migration lock and reads the revision again under it, so
    of several processes that start together one migrates and the others find the head (module docstring).

    A SQLite database written before the keys were enforced may hold rows without a parent. They are counted by table in one
    warning at every start and left as they are: the service goes on, and only a new or changed reference is refused."""
    with engine.connect() as conn:
        if _revision(conn) == head:
            _warn_violations(conn)
            return head
    with migration(engine) as conn:  # locked: what follows is read after any other process's migration has ended
        found = _revision(conn)
        cfg = alembic_config(conn)
        if found == head:
            pass  # another process brought it to the head while this one waited: nothing to change
        elif found is None and "analyses" in set(inspect(conn).get_table_names()):
            columns = {c["name"] for c in inspect(conn).get_columns("analyses")}
            missing = sorted(BASELINE_COLUMNS - columns)
            if missing:
                raise LegacySchemaError(
                    f"unversioned database does not match the v0.3 baseline (missing columns: {', '.join(missing)}); "
                    "back it up and migrate it manually before starting Veritas"
                )
            command.stamp(cfg, BASELINE_REVISION)
            log.info("db.legacy_stamped", revision=BASELINE_REVISION)
            command.upgrade(cfg, "head")
        else:
            command.upgrade(cfg, "head")
        _warn_violations(conn)
    return current_revision(engine)


def downgrade(engine: Engine, revision: str = "base") -> str | None:
    with migration(engine) as conn:
        command.downgrade(alembic_config(conn), revision)
    return current_revision(engine)
