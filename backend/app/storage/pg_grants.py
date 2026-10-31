"""Postgres grants for the application role (ARCHITECTURE §4.8 "application role granted INSERT/SELECT only"; code review P2-12;
BUILD_SPEC J-6).

The append-only triggers on `audit_events` protect the chain only while the role the service connects as cannot drop them. So:

- **Owner role:** the migration role owns every table and runs `alembic upgrade`.
- **Application role:** the service connects as a separate role that owns nothing. It gets:
  - custody tables (`APPEND_ONLY`): SELECT and INSERT only, so no UPDATE, DELETE, TRUNCATE or trigger changes;
  - `alembic_version`: SELECT only;
  - every other table: SELECT, INSERT, UPDATE and DELETE (no TRUNCATE, REFERENCES or TRIGGER);
  - the product's own sequences, by name: USAGE and SELECT (N-19g: `ON ALL SEQUENCES IN SCHEMA` was refused where another
    role's sequence stood in the schema, and with it every grant of the transaction).

`python -m app.cli db grants --app-role ROLE` prints the statements. With `--database-url` (an owner connection) and `--apply` it
runs them in one transaction, after refusing an application role that owns a table.

`python -m app.cli db setup --app-role ROLE` is the whole of the owner's step (`setup`; BUILD_SPEC N-14, SEC-17, PG-13): it
migrates to the head, sets the database's default time zone to UTC, creates the role or sets its password again when one is
given, takes CREATE on the schema from PUBLIC and applies the grants. `docker-compose.postgres.yml` runs it at every `up`,
before the API starts, so an upgrade needs no step by hand and a table that a migration added is granted at once.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, text

APPEND_ONLY = frozenset({"audit_events"})
READ_ONLY = frozenset({"alembic_version"})
_ROLE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class GrantError(ValueError):
    """An invalid role name, or an application role that owns tables."""


class NotBroughtAbout(RuntimeError):
    """The step ran, and what it was to bring about does not hold (N-19g, d): the application role may not use the schema,
    may create in it, or reads no table of it. The message is one sentence with the statements for the schema's owner."""


class RoleRefused(RuntimeError):
    """PostgreSQL refused the statement that makes the application role or sets its password (N-19g, a and b). It is the
    step's first statement, so nothing else has been changed. The message is one sentence: PostgreSQL's first line and
    what to do."""


def _ident(name: str) -> str:
    if not _ROLE.fullmatch(name):
        raise GrantError(f"{name!r} is not a plain lower-case Postgres identifier")
    return f'"{name}"'


def _quoted(name: str) -> str:
    """A name as it stands in a statement that is printed for an administrator (any role name, not only a plain one)."""
    return '"' + name.replace('"', '""') + '"'


def application_tables() -> list[str]:
    """The tables of the current models plus Alembic's version table."""
    from app.storage.db import Base

    return sorted({*Base.metadata.tables, *READ_ONLY})


def application_sequences() -> list[str]:
    """The sequences of the current models, by the names Postgres gives them: `<table>_<column>_seq` for the column of a
    table that counts itself (two today, `analysis_events_seq_seq` and `assessments_id_seq`)."""
    return sorted(_sequence_of_table().values())


def _sequence_of_table() -> dict[str, str]:
    from app.storage.db import Base

    return {table.name: f"{table.name}_{table.autoincrement_column.name}_seq" for table in Base.metadata.tables.values()
            if table.autoincrement_column is not None}


def statements(app_role: str, tables: Iterable[str] | None = None, *, schema: str = "public",
               sequences: Iterable[str] | None = None) -> list[str]:
    """The grants of the application role. `tables`, `sequences`: those of an earlier revision; by default the current
    models'. With `tables` and without `sequences`, the sequences of those tables."""
    role, ns = _ident(app_role), _ident(schema)
    out = [f"GRANT USAGE ON SCHEMA {ns} TO {role}"]
    for table in sorted(set(tables) if tables is not None else application_tables()):
        name = f"{ns}.{_ident(table)}"
        out.append(f"REVOKE ALL ON TABLE {name} FROM {role}")
        if table in APPEND_ONLY:
            out.append(f"GRANT SELECT, INSERT ON TABLE {name} TO {role}")
        elif table in READ_ONLY:
            out.append(f"GRANT SELECT ON TABLE {name} TO {role}")
        else:
            out.append(f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {name} TO {role}")
    # N-19g (k): the product's own sequences, as its tables: a sequence of another role in the schema (an administrator's,
    # an extension's) is not the step's business, and `ON ALL SEQUENCES IN SCHEMA` was refused there
    if sequences is None:
        granted = set(tables) if tables is not None else None
        sequences = [s for t, s in _sequence_of_table().items() if granted is None or t in granted]
    for sequence in sorted(set(sequences)):
        out.append(f"GRANT USAGE, SELECT ON SEQUENCE {ns}.{_ident(sequence)} TO {role}")
    return out


# N-19g (j): one owner's step at a time on a database. Two steps at one moment both sent the zone statement, and one
# ended at "duplicate key value violates unique constraint"; the other printed "MIGRATED" though it had migrated nothing
# (it read the revision before it waited). Another key than the migration's lock, which the step takes inside this one.
# An advisory lock is one database's.
STEP_LOCK_KEY = "hashtextextended('veritas.owner-step', 0)"
# the function that the custody log's two append-only triggers execute (migration 0007)
APPEND_ONLY_FUNCTION = "audit_events_append_only()"


@contextmanager
def step_lock(engine: Engine) -> Iterator[None]:
    """Held from before the step's first look to its end, on a connection of its own (a session lock: the step's own
    statements run in several transactions)."""
    with engine.connect() as holder:
        holder.exec_driver_sql(f"SELECT pg_advisory_lock({STEP_LOCK_KEY})")
        holder.commit()
        try:
            yield
        finally:
            holder.exec_driver_sql(f"SELECT pg_advisory_unlock({STEP_LOCK_KEY})")
            holder.commit()


def setup(engine: Engine, app_role: str, *, password: str | None = None, schema: str = "public") -> list[str]:
    """The owner's step, over an owner connection; returns one line for each thing done, in the order migration, zone, role,
    grants. One step runs at a time on a database (`step_lock`). Since N-19g it works in this order:

    1. The refusals, before anything is changed (`GrantError`): SQLite, a role name that is not a plain identifier, the
       connected role itself, a superuser, a role that does not exist when no password is given; a connection without the
       rights of the database's owner or without CREATE on the schema (`refuse_connection`); a table of the product that
       belongs to another role (`refuse_foreign_tables`); an application role that owns a table or is a member of a role
       that owns anything of the instance (`refuse_application_role`).
    2. The application role: created with LOGIN and `password`, or, when it exists, given `password` again (so the value
       the deployment holds is the one that works). Without `password` the role must exist and is left as it is. The
       password goes as a SCRAM-SHA-256 verifier (`role_statement`); a refusal is `RoleRefused`, with nothing else changed.
    3. Migrate to the head (the lock of `migrations.migration` holds: a service that starts beside it waits).
    4. `ALTER DATABASE … SET timezone TO 'UTC'`: the default of every new session. The service pins its own sessions at
       connect (N-4b); a pooler that resets a session would drop that pin, and this default is what the session falls to.
    5. CREATE on the schema is taken from PUBLIC (the default since PostgreSQL 15), and the grants are applied (`apply`).
    6. What the step was to bring about is read back (`read_back`: `NotBroughtAbout`)."""
    if engine.dialect.name != "postgresql":
        raise GrantError(f"db setup is for Postgres (two roles), not {engine.dialect.name}")
    role, ns = _ident(app_role), _ident(schema)
    with step_lock(engine):
        return _setup(engine, app_role, role, ns, password=password, schema=schema)


def _setup(engine: Engine, app_role: str, role: str, ns: str, *, password: str | None, schema: str) -> list[str]:
    from app.storage import migrations

    with engine.connect() as conn:
        owner = conn.execute(text("SELECT current_user")).scalar_one()
        found = conn.execute(text("SELECT rolsuper FROM pg_roles WHERE rolname = :r"), {"r": app_role}).first()
    if app_role == owner:
        raise GrantError(f"{app_role} is the role of this connection: the application role must be another one, which owns "
                         "nothing (it could otherwise drop the append-only triggers)")
    if found is not None and found.rolsuper:
        raise GrantError(f"role {app_role} is a superuser: no grant binds it")
    if found is None and password is None:
        raise GrantError(f"role {app_role} does not exist: create it (CREATE ROLE {app_role} LOGIN PASSWORD …) or pass "
                         "--password-env")
    with engine.connect() as conn:  # N-19g (c), (i), (h), then (e), (g): before anything is changed
        refuse_connection(conn, schema)
        refuse_foreign_tables(conn, schema)
        refuse_application_role(conn, app_role, schema)

    # N-19g: the role statement comes first. Refused (a role that another administrator made, from PostgreSQL 16; an
    # owner without CREATEROLE), it ended in a traceback after the migration, with the tables at the head and no grant
    # on them. The line of the role is printed where it was
    role_line = f"ROLE      {app_role} exists: left as it is"
    if password is not None:
        created = _role(engine, app_role, password, exists=found is not None)
        role_line = f"ROLE      {app_role} {'created' if created else 'exists: password set'}"

    before = migrations.current_revision(engine)
    head = migrations.upgrade_to_head(engine)
    done = [f"AT HEAD   {head}" if before == head else f"MIGRATED  {before or 'empty'} -> {head}"]
    with engine.begin() as conn:
        database = conn.execute(text("SELECT current_database()")).scalar_one()
        quoted = '"' + database.replace('"', '""') + '"'
        conn.execute(text(f"ALTER DATABASE {quoted} SET timezone TO 'UTC'"))
        done.append(f"ZONE      UTC is the default time zone of new sessions of {database}")
        done.append(role_line)
        conn.execute(text(f"REVOKE CREATE ON SCHEMA {ns} FROM PUBLIC"))
    statements_run = apply(engine, app_role, schema=schema)  # it ends with the reading back (`read_back`)
    done.append(f"GRANTS    {len(statements_run)} statements for {app_role} ({role} owns no table)")
    return done


def role_statement(driver, app_role: str, password: str, *, exists: bool):
    """`CREATE ROLE` or `ALTER ROLE … LOGIN PASSWORD` for the application role, without the clear text (N-19g, f): the
    password goes as the SCRAM-SHA-256 verifier that the driver makes of it. The algorithm is named: left out, the driver
    makes an md5 verifier on a server whose `password_encryption` is md5. The server stores a verifier unchanged. A
    refused statement is written to the server log with what it holds, under the default `log_min_error_statement`:
    that is now the verifier. `driver`: the psycopg connection."""
    from psycopg import sql

    verifier = driver.pgconn.encrypt_password(password.encode(), app_role.encode(), b"scram-sha-256").decode()
    # a utility statement takes no bound parameter: the driver composes the literal
    return sql.SQL(("ALTER" if exists else "CREATE") + " ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(app_role), sql.Literal(verifier))


CONNECTION_LOST = "the connection was lost at the role statement; nothing was changed; run the command again"


def _first_line(error: BaseException) -> str:
    return (str(error).strip().splitlines() or [type(error).__name__])[0]


def _role(engine: Engine, app_role: str, password: str, *, exists: bool) -> bool:
    """Make the application role, or set the password of the one that exists; returns whether it was made. The statement
    goes to the driver's connection, so that it is not echoed by the engine's logging, and is committed alone. A
    `CREATE ROLE` that is answered "already exists" is no refusal: another step made the role in the meantime (the
    step's lock is one database's, a role is the server's), and the role is then one that exists. N-19n (PG3-01): so is
    one refused for another reason while another session made the role: at one moment the server answers with the unique
    violation of `pg_authid_rolname_index`, not "already exists". After any refusal, and after its rollback, the step
    looks for the role and goes on as for one that exists where it finds it. Raises `RoleRefused`."""
    import psycopg

    with engine.connect() as conn:
        driver = conn.connection.driver_connection

        def lost() -> RoleRefused:
            # N-19o (PG4-02): the session ended under the statement (`pg_terminate_backend`): one line, and the connection
            # is dropped from the pool, whose reset printed tracebacks
            conn.invalidate()
            return RoleRefused(CONNECTION_LOST)

        # N-19p (PG5-01): only a broken connection is a lost one. A cancelled statement (a lock timeout, a deadlock, a
        # cancel request) is an OperationalError too and leaves the session alive: it goes to the handlers of a refusal
        if not exists:
            try:
                driver.execute(role_statement(driver, app_role, password, exists=False))
                driver.commit()
                return True
            except psycopg.Error as e:
                if driver.broken:
                    raise lost() from None
                try:
                    driver.rollback()
                    found = driver.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (app_role,)).fetchone()
                    driver.rollback()  # the look's own transaction
                except psycopg.OperationalError:
                    if driver.broken:
                        raise lost() from None
                    raise
                if found is None:
                    raise RoleRefused(
                        f"role {app_role} could not be created ({_first_line(e)}), and nothing was changed. Have an administrator "
                        f"create it (CREATE ROLE {app_role} LOGIN PASSWORD …), then run this command without --password-env.") from None
        try:
            driver.execute(role_statement(driver, app_role, password, exists=True))
            driver.commit()
        except psycopg.Error as e:
            if driver.broken:
                raise lost() from None
            driver.rollback()
            raise RoleRefused(
                f"the password of role {app_role} could not be set ({_first_line(e)}), and nothing was changed. Run this command "
                "without --password-env (the role is then left as it is); to change the password, have a role that may alter it "
                f"run ALTER ROLE {app_role} PASSWORD … (psql: \\password {app_role}).") from None
    return False


def database_owner_rights(conn) -> tuple[bool, str, str]:
    """Whether the role of the connection may set a default of the database, with the database and its owner. It may as
    a superuser, or with the rights of the database's owner: `pg_has_role(…, 'USAGE')`, which is what the server asks of
    `ALTER DATABASE … SET` (MEMBER is true for a member that does not inherit, whom the server refuses)."""
    row = conn.execute(text(
        "SELECT d.datname, r.rolname, (SELECT rolsuper FROM pg_roles WHERE rolname = current_user) OR "
        "pg_has_role(current_user, d.datdba, 'USAGE') FROM pg_database d JOIN pg_roles r ON r.oid = d.datdba "
        "WHERE d.datname = current_database()")).one()
    return bool(row[2]), row[0], row[1]


def refuse_connection(conn, schema: str) -> None:
    """N-19g (c), (i): a connection that cannot do the owner's step is refused before anything is changed (`GrantError`:
    exit code 2). (c) One that is no superuser and has not the rights of the database's owner: the zone statement was
    refused ("must be owner of database …") after the migration and before the role and the grants, at every run. (i)
    One that may not create in the schema: the migration was refused there, with a sentence that told the owner to run
    the step it was running."""
    me = conn.execute(text("SELECT current_user")).scalar_one()
    may, database, owner = database_owner_rights(conn)
    if not may:
        # N-19n (PG3-02): it sent the operator to run the step as another role ("Run it as …, as a role that inherits that
        # role's rights, or as a superuser."), and the role that runs the step owns the tables it makes
        raise GrantError(f"{me} has not the rights of {owner}, the owner of the database {database}: the owner's step sets "
                         "the default time zone of the database, which its owner or a superuser may. The role that runs the step "
                         f"owns the tables: have a superuser run ALTER DATABASE {_quoted(database)} OWNER TO {_quoted(me)} and run "
                         f"this command again, or run every owner's step as {owner}.")
    if not conn.execute(text("SELECT has_schema_privilege(current_user, :s, 'CREATE')"), {"s": schema}).scalar_one():
        raise GrantError(f"{me} may not create in the schema {schema}, where the step makes and changes the tables. Have the "
                         f"owner of the schema run: GRANT USAGE, CREATE ON SCHEMA {_ident(schema)} TO {_quoted(me)}")


def refuse_foreign_tables(conn, schema: str) -> None:
    """N-19g (h): one of the tables the step grants on (the models' tables and `alembic_version`) belongs to a role whose
    rights the connection has not, the connection being no superuser (a restore that a superuser ran). The grants on
    such tables were warnings: the step ended with exit code 0 and its four lines, and the application role read no
    table. Another role's table beside the product's is not the step's business."""
    if conn.execute(text("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")).scalar_one():
        return
    foreign = conn.execute(text(
        "SELECT r.rolname, c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_roles r ON r.oid = c.relowner WHERE n.nspname = :s AND c.relkind IN ('r', 'p') AND c.relname = ANY(:tables) "
        "AND NOT pg_has_role(current_user, c.relowner, 'USAGE') ORDER BY r.rolname, c.relname"),
        {"s": schema, "tables": application_tables()}).all()
    if foreign:
        me = conn.execute(text("SELECT current_user")).scalar_one()
        other = foreign[0][0]
        tables = [table for role, table in foreign if role == other]
        named = ", ".join(tables[:3]) + (f" and {len(tables) - 3} more" if len(tables) > 3 else "")
        raise GrantError(f"{named} in the schema {schema} belong{'s' if len(tables) == 1 else ''} to {other}, whose rights {me} "
                         "has not: only the owner of a table grants on it. A restore is run as the role that is to own the "
                         "tables (pg_restore --no-owner as that role); here, have a superuser give them to that role (ALTER TABLE "
                         f"… OWNER TO … for each table, ALTER FUNCTION {APPEND_ONLY_FUNCTION} OWNER TO …), or restore again.")


def refuse_application_role(conn, app_role: str, schema: str) -> None:
    """N-19g (e), (g): an application role that could switch off the triggers which keep the custody log append-only is
    refused before anything is changed (`GrantError`: exit code 2). That is a role that owns a table of the schema, and a
    role that is a member of the role of the connection or of a role that owns the database, the schema or one of the
    tables the step grants on. `pg_has_role(…, 'MEMBER')`: with or without INHERIT, since a member that does not inherit
    takes the owner's rights with `SET ROLE`. The step ended there with exit code 0 and "owns no table", or refused the
    owner of the tables after it had set the database's zone and changed that role's password. A role that does not
    exist yet is a member of nothing."""
    owned = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = :s AND tableowner = :r ORDER BY tablename"),
                         {"s": schema, "r": app_role}).scalars().all()
    if owned:
        raise GrantError(f"role {app_role} owns {', '.join(owned)}; the application role must not own tables (it could drop the "
                         f"append-only triggers). Transfer ownership of the tables and of the function {APPEND_ONLY_FUNCTION} to "
                         "the migration role first.")
    if conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": app_role}).first() is None:
        return
    # N-19p (PG5-02): a member of a superuser role, with or without INHERIT, may `SET ROLE` to it and switch the triggers off;
    # the owners below need not be superusers, so it is asked first. A superuser itself is refused by name before this
    superusers = conn.execute(text(
        "SELECT rolname FROM pg_roles WHERE rolsuper AND rolname <> :r AND pg_has_role(:r, oid, 'MEMBER') ORDER BY rolname"),
        {"r": app_role}).scalars().all()
    if superusers:
        raise GrantError(f"role {app_role} is a member of {superusers[0]}, a superuser: such a role can switch off the triggers "
                         "that keep the custody log append-only (a member may act as that role, with or without INHERIT). The "
                         "application role must be a member of no superuser role.")
    owners = conn.execute(text(
        "SELECT current_user AS role, 'the role of this connection' AS what "
        "UNION ALL SELECT r.rolname, 'the owner of the database' FROM pg_database d JOIN pg_roles r ON r.oid = d.datdba "
        "WHERE d.datname = current_database() "
        "UNION ALL SELECT r.rolname, 'the owner of the schema' FROM pg_namespace n JOIN pg_roles r ON r.oid = n.nspowner "
        "WHERE n.nspname = :s "
        "UNION ALL SELECT DISTINCT r.rolname, 'the owner of tables of the schema' FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace JOIN pg_roles r ON r.oid = c.relowner "
        "WHERE n.nspname = :s AND c.relkind IN ('r', 'p') AND c.relname = ANY(:tables)"),
        {"s": schema, "tables": application_tables()}).all()
    for owner, what in owners:
        if conn.execute(text("SELECT pg_has_role(:r, :o, 'MEMBER')"), {"r": app_role, "o": owner}).scalar_one():
            same = f"role {app_role} is {what}" if owner == app_role else f"role {app_role} is a member of {owner}, {what}"
            raise GrantError(f"{same}: such a role can switch off the triggers that keep the custody log append-only (a "
                             "member may act as that role, with or without INHERIT). The application role must own nothing "
                             "of the instance and be a member of no role that does.")
    # N-19n (PG3-03): the owner of the triggers' function is read as the owners of the tables are. An application role that
    # owned it was accepted ("owns no table") and could `DROP FUNCTION … CASCADE`, which drops both append-only triggers.
    # N-19o (PG4-01, R9-01): asked after the owners, and only where the function's owner is not the owner of `audit_events`:
    # asked first, a member of the owner of the tables (case g) got the function's sentence, whose ALTER FUNCTION changes
    # nothing there, and the same refusal followed
    function_owner = conn.execute(text(
        "SELECT r.rolname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace JOIN pg_roles r ON r.oid = p.proowner "
        "WHERE n.nspname = :s AND p.proname = :f"), {"s": schema, "f": APPEND_ONLY_FUNCTION.removesuffix("()")}).scalar()
    tables_owner = conn.execute(text("SELECT tableowner FROM pg_tables WHERE schemaname = :s AND tablename = 'audit_events'"),
                                {"s": schema}).scalar()
    if function_owner is not None and function_owner != tables_owner and conn.execute(
            text("SELECT pg_has_role(:r, :o, 'MEMBER')"), {"r": app_role, "o": function_owner}).scalar_one():
        same = (f"role {app_role} owns the function {APPEND_ONLY_FUNCTION}" if function_owner == app_role
                else f"role {app_role} is a member of {function_owner}, the owner of the function {APPEND_ONLY_FUNCTION}")
        raise GrantError(f"{same}, which the triggers that keep the custody log append-only execute: such a role can drop them. "
                         f"Have a superuser run ALTER FUNCTION {APPEND_ONLY_FUNCTION} OWNER TO "
                         f"{_quoted(tables_owner or 'the owner of the tables')}, then run this command again.")
    # N-19p (PG5-02): a serving role needs no CREATEROLE, and on PostgreSQL 15 such a role can grant itself the owner's
    # role. Asked last, so that a role that owns something, or may act as an owner, gets that sentence first
    if conn.execute(text("SELECT rolcreaterole FROM pg_roles WHERE rolname = :r"), {"r": app_role}).scalar():
        raise GrantError(f"role {app_role} has CREATEROLE: a serving role needs no such right, and such a role can make itself a "
                         f"member of the owner of the tables. Have an administrator run ALTER ROLE {app_role} NOCREATEROLE, then "
                         "run this command again.")


def apply(engine: Engine, app_role: str, *, schema: str = "public") -> list[str]:
    """Run the statements over an owner connection; returns them. Refuses SQLite, tables that belong to a role whose
    rights the connection has not (`refuse_foreign_tables`), and an application role that owns tables or is a member of
    a role that owns anything of the instance (`refuse_application_role`)."""
    if engine.dialect.name != "postgresql":
        raise GrantError(f"grants apply to Postgres only, not {engine.dialect.name}")
    sql = statements(app_role, schema=schema)
    with engine.begin() as conn:
        # N-19o (PG4-01): a superuser is refused by name, as `db setup` refuses it; it got the sentence of the function's owner
        if conn.execute(text("SELECT rolsuper FROM pg_roles WHERE rolname = :r"), {"r": app_role}).scalar():
            raise GrantError(f"role {app_role} is a superuser: no grant binds it")
        refuse_foreign_tables(conn, schema)
        refuse_application_role(conn, app_role, schema)
        for statement in sql:
            conn.execute(text(statement))
    with engine.connect() as conn:
        read_back(conn, app_role, schema)
    return sql


def read_back(conn, app_role: str, schema: str) -> None:
    """N-19g (d): last, the step reads back what it was to bring about: the application role may use the schema, may not
    create in it, and may read a table of it. In a schema that the owner of the tables does not own (as `public` in a
    database that came from PostgreSQL 14 or earlier) `REVOKE CREATE ON SCHEMA … FROM PUBLIC` and `GRANT USAGE ON SCHEMA
    …` are warnings: the step ended with exit code 0 and its four lines, and the role created a table, or read nothing.
    Raises `NotBroughtAbout` with the statements for the schema's owner."""
    row = conn.execute(text(
        "SELECT has_schema_privilege(:r, :s, 'USAGE'), has_schema_privilege(:r, :s, 'CREATE'), "
        "(SELECT r.rolname FROM pg_namespace n JOIN pg_roles r ON r.oid = n.nspowner WHERE n.nspname = :s), "
        "(SELECT r.rolname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace JOIN pg_roles r ON r.oid = c.relowner "
        " WHERE n.nspname = :s AND c.relname = 'alembic_version'), "
        "(SELECT has_table_privilege(:r, c.oid, 'SELECT') FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        " WHERE n.nspname = :s AND c.relname = 'alembic_version'), current_user"), {"r": app_role, "s": schema}).one()
    uses, creates, schema_owner, tables_owner, reads, me = row
    wrong = ([] if uses else [f"may not use the schema {schema}"]) + ([f"may create in the schema {schema}"] if creates else [])
    if reads is False:  # None: the schema holds no table of the product yet (grants printed for an empty database)
        wrong.append("may not read alembic_version")
    if not wrong:
        return
    ns = _ident(schema)
    raise NotBroughtAbout(
        f"the application role {app_role} {' and '.join(wrong)}: the schema belongs to {schema_owner}, and the step's statements "
        f"on it were not carried out. Have {schema_owner} or a superuser run: GRANT USAGE, CREATE ON SCHEMA {ns} TO "
        f"{_quoted(tables_owner or me)}; REVOKE CREATE ON SCHEMA {ns} FROM PUBLIC; GRANT USAGE ON SCHEMA {ns} TO {_ident(app_role)}; "
        "then run this command again.")
