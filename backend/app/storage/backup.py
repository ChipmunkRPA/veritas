"""Backup, backup verification and store verification (BUILD_SPEC N-12; ops review OPS-05; Postgres QA PG-13).

- **`python -m app.cli backup create <dir>`** writes a directory that has the layout of a data directory, so a restore is a
  copy (`docs/runbook.md` "Backup and restore"):
  1. `veritas.db`: the database first, through the SQLite backup API (one consistent snapshot of a running instance, the
     write-ahead log included). On Postgres the command refuses and prints the `pg_dump` procedure (`POSTGRES_PROCEDURE`);
  2. `objects/`: every store object that snapshot references (`retention.REFERENCE_COLUMNS`), re-hashed while it is copied.
     The database is taken before the objects, so an object written meanwhile is at worst an unreferenced extra, never a
     reference without its object;
  3. `uploads/`, `artifacts/`, `reports/`: the files of the snapshot's analyses that live outside the store (the upload the
     v1 routes serve, v1 figures and v1 PDFs). An upload that is a hard link of a store object is a hard link in the backup;
  4. `keys/`: the key files (`instance_ed25519.pem` signs reports and packages, `signing.key` review records and download
     links, `api_key_pepper.key` the principals' API keys), and `fusion/`: fuser artefacts and their review records;
  5. `backup.json`: every file with its size and SHA-256, the database revision and custody head, the id of the instance
     signing key, and the source and fingerprint of the signing key and of the API-key pepper. A key that comes from the
     environment (`VERITAS_SIGNING_KEY`, `VERITAS_API_KEY_PEPPER`) is not in the backup: the manifest names its fingerprint,
     and the restore needs the same value.
  Model weights are not backed up (`models fetch`, and the local fine-tune of the runbook), nor is `.env`, nor the staged
  uploads of a batch that is still expanding (`data/batches`).
- **`python -m app.cli backup verify <dir>`** needs no running instance: every listed file hashes to its line and nothing is
  unlisted; the database passes `PRAGMA integrity_check`; every object its `artefacts` name is in the backup under its own
  hash; the custody chain of the copy verifies and ends where the manifest says; the instance signing key is the one the
  database registered; the review signatures of the copy verify with the backed-up signing key.
- **`python -m app.cli backup restore <dir> [--data-dir PATH] [--replace]`** (N-12a; code re-check R3-01) puts a backup
  that verifies into a data directory on which no instance runs (`app.storage.running`). The target is empty, or
  `--replace` is given: the database that is there is then set aside with its write-ahead log
  (`veritas.db.replaced-<time>`, `…-wal`, `…-shm`), never deleted and never left beside the restored file. A good
  `veritas.db` put beside the `-wal` of the instance it replaces is not the backup: the old log is read over it, and the
  database is malformed or shows the earlier instance's rows. The files of the backup are written first and the database
  last, and its integrity is checked where it now is.
- **`python -m app.cli store verify`** re-hashes every object the running instance's database references.

The backup directory holds the keys in the clear (mode 0700): keep it where the keys may be.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import stat
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sqlalchemy import create_engine, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.canonical import canonical_json
from app.core.config import settings
from app.core.logging import get_logger
from app.storage.objects import CHUNK, ObjectIntegrityError, ObjectNotFound, ObjectStore, default_store

log = get_logger(__name__)

SCHEMA = "veritas.backup/1"
MANIFEST = "backup.json"
DATABASE = "veritas.db"
TREES = ("uploads", "artifacts", "reports")  # per-analysis files outside the store
INSTANCE_KEY_FILE = "instance_ed25519.pem"  # app.reports.signing.KEY_FILE
# N-19g (Postgres QA PG2-01, PG2-02): the command lines of the Postgres procedure. The runbook ("Backup and restore",
# "Postgres") gives the same lines word for word (`tests/test_n19g_restore_docs.py`), and the tests of the restore run them
# (`tests/test_n19g_restore_pg.py`): the text is what is tested.
PG_DUMP = "pg_dump -Fc -f veritas.pgc <database>"
PG_CREATEDB = "createdb -O <owner> <new database>"
PG_RESTORE = "pg_restore --exit-on-error --no-owner -d <new database> veritas.pgc"
PG_RESTORE_NO_ACL = "pg_restore --exit-on-error --no-owner --no-acl -d <new database> veritas.pgc"
PG_OWNER_STEP = "python -m app.cli db setup --app-role <role> --database-url <owner url of the new database> [--password-env NAME]"
POSTGRES_COMMANDS = (PG_DUMP, PG_CREATEDB, PG_RESTORE, PG_RESTORE_NO_ACL, PG_OWNER_STEP)
POSTGRES_PROCEDURE = f"""\
This instance runs on Postgres: `backup create` takes an SQLite database only. The procedure for Postgres (runbook, "Backup
and restore", "Postgres"):
  1. stop the writers: the api and the worker;
  2. as the owner of the database:  {PG_DUMP}
  3. copy the data directory with data/keys (objects, uploads, artifacts, reports, fusion, keys);
  4. keep VERITAS_API_KEY and, where they are set, VERITAS_SIGNING_KEY and VERITAS_API_KEY_PEPPER with the backup.
Restore, into a new database that is made before anything is dropped, as the role that is to own the tables (never as a
superuser that is not that role), with pg_dump and pg_restore of the server's own major version:
  1. {PG_CREATEDB}
  2. where the roles the dump grants to exist (the same server):  {PG_RESTORE}
     where they do not (another server, other role names):  {PG_RESTORE_NO_ACL}
  3. the data directory;
  4. with two roles, the owner's step:  {PG_OWNER_STEP}
  5. point the service at the new database, start and check; drop the old database when the checks have passed.
Never `pg_restore --data-only` into a migrated database (seven tables stay empty), never a plain restore into one, and
never `pg_restore --clean --if-exists` into a database of a later revision (the tables of the later revisions stay)."""


class BackupRefused(RuntimeError):
    """The backup cannot be made here: Postgres, or a destination that already holds files."""


class BackupUnreadable(ValueError):
    """The directory holds no readable backup manifest."""


class BackupNotVerified(RuntimeError):
    """A backup that fails a check is not restored."""

    def __init__(self, checks: list[Check]):
        self.checks = checks
        super().__init__("The backup does not verify; nothing was restored.")


@dataclass
class Restored:
    data_dir: Path
    files: int
    set_aside: list[str]  # the earlier database and its log, renamed
    manifest: dict[str, Any]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class StoreCheck:
    checked: int = 0
    bytes: int = 0
    missing: list[str] = field(default_factory=list)  # indexed in `artefacts`, not in the store
    altered: list[str] = field(default_factory=list)  # in the store, and its bytes no longer hash to its name
    elsewhere: int = 0  # referenced hashes that `artefacts` does not index and the store does not hold (an exhibit without a payload)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.altered


@dataclass
class Backup:
    path: Path
    manifest: dict[str, Any]

    @property
    def missing(self) -> list[str]:
        return list(self.manifest["objects"]["missing"])

    @property
    def altered(self) -> list[str]:
        return list(self.manifest["objects"]["altered"])

    @property
    def ok(self) -> bool:
        return not self.missing and not self.altered


# --- the store of the running instance ---------------------------------------------------------------------------------

def _reference_columns() -> list[tuple[str, str]]:
    from app import retention

    return [(column.class_.__tablename__, column.key) for column in retention.REFERENCE_COLUMNS]


def store_verify(*, store: ObjectStore | None = None, sessions: Callable[[], Session] | None = None) -> StoreCheck:
    """Re-hash every store object the database references. An object the `artefacts` index names and the store lacks is
    missing; one whose bytes no longer hash to its name is altered (it is left where it is: the record of what is there)."""
    from sqlalchemy import text

    from app.storage.db import get_session

    store = store or default_store()
    with (sessions or get_session)() as db:
        referenced = {sha for table, column in _reference_columns()
                      for sha in db.scalars(text(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL")) if sha}
        indexed = {sha for sha in db.scalars(text("SELECT DISTINCT sha256 FROM artefacts")) if sha}
    out = StoreCheck()
    for sha in sorted(referenced):
        try:
            out.bytes += store.verify(sha)
            out.checked += 1
        except ObjectNotFound:
            if sha in indexed:
                out.missing.append(sha)
            else:
                out.elsewhere += 1
        except ObjectIntegrityError:
            out.altered.append(sha)
    return out


# --- create ------------------------------------------------------------------------------------------------------------

def _sqlite_path() -> Path:
    url = make_url(settings.sqlalchemy_url)
    if url.get_backend_name() != "sqlite":
        raise BackupRefused(POSTGRES_PROCEDURE)
    if not url.database or url.database == ":memory:":
        raise BackupRefused("The database is in memory: there is nothing to back up.")
    return Path(url.database)


def _read_only(path: Path) -> sqlite3.Connection:
    """The copy as it is: `immutable` takes no lock and writes no `-wal` or `-shm` file beside it, so its hash stays its hash."""
    return sqlite3.connect(f"file:{quote(str(path.resolve()))}?immutable=1", uri=True, check_same_thread=False)


def _copy_database(source: Path, target: Path) -> None:
    src = sqlite3.connect(str(source), timeout=30.0)
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)  # the whole database in one step: a snapshot, whatever is written meanwhile
            # the copy is one file whatever the journal mode of the source: nothing is left in a `-wal` beside it
            dst.execute("PRAGMA journal_mode=DELETE")
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()
    for suffix in ("-wal", "-shm", "-journal"):
        Path(f"{target}{suffix}").unlink(missing_ok=True)


def _copy(source: Path, target: Path) -> tuple[int, str]:
    """Copy one file with its mode; returns the size and the SHA-256 of what was read."""
    target.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with open(source, "rb") as src, open(target, "wb") as dst:
        while chunk := src.read(CHUNK):
            digest.update(chunk)
            size += len(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    os.chmod(target, stat.S_IMODE(source.stat().st_mode))
    return size, digest.hexdigest()


def _sha256(path: Path) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _files(root: Path) -> Iterator[Path]:
    """Regular files under `root`, in path order; symbolic links are not followed and not copied."""
    if not root.is_dir():
        return
    for directory, names, files in os.walk(root, followlinks=False):
        names.sort()
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                yield path


def key_record() -> dict[str, Any]:
    """The keys of this instance as a manifest states them: ids and fingerprints, never a key. Nothing is generated."""
    from app.api import auth, tokens
    from app.reports import signing

    instance: dict[str, Any] = {"file": f"keys/{INSTANCE_KEY_FILE}", "key_id": None}
    if signing.key_path().is_file():
        instance["key_id"] = signing.instance_key().key_id
    out: dict[str, Any] = {"instance_signing_key": instance}
    for name, (source, key), file in (("signing_key", tokens.signing_key_source(), tokens.SIGNING_KEY_FILE),
                                      ("api_key_pepper", auth.pepper_source(), auth.PEPPER_FILE)):
        out[name] = {"source": source, "fingerprint": None if key is None else tokens.fingerprint(key),
                     "file": f"keys/{file}" if source == "file" else None}
    return out


def create(dest: Path, *, store: ObjectStore | None = None) -> Backup:
    """Write a backup of this instance into `dest` (module docstring). Raises `BackupRefused` on Postgres and for a
    destination that holds files. A backup whose `ok` is false was written and lacks objects the database names."""
    source_db = _sqlite_path()
    dest = Path(dest)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise BackupRefused(f"{dest} exists and is not an empty directory: a backup is written into a new or empty one.")
    store = store or default_store()
    dest.mkdir(parents=True, exist_ok=True)
    os.chmod(dest, 0o700)  # it will hold the keys
    started = datetime.now(UTC)
    files: dict[str, dict[str, Any]] = {}

    def listed(relative: str, size: int, sha256: str) -> None:
        files[relative] = {"bytes": size, "sha256": sha256}

    _copy_database(source_db, dest / DATABASE)  # first: everything below is read from this snapshot
    listed(DATABASE, *_sha256(dest / DATABASE))
    snapshot = _read_only(dest / DATABASE)
    try:
        referenced = {row[0] for table, column in _reference_columns()
                      for row in snapshot.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL") if row[0]}
        indexed = {row[0] for row in snapshot.execute("SELECT DISTINCT sha256 FROM artefacts")}
        analyses = [row[0] for row in snapshot.execute("SELECT id FROM analyses ORDER BY id")]
        revision = next(iter(snapshot.execute("SELECT version_num FROM alembic_version")), (None,))[0]
        head = next(iter(snapshot.execute("SELECT seq, event_hash FROM audit_events ORDER BY seq DESC LIMIT 1")), (0, None))
    finally:
        snapshot.close()

    missing, altered, inodes = [], [], {}
    for sha in sorted(referenced):
        try:
            source = store.path_for(sha)
            info = source.stat()
        except (OSError, ValueError):
            if sha in indexed:
                missing.append(sha)
            continue
        relative = f"objects/sha256/{sha[:2]}/{sha[2:4]}/{sha}"
        size, found = _copy(source, dest / relative)
        if found != sha:  # the damaged bytes are not carried into the backup under a name they do not have
            (dest / relative).unlink()
            altered.append(sha)
            continue
        listed(relative, size, found)
        inodes[(info.st_dev, info.st_ino)] = relative

    for tree in TREES:
        for analysis_id in analyses:
            for path in _files(settings.data_dir / tree / analysis_id):
                relative = path.relative_to(settings.data_dir).as_posix()
                info = path.stat()
                same = inodes.get((info.st_dev, info.st_ino))
                if same is not None:  # the upload is a hard link of its store object: the same in the backup
                    (dest / relative).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(dest / same, dest / relative)
                        listed(relative, files[same]["bytes"], files[same]["sha256"])
                        continue
                    except OSError:
                        pass
                listed(relative, *_copy(path, dest / relative))
    for tree in ("keys", "fusion"):
        for path in _files(settings.data_dir / tree):
            relative = path.relative_to(settings.data_dir).as_posix()
            listed(relative, *_copy(path, dest / relative))
    if (dest / "keys").is_dir():
        os.chmod(dest / "keys", 0o700)

    manifest = {
        "schema": SCHEMA, "created_at": started.isoformat().replace("+00:00", "Z"), "version": settings.version,
        "database": {"engine": "sqlite", "file": DATABASE, "revision": revision,
                     "custody_head": {"seq": int(head[0] or 0), "hash": head[1]}, "analyses": len(analyses)},
        "objects": {"copied": sum(1 for name in files if name.startswith("objects/")), "missing": missing, "altered": altered},
        "keys": key_record(),
        "not_included": ["model weights (python -m app.cli models fetch; the local fine-tune of the runbook)",
                         "environment secrets (VERITAS_API_KEY, VERITAS_SIGNING_KEY, VERITAS_API_KEY_PEPPER)",
                         "staged batch uploads (data/batches): a batch that was still expanding is submitted again"],
        "files": dict(sorted(files.items())),
    }
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("backup.created", path=str(dest), files=len(files), missing=len(missing), altered=len(altered))
    return Backup(dest, manifest)


# --- verify ------------------------------------------------------------------------------------------------------------

def read_manifest(directory: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((Path(directory) / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BackupUnreadable(f"{directory} holds no readable {MANIFEST} ({type(e).__name__}).") from None
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"), dict):
        raise BackupUnreadable(f"{Path(directory) / MANIFEST} is not a {SCHEMA} manifest.")
    return manifest


def _few(names: list[str], limit: int = 5) -> str:
    return ", ".join(names[:limit]) + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _sessions(path: Path) -> Callable[[], Session]:
    engine = create_engine("sqlite://", creator=lambda: _read_only(path), future=True)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _review_signatures(sessions: Callable[[], Session], key: bytes) -> tuple[int, list[str]]:
    """Every conclusion and sign-off of the copy against `key`, as `review.signature_valid` checks them on the instance."""
    from app.cases import review
    from app.storage.db import SignoffRow

    checked, invalid = 0, []
    with sessions() as db:
        # N-19n: the conclusions of a copy at an earlier revision are read with the columns that it holds (`stored_conclusions`)
        for rows, body, word in ((review.stored_conclusions(db), review.conclusion_body, "conclusion"),
                                 (db.scalars(select(SignoffRow)), review.signoff_body, "sign-off")):
            for row in rows:
                checked += 1
                expected = hmac.new(key, canonical_json(body(row)), hashlib.sha256).hexdigest()
                if not row.signature or not hmac.compare_digest(expected, str(row.signature)):
                    invalid.append(f"{word} {row.id}")
    return checked, invalid


def verify(directory: Path) -> list[Check]:
    """Check the backup in `directory` (module docstring). Raises `BackupUnreadable` when it has no manifest."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    listed: dict[str, dict[str, Any]] = manifest["files"]
    checks: list[Check] = []

    present = {p.relative_to(directory).as_posix() for p in _files(directory)} - {MANIFEST}
    absent = sorted(set(listed) - present)
    unlisted = sorted(present - set(listed))
    changed = []
    for name in sorted(set(listed) & present):
        size, sha = _sha256(directory / name)
        if sha != listed[name].get("sha256") or size != listed[name].get("bytes"):
            changed.append(name)
    problems = [text for text in (f"missing: {_few(absent)}" if absent else "", f"altered: {_few(changed)}" if changed else "",
                                  f"not listed: {_few(unlisted)}" if unlisted else "") if text]
    checks.append(Check("files", not problems, "; ".join(problems) or f"all {len(listed)} listed files are present and match"))

    database = directory / DATABASE
    if DATABASE in absent or not database.is_file():
        checks.append(Check("database", False, f"{DATABASE} is not in the backup"))
        return checks
    try:
        snapshot = _read_only(database)
        try:
            integrity = [row[0] for row in snapshot.execute("PRAGMA integrity_check")]
            indexed = {row[0] for row in snapshot.execute("SELECT DISTINCT sha256 FROM artefacts")}
            registered = {row[0] for row in snapshot.execute("SELECT id FROM signing_keys")}
            head = next(iter(snapshot.execute("SELECT seq, event_hash FROM audit_events ORDER BY seq DESC LIMIT 1")), (0, None))
            api_keys = next(iter(snapshot.execute("SELECT COUNT(*) FROM principals WHERE api_key_hash IS NOT NULL")))[0]
        finally:
            snapshot.close()
    except sqlite3.Error as e:
        checks.append(Check("database", False, f"{DATABASE} cannot be read ({e})"))
        return checks
    checks.append(Check("database", integrity == ["ok"],
                        f"integrity check passed; revision {manifest['database'].get('revision')}" if integrity == ["ok"]
                        else f"integrity check: {_few([str(i) for i in integrity])}"))

    recorded = manifest.get("objects") or {}
    lost = sorted(sha for sha in indexed if f"objects/sha256/{sha[:2]}/{sha[2:4]}/{sha}" not in present)
    misnamed = sorted(name for name, entry in listed.items() if name.startswith("objects/") and entry.get("sha256") != name.rsplit("/", 1)[-1])
    problems = [text for text in (f"not in the backup: {_few(lost)}" if lost else "",
                                  f"stored under another hash than their own: {_few(misnamed)}" if misnamed else "",
                                  f"{len(recorded.get('altered') or [])} object(s) were altered in the store when the backup was made"
                                  if recorded.get("altered") else "") if text]
    checks.append(Check("objects", not problems, "; ".join(problems)
                        or f"every object the database indexes is in the backup under its own hash ({len(indexed)})"))

    sessions = _sessions(database)
    try:
        from app.audit import chain

        walked = chain.verify_events(chain.iter_events(sessions))
        stated = manifest["database"].get("custody_head") or {}
        if not walked.valid:
            checks.append(Check("custody", False, f"broken at seq {walked.first_broken_seq}: {walked.reason}"))
        elif (int(head[0] or 0), head[1]) != (stated.get("seq"), stated.get("hash")):
            checks.append(Check("custody", False, f"the chain of the copy ends at seq {head[0]}, the manifest says {stated.get('seq')}"))
        else:
            checks.append(Check("custody", True, f"{walked.checked} event(s) verify; the chain ends at seq {int(head[0] or 0)}"))
    except Exception as e:  # noqa: BLE001  a copy this build cannot read (another revision) is a failed check, not a crash
        checks.append(Check("custody", False, f"the custody log of the copy cannot be read by this build ({type(e).__name__})"))

    checks.append(_instance_key(directory, manifest, registered))
    checks.append(_signing_key(directory, manifest, sessions))
    checks.append(_pepper(directory, manifest, int(api_keys or 0)))
    return checks


def _instance_key(directory: Path, manifest: dict[str, Any], registered: set[str]) -> Check:
    from app.reports import signing

    name = "instance signing key"
    stated = (manifest["keys"].get("instance_signing_key") or {}).get("key_id")
    path = directory / "keys" / INSTANCE_KEY_FILE
    if not path.is_file():
        return Check(name, not registered and stated is None,
                     "the instance had no signing key yet and signed nothing" if not registered and stated is None
                     else f"keys/{INSTANCE_KEY_FILE} is not in the backup, and the database holds records signed with "
                          f"{_few(sorted(registered)) or stated}")
    try:
        key_id = signing._load(path).key_id
    except signing.SigningKeyError as e:
        return Check(name, False, str(e))
    if key_id != stated:
        return Check(name, False, f"keys/{INSTANCE_KEY_FILE} is {key_id}, the manifest says {stated}")
    if registered and key_id not in registered:
        return Check(name, False, f"{key_id} is not a key the database registered ({_few(sorted(registered))})")
    return Check(name, True, f"{key_id}, the key the database registered" if registered else f"{key_id}; nothing was signed yet")


def _backed_up_key(directory: Path, entry: dict[str, Any]) -> bytes | None:
    if entry.get("source") != "file" or not entry.get("file"):
        return None
    try:
        return (directory / entry["file"]).read_bytes()
    except OSError:
        return None


def _signing_key(directory: Path, manifest: dict[str, Any], sessions: Callable[[], Session]) -> Check:
    from app.api import tokens

    name = "review signatures"
    entry = manifest["keys"].get("signing_key") or {}
    source, stated = entry.get("source"), entry.get("fingerprint")
    key = _backed_up_key(directory, entry)
    where = "the backed-up signing key"
    if source == "file" and (key is None or tokens.fingerprint(key) != stated):
        return Check(name, False, f"{entry.get('file')} is missing or is not the key the manifest names ({stated})")
    if key is None and stated is not None:  # the key is held outside the backup: this environment may hold the same one
        here_source, here = tokens.signing_key_source()
        if here is not None and tokens.fingerprint(here) == stated:
            key, where = here, f"the signing key of this environment ({here_source})"
    try:
        if key is None:
            from app.storage.db import ConclusionRow, SignoffRow

            with sessions() as db:
                signed = sum(len(db.scalars(select(model.id)).all()) for model in (ConclusionRow, SignoffRow))
            if signed == 0:
                return Check(name, True, "the copy holds no conclusion and no sign-off")
            return Check(name, False, f"{signed} signed record(s), and the signing key is not in the backup (source: {source}, "
                                      f"fingerprint {stated}): verify where that key is set, and restore with the same value")
        checked, invalid = _review_signatures(sessions, key)
    except Exception as e:  # noqa: BLE001
        return Check(name, False, f"the review records of the copy cannot be read by this build ({type(e).__name__})")
    if invalid:
        return Check(name, False, f"{len(invalid)} of {checked} do not verify with {where}: {_few(invalid)}")
    return Check(name, True, f"{checked} conclusion(s) and sign-off(s) verify with {where} (fingerprint {stated})")


def _pepper(directory: Path, manifest: dict[str, Any], api_keys: int) -> Check:
    from app.api import tokens

    name = "API-key pepper"
    entry = manifest["keys"].get("api_key_pepper") or {}
    source, stated = entry.get("source"), entry.get("fingerprint")
    if source == "file":
        key = _backed_up_key(directory, entry)
        ok = key is not None and tokens.fingerprint(key) == stated
        return Check(name, ok, f"{entry.get('file')}, fingerprint {stated}; {api_keys} principal key(s) depend on it" if ok
                     else f"{entry.get('file')} is missing or is not the pepper the manifest names ({stated})")
    if source == "environment":
        return Check(name, True, f"held in the environment, not in the backup (fingerprint {stated}): restore with the same "
                                 f"VERITAS_API_KEY_PEPPER; {api_keys} principal key(s) depend on it")
    return Check(name, api_keys == 0, "no pepper yet and no principal key" if api_keys == 0
                 else f"{api_keys} principal key(s) and no pepper: they cannot be checked after a restore")



# --- restore ---------------------------------------------------------------------------------------------------------------

DATABASE_FILES = (DATABASE, f"{DATABASE}-wal", f"{DATABASE}-shm", f"{DATABASE}-journal")


def restore(directory: Path, data_dir: Path | None = None, *, replace: bool = False) -> Restored:
    """Put the backup in `directory` into `data_dir` (default: the configured data directory); module docstring.

    Raises `BackupUnreadable` (no manifest), `BackupNotVerified` (a check fails: nothing is written), `BackupRefused` (a
    target that holds files without `replace`, a backup inside its target, Postgres) and `running.InstanceRunning`."""
    from app.storage import running

    directory = Path(directory).resolve()
    configured = data_dir is None
    data_dir = Path(data_dir or settings.data_dir).resolve()
    if configured and make_url(settings.sqlalchemy_url).get_backend_name() != "sqlite":
        raise BackupRefused("This instance runs on Postgres: its database is restored with pg_restore (runbook, \"Backup and "
                            "restore\", \"Postgres\"), its data directory by a copy.")
    manifest = read_manifest(directory)
    if directory == data_dir or data_dir in directory.parents:
        raise BackupRefused(f"The backup {directory} lies inside the data directory {data_dir}: move it out first.")
    checks = verify(directory)
    if not all(check.ok for check in checks):
        raise BackupNotVerified(checks)

    with running.exclusive(data_dir):
        held = [p for p in data_dir.iterdir() if p.name != running.LOCK_FILE]
        if held and not replace:
            raise BackupRefused(f"{data_dir} is not empty. Restore into an empty data directory, or pass --replace: the database "
                                "that is there is then set aside as veritas.db.replaced-<time>, with its log.")
        # the earlier database leaves with its log: beside another database's -wal the restored file would not be read as it is
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        set_aside = []
        for name in DATABASE_FILES:
            old = data_dir / name
            if old.exists():
                new = data_dir / name.replace(DATABASE, f"{DATABASE}.replaced-{stamp}", 1)
                old.rename(new)
                set_aside.append(new.name)
        linked: dict[tuple[int, int], Path] = {}
        for name in sorted(manifest["files"], key=lambda n: (n == DATABASE, n)):  # the database last: no row before its objects
            source, target = directory / name, data_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(f".{target.name}.restoring")
            staged.unlink(missing_ok=True)
            info = source.stat()
            first = linked.get((info.st_dev, info.st_ino))
            if first is not None:
                os.link(first, staged)  # an upload that is a hard link of its store object stays one
            else:
                _copy(source, staged)
            os.replace(staged, target)
            linked.setdefault((info.st_dev, info.st_ino), target)
        for name in ("keys",):
            if (data_dir / name).is_dir():
                os.chmod(data_dir / name, 0o700)
        snapshot = _read_only(data_dir / DATABASE)
        try:
            integrity = [row[0] for row in snapshot.execute("PRAGMA integrity_check")]
        finally:
            snapshot.close()
        if integrity != ["ok"]:
            raise BackupRefused(f"The restored database fails its integrity check ({_few([str(i) for i in integrity])}).")
    log.info("backup.restored", data_dir=str(data_dir), files=len(manifest["files"]), set_aside=set_aside)
    return Restored(data_dir, len(manifest["files"]), set_aside, manifest)
