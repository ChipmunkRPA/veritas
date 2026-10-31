"""Fail closed before serving: weights, fuser compatibility, release identity, schema and database privileges."""
from __future__ import annotations

import json
import time
from pathlib import Path


def _existing_key(path: Path, *, maximum: int, name: str) -> bytes:
    """Read bounded existing bytes only; never call a key-creation helper."""
    try:
        if not path.is_file():
            raise OSError("not a regular key file")
        with path.open("rb") as handle:
            value = handle.read(maximum + 1)
    except OSError:
        raise RuntimeError(f"Restore refused: the existing {name} key is missing or unreadable") from None
    if len(value) > maximum:
        raise RuntimeError(f"Restore refused: the existing {name} key is invalid")
    return value


def validate_restore_keys(conn, data_dir: Path, api_key_pepper: str | None = None) -> None:
    """Check restored key prerequisites without generating keys or changing SQL.

    A pepper's presence/format is testable, but its continuity cannot be proved
    from stored HMACs without an original API credential. Ed25519 continuity is
    checked against the registered public key. Historical rows are not removed
    or required to have their old private files; the current key cannot be retired.
    """
    import hashlib

    from cryptography.exceptions import UnsupportedAlgorithm
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from sqlalchemy import text

    has_api_keys = conn.execute(text("SELECT 1 FROM principals WHERE api_key_hash IS NOT NULL LIMIT 1")).first()
    if has_api_keys is not None:
        # Match auth._pepper: a nonempty configured string takes precedence.
        # Its length need not be 32; that fixed size belongs to generated files.
        if api_key_pepper:
            try:
                api_key_pepper.encode("utf-8")
            except (AttributeError, UnicodeEncodeError):
                raise RuntimeError("Restore refused: the configured API-key pepper is unusable") from None
        elif len(_existing_key(data_dir / "keys" / "api_key_pepper.key", maximum=32, name="API-key pepper")) != 32:
            raise RuntimeError("Restore refused: the existing API-key pepper must contain exactly 32 bytes")

    if conn.execute(text("SELECT 1 FROM signing_keys LIMIT 1")).first() is None:
        return  # A fresh instance may create/register its first signing key later.
    encoded = _existing_key(data_dir / "keys" / "instance_ed25519.pem", maximum=16384, name="Ed25519 signing")
    try:
        key = serialization.load_pem_private_key(encoded, password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm):
        raise RuntimeError("Restore refused: the existing Ed25519 signing key is invalid") from None
    if not isinstance(key, Ed25519PrivateKey):
        raise RuntimeError("Restore refused: the existing signing key is not Ed25519")
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    key_id = "ed25519:" + hashlib.sha256(public).hexdigest()[:16]
    row = conn.execute(text("SELECT public_key, retired_at FROM signing_keys WHERE id = :id"), {"id": key_id}).first()
    if row is None or row.public_key != public:
        raise RuntimeError("Restore refused: the existing signing key does not match its registered public key")
    if row.retired_at is not None:
        raise RuntimeError("Restore refused: the existing signing key is retired")


def validate_required_grants(conn) -> None:
    """Read effective privileges required by the existing owner-grant policy.

    This grants nothing and performs no trial writes. Check each privilege
    separately: PostgreSQL's comma-separated privilege argument means ANY,
    whereas this startup contract needs ALL. Inherited grants are sufficient;
    audit events remain SELECT/INSERT only and the schema version SELECT only.
    """
    from sqlalchemy import text

    from app.storage.pg_grants import APPEND_ONLY, READ_ONLY, application_sequences, application_tables

    def require(sql: str, params: dict, label: str) -> None:
        if conn.execute(text(sql), params).scalar_one() is not True:
            raise RuntimeError(f"runtime is missing required {label} privilege; rerun the separate owner migration/grants helper")

    require("SELECT has_schema_privilege(current_user, :schema, 'USAGE')",
            {"schema": "public"}, "public schema USAGE")
    for table in application_tables():
        privileges = ("SELECT", "INSERT") if table in APPEND_ONLY else (
            ("SELECT",) if table in READ_ONLY else ("SELECT", "INSERT", "UPDATE", "DELETE"))
        relation = 'public."' + table.replace('"', '""') + '"'
        for privilege in privileges:
            require("SELECT has_table_privilege(current_user, :relation, :privilege)",
                    {"relation": relation, "privilege": privilege}, f"{relation} {privilege}")
    # N-20b: the product's own sequences, by name, as the owner's step grants them (N-19g (k)). Every sequence of the schema
    # was required here, while the step grants no sequence of another role (an administrator's, an extension's), so such a
    # sequence refused the start of a deployment whose grants were complete
    for sequence in application_sequences():
        relation = 'public."' + sequence.replace('"', '""') + '"'
        for privilege in ("USAGE", "SELECT"):
            require("SELECT has_sequence_privilege(current_user, :relation, :privilege)",
                    {"relation": relation, "privilege": privilege}, f"{relation} {privilege}")


def main() -> None:
    from release import verify
    from sqlalchemy import create_engine, text

    from app.core.config import settings
    from app.engine.fetchers import ensure_verified
    from app.engine.fusion.registry import select_for_registry
    from app.engine.registry import get_registry
    from app.storage.migrations import current_revision, head_revision

    config = json.loads(Path("/opt/veritas/config.json").read_text())
    if settings.build_commit != config["build_commit"] or settings.licence_policy != "commercial":
        raise RuntimeError("backend build or licence policy differs from deployment configuration")
    verify(settings.data_dir, settings.build_commit)
    registry = get_registry()
    enabled = {entry.detector_id for entry in registry.decisions() if entry.enabled}
    wanted = {mid for det in registry.manifest.detectors if det.id in enabled and "full" in det.profiles
              for mid in registry.required_models(det)}
    for mid in sorted(wanted):
        # Unlike `models verify --installed`, a missing required model is fatal here.
        ensure_verified(registry.manifest.model(mid), force=True)
    for kind in ("image", "video", "audio"):
        selection = select_for_registry(registry, kind, "full")
        if selection.mode != "artefact":
            raise RuntimeError(f"no compatible reviewed {kind} fuser: {selection.reason}")
        # K-3, D-57: an artefact is pending human review until a review sidecar names its sha256 (`Selection.pending_review`).
        # Local serving says so in every result; production needs the review unless the operator sets
        # VERITAS_ACCEPT_PENDING_FUSER_REVIEW (README, "Fuser review"), which is printed here
        if settings.environment == "prod" and selection.pending_review:
            name = selection.path.name
            if not settings.accept_pending_fuser_review:
                raise RuntimeError(f"the {kind} fuser {name} is pending human review; record the review with `python -m app.cli "
                                   f"fusers review --modality {kind} …` (a sidecar beside the artefacts in the fuser directory), or "
                                   "set VERITAS_ACCEPT_PENDING_FUSER_REVIEW=true to accept it")
            print(f"WARNING: serving the {kind} fuser {name} pending human review (VERITAS_ACCEPT_PENDING_FUSER_REVIEW=true).")
    engine = create_engine(settings.sqlalchemy_url)
    try:
        for attempt in range(30):
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                break
            except Exception:
                if attempt == 29:
                    raise RuntimeError("Cloud SQL proxy/database not ready") from None
                time.sleep(2)
        if current_revision(engine) != head_revision():
            raise RuntimeError("run the separate owner migration helper before starting runtime")
        with engine.connect() as conn:
            unsafe = conn.execute(text("SELECT has_table_privilege(current_user, 'audit_events', 'UPDATE') "
                                       "OR has_table_privilege(current_user, 'audit_events', 'DELETE') "
                                       "OR has_table_privilege(current_user, 'audit_events', 'TRUNCATE') "
                                       "OR has_table_privilege(current_user, 'audit_events', 'TRIGGER')")).scalar_one()
            owns = conn.execute(text("SELECT count(*) FROM pg_tables WHERE schemaname='public' "
                                     "AND pg_has_role(current_user, tableowner, 'MEMBER')")).scalar_one()
            can_create = conn.execute(text("SELECT has_schema_privilege(current_user, 'public', 'CREATE')")).scalar_one()
            if unsafe or owns or can_create:
                raise RuntimeError("runtime must inherit only the non-owner veritas_app grants; audit mutation/DDL privileges refused")
            validate_required_grants(conn)
            validate_restore_keys(conn, settings.data_dir, settings.api_key_pepper)
    finally:
        engine.dispose()
    print("Runtime preflight passed; no inference or new benchmark evaluation was performed.")


if __name__ == "__main__":
    main()
