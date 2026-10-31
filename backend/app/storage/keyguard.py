"""A key that records depend on is never generated anew (BUILD_SPEC N-12; ops review OPS-05).

Three keys are generated at their first use when they are not there:
- the **instance signing key** (`data/keys/instance_ed25519.pem`): the Ed25519 signatures of reports and packages;
- the **signing key** (`data/keys/signing.key`, unless `VERITAS_SIGNING_KEY` is set): conclusions, sign-offs, webhook secrets;
- the **API-key pepper** (`data/keys/api_key_pepper.key`, unless `VERITAS_API_KEY_PEPPER` is set): the principals' API keys.

On a database that already holds records made with such a key, a missing key is a restore without its keys, not a first
start. A new key would come up with no warning and leave every review record "signature does not verify", every principal
key refused, and new reports signed by a key no recipient has pinned. In prod the start is refused (`check_start`: the API's
lifespan and the worker), and so is the generation itself (`refuse_generation`) for a key file that goes missing while the
instance runs. Dev generates as before. A key that is present and is not the earlier one is the operator's act: it is
logged at the start, not refused.

**An instance of an earlier build (N-17a; ops re-review O2-01).** Before D-59 the signing key of a prod instance without
`VERITAS_SIGNING_KEY` was derived from the API key, so no key file and no value ever existed to put back. When every
record a signing key made verifies under that derived key, the refusal says so and names
`python -m app.cli keys adopt-derived` (`adopt_derived`): it writes the derived key to `data/keys/signing.key` once, with
a custody event, and from then on the key is a file like any other and no longer follows the API key.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError, ProgrammingError

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

INSTANCE_KEY, SIGNING_KEY, PEPPER = "instance signing key", "signing key", "API-key pepper"
RESTORE = 'Restore it from the backup (runbook, "Backup and restore").'
_HOW = {
    INSTANCE_KEY: f"Put data/keys/instance_ed25519.pem back. {RESTORE}",
    SIGNING_KEY: f"Put data/keys/signing.key back, or set VERITAS_SIGNING_KEY to the value the instance ran with. {RESTORE}",
    PEPPER: f"Put data/keys/api_key_pepper.key back, or set VERITAS_API_KEY_PEPPER to the value the instance ran with. {RESTORE}",
}
ADOPT = "python -m app.cli keys adopt-derived"
_ADOPT_HOW = ("They verify under the key that an earlier build derived from VERITAS_API_KEY, which prod no longer derives: no "
              f"key file ever existed. Run `{ADOPT}` once: it writes that key to data/keys/signing.key. Then start again "
              '(runbook, "Upgrades").')


class KeyMissing(RuntimeError):
    """A key that existing records were made with is not there, and a new one would be generated in its place."""


def dependants(kind: str) -> tuple[int, str]:
    """How many records of the database were made with the key of this kind, and what they are."""
    from app.storage.db import ConclusionRow, PrincipalRow, SigningKeyRow, SignoffRow, WebhookRow, get_session

    def count(db, model, *where) -> int:
        return int(db.scalar(select(func.count()).select_from(model).where(*where)) or 0)

    try:
        with get_session() as db:
            if kind == INSTANCE_KEY:
                return count(db, SigningKeyRow), "registered key(s) that signed its reports and packages"
            if kind == SIGNING_KEY:
                return (count(db, ConclusionRow) + count(db, SignoffRow) + count(db, WebhookRow),
                        "conclusion(s), sign-off(s) and webhook(s) signed or sealed with it")
            if kind == PEPPER:
                return count(db, PrincipalRow, PrincipalRow.api_key_hash.is_not(None)), "principal API key(s) hashed with it"
    except (OperationalError, ProgrammingError):  # a database with no tables yet holds no record
        return 0, ""
    raise ValueError(f"unknown key kind {kind!r}")


@dataclass(frozen=True)
class Signed:
    """What a signing key made, checked against one key: the signatures of conclusions and sign-offs, the sealed secrets of
    webhooks. `invalid` names the records that this key did not make."""
    conclusions: int
    signoffs: int
    webhooks: int
    invalid: tuple[str, ...]

    @property
    def total(self) -> int:
        return self.conclusions + self.signoffs + self.webhooks

    @property
    def verified(self) -> bool:
        return self.total > 0 and not self.invalid


def signed_under(key: bytes) -> Signed:
    """Every conclusion, sign-off and webhook of the database against `key`."""
    import hmac

    from app.cases import review
    from app.storage.db import SignoffRow, WebhookRow, get_session
    from app.webhooks import service as webhooks

    def sealed(row: WebhookRow) -> bool:
        return hmac.compare_digest(webhooks.secret_hash(webhooks.secret_for(row.id, row.secret_nonce, key)), str(row.secret_hash))

    counts, invalid = [], []
    with get_session() as db:
        # N-19n: the guard reads the database before the start migrates it: the conclusions with the columns that it holds
        for word, records, valid in (
                ("conclusion", lambda: review.stored_conclusions(db), lambda row: review.signature_valid(review.conclusion_body(row), row.signature, key)),
                ("sign-off", lambda: db.scalars(select(SignoffRow)), lambda row: review.signature_valid(review.signoff_body(row), row.signature, key)),
                ("webhook", lambda: db.scalars(select(WebhookRow)), sealed)):
            rows = 0
            for row in records():
                rows += 1
                if not valid(row):
                    invalid.append(f"{word} {row.id}")
            counts.append(rows)
    return Signed(*counts, invalid=tuple(invalid))


def signed_by_derived_key() -> bool:
    """Whether the signing key that is missing is the one an earlier build derived from `VERITAS_API_KEY`: every record a
    signing key made verifies under it. False when the records cannot be read."""
    from app.api import tokens

    if not settings.api_key:
        return False
    try:
        return signed_under(tokens.derived_key(settings.api_key)).verified
    except Exception as e:  # noqa: BLE001 - the refusal is then the general one
        log.warning("keys.derived_check_skipped", error=f"{type(e).__name__}: {e}"[:200])
        return False


class NotAdopted(RuntimeError):
    """`adopt_derived` wrote nothing, and why."""


@dataclass(frozen=True)
class Adopted:
    path: str
    fingerprint: str
    signed: Signed
    seq: int


def adopt_derived(api_key: str | None, *, named: str = "VERITAS_API_KEY") -> Adopted:
    """Keep the signing key of an instance that an earlier build ran (module docstring): the key derived from `api_key`
    goes to `data/keys/signing.key` (0600) with a `signing_key_adopted` event of the system, in one step: the file is
    written inside the event's transaction and removed again when the commit fails.

    Refused (`NotAdopted`, nothing written) when the instance has a signing key (`VERITAS_SIGNING_KEY`, or the file), when no
    record was signed, and when one record does not verify under the derived key."""
    from app.api import tokens
    from app.audit import chain as audit
    from app.storage.files import write_new_file

    path = settings.data_dir / "keys" / tokens.SIGNING_KEY_FILE
    if settings.signing_key:
        raise NotAdopted("VERITAS_SIGNING_KEY is set: that value is the signing key of this instance, and no key file is "
                         "written beside it.")
    if path.exists():
        raise NotAdopted(f"{path} exists: this instance has its signing key, and the file is not replaced.")
    if not api_key:
        raise NotAdopted(f"{named} is not set: there is no API key to derive the key from.")
    key = tokens.derived_key(api_key)
    signed = signed_under(key)
    if signed.total == 0:
        raise NotAdopted("the database holds no conclusion, sign-off or webhook: no signing key was used here, and a start "
                         "generates data/keys/signing.key.")
    if signed.invalid:
        few = ", ".join(signed.invalid[:5]) + (f" and {len(signed.invalid) - 5} more" if len(signed.invalid) > 5 else "")
        raise NotAdopted(f"{len(signed.invalid)} of {signed.total} record(s) do not verify under the key derived from {named} "
                         f"({few}): another key signed them, or the API key was another one then. Give the API key the "
                         "instance ran with (--api-key-env), or put its signing key back.")
    event = {"action": "signing_key_adopted", "actor_id": audit.SYSTEM_ACTOR,
             "details": {"key": "signing key", "file": f"keys/{tokens.SIGNING_KEY_FILE}", "fingerprint": tokens.fingerprint(key),
                         "derivation": "HKDF-SHA256 of the API key, as builds before 2.1.1 derived it",
                         "conclusions": signed.conclusions, "signoffs": signed.signoffs, "webhooks": signed.webhooks,
                         "by": "command line"}}

    def unit() -> int:
        written = False

        def write(_seqs: list[int]) -> None:
            nonlocal written
            if not write_new_file(path, key):
                raise NotAdopted(f"{path} exists: another process wrote it first, and the file is not replaced.")
            written = True

        with audit.writing() as db:
            try:
                return audit.commit_with(db, [event], before_commit=write)[0]
            except BaseException:
                if written:  # the event was not stored: no key file without its event
                    path.unlink(missing_ok=True)
                raise

    seq = audit.repeated(unit)
    log.info("keys.derived_adopted", file=str(path), fingerprint=tokens.fingerprint(key), seq=seq)
    return Adopted(str(path), tokens.fingerprint(key), signed, seq)


def refusal(kind: str) -> str | None:
    """In prod, why a new key of this kind must not be generated; None where generating is right (dev, or no record yet)."""
    if settings.environment != "prod":
        return None
    held, what = dependants(kind)
    if held == 0:
        return None
    how = _ADOPT_HOW if kind == SIGNING_KEY and signed_by_derived_key() else _HOW[kind]
    return f"The {kind} is missing and a new one is not generated: the database holds {held} {what}. {how}"


def refuse_generation(kind: str) -> None:
    message = refusal(kind)
    if message is not None:
        log.error("keys.generation_refused", kind=kind)
        raise KeyMissing(message)


def check_start() -> None:
    """Refuse a prod start that would generate a key over records made with the earlier one (the API and the worker)."""
    if settings.environment != "prod":
        return
    from app.api import auth, tokens
    from app.reports import signing

    absent = [kind for kind, missing in ((INSTANCE_KEY, not signing.key_path().is_file()),
                                         (SIGNING_KEY, tokens.signing_key_source()[0] == "absent"),
                                         (PEPPER, auth.pepper_source()[0] == "absent")) if missing]
    problems = {kind: message for kind in absent if (message := refusal(kind)) is not None}
    if problems:
        log.error("startup.keys_missing", kinds=list(problems),
                  derived=_ADOPT_HOW in problems.get(SIGNING_KEY, ""))  # N-17a: an earlier build's key; the sentence names the command
        raise KeyMissing("Start refused. " + " ".join(problems.values()))
    _say_replaced(signing, absent)


def _say_replaced(signing, absent: list[str]) -> None:
    """Keys that are present and are not the ones the records were made with: logged once, never refused."""
    from app.cases import review
    from app.storage.db import SigningKeyRow, SignoffRow, get_session

    try:
        with get_session() as db:
            registered = set(db.scalars(select(SigningKeyRow.id)))
            newest = db.scalars(select(SignoffRow).order_by(SignoffRow.signed_at.desc()).limit(1)).first()
            if INSTANCE_KEY not in absent and registered:
                key_id = signing.instance_key().key_id
                if key_id not in registered:
                    log.warning("startup.instance_key_replaced", key_id=key_id, registered=sorted(registered),
                                detail="reports and packages signed earlier verify against the earlier key only")
            if SIGNING_KEY not in absent and newest is not None and not review.signoff_valid(newest):
                log.error("startup.signing_key_mismatch", signoff_id=newest.id,
                          detail="the newest sign-off does not verify with the configured signing key; with another key than "
                                 "the one the instance ran with, every review record reads 'signature does not verify'")
    except (OperationalError, ProgrammingError, signing.SigningKeyError) as e:
        log.warning("startup.key_check_skipped", error=f"{type(e).__name__}: {e}"[:200])
