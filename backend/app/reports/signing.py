"""Ed25519 instance key: report and package signatures (D-11; ARCHITECTURE §4.7 `signing_keys`, §6.3 step 8, §6.5; PRD RP-06;
BUILD_SPEC R-4).

- **Key.** Generated on first use at `data/keys/instance_ed25519.pem` (PKCS#8 PEM, mode 0600, created exclusively so two
  processes never both write a key). It never enters the database, the artefact store or a package. Its public half is indexed
  in `signing_keys` (raw 32 bytes), exported at `GET /api/v2/instance/public-key` and embedded in every package.
- **Key id.** `ed25519:` + the first 16 hex characters of SHA-256 over the raw public key.
- **Signatures** are detached and base64 over the exact bytes (a PDF, `manifest.sha256`); embedding would change the bytes.
- HMAC stays for resource tokens and review records (ARCHITECTURE §5.3–5.4).

Only `cryptography` is imported at module level, so the offline verifier (`app.reports.verify`) loads no settings or database.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ALGORITHM = "Ed25519"
KEY_FILE = "instance_ed25519.pem"
_lock = threading.Lock()


class SigningKeyError(RuntimeError):
    """The instance key cannot be created or read, or a public key is not an Ed25519 key."""


def key_id_of(public: Ed25519PublicKey) -> str:
    raw = public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return f"ed25519:{hashlib.sha256(raw).hexdigest()[:16]}"


@dataclass(frozen=True)
class InstanceKey:
    private: Ed25519PrivateKey
    path: Path

    @property
    def public(self) -> Ed25519PublicKey:
        return self.private.public_key()

    @property
    def key_id(self) -> str:
        return key_id_of(self.public)

    @property
    def public_raw(self) -> bytes:
        return self.public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def public_pem(self) -> bytes:
        return self.public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)

    def sign(self, data: bytes) -> str:
        return base64.b64encode(self.private.sign(data)).decode("ascii")


def key_path() -> Path:
    from app.core.config import settings

    return settings.data_dir / "keys" / KEY_FILE


def instance_key(path: Path | None = None) -> InstanceKey:
    """The instance key, generated on first use. In prod it is not generated over a database that registered an earlier key
    (N-12): a restore without its keys is said, not papered over."""
    own = path is None
    path = path or key_path()
    with _lock:
        if not path.is_file():
            if own:
                from app.storage import keyguard

                message = keyguard.refusal(keyguard.INSTANCE_KEY)
                if message is not None:
                    raise SigningKeyError(message)
            _generate(path)
        return _load(path)


def _generate(path: Path) -> None:
    pem = Ed25519PrivateKey.generate().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                     serialization.NoEncryption())
    from app.storage.files import write_new_file

    try:
        write_new_file(path, pem)  # complete or absent: a concurrent reader never loads a partial key; another process may win
    except OSError as e:
        raise SigningKeyError(f"The instance signing key could not be created at {path}: {e.strerror}.") from None


def _load(path: Path) -> InstanceKey:
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as e:
        raise SigningKeyError(f"The instance signing key at {path} cannot be read ({type(e).__name__}).") from None
    if not isinstance(key, Ed25519PrivateKey):
        raise SigningKeyError(f"The instance signing key at {path} is not an Ed25519 key.")
    return InstanceKey(key, path)


def load_public_key(pem: bytes) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(pem)
    except (ValueError, TypeError):
        raise SigningKeyError("The public key is not a PEM-encoded public key.") from None
    if not isinstance(key, Ed25519PublicKey):
        raise SigningKeyError("The public key is not an Ed25519 key.")
    return key


def verify(public: Ed25519PublicKey, data: bytes, signature: str) -> bool:
    try:
        raw = base64.b64decode(signature.strip(), validate=True)
    except (binascii.Error, ValueError):
        return False
    try:
        public.verify(raw, data)
    except InvalidSignature:
        return False
    return True


def ensure_registered(key: InstanceKey, sessions: Any = None) -> None:
    """Index the key's public half in `signing_keys` once (idempotent across processes)."""
    from sqlalchemy.exc import IntegrityError

    from app.storage.db import SigningKeyRow, get_session, utcnow

    with (sessions or get_session)() as db:
        if db.get(SigningKeyRow, key.key_id) is not None:
            return
        db.add(SigningKeyRow(id=key.key_id, public_key=key.public_raw, key_path=str(key.path), created_at=utcnow()))
        try:
            db.commit()
        except IntegrityError:  # registered concurrently
            db.rollback()
