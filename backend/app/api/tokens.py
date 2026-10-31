"""Resource-scoped tokens and the HMAC signing key (ARCHITECTURE §5.3).

- **Format.** `base64url(json).base64url(hmac_sha256(signing_key, first segment))` with payload `{res, sub, exp, scope}`.
  `res` is `analysis:<id>`, `events:<analysis_id>`, `exhibit:<id>`, `media:<id>` or `report:<id>`; `scope` is `read`.
- **Use.** Tokens authenticate GET requests only, on routes whose resource matches `res` (`security.require_api_key`).
  Browsers need them because `<img>`, `<video>` and `EventSource` cannot send headers.
- **Signing key.** `VERITAS_SIGNING_KEY` when set. Otherwise, in dev only, HKDF-SHA256 from `VERITAS_API_KEY` (so existing
  dev databases keep valid review signatures). Otherwise a random 32-byte key generated once at `data/keys/signing.key`
  (0600). In prod the key is never derived from the API key (D-59): every principal holds HMAC outputs (tokens, review
  signatures), and a derived key would let any of them test API-key candidates offline.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.core.config import settings

MAX_TTL_S = 600
RESOURCE_PATTERN = r"(analysis|events|exhibit|media|report):[A-Za-z0-9._-]{1,128}"
_RESOURCE = re.compile(RESOURCE_PATTERN)
_HKDF_INFO = b"veritas signing key v1"
_KEY_BYTES = 32
SIGNING_KEY_FILE = "signing.key"
_MAX_TOKEN_CHARS = 1024
_key_lock = threading.Lock()


class TokenError(ValueError):
    """A token cannot be issued for this resource or lifetime."""


@dataclass(frozen=True)
class ResourceToken:
    res: str
    sub: str
    exp: int
    scope: str


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _read_key(path: Path) -> bytes:
    key = path.read_bytes()
    if len(key) != _KEY_BYTES:
        raise RuntimeError(f"the signing key file {path.name} is corrupt ({len(key)} bytes)")
    return key


def instance_key(path: Path) -> bytes:
    """A random 32-byte instance key at `path` (0600), created once; also the API-key pepper file (C-1)."""
    with _key_lock:
        try:
            return _read_key(path)
        except FileNotFoundError:
            pass
        from app.storage.files import write_new_file

        key = secrets.token_bytes(_KEY_BYTES)
        if not write_new_file(path, key):  # another process created it first; a reader never sees a partial file (CR P2-11)
            return _read_key(path)
        return key


def derived_key(api_key: str) -> bytes:
    """The key derived from an API key: what dev signs with, and what every build before D-59 signed with in prod when
    `VERITAS_SIGNING_KEY` was not set (`python -m app.cli keys adopt-derived` keeps it for such an instance, N-17a)."""
    return HKDF(algorithm=hashes.SHA256(), length=_KEY_BYTES, salt=None, info=_HKDF_INFO).derive(api_key.encode())


def signing_key() -> bytes:
    if settings.signing_key:
        return settings.signing_key.encode()
    if settings.api_key and settings.environment != "prod":
        return derived_key(settings.api_key)
    path = settings.data_dir / "keys" / SIGNING_KEY_FILE
    if not path.is_file():  # N-12: in prod no new key over records signed with the earlier one
        from app.storage import keyguard

        keyguard.refuse_generation(keyguard.SIGNING_KEY)
    return instance_key(path)


def signing_key_source() -> tuple[str, bytes | None]:
    """Where `signing_key()` takes the key from, and the key, without creating one (N-12): `environment`
    (`VERITAS_SIGNING_KEY`), `derived` (dev: from the API key), `file` (`data/keys/signing.key`) or `absent`: the next use
    would generate the file."""
    if settings.signing_key:
        return "environment", settings.signing_key.encode()
    if settings.api_key and settings.environment != "prod":
        return "derived", signing_key()
    try:
        return "file", _read_key(settings.data_dir / "keys" / SIGNING_KEY_FILE)
    except FileNotFoundError:
        return "absent", None


def fingerprint(key: bytes) -> str:
    """16 hexadecimal characters that name a secret key in a backup manifest without holding it (N-12)."""
    return hmac.new(key, b"veritas key fingerprint v1", hashlib.sha256).hexdigest()[:16]


def valid_resource(res: str) -> bool:
    return isinstance(res, str) and _RESOURCE.fullmatch(res) is not None


def _mac(body: str) -> bytes:
    return hmac.new(signing_key(), body.encode("ascii"), hashlib.sha256).digest()


def make_token(res: str, sub: str, ttl_s: int = MAX_TTL_S, *, now: float | None = None) -> tuple[str, int]:
    """Returns `(token, exp)`; `exp` is a Unix timestamp in seconds."""
    if not valid_resource(res):
        raise TokenError("resource must look like analysis:<id>, events:<id>, exhibit:<id>, media:<id> or report:<id>")
    if not 1 <= ttl_s <= MAX_TTL_S:
        raise TokenError(f"ttl_s must be between 1 and {MAX_TTL_S} seconds")
    exp = int(time.time() if now is None else now) + int(ttl_s)
    payload = json.dumps({"res": res, "sub": sub, "exp": exp, "scope": "read"}, separators=(",", ":"), sort_keys=True)
    body = _b64(payload.encode())
    return f"{body}.{_b64(_mac(body))}", exp


def verify_token(token: str | None, *, now: float | None = None) -> ResourceToken | None:
    """The token's claims when the signature, expiry, scope and resource format are valid; otherwise None."""
    if not token or len(token) > _MAX_TOKEN_CHARS or token.count(".") != 1:
        return None
    body, mac = token.split(".")
    try:
        given = _unb64(mac)
        if not hmac.compare_digest(given, _mac(body)):
            return None
        claims = json.loads(_unb64(body))
        parsed = ResourceToken(res=str(claims["res"]), sub=str(claims["sub"]), exp=int(claims["exp"]),
                               scope=str(claims["scope"]))
    except (ValueError, KeyError, TypeError, UnicodeError):
        return None
    if parsed.scope != "read" or not valid_resource(parsed.res) or parsed.exp < (time.time() if now is None else now):
        return None
    return parsed
