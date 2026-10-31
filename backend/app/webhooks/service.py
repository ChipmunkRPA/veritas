"""Signed webhooks (ARCHITECTURE §4.7 `webhooks`, `webhook_deliveries`, §5.2; PRD API-02; BUILD_SPEC X-1).

- **Off by default.** Deliveries run only while `VERITAS_WEBHOOKS_ENABLED=true` (the only outbound network path, PL-01) and the hook
  is enabled; `POST /webhooks` is refused while the instance switch is off.
- **Events:** `analysis.completed`, `analysis.failed`, `analysis.signed_off`, `batch.completed`. The body is compact JSON
  `{"created_at", "data", "event", "id"}`; every attempt sends the same bytes.
- **Signature.** `X-Veritas-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256(secret, "<t>." + body)>`, with `X-Veritas-Event` and
  `X-Veritas-Delivery` (the event id). A receiver compares the HMAC in constant time and refuses a timestamp more than five minutes
  away (`verify_signature`; the sample in `docs/v2/api/webhooks.md`).
- **Secret.** `whsec_` + base64url(HMAC-SHA256(instance HMAC key, "veritas.webhook/1:<hook id>:<nonce>")), shown once when the hook is
  created or its secret rotated. The database keeps the nonce and the secret's SHA-256, never the secret.
- **Delivery.** One `webhook_deliveries` row per attempt. A 2xx answer is `delivered`. Otherwise the attempt is `failed` with its
  status code or error and attempt n+1 is queued `webhook_retry_base_s × 2^(n-1)` seconds later, while that lies within
  `webhook_retry_window_s` (24 h) of the first attempt; past the window the attempt is `abandoned`. Redirects are not followed.
- **Addresses (Q-2 W-1).** Unless `webhook_allow_private_networks`, a URL whose host is a non-global IP literal is refused at
  configuration, and at delivery the name is resolved once, every address must be global (an IPv4-mapped IPv6 address counts as
  its IPv4 address), and the request goes to the first checked address with the original `Host` header and, for https, the name
  as SNI and certificate name; a refused address is an `abandoned` attempt and nothing is sent. Proxy environment variables are
  ignored. The row keeps a generic error category; the exception text goes to the log only.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
import socket
import ssl
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import select

from app.core.config import settings
from app.core.logging import get_logger
from app.storage.db import WebhookDeliveryRow, WebhookRow, get_session, utcnow

log = get_logger(__name__)
EVENTS = ("analysis.completed", "analysis.failed", "analysis.signed_off", "batch.completed")
SIGNATURE_HEADER, EVENT_HEADER, DELIVERY_HEADER = "X-Veritas-Signature", "X-Veritas-Event", "X-Veritas-Delivery"
TOLERANCE_S = 300
REFUSED_ADDRESS = "The webhook address is not allowed"


class AddressNotAllowedError(Exception):
    """The webhook host is, or resolves to, an address deliveries may not reach."""


def address_allowed(address: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether deliveries may reach this IP address: a global address, or any address with `webhook_allow_private_networks`."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return settings.webhook_allow_private_networks or ip.is_global


def _literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The address an IP-literal host denotes, including the forms the C resolver accepts (`2130706433`, `0x7f.1`, `127.1`)."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except (OSError, ValueError):
        return None


def url_problem(url: str) -> str | None:
    """Why the URL cannot be a webhook endpoint, else None. Names are resolved and checked at delivery, not here."""
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - raises ValueError for a port outside 0-65535
    except ValueError:
        return "The URL cannot be parsed."
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "The URL needs an http or https scheme and a host."
    if parts.username or parts.password:
        return "The URL must not carry credentials; the signature authenticates deliveries."
    if (literal := _literal(parts.hostname)) is not None and not address_allowed(literal):
        return ("The URL points at a loopback, private, link-local or otherwise non-global address; set "
                "VERITAS_WEBHOOK_ALLOW_PRIVATE_NETWORKS=true to deliver to an on-premises receiver.")
    return None


def new_nonce() -> str:
    return secrets.token_hex(16)


def secret_for(hook_id: str, nonce: str, key: bytes | None = None) -> str:
    """The secret of a webhook under the signing key; `key` names another key to try (`keyguard.signed_under`)."""
    from app.api.tokens import signing_key

    mac = hmac.new(key or signing_key(), f"veritas.webhook/1:{hook_id}:{nonce}".encode(), hashlib.sha256).digest()
    return "whsec_" + base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")


def secret_hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("ascii")).hexdigest()


def sign(secret: str, body: bytes, timestamp: int) -> str:
    mac = hmac.new(secret.encode("ascii"), str(timestamp).encode("ascii") + b"." + body, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={mac}"


def verify_signature(secret: str, header: str, body: bytes, *, now: float | None = None, tolerance_s: int = TOLERANCE_S) -> bool:
    try:
        fields = dict(part.split("=", 1) for part in header.split(","))
        timestamp = int(fields["t"])
    except (ValueError, KeyError):
        return False
    if abs((time.time() if now is None else now) - timestamp) > tolerance_s:
        return False
    expected = hmac.new(secret.encode("ascii"), f"{timestamp}.".encode("ascii") + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, fields.get("v1", ""))


def backoff_s(attempt: int) -> float:
    return settings.webhook_retry_base_s * 2 ** (attempt - 1)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def record_event(event: str, data: dict[str, Any]) -> list[str]:
    """Attempt 1 for every enabled hook subscribed to the event; returns the delivery ids (none while webhooks are off)."""
    if event not in EVENTS:
        raise ValueError(f"unknown webhook event {event!r}")
    if not settings.webhooks_enabled:
        return []
    now, event_id = utcnow(), str(uuid.uuid4())
    body = json.dumps({"id": event_id, "event": event, "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "data": data},
                      sort_keys=True, separators=(",", ":"), default=str)
    ids: list[str] = []
    with get_session() as db:
        for hook in db.scalars(select(WebhookRow).where(WebhookRow.enabled.is_(True)).order_by(WebhookRow.created_at)):
            if event not in (hook.events or []):
                continue
            row = WebhookDeliveryRow(id=str(uuid.uuid4()), hook_id=hook.id, event=event, event_id=event_id, payload=body,
                                     payload_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(), attempt=1, status="pending",
                                     created_at=now, next_attempt_at=now)
            db.add(row)
            ids.append(row.id)
        db.commit()
    return ids


def _getaddrinfo(host: str, port: int) -> list[tuple[Any, ...]]:
    """The resolver, behind a seam tests patch."""
    return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)


def _client(**kwargs: Any) -> Any:
    """The HTTP client factory, behind a seam tests patch to inject a transport."""
    import httpx

    return httpx.Client(**kwargs)


def _pinned(url: str) -> tuple[str, str, str | None]:
    """(URL with the host replaced by its first checked address, the original Host header, the TLS server name or None).

    Raises AddressNotAllowedError when any address the host resolves to is not allowed; the resolved address is used as is, so
    a second lookup cannot answer differently (DNS rebinding)."""
    parts = urlsplit(url)
    host, port = parts.hostname or "", parts.port or (443 if parts.scheme == "https" else 80)
    addresses = [str(info[4][0]) for info in _getaddrinfo(host, port) if info[0] in (socket.AF_INET, socket.AF_INET6)]
    if not addresses:
        raise socket.gaierror(f"no IPv4 or IPv6 address for {host!r}")
    if refused := [a for a in addresses if not address_allowed(a)]:
        raise AddressNotAllowedError(f"{host!r} resolves to {', '.join(refused)}")
    address = addresses[0]
    netloc = (f"[{address}]" if ":" in address else address) + (f":{parts.port}" if parts.port is not None else "")
    sni = host if parts.scheme == "https" and _literal(host) is None else None
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, "")), parts.netloc.rpartition("@")[2], sni


def _post(url: str, body: bytes, headers: dict[str, str]) -> Any:
    target, host_header, sni = _pinned(url)
    with _client(trust_env=False, follow_redirects=False, timeout=settings.webhook_timeout_s) as client:
        return client.post(target, content=body, headers={**headers, "Host": host_header},
                           extensions={"sni_hostname": sni} if sni is not None else None)


def _error_category(e: BaseException) -> str:
    """The generic error recorded on the delivery row; the exception text is logged, never returned by the API."""
    import httpx

    chain, seen = [], set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        chain.append(cur)
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    if any(isinstance(x, ssl.SSLError) for x in chain):
        return "TLS handshake failed"
    if any(isinstance(x, httpx.TimeoutException | TimeoutError) for x in chain):
        return "Timed out"
    if any(isinstance(x, httpx.ConnectError | httpx.NetworkError | ConnectionError | socket.gaierror) for x in chain):
        return "Connection failed"
    return "Delivery failed"


def deliver(delivery_id: str, *, retry: Callable[[str, datetime], None], post: Callable[[str, bytes, dict[str, str]], Any] | None = None) -> str:
    """The `webhook_delivery` job: one attempt. Returns the job outcome (always done: the row records the attempt)."""
    with get_session() as db:
        row = db.get(WebhookDeliveryRow, delivery_id)
        if row is None or row.status != "pending":
            return "done"
        hook = db.get(WebhookRow, row.hook_id)
        if hook is None or not hook.enabled or not settings.webhooks_enabled:
            row.status, row.error, row.attempted_at = "abandoned", "The webhook is disabled or deleted.", utcnow()
            db.commit()
            return "done"
        url, secret, body = hook.url, secret_for(hook.id, hook.secret_nonce), row.payload.encode("utf-8")
        event, event_id, attempt = row.event, row.event_id, row.attempt
        first = db.scalars(select(WebhookDeliveryRow.created_at).where(WebhookDeliveryRow.event_id == event_id,
                                                                       WebhookDeliveryRow.hook_id == hook.id,
                                                                       WebhookDeliveryRow.attempt == 1)).first() or row.created_at
    headers = {"Content-Type": "application/json", "User-Agent": f"Veritas/{settings.version}", SIGNATURE_HEADER: sign(secret, body, int(time.time())),
               EVENT_HEADER: event, DELIVERY_HEADER: event_id}
    now, started, code, error, refused = utcnow(), time.perf_counter(), None, None, False
    try:
        code = int((post or _post)(url, body, headers).status_code)
        if not 200 <= code < 300:
            error = f"HTTP {code}"
    except AddressNotAllowedError as e:  # nothing was sent; retrying would not change the answer
        error, refused = REFUSED_ADDRESS, True
        log.warning("webhooks.address_refused", delivery_id=delivery_id, webhook_event=event, attempt=attempt, reason=str(e))
    except Exception as e:  # noqa: BLE001 - any transport failure is an attempt that failed, recorded on its row
        error = _error_category(e)
        log.warning("webhooks.attempt_error", delivery_id=delivery_id, webhook_event=event, attempt=attempt, error_category=error,
                    exception=f"{type(e).__name__}: {e}")
    next_at = now + timedelta(seconds=backoff_s(attempt))
    retry_id = None
    with get_session() as db:
        row = db.get(WebhookDeliveryRow, delivery_id)
        if row is None:  # the hook was deleted during the attempt
            return "done"
        row.attempted_at, row.status_code, row.error, row.duration_ms = now, code, error, round((time.perf_counter() - started) * 1000)
        if error is None:
            row.status, row.delivered_at = "delivered", utcnow()
        elif refused or next_at - _aware(first) > timedelta(seconds=settings.webhook_retry_window_s):
            row.status = "abandoned"
        else:
            row.status, row.next_attempt_at = "failed", next_at
            follow = WebhookDeliveryRow(id=str(uuid.uuid4()), hook_id=row.hook_id, event=row.event, event_id=row.event_id, payload=row.payload,
                                        payload_sha256=row.payload_sha256, attempt=attempt + 1, status="pending", created_at=now,
                                        next_attempt_at=next_at)
            db.add(follow)
            retry_id = follow.id
        status = row.status
        db.commit()
    if retry_id is not None:
        retry(retry_id, next_at)
    log.info("webhooks.attempt", delivery_id=delivery_id, webhook_event=event, attempt=attempt, status=status, status_code=code)
    return "done"
