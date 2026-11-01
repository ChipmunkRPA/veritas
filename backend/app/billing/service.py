"""USD 9.99 per analysis, with payment verified before any analysis job exists.

Only Stripe receives payment details. Uploaded content, filenames and findings never
leave this service. A signed event is checked against a fresh Stripe Session read;
its order, price, quantity and mode must all agree with the immutable local order.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
import uuid
from datetime import UTC, timedelta
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api import jobs
from app.api.auth import Principal
from app.audit import chain as audit
from app.cases import intake
from app.core.config import settings
from app.storage import files, locks
from app.storage.db import Analysis, BillingEventRow, BillingOrderRow, CaseRow, get_session, utcnow
from app.storage.objects import ObjectStoreError, default_store

AMOUNT_MINOR, CURRENCY = 999, "usd"
STRIPE_API_VERSION = "2026-10-28.endive"
SESSION_ID = re.compile(r"cs_(?:test_|live_)?[A-Za-z0-9_]+\Z")
EVENT_ID = re.compile(r"evt_[A-Za-z0-9_]+\Z")
SIGNATURE_TOLERANCE_S = 300
BLOCKED_SUBMISSION_ROUTES = frozenset({
    "/api/analyze/{kind}", "/api/analyze-text", "/api/demo/start", "/api/v2/cases/{case_id}/items",
    "/api/v2/batches", "/api/v2/analyses/{aid}/rerun", "/api/v2/analyses/{aid}/reanalyse",
    "/api/v2/analyses/{aid}/refusion",
})


class BillingError(Exception):
    def __init__(self, status: int, detail: str, *, order_id: str | None = None):
        super().__init__(detail)
        self.status, self.detail, self.order_id = status, detail, order_id


def enabled() -> bool:
    return settings.billing_mode != "disabled"


def guard_submission(method: str, route: str) -> None:
    if enabled() and method == "POST" and route in BLOCKED_SUBMISSION_ROUTES:
        raise HTTPException(402, "Each new analysis costs US$9.99. Start it through secure checkout.")


def _key() -> str:
    return settings.stripe_secret_key.get_secret_value() if settings.stripe_secret_key else ""


def _webhook_key() -> str:
    return settings.stripe_webhook_secret.get_secret_value() if settings.stripe_webhook_secret else ""


def ready() -> bool:
    if not enabled():
        return False
    mode = settings.billing_mode
    origin = urlsplit(settings.billing_app_origin)
    good_origin = (origin.scheme == "https" and bool(origin.hostname)) or (
        mode == "test" and origin.scheme == "http" and origin.hostname in {"localhost", "127.0.0.1", "::1"})
    good_origin = good_origin and origin.path in {"", "/"} and not (origin.query or origin.fragment or origin.username or origin.password)
    return bool(good_origin and _key().startswith((f"sk_{mode}_", f"rk_{mode}_")) and _webhook_key().startswith("whsec_"))


def configuration() -> dict:
    return {"enabled": enabled(), "mode": settings.billing_mode, "ready": ready(), "amount_minor": AMOUNT_MINOR,
            "currency": CURRENCY, "price_display": "US$9.99", "requires_sign_in": True}


def require_ready() -> None:
    if not enabled():
        raise BillingError(409, "Payments are not enabled on this instance.")
    if not ready():
        raise BillingError(503, "Secure checkout is not configured yet. No payment or analysis was started.")


def require_owner_principal(principal: Principal) -> None:
    if principal.unauthenticated_instance or principal.auth in {"single_user", "token"}:
        raise BillingError(401, "Sign in before starting a paid analysis.")


def _stripe(method: str, path: str, *, data: dict | None = None, params: dict | None = None,
            idempotency_key: str | None = None) -> dict:
    headers = {"Stripe-Version": STRIPE_API_VERSION}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    try:
        # Never follow a redirect with the secret; the API host is a constant.
        with httpx.Client(timeout=20.0, follow_redirects=False) as client:
            response = client.request(method, f"https://api.stripe.com/v1/{path}", auth=(_key(), ""), headers=headers,
                                      data=data, params=params)
        if response.status_code >= 300:
            raise BillingError(503, "Stripe could not complete this request. Retry this saved checkout; no new order is needed.")
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError("Unexpected response")
        return result
    except (httpx.HTTPError, ValueError):
        # Stripe response bodies and request objects can contain sensitive details. Do not log or return them.
        raise BillingError(503, "Stripe is temporarily unavailable. Retry this saved checkout; no new order is needed.") from None


def _owned(order_id: str, owner_id: str) -> BillingOrderRow:
    with get_session() as db:
        order = db.get(BillingOrderRow, order_id)
        if order is None or order.owner_id != owner_id:
            raise BillingError(404, "Order not found.")
        db.expunge(order)
        return order


def order_status(order_id: str, owner_id: str) -> dict:
    order = _owned(order_id, owner_id)
    with get_session() as db:
        analysis = db.get(Analysis, order.analysis_id) if order.analysis_id else None
        return {"order_id": order.id, "status": order.status, "analysis_id": order.analysis_id,
                "analysis_status": analysis.status if analysis else None, "amount_minor": order.amount_minor,
                "currency": order.currency}


def _metadata(order: BillingOrderRow) -> dict[str, str]:
    return {"chipmunk_order_id": order.id, "chipmunk_owner_id": order.owner_id, "chipmunk_sha256": order.sha256}


def _same_upload(order: BillingOrderRow, *, sha256: str, kind: str, profile: str) -> None:
    if (order.sha256, order.kind, order.profile) != (sha256, kind, profile):
        raise BillingError(409, "This checkout request already belongs to another file or analysis profile.", order_id=order.id)


def create_order(src: BinaryIO, filename: str, kind: str, *, owner_id: str, request_key: str, profile: str,
                 client_sha256: str | None = None) -> BillingOrderRow:
    require_ready()
    if kind not in intake.KIND_LIMITS:
        raise BillingError(422, "Choose image, video, audio or document.")
    # Bind a repeated upload request to its bytes before accepting its idempotency key.
    _head, sha256, _md5, _size = intake._digest(src, intake.KIND_LIMITS[kind])
    if client_sha256 and client_sha256.lower() != sha256:
        raise BillingError(422, "The file changed in transit (hash mismatch).")
    with get_session() as db:
        previous = db.scalar(select(BillingOrderRow).where(BillingOrderRow.owner_id == owner_id,
                                                          BillingOrderRow.request_key == request_key))
        if previous is not None:
            _same_upload(previous, sha256=sha256, kind=kind, profile=profile)
            db.expunge(previous)
            return previous
    src.seek(0)
    if _size == 0:
        raise BillingError(400, "The uploaded file is empty.")
    try:
        files.validate_kind(kind, _head, filename, size=_size)
    except ValueError as error:
        raise BillingError(415, str(error)) from None
    with intake.upload_slot():
        from app.billing.preflight import PreflightError, validate_upload

        try:
            validate_upload(src, filename, kind)
        except PreflightError as error:
            raise BillingError(422, str(error)) from None
        received = intake.take_in(src, filename, kind, principal_id=owner_id, analyze=False,
                                 acquisition=intake.Acquisition(method="quick_upload", client_sha256=client_sha256))
    # The shared media object's first-ever filename may belong to another customer.
    # Preserve this upload's name instead of exposing that other customer's name.
    shown = files.received_filename(filename)
    if kind == "document":
        _mime, ext = files.validate_kind(kind, _head, filename, size=_size)
        if not shown.lower().endswith(ext):
            shown = files.received_filename(f"{Path(shown).stem}{ext}")
    order = BillingOrderRow(id=str(uuid.uuid4()), owner_id=owner_id, request_key=request_key, case_id=received.case.id,
                            media_id=received.media.id, kind=kind, filename=shown,
                            content_type=received.media.mime_sniffed or "application/octet-stream", sha256=received.media.sha256,
                            size_bytes=received.media.size_bytes, profile=profile, amount_minor=AMOUNT_MINOR, currency=CURRENCY,
                            mode=settings.billing_mode, status="pending", created_at=utcnow())
    try:
        with get_session() as db:
            db.add(order)
            db.commit()
            db.refresh(order)
            db.expunge(order)
    except IntegrityError:
        # Two simultaneous retries may stage the same bytes; only the unique owner/request pair can purchase them.
        with get_session() as db:
            order = db.scalar(select(BillingOrderRow).where(BillingOrderRow.owner_id == owner_id,
                                                           BillingOrderRow.request_key == request_key))
            if order is None:
                raise
            _same_upload(order, sha256=sha256, kind=kind, profile=profile)
            db.expunge(order)
    return order


def checkout(order_id: str, owner_id: str) -> dict:
    require_ready()
    order = _owned(order_id, owner_id)
    if order.mode != settings.billing_mode:
        raise BillingError(409, "This order belongs to another payment environment. Start a new checkout.", order_id=order.id)
    if order.status in {"fulfilled", "expired", "payment_failed"}:
        raise BillingError(409, "This checkout has already completed, expired or failed.", order_id=order.id)
    if order.checkout_url:
        return {"order_id": order.id, "checkout_url": order.checkout_url, "status": order.status}
    # Stripe keeps idempotency keys for at least 24 hours. Never reuse an unknown-outcome request after that window.
    created = order.created_at.replace(tzinfo=UTC) if order.created_at.tzinfo is None else order.created_at
    if utcnow() > created + timedelta(hours=23):
        raise BillingError(409, "This saved checkout is too old to retry. Contact sales@chipmunkrpa.com before paying again.", order_id=order.id)
    origin = settings.billing_app_origin.rstrip("/")
    # Stable expiration and URLs ensure an unknown-outcome retry has exactly the same Stripe parameters.
    expires_at = int(created.timestamp()) + 24 * 3600
    data = {"mode": "payment", "locale": "en", "client_reference_id": order.id,
            "success_url": f"{origin}/billing/return?order_id={order.id}",
            "cancel_url": f"{origin}/billing/return?order_id={order.id}&cancelled=1", "expires_at": expires_at,
            "line_items[0][price_data][currency]": order.currency, "line_items[0][price_data][unit_amount]": order.amount_minor,
            "line_items[0][price_data][product_data][name]": "Chipmunk Verify — one analysis",
            "line_items[0][quantity]": 1, "allow_promotion_codes": "false",
            "automatic_tax[enabled]": "false", "adaptive_pricing[enabled]": "false"}
    data.update({f"metadata[{key}]": value for key, value in _metadata(order).items()})
    try:
        session = _stripe("POST", "checkout/sessions", data=data, idempotency_key=f"chipmunk-order-{order.id}")
        session_id, url = session.get("id", ""), session.get("url", "")
        target = urlsplit(url)
        if (not SESSION_ID.fullmatch(session_id) or session.get("livemode") is not (order.mode == "live")
                or target.scheme != "https" or target.hostname != "checkout.stripe.com" or target.username or target.password):
            raise BillingError(503, "Stripe returned an unexpected checkout. No analysis was started.")
        with get_session() as db:
            locks.serialise(db, "billing-order", order.id)
            saved = db.get(BillingOrderRow, order.id)
            if saved.stripe_session_id not in {None, session_id}:
                raise BillingError(409, "The saved checkout has changed. Contact sales@chipmunkrpa.com.")
            saved.stripe_session_id, saved.checkout_url = session_id, url
            if saved.status == "pending":
                saved.status = "awaiting_payment"
            db.commit()
        return {"order_id": order.id, "checkout_url": url, "status": "awaiting_payment"}
    except BillingError as error:
        error.order_id = order.id
        raise


def verify_event(payload: bytes, signature: str) -> dict:
    require_ready()
    try:
        fields = [piece.split("=", 1) for piece in signature.split(",")]
        timestamps = [value for key, value in fields if key == "t"]
        signatures = [value for key, value in fields if key == "v1"]
        if len(timestamps) != 1:
            raise ValueError("Timestamp")
        timestamp = int(timestamps[0])
        if abs(time.time() - timestamp) > SIGNATURE_TOLERANCE_S:
            raise ValueError("Expired")
        expected = hmac.new(_webhook_key().encode(), str(timestamp).encode() + b"." + payload, hashlib.sha256).hexdigest()
        if not any(hmac.compare_digest(expected, value) for value in signatures):
            raise ValueError("Signature")
        event = json.loads(payload)
        if not isinstance(event, dict) or not EVENT_ID.fullmatch(event.get("id", "")):
            raise ValueError("Event")
        if event.get("livemode") is not (settings.billing_mode == "live"):
            raise ValueError("Mode")
        return event
    except (ValueError, TypeError, KeyError):
        raise BillingError(400, "Invalid Stripe webhook signature or payment environment.") from None


def _validate_session(session: dict, order: BillingOrderRow, session_id: str, *, paid: bool) -> None:
    checks = [session.get("id") == session_id == order.stripe_session_id,
              session.get("object") == "checkout.session", session.get("mode") == "payment",
              session.get("livemode") is (order.mode == "live"), order.mode == settings.billing_mode,
              session.get("client_reference_id") == order.id,
              session.get("amount_total") == order.amount_minor == AMOUNT_MINOR,
              session.get("amount_subtotal") == order.amount_minor, session.get("currency") == order.currency == CURRENCY]
    metadata = session.get("metadata") or {}
    checks.append(all(metadata.get(k) == v for k, v in _metadata(order).items()))
    if paid:
        items = session.get("line_items") or {}
        rows = items.get("data") or []
        checks.extend([session.get("payment_status") == "paid", session.get("status") == "complete",
                       isinstance(session.get("payment_intent"), str) and session["payment_intent"].startswith("pi_"),
                       items.get("has_more") is False, len(rows) == 1])
        if len(rows) == 1:
            item, price = rows[0], rows[0].get("price") or {}
            checks.extend([item.get("quantity") == 1, item.get("amount_total") == order.amount_minor,
                           item.get("amount_subtotal") == order.amount_minor, item.get("currency") == order.currency,
                           price.get("unit_amount") == order.amount_minor, price.get("currency") == order.currency,
                           price.get("type") == "one_time"])
    if not all(checks):
        raise BillingError(400, "Payment does not match the saved analysis order.")


def process_event(event: dict) -> dict:
    event_type = event.get("type")
    if event_type not in {"checkout.session.completed", "checkout.session.async_payment_succeeded", "checkout.session.expired",
                          "checkout.session.async_payment_failed"}:
        return {"received": True}
    with get_session() as db:
        if db.get(BillingEventRow, event["id"]) is not None:
            return {"received": True}
    event_session = (event.get("data") or {}).get("object") or {}
    session_id = event_session.get("id", "")
    if not SESSION_ID.fullmatch(session_id):
        raise BillingError(400, "Invalid checkout session.")
    # Do not trust a browser redirect or even event fields as the final payment state.
    session = _stripe("GET", f"checkout/sessions/{session_id}", params={"expand[]": "line_items"})
    order_id = (session.get("metadata") or {}).get("chipmunk_order_id")
    with get_session() as db:
        order = db.get(BillingOrderRow, order_id) if isinstance(order_id, str) else None
        if order is None:
            # Another Stripe product on the same account has no local order to fulfill.
            return {"received": True}
        if order.stripe_session_id is None:
            raise BillingError(503, "Checkout registration is pending. Retry this webhook.")
        _validate_session(session, order, session_id, paid=session.get("payment_status") == "paid")
        db.expunge(order)
    if event_type in {"checkout.session.expired", "checkout.session.async_payment_failed"}:
        expected = "expired" if event_type == "checkout.session.expired" else "complete"
        if session.get("status") != expected or session.get("payment_status") == "paid":
            return {"received": True}
        with get_session() as db:
            locks.serialise(db, "billing-order", order.id)
            current = db.get(BillingOrderRow, order.id)
            if current.analysis_id is None:
                current.status = "expired" if event_type == "checkout.session.expired" else "payment_failed"
            if db.get(BillingEventRow, event["id"]) is None:
                db.add(BillingEventRow(id=event["id"], order_id=order.id))
            db.commit()
        return {"received": True}
    if session.get("payment_status") != "paid":
        # Dynamic payment methods can complete Checkout before funds clear. Stripe
        # sends async_payment_succeeded later; completion alone never releases work.
        return {"received": True}
    with get_session() as db:
        locks.serialise(db, "billing-order", order.id)
        current = db.get(BillingOrderRow, order.id)
        if current.analysis_id is not None:
            # A later distinct event must still acknowledge a fulfilled purchase
            # after the analysis/original has reached its retention date.
            if db.get(BillingEventRow, event["id"]) is None:
                db.add(BillingEventRow(id=event["id"], order_id=order.id))
            db.commit()
            return {"received": True}
    store = default_store()
    try:
        valid_original = store.verify(order.sha256) == order.size_bytes
    except ObjectStoreError:
        valid_original = False
    if not valid_original:
        raise BillingError(503, "The uploaded file failed its integrity check. Payment fulfillment will be retried.")
    aid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"chipmunk-verify:paid-analysis:{order.id}"))

    def fulfill() -> None:
        with audit.writing() as db:
            locks.serialise(db, "billing-order", order.id)
            current = db.get(BillingOrderRow, order.id)
            if db.get(BillingEventRow, event["id"]) is not None:
                return
            if current.analysis_id is not None:
                db.add(BillingEventRow(id=event["id"], order_id=order.id))
                db.commit()
                return
            if db.get(CaseRow, order.case_id) is None:
                raise BillingError(503, "The saved upload is unavailable. Payment fulfillment needs assistance.")
            intake._link_upload(aid, files.safe_filename(order.filename), store.path_for(order.sha256))
            analysis = Analysis(id=aid, kind=order.kind, filename=order.filename, content_type=order.content_type,
                                size_bytes=order.size_bytes, sha256=order.sha256, status="queued", case_id=order.case_id,
                                media_id=order.media_id, submitted_by=order.owner_id, profile=order.profile)
            db.add(analysis)
            db.flush()
            jobs.submit_in(db, aid)
            current.analysis_id, current.status, current.paid_at = aid, "fulfilled", utcnow()
            current.payment_intent_id = session["payment_intent"]
            db.add(BillingEventRow(id=event["id"], order_id=order.id))
            audit.commit_with(db, [{"action": "intake", "actor_id": order.owner_id, "case_id": order.case_id,
                                    "analysis_id": aid, "media_id": order.media_id, "after_hash": order.sha256,
                                    "details": {"acquisition_method": "quick_upload", "billing_order_id": order.id,
                                                "payment_verified": True, "amount_minor": order.amount_minor,
                                                "currency": order.currency}}])
    audit.repeated(fulfill)
    jobs.wake()
    return {"received": True}
