"""Isolated real database/auth/queue tests; Stripe HTTP responses are test doubles, not a live charge."""
from __future__ import annotations

import copy
import hashlib
import hmac
import io
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api import auth, jobs, routes, security, tokens, v2_billing
from app.api.v2 import install_problem_handlers
from app.audit import chain as audit
from app.billing import purchase_notice, service
from app.cases import intake
from app.core.canonical import canonical_hash
from app.core.config import settings
from app.engine.queue import JobQueue
from app.storage.db import Analysis, Base, BillingEventRow, BillingOrderRow, Job
from app.storage.objects import ObjectStore


@pytest.fixture
def billing(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'billing.sqlite'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    store = ObjectStore(tmp_path / "objects")
    for key, value in {"billing_mode": "test", "stripe_secret_key": SecretStr("sk_test_isolated_test_only"),
                       "stripe_webhook_secret": SecretStr("whsec_isolated_test_only"),
                       "billing_app_origin": "http://127.0.0.1:5230", "data_dir": tmp_path,
                       "min_free_bytes": 0, "api_key": "test-bootstrap-key", "single_user": True,
                       "api_key_pepper": "isolated-test-pepper", "signing_key": "isolated-token-key"}.items():
        monkeypatch.setattr(settings, key, value)

    @contextmanager
    def sessions():
        with Session(engine, expire_on_commit=False) as db:
            yield db

    for module in (auth, security, service, intake, audit, routes):
        monkeypatch.setattr(module, "get_session", sessions)
    for module in (service, intake, audit):
        monkeypatch.setattr(module, "default_store", lambda: store)
    monkeypatch.setattr(jobs, "get_queue", lambda: JobQueue(engine))
    monkeypatch.setattr(jobs, "wake", lambda: None)
    monkeypatch.setattr(security, "limiter", security.RateLimiter(10000))
    monkeypatch.setattr(security, "credential_limiter", security.RateLimiter(10000))
    with sessions() as db:
        owner, outsider = auth.new_principal("payer", "examiner"), auth.new_principal("outsider", "examiner")
        db.add_all([owner, outsider])
        db.flush()
        key, _ = auth.issue_api_key(db, owner)
        other_key, _ = auth.issue_api_key(db, outsider)
        db.commit()
    png = io.BytesIO()
    Image.new("RGB", (16, 16), (30, 50, 90)).save(png, "PNG")
    app = FastAPI()
    install_problem_handlers(app)
    app.include_router(v2_billing.public_router)
    app.include_router(v2_billing.router)
    app.include_router(routes.router)
    fixture = SimpleNamespace(engine=engine, sessions=sessions, store=store, client=TestClient(app), owner=owner,
                              outsider=outsider, headers={"X-API-Key": key}, other_headers={"X-API-Key": other_key},
                              image=png.getvalue(), created=[], retrieved=[], remote={})

    def provider(method, path, *, data=None, params=None, idempotency_key=None):
        if method == "POST":
            fixture.created.append((copy.deepcopy(data), idempotency_key))
            sid = "cs_test_" + data["client_reference_id"].replace("-", "")
            session = {"id": sid, "object": "checkout.session", "mode": "payment", "livemode": False,
                       "client_reference_id": data["client_reference_id"], "amount_total": 999, "amount_subtotal": 999,
                       "currency": "usd", "metadata": {k[9:-1]: v for k, v in data.items() if k.startswith("metadata[")},
                       "payment_status": "unpaid", "status": "open", "payment_intent": "pi_test_purchase",
                       "url": f"https://checkout.stripe.com/c/pay/{sid}",
                       "line_items": {"has_more": False, "data": [{"quantity": 1, "amount_total": 999,
                          "amount_subtotal": 999, "currency": "usd", "price": {"unit_amount": 999, "currency": "usd", "type": "one_time"}}]}}
            fixture.remote.setdefault(sid, session)
            return copy.deepcopy(fixture.remote[sid])
        fixture.retrieved.append(path)
        return copy.deepcopy(fixture.remote[path.rsplit("/", 1)[-1]])

    monkeypatch.setattr(service, "_stripe", provider)
    yield fixture
    fixture.client.close()
    engine.dispose()


def upload(f, *, request_key=None, headers=None, content=None, filename="private-name.png", acknowledgement=True,
           notice_version=None, notice_sha256=None):
    data = {"kind": "image", "profile": "full"}
    if acknowledgement is not None:
        data.update(risk_accepted=str(acknowledgement).lower(), risk_notice_version=notice_version or purchase_notice.VERSION,
                    risk_notice_sha256=notice_sha256 or purchase_notice.SHA256)
    return f.client.post("/api/v2/billing/checkout", headers={**(headers or f.headers),
                         "Idempotency-Key": request_key or str(uuid.uuid4())},
                         data=data,
                         files={"file": (filename, content if content is not None else f.image, "image/png")})


@pytest.mark.parametrize("options", [
    {"acknowledgement": None}, {"acknowledgement": False},
    {"notice_version": "outdated-notice"}, {"notice_sha256": "0" * 64},
])
def test_no_order_payment_or_analysis_without_current_explicit_acknowledgement(billing, options):
    response = upload(billing, **options)
    assert response.status_code == 422
    assert "accept" in response.json()["detail"].lower()
    assert billing.created == []
    with billing.sessions() as db:
        assert db.query(BillingOrderRow).count() == db.query(Analysis).count() == db.query(Job).count() == 0


def test_purchase_receipt_and_hosted_warning_are_bound_to_the_order(billing):
    key = str(uuid.uuid4())
    first = upload(billing, request_key=key)
    assert first.status_code == 201
    oid = first.json()["order_id"]
    with billing.sessions() as db:
        receipt = copy.deepcopy(db.get(BillingOrderRow, oid).purchase_acknowledgement)
    assert receipt["accepted_by"] == billing.owner.id
    assert receipt["notice"] == purchase_notice.NOTICE
    assert receipt["notice_sha256"] == purchase_notice.SHA256
    assert receipt["accepted_at"]
    assert billing.created[0][0]["custom_text[submit][message]"] == purchase_notice.NOTICE["stripe_message"]
    assert "private-name.png" not in json.dumps(billing.created)
    assert upload(billing, request_key=key).json()["order_id"] == oid
    with billing.sessions() as db:
        assert db.get(BillingOrderRow, oid).purchase_acknowledgement == receipt
    assert len(billing.created) == 1


def test_saved_v2_receipt_remains_valid_when_new_checkout_requires_v3(billing):
    archived = json.loads(Path(purchase_notice.__file__).with_name("purchase_notice_v2.json").read_text())
    digest = canonical_hash(archived)
    assert digest == "22024573c8f354c278dd20bdf387c7cf3de9efcaf8a425629c56f409f4d6d826"
    assert upload(billing, notice_version=archived["version"], notice_sha256=digest).status_code == 422

    order_id = upload(billing).json()["order_id"]
    with billing.sessions() as db:
        order = db.get(BillingOrderRow, order_id)
        old_receipt = {**order.purchase_acknowledgement, "notice": archived, "notice_sha256": digest}
        order.purchase_acknowledgement = old_receipt
        db.commit()
    path = f"/api/v2/billing/orders/{order_id}/checkout"
    assert billing.client.post(path, headers=billing.headers).status_code == 200
    with billing.sessions() as db:
        assert db.get(BillingOrderRow, order_id).purchase_acknowledgement == old_receipt


def test_old_checkout_requires_acknowledgement_on_resume(billing):
    oid = upload(billing).json()["order_id"]
    with billing.sessions() as db:
        db.get(BillingOrderRow, oid).purchase_acknowledgement = None
        db.commit()
    path = f"/api/v2/billing/orders/{oid}/checkout"
    assert billing.client.post(path, headers=billing.headers).status_code == 422
    accepted = {"risk_accepted": True, "risk_notice_version": purchase_notice.VERSION,
                "risk_notice_sha256": purchase_notice.SHA256}
    assert billing.client.post(path, headers=billing.other_headers, json=accepted).status_code == 404
    assert billing.client.post(path, headers=billing.headers, json=accepted).status_code == 200
    with billing.sessions() as db:
        receipt = db.get(BillingOrderRow, oid).purchase_acknowledgement
        assert purchase_notice.recorded(receipt, billing.owner.id)
        assert receipt["stripe_notice"] is False
    assert len(billing.created) == 1


def test_cannot_record_retrospective_acceptance_for_a_completed_order(billing):
    oid = upload(billing).json()["order_id"]
    with billing.sessions() as db:
        order = db.get(BillingOrderRow, oid)
        order.purchase_acknowledgement, order.status = None, "fulfilled"
        db.commit()
    response = billing.client.post(f"/api/v2/billing/orders/{oid}/checkout", headers=billing.headers,
                                   json={"risk_accepted": True, "risk_notice_version": purchase_notice.VERSION,
                                         "risk_notice_sha256": purchase_notice.SHA256})
    assert response.status_code == 409
    with billing.sessions() as db:
        assert db.get(BillingOrderRow, oid).purchase_acknowledgement is None


def event_for(f, order_id, *, event_id="evt_test_paid", paid=True):
    with f.sessions() as db:
        order = db.get(BillingOrderRow, order_id)
        sid = order.stripe_session_id
    if paid:
        f.remote[sid].update(payment_status="paid", status="complete")
    return {"id": event_id, "type": "checkout.session.completed", "livemode": False,
            "data": {"object": {"id": sid}}}


def signed(f, event, *, timestamp=None, bad=False):
    payload = json.dumps(event, separators=(",", ":")).encode()
    timestamp = int(time.time()) if timestamp is None else timestamp
    signature = hmac.new(b"whsec_isolated_test_only", str(timestamp).encode() + b"." + payload, hashlib.sha256).hexdigest()
    return f.client.post("/api/v2/billing/webhook", content=payload,
                         headers={"Stripe-Signature": f"t={timestamp},v1={'0' * 64 if bad else signature}"})


def counts(f):
    with f.sessions() as db:
        return db.query(Analysis).count(), db.query(Job).count(), db.query(BillingEventRow).count()


def test_no_analysis_until_verified_payment_and_owner_isolation(billing):
    f = billing
    response = upload(f)
    assert response.status_code == 201, response.text
    oid = response.json()["order_id"]
    assert counts(f) == (0, 0, 0)
    assert f.client.get(f"/api/v2/billing/orders/{oid}", headers=f.other_headers).status_code == 404
    assert f.client.get(f"/api/v2/billing/orders/{oid}").status_code == 401
    assert signed(f, event_for(f, oid, paid=False)).status_code == 200
    assert counts(f) == (0, 0, 0)
    event = event_for(f, oid)
    assert signed(f, event).status_code == 200
    assert counts(f) == (1, 1, 1)
    status = f.client.get(f"/api/v2/billing/orders/{oid}", headers=f.headers).json()
    assert status["status"] == "fulfilled" and status["analysis_status"] == "queued"
    aid = status["analysis_id"]
    assert f.client.get(f"/api/analyses/{aid}", headers=f.headers).status_code == 200
    assert f.client.get(f"/api/analyses/{aid}", headers=f.other_headers).status_code == 404
    assert f.client.get(f"/api/analyses/{aid}").status_code == 401
    # Returning to a URL / asking for status never creates another job.
    assert counts(f) == (1, 1, 1)
    with f.sessions() as db:
        analysis = db.get(Analysis, aid)
        # The real fulfillment transaction must produce the receipt the worker
        # requires, not merely an analysis id for the browser's return page.
        from app.billing.entitlements import require_analysis_payment

        require_analysis_payment(db, analysis)
        assert analysis.submitted_by == f.owner.id
        assert (settings.uploads_dir / aid / "private-name.png").read_bytes() == f.image
        analysis.status = "failed"
        db.commit()
    assert f.client.get(f"/api/v2/billing/orders/{oid}", headers=f.headers).json()["analysis_status"] == "failed"
    assert upload(f).json()["order_id"] != oid  # explicit new upload is a separate analysis purchase


def test_repeated_uploads_and_unknown_outcome_retry_share_one_checkout(billing, monkeypatch):
    f = billing
    request_key = str(uuid.uuid4())
    real_provider = service._stripe
    raised = False

    def timeout_after_creation(*args, **kwargs):
        nonlocal raised
        result = real_provider(*args, **kwargs)
        if not raised:
            raised = True
            raise service.BillingError(503, "Temporary test timeout")
        return result

    monkeypatch.setattr(service, "_stripe", timeout_after_creation)
    first = upload(f, request_key=request_key)
    assert first.status_code == 503 and first.json()["order_id"]
    second = upload(f, request_key=request_key)
    third = upload(f, request_key=request_key)
    assert second.status_code == third.status_code == 201
    assert first.json()["order_id"] == second.json()["order_id"] == third.json()["order_id"]
    assert len(f.remote) == 1 and f.created[0] == f.created[1]
    assert counts(f) == (0, 0, 0)
    assert upload(f, request_key=request_key, content=f.image + b"different").status_code == 409
    assert "private-name" not in json.dumps(f.created)
    with f.sessions() as db:
        assert db.query(BillingOrderRow).count() == 1


def test_duplicate_and_concurrent_events_queue_once(billing):
    f = billing
    oid = upload(f).json()["order_id"]
    event = event_for(f, oid)
    events = [event, event, {**event, "id": "evt_test_second"}, {**event, "id": "evt_test_third"}]
    with ThreadPoolExecutor(max_workers=4) as executor:
        statuses = list(executor.map(lambda e: signed(f, e).status_code, events))
    assert statuses == [200] * 4
    assert counts(f) == (1, 1, 3)
    assert signed(f, event).status_code == 200
    assert counts(f) == (1, 1, 3)


@pytest.mark.parametrize("mutation", ["amount", "currency", "quantity", "unit_amount", "livemode", "owner", "session", "metadata", "line_items"])
def test_wrong_price_quantity_mode_and_binding_never_fulfill(billing, mutation):
    f = billing
    oid = upload(f).json()["order_id"]
    event = event_for(f, oid)
    session = f.remote[event["data"]["object"]["id"]]
    if mutation == "amount":
        session["amount_total"] = 998
    elif mutation == "currency":
        session["currency"] = "eur"
    elif mutation == "quantity":
        session["line_items"]["data"][0]["quantity"] = 2
    elif mutation == "unit_amount":
        session["line_items"]["data"][0]["price"]["unit_amount"] = 499
    elif mutation == "livemode":
        session["livemode"] = True
    elif mutation == "owner":
        session["metadata"]["chipmunk_owner_id"] = f.outsider.id
    elif mutation == "session":
        session["id"] = "cs_test_other"
    elif mutation == "metadata":
        session["metadata"]["chipmunk_sha256"] = "0" * 64
    elif mutation == "line_items":
        session["line_items"]["has_more"] = True
    assert signed(f, event).status_code == 400
    assert counts(f) == (0, 0, 0)


def test_bad_signatures_stale_signatures_and_unsigned_success_are_refused(billing):
    f = billing
    oid = upload(f).json()["order_id"]
    event = event_for(f, oid)
    assert signed(f, event, bad=True).status_code == 400
    assert signed(f, event, timestamp=int(time.time()) - 400).status_code == 400
    assert f.client.post("/api/v2/billing/webhook", json=event).status_code == 400
    assert counts(f) == (0, 0, 0) and not f.retrieved


def test_browser_success_claims_and_status_polling_never_release_unpaid_work(billing):
    f = billing
    oid = upload(f).json()["order_id"]
    for suffix in ("", "?paid=true&payment_status=paid&status=complete", "?cancelled=1"):
        response = f.client.get(f"/api/v2/billing/orders/{oid}{suffix}", headers=f.headers)
        assert response.status_code == 200
        assert response.json()["status"] == "awaiting_payment"
        assert response.json()["analysis_id"] is None
        assert counts(f) == (0, 0, 0)


def test_stripe_verification_outage_does_not_release_work(billing, monkeypatch):
    f = billing
    oid = upload(f).json()["order_id"]
    event = event_for(f, oid)
    provider = service._stripe

    def unavailable(*args, **kwargs):
        raise service.BillingError(503, "Stripe verification is temporarily unavailable.")

    monkeypatch.setattr(service, "_stripe", unavailable)
    assert signed(f, event).status_code == 503
    assert counts(f) == (0, 0, 0)
    assert f.client.get(f"/api/v2/billing/orders/{oid}", headers=f.headers).json()["analysis_id"] is None
    monkeypatch.setattr(service, "_stripe", provider)
    assert signed(f, event).status_code == 200
    assert counts(f) == (1, 1, 1)


def test_unconfigured_and_wrong_mode_fail_closed_before_upload(billing, monkeypatch):
    f = billing
    monkeypatch.setattr(settings, "stripe_webhook_secret", None)
    assert upload(f).status_code == 503
    assert not f.created
    with f.sessions() as db:
        assert db.query(BillingOrderRow).count() == 0
    monkeypatch.setattr(settings, "stripe_webhook_secret", SecretStr("whsec_fixture"))
    monkeypatch.setattr(settings, "billing_mode", "live")
    assert not service.ready() and upload(f).status_code == 503


def test_invalid_upload_refused_before_stripe(billing):
    assert upload(billing, content=b"not an image").status_code == 415
    assert not billing.created and counts(billing) == (0, 0, 0)


def test_paid_mode_closes_legacy_submission_routes_and_anonymous_tokens(billing, monkeypatch):
    f = billing
    assert f.client.post("/api/analyze/image", headers=f.headers).status_code == 402
    assert f.client.post("/api/analyze-text", headers=f.headers, json={"text": "hello"}).status_code == 402
    assert f.client.post("/api/analyze/image").status_code == 401
    assert not auth.single_user_enabled()
    old, _ = tokens.make_token("analysis:any", auth.LOCAL_ID, 60)
    assert f.client.get(f"/api/analyses/any?token={old}").status_code == 401
    for route in service.BLOCKED_SUBMISSION_ROUTES:
        with pytest.raises(Exception) as refused:
            service.guard_submission("POST", route)
        assert refused.value.status_code == 402
    monkeypatch.setattr(settings, "billing_mode", "disabled")
    assert auth.single_user_enabled()
    for route in service.BLOCKED_SUBMISSION_ROUTES:
        service.guard_submission("POST", route)
    # Existing free upload handler can run again, although this test intentionally omits its body.
    assert f.client.post("/api/analyze/image", headers=f.headers).status_code == 422


def test_filename_of_another_customers_identical_file_is_never_exposed(billing):
    first = upload(billing, filename="secret-other-customer-name.png").json()["order_id"]
    second = upload(billing, headers=billing.other_headers, filename="my-own-name.png").json()["order_id"]
    with billing.sessions() as db:
        a, b = db.get(BillingOrderRow, first), db.get(BillingOrderRow, second)
        assert a.media_id == b.media_id
        assert b.filename == "my-own-name.png"


def test_expired_checkout_does_not_queue_and_cannot_be_reused(billing):
    f = billing
    oid = upload(f).json()["order_id"]
    event = event_for(f, oid, paid=False)
    event["type"] = "checkout.session.expired"
    f.remote[event["data"]["object"]["id"]]["status"] = "expired"
    assert signed(f, event).status_code == 200
    assert f.client.get(f"/api/v2/billing/orders/{oid}", headers=f.headers).json()["status"] == "expired"
    assert f.client.post(f"/api/v2/billing/orders/{oid}/checkout", headers=f.headers).status_code == 409
    assert counts(f) == (0, 0, 1)


def test_billing_migration_matches_tables(tmp_path):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from app.storage.migrations import upgrade_to_head

    engine = create_engine(f"sqlite:///{tmp_path / 'migration.sqlite'}")
    upgrade_to_head(engine)
    with engine.connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert not differences, differences
    engine.dispose()


def test_real_app_all_analysis_entrypoints_are_payment_gated(billing):
    from app.main import app

    client = TestClient(app)
    for template in service.BLOCKED_SUBMISSION_ROUTES:
        path = template.replace("{kind}", "image").replace("{aid}", "any-analysis").replace("{case_id}", "any-case")
        response = client.post(path, headers={"X-API-Key": settings.api_key})
        assert response.status_code == 402, (path, response.status_code, response.text)
    client.close()


def test_disabled_mode_keeps_existing_free_upload_working(billing, monkeypatch):
    monkeypatch.setattr(settings, "billing_mode", "disabled")
    response = billing.client.post("/api/analyze/image", headers=billing.headers,
                                   files={"file": ("free.png", billing.image, "image/png")})
    assert response.status_code == 202, response.text
    assert counts(billing) == (1, 1, 0)
    assert not billing.created


def test_failed_queue_transaction_retries_without_duplicate_analysis(billing, monkeypatch):
    f = billing
    oid = upload(f).json()["order_id"]
    event = event_for(f, oid)
    submit = jobs.submit_in

    def unavailable_queue(*args, **kwargs):
        raise RuntimeError("isolated queue failure")

    monkeypatch.setattr(jobs, "submit_in", unavailable_queue)
    with pytest.raises(RuntimeError, match="isolated queue failure"):
        signed(f, event)
    assert counts(f) == (0, 0, 0)
    monkeypatch.setattr(jobs, "submit_in", submit)
    assert signed(f, event).status_code == 200
    assert counts(f) == (1, 1, 1)


def test_public_config_never_contains_credentials_and_secrets_do_not_enter_provenance(billing):
    from app.engine.build_info import sanitised_settings

    response = billing.client.get("/api/v2/billing/config")
    assert response.status_code == 200
    assert response.json() == {"enabled": True, "mode": "test", "ready": True, "amount_minor": 999,
                               "currency": "usd", "price_display": "US$9.99", "requires_sign_in": True, "purchase_notice": purchase_notice.public_notice()}
    assert "sk_test_" not in response.text and "whsec_" not in response.text
    assert "stripe_secret_key" not in sanitised_settings()
    assert "stripe_webhook_secret" not in sanitised_settings()


def test_delayed_payment_waits_for_paid_session_and_failed_payment_is_visible(billing):
    f = billing
    oid = upload(f).json()["order_id"]
    assert "payment_method_types[0]" not in f.created[0][0]
    pending = event_for(f, oid, event_id="evt_async_pending", paid=False)
    sid = pending["data"]["object"]["id"]
    f.remote[sid]["status"] = "complete"
    assert signed(f, pending).status_code == 200
    assert counts(f) == (0, 0, 0)
    failed = {**pending, "type": "checkout.session.async_payment_failed", "id": "evt_async_failed"}
    assert signed(f, failed).status_code == 200
    assert f.client.get(f"/api/v2/billing/orders/{oid}", headers=f.headers).json()["status"] == "payment_failed"
    assert counts(f) == (0, 0, 1)
    succeeded = event_for(f, oid, event_id="evt_async_success", paid=True)
    succeeded["type"] = "checkout.session.async_payment_succeeded"
    assert signed(f, succeeded).status_code == 200
    assert counts(f) == (1, 1, 2)


def test_distinct_duplicate_after_original_retention_still_acknowledges(billing):
    f = billing
    oid = upload(f).json()["order_id"]
    paid = event_for(f, oid)
    assert signed(f, paid).status_code == 200
    with f.sessions() as db:
        sha256 = db.get(BillingOrderRow, oid).sha256
    f.store.path_for(sha256).unlink()
    assert signed(f, {**paid, "id": "evt_later_duplicate"}).status_code == 200
    assert counts(f) == (1, 1, 2)


def test_missing_original_before_fulfillment_is_retryable_without_starting_work(billing):
    f = billing
    oid = upload(f).json()["order_id"]
    paid = event_for(f, oid)
    with f.sessions() as db:
        sha256 = db.get(BillingOrderRow, oid).sha256
    f.store.path_for(sha256).unlink()
    response = signed(f, paid)
    assert response.status_code == 503 and "integrity check" in response.json()["detail"]
    assert counts(f) == (0, 0, 0)


def test_truncated_image_never_reaches_payment_provider(billing):
    response = upload(billing, filename="truncated.png", content=billing.image[:40])
    assert response.status_code == 422
    assert "decoded" in response.json()["detail"]
    assert not billing.created and counts(billing) == (0, 0, 0)
    with billing.sessions() as db:
        assert db.query(BillingOrderRow).count() == 0
