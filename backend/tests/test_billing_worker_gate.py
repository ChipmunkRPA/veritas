"""Payment must precede worker dispatch, including queued work from free mode."""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api import jobs
from app.audit import chain as audit
from app.billing.entitlements import BATCH_PAYMENT_REQUIRED, PAYMENT_REQUIRED, PaymentRequired
from app.core.config import settings
from app.engine.boundary import Deadline
from app.engine.queue import JobQueue
from app.storage.db import Analysis, Base, BillingOrderRow, utcnow


@pytest.fixture
def worker_gate(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'worker-payments.sqlite'}")
    Base.metadata.create_all(engine)

    @contextmanager
    def sessions():
        with Session(engine, expire_on_commit=False) as session:
            yield session

    monkeypatch.setattr(settings, "billing_mode", "test")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(jobs, "get_session", sessions)
    monkeypatch.setattr(jobs, "_sweep_tmp", lambda *args: None)
    monkeypatch.setattr(jobs, "_finished", lambda *args: None)
    recorded = Mock()
    monkeypatch.setattr(audit, "record", recorded)
    queue = JobQueue(engine)
    monkeypatch.setattr(jobs, "get_queue", lambda: queue)
    model_run = Mock(return_value="done")
    monkeypatch.setattr(jobs, "_run_analysis_job", model_run)
    monkeypatch.setattr(jobs, "_run_refusion_job", model_run)

    def source(*, paid=False, status="queued", kind="audio"):
        aid = str(uuid.uuid4())
        with sessions() as session:
            analysis = Analysis(id=aid, kind=kind, filename="sample.wav", content_type="audio/wav", size_bytes=8000,
                                sha256="a" * 64, status=status, case_id="case-one", media_id="media-one",
                                submitted_by="payer", profile="full")
            session.add(analysis)
            if paid:
                session.add(BillingOrderRow(id=str(uuid.uuid4()), request_key=aid, owner_id="payer", case_id="case-one",
                    media_id="media-one", kind=kind, filename="sample.wav", content_type="audio/wav", size_bytes=8000,
                    sha256="a" * 64, profile="full", amount_minor=999, currency="usd", mode="test", status="fulfilled",
                    stripe_session_id="cs_test_" + aid.replace("-", ""), payment_intent_id="pi_" + aid.replace("-", ""),
                    analysis_id=aid, paid_at=utcnow()))
            session.commit()
        return aid

    def context(aid, kind="analysis"):
        return SimpleNamespace(job=queue.enqueue(kind, aid), deadline=Deadline(), lease_lost=False)

    yield SimpleNamespace(sessions=sessions, queue=queue, source=source, context=context,
                          model_run=model_run, recorded=recorded)
    engine.dispose()


@pytest.mark.parametrize("kind", ["image", "video", "audio", "document"])
@pytest.mark.parametrize("handler,job_type", [(jobs._handle_analysis, "analysis"), (jobs._handle_refusion, "refusion")])
def test_unpaid_queued_work_never_reaches_models_or_analysis_started(worker_gate, kind, handler, job_type):
    f = worker_gate
    aid = f.source(kind=kind)
    with pytest.raises(PaymentRequired, match="Payment has not been confirmed"):
        handler(f.context(aid, job_type))
    f.model_run.assert_not_called()
    f.recorded.assert_not_called()
    with f.sessions() as session:
        analysis = session.get(Analysis, aid)
        assert analysis.status == "failed" and analysis.error == PAYMENT_REQUIRED
        assert analysis.started_at is None and analysis.attempts == 0


@pytest.mark.parametrize("field,value", [
    ("status", "awaiting_payment"), ("paid_at", None), ("payment_intent_id", None),
    ("payment_intent_id", "pi_"), ("stripe_session_id", None), ("owner_id", "another-payer"),
    ("case_id", "another-case"), ("media_id", "another-source"), ("sha256", "b" * 64),
    ("size_bytes", 9000), ("kind", "video"), ("profile", "deep"), ("amount_minor", 998),
    ("currency", "eur"), ("mode", "live"), ("analysis_id", "another-analysis"),
])
def test_incomplete_or_mismatched_receipt_cannot_authorize_work(worker_gate, field, value):
    f = worker_gate
    aid = f.source(paid=True)
    with f.sessions() as session:
        order = session.query(BillingOrderRow).filter_by(analysis_id=aid).one()
        setattr(order, field, value)
        session.commit()
    with pytest.raises(PaymentRequired):
        jobs._handle_analysis(f.context(aid))
    f.model_run.assert_not_called()


def test_test_payment_cannot_authorize_live_worker(worker_gate, monkeypatch):
    aid = worker_gate.source(paid=True)
    monkeypatch.setattr(settings, "billing_mode", "live")
    with pytest.raises(PaymentRequired):
        jobs._handle_analysis(worker_gate.context(aid))
    worker_gate.model_run.assert_not_called()


def test_paid_analysis_can_resume_without_another_payment(worker_gate):
    f = worker_gate
    aid = f.source(paid=True)
    f.model_run.side_effect = [jobs.LEASE_LOST, "done"]
    ctx = f.context(aid)
    assert jobs._handle_analysis(ctx) == jobs.LEASE_LOST
    assert jobs._handle_analysis(ctx) == "done"
    assert f.model_run.call_count == 2
    with f.sessions() as session:
        assert session.query(BillingOrderRow).count() == 1
        assert session.query(Analysis).count() == 1


def test_paid_source_does_not_pay_for_new_revision(worker_gate):
    f = worker_gate
    paid = f.source(paid=True, status="done")
    revision = f.source()
    with f.sessions() as session:
        session.get(Analysis, revision).supersedes_id = paid
        session.commit()
    with pytest.raises(PaymentRequired):
        jobs._handle_refusion(f.context(revision, "refusion"))
    f.model_run.assert_not_called()
    with f.sessions() as session:
        assert session.get(Analysis, paid).status == "done"


def test_recovery_adopts_only_paid_unfinished_work(worker_gate):
    f = worker_gate
    unpaid = f.source(status="running")
    paid = f.source(paid=True, status="running")
    historical = f.source(status="done")
    assert jobs.adopt_orphans(f.queue) == 1
    assert f.queue.latest(unpaid) is None
    assert f.queue.latest(paid).status == "queued"
    with f.sessions() as session:
        assert session.get(Analysis, unpaid).status == "failed"
        assert session.get(Analysis, unpaid).error == PAYMENT_REQUIRED
        assert session.get(Analysis, paid).status == "queued"
        assert session.get(Analysis, historical).status == "done"
    f.model_run.assert_not_called()


def test_stale_batch_expansion_is_refused_before_intake(worker_gate, monkeypatch):
    from app.batches import service as batches

    expand, failed = Mock(), Mock()
    monkeypatch.setattr(batches, "expand", expand)
    monkeypatch.setattr(batches, "mark_failed", failed)
    with pytest.raises(PaymentRequired):
        jobs._handle_batch_expand(worker_gate.context("old-free-batch", "batch_expand"))
    expand.assert_not_called()
    failed.assert_called_once_with("old-free-batch", BATCH_PAYMENT_REQUIRED)


def test_finished_historical_result_is_not_rerun_or_payment_blocked(worker_gate):
    aid = worker_gate.source(status="done")
    assert jobs._handle_analysis(worker_gate.context(aid)) == "done"
    worker_gate.model_run.assert_not_called()


def test_disabled_billing_preserves_free_worker_and_orphan_behavior(worker_gate, monkeypatch):
    f = worker_gate
    monkeypatch.setattr(settings, "billing_mode", "disabled")
    aid = f.source()
    assert jobs.adopt_orphans(f.queue) == 1
    assert jobs._handle_analysis(f.context(aid)) == "done"
    f.model_run.assert_called_once()
