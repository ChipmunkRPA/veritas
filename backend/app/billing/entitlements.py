"""Local payment receipts checked again before a worker may run an analysis.

Only verified Stripe fulfillment writes these receipts. The worker checks the
committed order, not browser state, so queued work from an earlier free instance
cannot silently become paid work after billing is enabled. Retrying the same paid
analysis reuses its receipt; a new analysis needs its own order.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.storage.db import Analysis, BillingOrderRow

PAYMENT_REQUIRED = "Payment has not been confirmed for this analysis. Complete a US$9.99 checkout before analysis can start."
BATCH_PAYMENT_REQUIRED = "Batch analysis cannot start without payment. Submit each file through its own US$9.99 checkout."


class PaymentRequired(ValueError):
    """Safe failure text for an unpaid or mismatched queued analysis."""


def require_analysis_payment(session: Session, analysis: Analysis) -> None:
    if settings.billing_mode == "disabled":
        return
    # Import at call time: fulfillment itself imports the queue facade.
    from app.billing.service import AMOUNT_MINOR, CURRENCY

    order = session.scalar(select(BillingOrderRow).where(BillingOrderRow.analysis_id == analysis.id))
    if order is None:
        raise PaymentRequired(PAYMENT_REQUIRED)
    receipt_matches = (
        order.status == "fulfilled"
        and order.paid_at is not None
        and isinstance(order.payment_intent_id, str)
        and order.payment_intent_id.startswith("pi_")
        and len(order.payment_intent_id) > 3
        and bool(order.stripe_session_id)
        and order.mode == settings.billing_mode
        and order.amount_minor == AMOUNT_MINOR
        and order.currency == CURRENCY
        and bool(analysis.submitted_by)
        and order.owner_id == analysis.submitted_by
        and order.case_id == analysis.case_id
        and order.media_id == analysis.media_id
        and order.sha256 == analysis.sha256
        and order.size_bytes == analysis.size_bytes
        and order.kind == analysis.kind
        and order.profile == analysis.profile
    )
    if not receipt_matches:
        raise PaymentRequired(PAYMENT_REQUIRED)


def require_batch_payment() -> None:
    # Paid batch checkout does not exist. No stale expansion job can create free
    # child analyses; exports of completed historical work remain available.
    if settings.billing_mode != "disabled":
        raise PaymentRequired(BATCH_PAYMENT_REQUIRED)
