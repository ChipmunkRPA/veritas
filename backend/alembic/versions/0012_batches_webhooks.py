"""Batches and webhooks (ARCHITECTURE §4.7 `batches`, `batch_items`, `webhooks`, `webhook_deliveries`; PRD API-01, API-02; BUILD_SPEC X-1).

`batches` holds one row per batch submission (upload, archive or local path) and `batch_items` one row per file with its intake
outcome: staged, rejected with a reason, or submitted with its analysis. `webhooks` holds user-configured endpoints (the secret
is derived, never stored: only a nonce and a hash) and `webhook_deliveries` one row per delivery attempt with its response code.

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-13
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "batches",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("case_id", sa.String(36), nullable=False),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("profile", sa.String(16), nullable=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("expanded_at", TS, nullable=True),
        sa.Column("finished_at", TS, nullable=True),
    )
    op.create_index("ix_batches_created_at", "batches", ["created_at"])
    op.create_table(
        "batch_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("batch_id", sa.String(36), sa.ForeignKey("batches.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("analysis_id", sa.String(36), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=True),
    )
    op.create_index("ix_batch_items_batch_id_seq", "batch_items", ["batch_id", "seq"], unique=True)
    op.create_index("ix_batch_items_analysis_id", "batch_items", ["analysis_id"])
    op.create_table(
        "webhooks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("events", sa.JSON(), nullable=False),
        sa.Column("secret_nonce", sa.String(64), nullable=False),
        sa.Column("secret_hash", sa.String(64), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
    )
    op.create_table(
        "webhook_deliveries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("hook_id", sa.String(36), sa.ForeignKey("webhooks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event", sa.String(32), nullable=False),
        sa.Column("event_id", sa.String(36), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("payload_sha256", sa.String(64), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("next_attempt_at", TS, nullable=True),
        sa.Column("attempted_at", TS, nullable=True),
        sa.Column("delivered_at", TS, nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
    )
    op.create_index("ix_webhook_deliveries_hook_id_created_at", "webhook_deliveries", ["hook_id", "created_at"])
    op.create_index("ix_webhook_deliveries_event_id", "webhook_deliveries", ["event_id"])


def downgrade() -> None:
    op.drop_index("ix_webhook_deliveries_event_id", table_name="webhook_deliveries")
    op.drop_index("ix_webhook_deliveries_hook_id_created_at", table_name="webhook_deliveries")
    op.drop_table("webhook_deliveries")
    op.drop_table("webhooks")
    op.drop_index("ix_batch_items_analysis_id", table_name="batch_items")
    op.drop_index("ix_batch_items_batch_id_seq", table_name="batch_items")
    op.drop_table("batch_items")
    op.drop_index("ix_batches_created_at", table_name="batches")
    op.drop_table("batches")
