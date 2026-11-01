"""Durable fixed-price analysis orders and idempotent Stripe fulfillment."""
import sqlalchemy as sa
from alembic import op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("billing_orders",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(36), nullable=False),
        sa.Column("request_key", sa.String(64), nullable=False),
        sa.Column("case_id", sa.String(36), nullable=False),
        sa.Column("media_id", sa.String(36), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("profile", sa.String(16), nullable=False),
        sa.Column("amount_minor", sa.Integer(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("mode", sa.String(8), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("stripe_session_id", sa.String(255), nullable=True, unique=True),
        sa.Column("checkout_url", sa.Text(), nullable=True),
        sa.Column("payment_intent_id", sa.String(255), nullable=True, unique=True),
        sa.Column("analysis_id", sa.String(36), nullable=True, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ux_billing_owner_request", "billing_orders", ["owner_id", "request_key"], unique=True)
    op.create_index("ix_billing_orders_owner_id", "billing_orders", ["owner_id"])
    op.create_index("ix_billing_orders_status", "billing_orders", ["status"])
    op.create_table("billing_events",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("order_id", sa.String(36), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_billing_events_order_id", "billing_events", ["order_id"])


def downgrade() -> None:
    op.drop_table("billing_events")
    op.drop_table("billing_orders")
