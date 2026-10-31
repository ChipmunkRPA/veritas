"""Review and sign-off (ARCHITECTURE §4.7 `conclusions`, `signoffs`, §5.4; PRD RW-03, RW-04; D-27; BUILD_SPEC C-5).

Adds `machine_code` and `signature` to conclusions, `signature` to signoffs (HMAC-SHA256 review records), and the withdrawal
fields of D-27 to `analyses`.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-13
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "conclusions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("analysis_revision", sa.Integer(), nullable=False),
        sa.Column("author_id", sa.String(64), nullable=False),
        sa.Column("scale_position", sa.String(32), nullable=False),
        sa.Column("machine_code", sa.String(32), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("limitations_acknowledged", sa.JSON(), nullable=False),
        sa.Column("next_steps", sa.Text(), nullable=True),
        sa.Column("declaration_text_id", sa.String(64), nullable=True),
        sa.Column("deviation_steps", sa.Integer(), nullable=True),
        sa.Column("deviation_justification", sa.Text(), nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("signature", sa.String(64), nullable=False),
    )
    op.create_index("ix_conclusions_analysis_id", "conclusions", ["analysis_id"])
    op.create_table(
        "signoffs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("analysis_revision", sa.Integer(), nullable=False),
        sa.Column("conclusion_id", sa.String(36), sa.ForeignKey("conclusions.id"), nullable=False),
        sa.Column("principal_id", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("result_hash", sa.String(64), nullable=True),
        sa.Column("annotations_seq", sa.Integer(), nullable=False),
        sa.Column("signed_at", TS, nullable=False),
        sa.Column("audit_event_seq", sa.BigInteger(), nullable=True),
        sa.Column("signature", sa.String(64), nullable=False),
    )
    op.create_index("ix_signoffs_analysis_id", "signoffs", ["analysis_id"])
    op.add_column("analyses", sa.Column("withdrawn_at", TS, nullable=True))
    op.add_column("analyses", sa.Column("withdrawn_reason", sa.Text(), nullable=True))
    op.add_column("analyses", sa.Column("withdrawn_by", sa.String(64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.drop_column("withdrawn_by")
        batch.drop_column("withdrawn_reason")
        batch.drop_column("withdrawn_at")
    op.drop_index("ix_signoffs_analysis_id", table_name="signoffs")
    op.drop_table("signoffs")
    op.drop_index("ix_conclusions_analysis_id", table_name="conclusions")
    op.drop_table("conclusions")
