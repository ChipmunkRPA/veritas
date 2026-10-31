"""Cases, per-case membership, case items and intake records (ARCHITECTURE §4.7 `cases`, `case_members`, `intake_records`;
PRD CM-01, CM-02, CC-01; BUILD_SPEC C-2). `case_media` holds the items of a case (a media object once per case, with its
exhibit label); `media_objects` exists since 0002.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-13
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("reference", sa.String(64), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("matter_reference", sa.String(255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("classification", sa.String(16), nullable=False),
        sa.Column("case_type", sa.String(64), nullable=True),
        sa.Column("requires_second_review", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("legal_hold", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("retention_until", TS, nullable=True),
        sa.Column("implicit", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_by", sa.String(36), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
    )
    op.create_index("ix_cases_reference", "cases", ["reference"], unique=True)
    op.create_index("ix_cases_created_at", "cases", ["created_at"])
    op.create_table(
        "case_members",
        sa.Column("case_id", sa.String(36), sa.ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("principal_id", sa.String(36), sa.ForeignKey("principals.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("added_by", sa.String(36), nullable=False),
        sa.Column("added_at", TS, nullable=False),
    )
    op.create_index("ix_case_members_principal_id", "case_members", ["principal_id"])
    op.create_table(
        "case_media",
        sa.Column("case_id", sa.String(36), sa.ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("media_id", sa.String(36), sa.ForeignKey("media_objects.id"), primary_key=True),
        sa.Column("label", sa.String(8), nullable=False),
        sa.Column("added_by", sa.String(36), nullable=False),
        sa.Column("added_at", TS, nullable=False),
    )
    op.create_index("ix_case_media_case_id_label", "case_media", ["case_id", "label"], unique=True)
    op.create_index("ix_case_media_media_id", "case_media", ["media_id"])
    op.create_table(
        "intake_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("media_id", sa.String(36), sa.ForeignKey("media_objects.id"), nullable=False),
        sa.Column("case_id", sa.String(36), sa.ForeignKey("cases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("supplied_by", sa.String(255), nullable=True),
        sa.Column("acquisition_method", sa.String(32), nullable=False),
        sa.Column("source", sa.String(2048), nullable=True),
        sa.Column("date_acquired", TS, nullable=True),
        sa.Column("custodian_stated_hash", sa.String(64), nullable=True),
        sa.Column("custodian_hash_match", sa.String(16), nullable=False),
        sa.Column("client_sha256", sa.String(64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("intake_principal_id", sa.String(36), nullable=False),
        sa.Column("intake_at", TS, nullable=False),
        sa.Column("audit_event_seq", sa.BigInteger(), nullable=True),
    )
    op.create_index("ix_intake_records_case_id_intake_at", "intake_records", ["case_id", "intake_at"])
    op.create_index("ix_intake_records_media_id", "intake_records", ["media_id"])


def downgrade() -> None:
    op.drop_index("ix_intake_records_media_id", table_name="intake_records")
    op.drop_index("ix_intake_records_case_id_intake_at", table_name="intake_records")
    op.drop_table("intake_records")
    op.drop_index("ix_case_media_media_id", table_name="case_media")
    op.drop_index("ix_case_media_case_id_label", table_name="case_media")
    op.drop_table("case_media")
    op.drop_index("ix_case_members_principal_id", table_name="case_members")
    op.drop_table("case_members")
    op.drop_index("ix_cases_created_at", table_name="cases")
    op.drop_index("ix_cases_reference", table_name="cases")
    op.drop_table("cases")
