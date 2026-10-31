"""Reports (ARCHITECTURE §4.7 `reports`, §6; PRD RP-03, RP-06; D-10, D-25; BUILD_SPEC R-1).

One row per report revision: the frozen snapshot it renders (`snapshot_hash`, `snapshot_artefact_id`), the rendered PDF
(`artefact_id`, `sha256`), the job status, and the D-25 display id with the review state frozen at request time. Revisions
count per (analysis, profile). `signature` and `signing_key_id` stay empty until R-4 signs reports.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-13
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "reports",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("display_id", sa.String(48), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("supersedes_report_id", sa.String(32), nullable=True),
        sa.Column("revision_reason", sa.Text(), nullable=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("analysis_revision", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.String(36), nullable=True),
        sa.Column("profile", sa.String(16), nullable=False),
        sa.Column("redacted", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("review_state", sa.String(16), nullable=False),
        sa.Column("template_set", sa.String(32), nullable=False),
        sa.Column("engine", sa.String(16), nullable=False),
        sa.Column("engine_version", sa.String(32), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("snapshot_hash", sa.String(64), nullable=False),
        sa.Column("snapshot_artefact_id", sa.String(36), nullable=False),
        sa.Column("artefact_id", sa.String(36), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("signature", sa.Text(), nullable=True),
        sa.Column("signing_key_id", sa.String(64), nullable=True),
        sa.Column("requested_by", sa.String(64), nullable=False),
        sa.Column("requested_at", TS, nullable=False),
        sa.Column("finished_at", TS, nullable=True),
        sa.Column("audit_event_seq", sa.BigInteger(), nullable=True),
    )
    op.create_index("ix_reports_analysis_id_revision", "reports", ["analysis_id", "revision"])
    op.create_index("ix_reports_analysis_id_profile_revision", "reports", ["analysis_id", "profile", "revision"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_reports_analysis_id_profile_revision", table_name="reports")
    op.drop_index("ix_reports_analysis_id_revision", table_name="reports")
    op.drop_table("reports")
