"""Baseline: the v0.3 `analyses` table (ARCHITECTURE §4.9 step 1).

Unversioned v0.3 databases are stamped at this revision by `init_db()` instead of running it.

Revision ID: 0001
Revises:
Create Date: 2026-09-11
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analyses",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("probability_ai", sa.Float(), nullable=True),
        sa.Column("band", sa.String(32), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("report_path", sa.String(512), nullable=True),
    )
    op.create_index("ix_analyses_kind", "analyses", ["kind"])
    op.create_index("ix_analyses_sha256", "analyses", ["sha256"])
    op.create_index("ix_analyses_status", "analyses", ["status"])


def downgrade() -> None:
    op.drop_index("ix_analyses_status", table_name="analyses")
    op.drop_index("ix_analyses_sha256", table_name="analyses")
    op.drop_index("ix_analyses_kind", table_name="analyses")
    op.drop_table("analyses")
