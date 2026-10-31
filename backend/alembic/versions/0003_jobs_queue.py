"""DB job queue with leases (ARCHITECTURE §3.4).

Analyses that are still queued or running when the database is upgraded have no job row. `jobs.start()` adopts
them, using the same poison guard as v0.3.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-12
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("subject_id", sa.String(64), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lease_owner", sa.String(100), nullable=True),
        sa.Column("lease_expires_at", TS, nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("not_before", TS, nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
    )
    op.create_index("ix_jobs_status_priority_created_at", "jobs", ["status", "priority", "created_at"])
    op.create_index("ix_jobs_subject_id", "jobs", ["subject_id"])


def downgrade() -> None:
    op.drop_index("ix_jobs_subject_id", table_name="jobs")
    op.drop_index("ix_jobs_status_priority_created_at", table_name="jobs")
    op.drop_table("jobs")
