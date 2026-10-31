"""Fuser artefacts in use (ARCHITECTURE §4.7 `fusers`; BUILD_SPEC M-2).

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-12
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "fusers",
        sa.Column("sha256", sa.String(64), primary_key=True),
        sa.Column("modality", sa.String(16), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("bench_version", sa.String(64), nullable=True),
        sa.Column("reviewed_by", sa.String(255), nullable=True),
        sa.Column("loaded_at", TS, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("fusers")
