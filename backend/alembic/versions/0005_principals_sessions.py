"""Principals and browser sessions (ARCHITECTURE §4.7 `principals`, `sessions`, §5.3; PRD CM-02, PL-04; BUILD_SPEC C-1).

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-13
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "principals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=True),
        sa.Column("api_key_hash", sa.String(128), nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("disabled_at", TS, nullable=True),
        sa.Column("last_login_at", TS, nullable=True),
    )
    op.create_index("ix_principals_username", "principals", ["username"], unique=True)
    op.create_index("ix_principals_api_key_hash", "principals", ["api_key_hash"])
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("principal_id", sa.String(36), sa.ForeignKey("principals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("expires_at", TS, nullable=False),
        sa.Column("csrf_secret", sa.String(64), nullable=False),
        sa.Column("user_agent", sa.String(256), nullable=True),
        sa.Column("revoked_at", TS, nullable=True),
    )
    op.create_index("ix_sessions_principal_id", "sessions", ["principal_id"])


def downgrade() -> None:
    op.drop_index("ix_sessions_principal_id", table_name="sessions")
    op.drop_table("sessions")
    op.drop_index("ix_principals_api_key_hash", table_name="principals")
    op.drop_index("ix_principals_username", table_name="principals")
    op.drop_table("principals")
