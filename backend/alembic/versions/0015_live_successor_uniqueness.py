"""One live newer revision per revision (K-2; v2.1.1 final code review P1).

`ux_analyses_supersedes_id` (0014) made any successor row, whatever its status, take the one newer-revision slot. A re-run,
re-fusion or reanalyse that was cancelled or failed never published, yet it left its source "superseded" for good. The index
becomes partial: rows whose status is `cancelled` or `failed` are outside it, so a concurrent duplicate of a queued, running or
published successor still fails with a unique violation, and a new successor can follow an unpublished one.

The downgrade restores the full index; it fails when a revision has an unpublished successor and a newer one (resolve by hand).

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-22
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LIVE = "status NOT IN ('cancelled', 'failed')"


def upgrade() -> None:
    op.drop_index("ux_analyses_supersedes_id", table_name="analyses")
    op.create_index("ux_analyses_supersedes_id", "analyses", ["supersedes_id"], unique=True,
                    sqlite_where=sa.text(LIVE), postgresql_where=sa.text(LIVE))


def downgrade() -> None:
    op.drop_index("ux_analyses_supersedes_id", table_name="analyses")
    op.create_index("ux_analyses_supersedes_id", "analyses", ["supersedes_id"], unique=True)
