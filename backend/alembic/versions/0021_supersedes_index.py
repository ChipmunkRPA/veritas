"""An ordinary index on `analyses.supersedes_id` (N-10a; code re-verification R2-01).

`cases/review.state_expression` finds what a revision was replaced by with a recursive query whose step joins
`later.supersedes_id = reach.node`. The only index on the column was the partial unique one of migrations 0014 and 0015
(`WHERE status NOT IN ('cancelled', 'failed')`), which cannot serve that join: the step read the whole table once for each
revision that did not publish, on every request for the list of analyses or of cases (the reviewer: 18 s at 48,000 analyses
with 1,670 such revisions, 0.2 s with this index and the same answers).

The partial unique index stays: one live successor per revision is still the rule.

Revision ID: 0021
Revises: 0020
Create Date: 2026-10-01
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_analyses_supersedes_id", "analyses", ["supersedes_id"])


def downgrade() -> None:
    op.drop_index("ix_analyses_supersedes_id", table_name="analyses")
