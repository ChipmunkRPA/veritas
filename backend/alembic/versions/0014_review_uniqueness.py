"""Uniqueness of review records and revisions (D-47; v2.1 code review P2-9).

- `signoffs(analysis_id, kind)`: one examiner sign-off and one peer review per revision, so concurrent requests cannot record two.
- `analyses.supersedes_id`: one newer revision per revision, so a concurrent reanalyse or re-fusion cannot create two r+1 rows.
  NULL (a first revision) is not unique-constrained in SQLite or Postgres.

A database that already holds duplicates fails this upgrade; resolve them by hand first.

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-16
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ux_signoffs_analysis_id_kind", "signoffs", ["analysis_id", "kind"], unique=True)
    op.create_index("ux_analyses_supersedes_id", "analyses", ["supersedes_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ux_analyses_supersedes_id", table_name="analyses")
    op.drop_index("ux_signoffs_analysis_id_kind", table_name="signoffs")
