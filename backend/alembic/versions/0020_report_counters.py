"""A report id is given once (N-9a; Postgres QA X-3).

`reports/service.next_report_id` took the highest remaining id plus one, so after a delete or a purge an id was given again
and one custody log named two reports `VR-<year>-<n>`. `report_counters` holds, per year, the last number given; a delete
does not lower it.

Existing instances: the table starts empty. The first allocation of a year starts above every id of that year that a report
row or the custody log names (`report_generated`, `redaction` and the reports of a `retention_purge` event).

Revision ID: 0020
Revises: 0019
Create Date: 2026-10-01
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "report_counters",
        sa.Column("year", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("last", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("report_counters")
