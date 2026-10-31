"""Versioned conclusions (L-5; v2.1.1 final browser QA P1-6).

`conclusions.version` numbers the conclusions of an analysis revision 1, 2, … in the order they were recorded. A conclusion PUT
names the version it was edited from and a sign-off names the conclusion id and version its signer saw, so a stale tab is refused
(409) instead of silently overwriting, or locking, a conclusion its user never saw. `ux_conclusions_analysis_id_version` turns a
concurrent second PUT from the same base into a unique violation.

Existing rows are numbered by (created_at, id), the order `latest_conclusion` already used. The version is not part of the signed
body, so existing conclusion signatures keep verifying.

Revision ID: 0016
Revises: 0015
Create Date: 2026-09-22
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("conclusions", sa.Column("version", sa.Integer(), nullable=True))
    op.execute(sa.text(
        "UPDATE conclusions SET version = (SELECT COUNT(*) FROM conclusions AS earlier WHERE earlier.analysis_id = conclusions.analysis_id "
        "AND (earlier.created_at < conclusions.created_at OR (earlier.created_at = conclusions.created_at AND earlier.id <= conclusions.id)))"))
    with op.batch_alter_table("conclusions") as batch:
        batch.alter_column("version", existing_type=sa.Integer(), nullable=False)
    op.create_index("ux_conclusions_analysis_id_version", "conclusions", ["analysis_id", "version"], unique=True)


def downgrade() -> None:
    op.drop_index("ux_conclusions_analysis_id_version", table_name="conclusions")
    with op.batch_alter_table("conclusions") as batch:
        batch.drop_column("version")
