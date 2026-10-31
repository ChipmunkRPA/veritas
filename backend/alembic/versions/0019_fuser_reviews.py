"""The review of a fuser artefact as this instance has recorded it in the custody log (N-9; D-57).

A review is a sidecar file beside the artefact (`fuser-<kind>-<version>.review.json`); it can be written by the review
command, come with a release or be deleted. `fuser_reviews` holds, per artefact sha256, the review the custody log names last:
the digest of the review record, the reviewer, the hash of the statement and the seq of the event. A review found on disk that
differs from this row is recorded once (`fuser_review_recorded`, `fuser_review_withdrawn`), with the row, in one transaction.

Existing instances: the table starts empty. An artefact served under a review that was written before this migration gets its
event the first time it is used afterwards.

Revision ID: 0019
Revises: 0018
Create Date: 2026-10-01
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "fuser_reviews",
        sa.Column("artefact_sha256", sa.String(64), primary_key=True),
        sa.Column("review_sha256", sa.String(64), nullable=True),
        sa.Column("reviewer", sa.String(255), nullable=True),
        sa.Column("statement_sha256", sa.String(64), nullable=True),
        sa.Column("audit_event_seq", sa.BigInteger(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("fuser_reviews")
