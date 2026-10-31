"""The report row stores whether its snapshot was issuable (N-2; D-56; production-readiness code review 2, SEC-01).

`reports.issuable` is the `review.issuable` flag frozen in the report's snapshot when it was requested: approved, or reviewed on
a case that does not require a second review. The `issue` package gate reads it, and the report's id and its word "issued" follow
the same flag, so a report signed while a required second review was outstanding, or after a peer review that does not concur,
is a draft.

Existing rows: the flag is rebuilt from what the earlier gate read. A report requested in the `approved` state is issuable
unless the revision's peer review does not concur (the earlier rule called any peer review an approval; sign-offs are
write-once, so today's decision is the one the report was requested under). A report requested in the `reviewed` state is
issuable unless its analysis's case requires a second review. Every other report is a draft. `review_state` is left as recorded:
it names the state the snapshot holds, and the PDFs of earlier reports are immutable.

Revision ID: 0017
Revises: 0016
Create Date: 2026-09-30
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

PEER_APPROVES = ("concur", "concur_with_changes")


def upgrade() -> None:
    op.add_column("reports", sa.Column("issuable", sa.Boolean(), nullable=False, server_default=sa.false()))
    reports = sa.table("reports", sa.column("analysis_id", sa.String), sa.column("review_state", sa.String), sa.column("issuable", sa.Boolean))
    signoffs = sa.table("signoffs", sa.column("analysis_id", sa.String), sa.column("kind", sa.String), sa.column("decision", sa.String))
    analyses = sa.table("analyses", sa.column("id", sa.String), sa.column("case_id", sa.String))
    cases = sa.table("cases", sa.column("id", sa.String), sa.column("requires_second_review", sa.Boolean))
    disputed = sa.select(sa.literal(1)).where(signoffs.c.analysis_id == reports.c.analysis_id, signoffs.c.kind == "peer_review",
                                              signoffs.c.decision.not_in(PEER_APPROVES)).exists()
    second_review = sa.select(sa.literal(1)).select_from(analyses.join(cases, cases.c.id == analyses.c.case_id)).where(
        analyses.c.id == reports.c.analysis_id, cases.c.requires_second_review.is_(sa.true())).exists()
    op.execute(reports.update().where(sa.or_(sa.and_(reports.c.review_state == "approved", sa.not_(disputed)),
                                             sa.and_(reports.c.review_state == "reviewed", sa.not_(second_review))))
               .values(issuable=sa.true()))


def downgrade() -> None:
    with op.batch_alter_table("reports") as batch:
        batch.drop_column("issuable")
