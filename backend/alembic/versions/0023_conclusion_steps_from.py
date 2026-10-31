"""A conclusion says what its steps are counted from (N-19n; design review D4-01, P1).

Under a trusted capture record and under a trusted declaration the examiner's position is compared with the step of that
record on the scale (the first and the fifth), not with the decision code: on the uncalibrated fallback the code stands
one step inside (the fallback issues no strongest code), so that an examiner on the record's own step was one step from
"the machine decision" and could not sign as concurring. `conclusions.steps_from` names the record whose step the
conclusion's `deviation_steps` are counted from (`capture_record`, `declaration`), and is part of the signed body where it
is set (`app.cases.review.conclusion_body`).

Existing instances: every conclusion recorded before keeps NULL, its steps and its signature: its steps were counted
from the decision code, and its signed body holds no such key.

Revision ID: 0023
Revises: 0022
Create Date: 2026-10-05
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("conclusions", sa.Column("steps_from", sa.String(32), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("conclusions") as batch:
        batch.drop_column("steps_from")
