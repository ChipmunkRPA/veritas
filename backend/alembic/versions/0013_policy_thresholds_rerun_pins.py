"""Workspace policy thresholds and re-run pins (ARCHITECTURE §4.4 "Abstention", §5.2 `rerun`, §6.6; PRD API-03, CC-03; BUILD_SPEC X-2).

`policy_thresholds` holds one row per version of the workspace decision policy: the conformal α levels of the strong and moderate
decision codes and the minimum slice AUC of the validated envelope. Rows are never updated; a change adds the next version and is
recorded as `policy_changed`. `analyses.rerun_pins` holds what a re-run revision reuses from the analysis it re-runs: the fusion
mode and fuser artefact, the policy thresholds version, the profile and the devices.

Revision ID: 0013
Revises: 0012
Create Date: 2026-09-13
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "policy_thresholds",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("thresholds", sa.JSON(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_policy_thresholds_version", "policy_thresholds", ["version"], unique=True)
    op.add_column("analyses", sa.Column("rerun_pins", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.drop_column("rerun_pins")
    op.drop_index("ix_policy_thresholds_version", table_name="policy_thresholds")
    op.drop_table("policy_thresholds")
