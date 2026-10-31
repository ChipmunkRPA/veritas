"""A worker says that it lives, and what it runs on, where the API reads it (N-13; ops review OPS-06, OPS-18).

In a split deployment (`VERITAS_RUN_WORKER=false` on the API, `python -m app.engine.worker` beside it) the API could not
tell whether a worker runs, and `/api/ready` described the device, the profile and the fusion state of the API process.
`worker_heartbeats` holds one row per running worker, written by that worker every few seconds
(`app.engine.heartbeat`): when it started and last wrote, its device, default profile and thread count, whether every
thread reached the database lately, and its fusion state as JSON. A worker that stops removes its row; the row of one that
was killed stays, grows old and is removed by a later start.

Existing instances: the table starts empty; a running worker writes its row at its start. With two Postgres roles the grants
are applied again after this migration (`python -m app.cli db grants`), or the worker cannot write its row and says so.

Revision ID: 0022
Revises: 0021
Create Date: 2026-10-01
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "worker_heartbeats",
        sa.Column("worker_id", sa.String(100), primary_key=True),
        sa.Column("hostname", sa.String(255), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=False),
        sa.Column("in_process", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("beat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("device", sa.String(32), nullable=False),
        sa.Column("profile", sa.String(16), nullable=False),
        sa.Column("threads", sa.Integer(), nullable=False),
        sa.Column("ready", sa.Boolean(), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
    )
    op.create_index("ix_worker_heartbeats_beat_at", "worker_heartbeats", ["beat_at"])


def downgrade() -> None:
    op.drop_index("ix_worker_heartbeats_beat_at", table_name="worker_heartbeats")
    op.drop_table("worker_heartbeats")
