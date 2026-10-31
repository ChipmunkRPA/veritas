"""Hash-chained audit log (ARCHITECTURE §4.7 `audit_events`, §4.8; PRD CC-01, CC-02; BUILD_SPEC C-3).

`seq` is assigned by the writer (the chain tail + 1) because it is part of the hashed body. Triggers make the table
append-only: SQLite aborts UPDATE and DELETE; Postgres raises on UPDATE, DELETE and TRUNCATE.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-13
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)
SEQ = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "audit_events",
        sa.Column("seq", SEQ, primary_key=True, autoincrement=False),
        sa.Column("ts", TS, nullable=False),
        sa.Column("actor_id", sa.String(64), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("case_id", sa.String(36), nullable=True),
        sa.Column("media_id", sa.String(36), nullable=True),
        sa.Column("analysis_id", sa.String(36), nullable=True),
        sa.Column("report_id", sa.String(32), nullable=True),
        sa.Column("object_ids", sa.JSON(), nullable=False),
        sa.Column("before_hash", sa.String(64), nullable=True),
        sa.Column("after_hash", sa.String(64), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("prev_hash", sa.String(64), nullable=False),
        sa.Column("event_hash", sa.String(64), nullable=False),
    )
    op.create_index("ix_audit_events_case_id_seq", "audit_events", ["case_id", "seq"])
    op.create_index("ix_audit_events_ts", "audit_events", ["ts"])
    dialect = op.get_bind().dialect.name
    if dialect == "sqlite":
        op.execute("CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'append-only'); END;")
        op.execute("CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'append-only'); END;")
    elif dialect == "postgresql":
        op.execute("CREATE FUNCTION audit_events_append_only() RETURNS trigger LANGUAGE plpgsql AS "
                   "$$ BEGIN RAISE EXCEPTION 'append-only'; END; $$;")
        op.execute("CREATE TRIGGER audit_no_update_delete BEFORE UPDATE OR DELETE ON audit_events "
                   "FOR EACH ROW EXECUTE FUNCTION audit_events_append_only();")
        op.execute("CREATE TRIGGER audit_no_truncate BEFORE TRUNCATE ON audit_events "
                   "FOR EACH STATEMENT EXECUTE FUNCTION audit_events_append_only();")


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS audit_no_delete;")
        op.execute("DROP TRIGGER IF EXISTS audit_no_update;")
    elif dialect == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS audit_no_truncate ON audit_events;")
        op.execute("DROP TRIGGER IF EXISTS audit_no_update_delete ON audit_events;")
        op.execute("DROP FUNCTION IF EXISTS audit_events_append_only();")
    op.drop_index("ix_audit_events_ts", table_name="audit_events")
    op.drop_index("ix_audit_events_case_id_seq", table_name="audit_events")
    op.drop_table("audit_events")
