"""Examiner layer: annotations and their revisions (ARCHITECTURE §4.7 `annotations`, `annotation_revisions`; PRD RW-01, RW-02;
BUILD_SPEC C-4). A CHECK constraint backs the API rule that disputed and not relevant statuses carry a note.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-13
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)
NOTE_RULE = "examiner_status IS NULL OR examiner_status NOT IN ('disputed', 'not_relevant') OR length(trim(coalesce(body, ''))) > 0"


def upgrade() -> None:
    op.create_table(
        "annotations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("case_id", sa.String(36), nullable=True),
        sa.Column("author_id", sa.String(64), nullable=False),
        sa.Column("target_type", sa.String(16), nullable=False),
        sa.Column("target_id", sa.String(255), nullable=True),
        sa.Column("shape", sa.String(16), nullable=False),
        sa.Column("geometry", sa.JSON(), nullable=True),
        sa.Column("examiner_status", sa.String(16), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("visibility", sa.String(16), nullable=False, server_default="internal"),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("edited_at", TS, nullable=True),
        sa.Column("deleted_at", TS, nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.CheckConstraint(NOTE_RULE, name="ck_annotations_note_required"),
    )
    op.create_index("ix_annotations_analysis_id", "annotations", ["analysis_id"])
    op.create_index("ix_annotations_case_id", "annotations", ["case_id"])
    op.create_table(
        "annotation_revisions",
        sa.Column("annotation_id", sa.String(36), sa.ForeignKey("annotations.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("revision", sa.Integer(), primary_key=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("examiner_status", sa.String(16), nullable=True),
        sa.Column("geometry", sa.JSON(), nullable=True),
        sa.Column("visibility", sa.String(16), nullable=False),
        sa.Column("edited_by", sa.String(64), nullable=False),
        sa.Column("edited_at", TS, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("annotation_revisions")
    op.drop_index("ix_annotations_case_id", table_name="annotations")
    op.drop_index("ix_annotations_analysis_id", table_name="annotations")
    op.drop_table("annotations")
