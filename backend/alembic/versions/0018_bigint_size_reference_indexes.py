"""`analyses.size_bytes` holds a file of 2 GiB or more; the six referencing columns without an index get one (N-4a; D-72;
Postgres QA PG-07, PG-08).

- `analyses.size_bytes` was INTEGER while `media_objects.size_bytes`, `artefacts.bytes` and `batch_items.size_bytes` are BIGINT.
  Postgres refused an analysis of a file above 2,147,483,647 bytes ("integer out of range"), which an operator meets after
  raising the upload limits. SQLite stores any 64-bit value in either type; its table is rebuilt so that both engines declare
  the same column (the rebuild runs with foreign keys unenforced, `migrations.migration`, so no referencing row is touched).
- Postgres checks every referencing column when a parent row is deleted, and SQLite does so since its keys are enforced. Without
  an index each check reads the whole child table: the delete of one analysis cost 645 ms in key checks at 20,000 analyses and
  2 ms with the indexes (the QA's measurement).

The downgrade narrows the column again; Postgres refuses it while a stored size exceeds 32 bits.

Revision ID: 0018
Revises: 0017
Create Date: 2026-09-30
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

REFERENCE_INDEXES = (
    ("signals", "detector_run_id"),
    ("analysis_stages", "output_artefact_id"),
    ("detector_runs", "raw_output_artefact_id"),
    ("exhibits", "payload_artefact_id"),
    ("media_objects", "storage_artefact_id"),
    ("signoffs", "conclusion_id"),
)


def upgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.alter_column("size_bytes", existing_type=sa.Integer(), type_=sa.BigInteger(), existing_nullable=False)
    for table, column in REFERENCE_INDEXES:
        op.create_index(f"ix_{table}_{column}", table, [column])


def downgrade() -> None:
    for table, column in REFERENCE_INDEXES:
        op.drop_index(f"ix_{table}_{column}", table_name=table)
    with op.batch_alter_table("analyses") as batch:
        batch.alter_column("size_bytes", existing_type=sa.BigInteger(), type_=sa.Integer(), existing_nullable=False)
