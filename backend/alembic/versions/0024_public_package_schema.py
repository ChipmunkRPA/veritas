"""Distinguish new public evidence derivatives from immutable internal packages.

Existing rows retain the legacy format; their bytes and signatures are unchanged.
"""
import sqlalchemy as sa
from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("packages", sa.Column("schema_id", sa.String(64), nullable=False,
                                       server_default="veritas.evidence/2.0"))


def downgrade() -> None:
    with op.batch_alter_table("packages") as batch:
        batch.drop_column("schema_id")
