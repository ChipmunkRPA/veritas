"""Record the exact analysis limitations accepted before checkout."""
import sqlalchemy as sa
from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("billing_orders", sa.Column("purchase_acknowledgement", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("billing_orders", "purchase_acknowledgement")
