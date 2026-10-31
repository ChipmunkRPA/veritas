"""Signing keys and evidence packages (ARCHITECTURE §4.7 `signing_keys`, `packages`; PRD RP-05, RP-06; D-11, D-27, D-30; BUILD_SPEC R-4).

`signing_keys` indexes the public half of each instance Ed25519 key; the private key stays in `data/keys`. `packages` holds one row
per evidence-package export: the report it packages, draft or issue, whether the original is included, the job status, the ZIP
artefact with its SHA-256, the manifest hash and its signature, and the `package_exported` event.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-13
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "signing_keys",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("public_key", sa.LargeBinary(32), nullable=False),
        sa.Column("key_path", sa.Text(), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("retired_at", TS, nullable=True),
    )
    op.create_table(
        "packages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("report_id", sa.String(32), nullable=False),
        sa.Column("case_id", sa.String(36), nullable=True),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("include_original", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("artefact_id", sa.String(36), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("manifest_sha256", sa.String(64), nullable=True),
        sa.Column("signature", sa.Text(), nullable=True),
        sa.Column("signing_key_id", sa.String(64), nullable=True),
        sa.Column("file_count", sa.Integer(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("finished_at", TS, nullable=True),
        sa.Column("audit_event_seq", sa.BigInteger(), nullable=True),
    )
    op.create_index("ix_packages_analysis_id", "packages", ["analysis_id"])


def downgrade() -> None:
    op.drop_index("ix_packages_analysis_id", table_name="packages")
    op.drop_table("packages")
    op.drop_table("signing_keys")
