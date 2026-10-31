"""Evidence model v2 tables and the extended `analyses` columns (ARCHITECTURE §4.7, §4.9 step 2).

- New tables: `artefacts`, `media_objects`, `analysis_stages`, `detector_runs`, `signals`, `assessments`, `findings`,
  `exhibits`, `analysis_events`.
- Existing analyses get `result_schema_version = 1` and `revision = 1`.
- `media_objects` rows are backfilled from `analyses.sha256/filename/content_type/size_bytes`. Originals are copied
  into the object store lazily, on first access (`app.storage.artefacts.ensure_original_in_store`).

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-12
"""
import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)
SEQ = sa.BigInteger().with_variant(sa.Integer(), "sqlite")  # SQLite only autoincrements INTEGER PRIMARY KEY

# (name, type, server_default)
ANALYSIS_COLUMNS: list[tuple[str, sa.types.TypeEngine, str | None]] = [
    ("case_id", sa.String(36), None),
    ("media_id", sa.String(36), None),
    ("revision", sa.Integer(), "1"),
    ("supersedes_id", sa.String(36), None),
    ("profile", sa.String(16), None),
    ("effective_profile", sa.String(16), None),
    ("licence_posture", sa.String(32), None),
    ("pipeline_version", sa.String(32), None),
    ("engine_version", sa.String(32), None),
    ("build_commit", sa.String(40), None),
    ("config_hash", sa.String(64), None),
    ("manifest_lock_hash", sa.String(64), None),
    ("fuser_sha256", sa.String(64), None),
    ("fusion_mode", sa.String(16), None),
    ("template_set", sa.String(32), None),
    ("policy_thresholds_id", sa.String(36), None),
    ("result_schema_version", sa.Integer(), "1"),
    ("result_hash", sa.String(64), None),
    ("decision_code", sa.String(32), None),
    ("driving_hypothesis", sa.String(40), None),
    ("slice_id", sa.String(120), None),
    ("progress", sa.Float(), None),
    ("current_stage", sa.String(32), None),
    ("submitted_by", sa.String(36), None),
    ("locked_at", TS, None),
]


def _analysis_fk() -> sa.ForeignKey:
    return sa.ForeignKey("analyses.id", ondelete="CASCADE")


def upgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        for name, type_, default in ANALYSIS_COLUMNS:
            batch.add_column(sa.Column(name, type_, nullable=default is None, server_default=default))
        batch.create_index("ix_analyses_case_id_created_at", ["case_id", "created_at"])
        batch.create_index("ix_analyses_media_id", ["media_id"])

    op.create_table(
        "artefacts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("media_type", sa.String(100), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("analysis_id", sa.String(36), _analysis_fk(), nullable=True),
        sa.Column("created_by", sa.String(100), nullable=False),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_artefacts_sha256", "artefacts", ["sha256"])
    op.create_index("ix_artefacts_analysis_id", "artefacts", ["analysis_id"])

    op.create_table(
        "media_objects",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("md5", sa.String(32), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("mime_sniffed", sa.String(100), nullable=True),
        sa.Column("original_filename", sa.String(255), nullable=True),
        sa.Column("storage_artefact_id", sa.String(36), sa.ForeignKey("artefacts.id"), nullable=True),
        sa.Column("first_seen_at", TS, nullable=False),
    )

    op.create_table(
        "analysis_stages",
        sa.Column("analysis_id", sa.String(36), _analysis_fk(), primary_key=True),
        sa.Column("stage", sa.String(32), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", TS, nullable=True),
        sa.Column("finished_at", TS, nullable=True),
        sa.Column("output_artefact_id", sa.String(36), sa.ForeignKey("artefacts.id"), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )

    op.create_table(
        "detector_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), _analysis_fk(), nullable=False),
        sa.Column("detector_id", sa.String(100), nullable=False),
        sa.Column("detector_version", sa.String(32), nullable=False),
        sa.Column("model_revisions", sa.JSON(), nullable=False),
        sa.Column("device", sa.String(32), nullable=True),
        sa.Column("dtype", sa.String(16), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("started_at", TS, nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("peak_memory_mb", sa.Float(), nullable=True),
        sa.Column("raw_output_artefact_id", sa.String(36), sa.ForeignKey("artefacts.id"), nullable=True),
    )
    op.create_index("ix_detector_runs_analysis_id", "detector_runs", ["analysis_id"])

    op.create_table(
        "signals",
        sa.Column("analysis_id", sa.String(36), _analysis_fk(), primary_key=True),
        sa.Column("signal_id", sa.String(255), primary_key=True),
        sa.Column("detector_run_id", sa.String(36), sa.ForeignKey("detector_runs.id", ondelete="CASCADE"), nullable=True),
        sa.Column("hypothesis", sa.String(40), nullable=True),
        sa.Column("lane", sa.String(16), nullable=True),
        sa.Column("track_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("raw_score", sa.Float(), nullable=True),
        sa.Column("calibrated_probability", sa.Float(), nullable=True),
        sa.Column("contribution_logit", sa.Float(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_signals_analysis_id_hypothesis_lane", "signals", ["analysis_id", "hypothesis", "lane"])

    op.create_table(
        "assessments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("analysis_id", sa.String(36), _analysis_fk(), nullable=False),
        sa.Column("hypothesis", sa.String(40), nullable=False),
        sa.Column("lane", sa.String(16), nullable=True),
        sa.Column("track_id", sa.String(64), nullable=True),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.Column("ci_low", sa.Float(), nullable=True),
        sa.Column("ci_high", sa.Float(), nullable=True),
        sa.Column("band", sa.String(32), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_assessments_analysis_id_hypothesis", "assessments", ["analysis_id", "hypothesis"])

    op.create_table(
        "findings",
        sa.Column("analysis_id", sa.String(36), _analysis_fk(), primary_key=True),
        sa.Column("finding_id", sa.String(16), primary_key=True),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("family", sa.String(48), nullable=False),
        sa.Column("hypothesis", sa.String(40), nullable=True),
        sa.Column("lane", sa.String(16), nullable=True),
        sa.Column("strength", sa.String(16), nullable=False),
        sa.Column("review_priority", sa.String(8), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )

    op.create_table(
        "exhibits",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("analysis_id", sa.String(36), _analysis_fk(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("payload_artefact_id", sa.String(36), sa.ForeignKey("artefacts.id"), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=True),
    )
    op.create_index("ix_exhibits_analysis_id", "exhibits", ["analysis_id"])

    op.create_table(
        "analysis_events",
        sa.Column("seq", SEQ, primary_key=True, autoincrement=True),
        sa.Column("analysis_id", sa.String(36), _analysis_fk(), nullable=False),
        sa.Column("ts", TS, nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_analysis_events_analysis_id_seq", "analysis_events", ["analysis_id", "seq"])

    _backfill_media_objects()


def _backfill_media_objects() -> None:
    """One `media_objects` row per distinct analysed sha256, taken from its earliest analysis."""
    conn = op.get_bind()
    rows = conn.execute(sa.text(
        "SELECT id, sha256, filename, content_type, size_bytes, created_at FROM analyses ORDER BY created_at, id")).fetchall()
    media_ids: dict[str, str] = {}
    for analysis_id, sha256, filename, content_type, size_bytes, created_at in rows:
        sha = sha256.lower()
        if sha not in media_ids:
            media_ids[sha] = str(uuid.uuid4())
            conn.execute(sa.text(
                "INSERT INTO media_objects (id, sha256, md5, size_bytes, mime_sniffed, original_filename, storage_artefact_id, "
                "first_seen_at) VALUES (:id, :sha, NULL, :size, :mime, :name, NULL, :ts)"),
                {"id": media_ids[sha], "sha": sha, "size": size_bytes, "mime": content_type, "name": filename, "ts": created_at})
        conn.execute(sa.text("UPDATE analyses SET media_id = :mid WHERE id = :aid"), {"mid": media_ids[sha], "aid": analysis_id})


def downgrade() -> None:
    op.drop_index("ix_analysis_events_analysis_id_seq", table_name="analysis_events")
    op.drop_table("analysis_events")
    op.drop_index("ix_exhibits_analysis_id", table_name="exhibits")
    op.drop_table("exhibits")
    op.drop_table("findings")
    op.drop_index("ix_assessments_analysis_id_hypothesis", table_name="assessments")
    op.drop_table("assessments")
    op.drop_index("ix_signals_analysis_id_hypothesis_lane", table_name="signals")
    op.drop_table("signals")
    op.drop_index("ix_detector_runs_analysis_id", table_name="detector_runs")
    op.drop_table("detector_runs")
    op.drop_table("analysis_stages")
    op.drop_table("media_objects")
    op.drop_index("ix_artefacts_analysis_id", table_name="artefacts")
    op.drop_index("ix_artefacts_sha256", table_name="artefacts")
    op.drop_table("artefacts")
    with op.batch_alter_table("analyses") as batch:
        batch.drop_index("ix_analyses_media_id")
        batch.drop_index("ix_analyses_case_id_created_at")
        for name, _type, _default in reversed(ANALYSIS_COLUMNS):
            batch.drop_column(name)
