"""The examiner layer joined into machine findings at read time (ARCHITECTURE §4.5 `examiner`; PRD RW-01; BUILD_SPEC C-4).

For each target id: every live annotation that targets it (`annotation_ids`, oldest first), and the examiner status most
recently set among them with that annotation's note. The machine layer is never changed.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.db import AnnotationRow


def examiner_layers(db: Session, analysis_id: str, target_type: str = "finding") -> dict[str, dict[str, Any]]:
    rows = db.scalars(select(AnnotationRow).where(AnnotationRow.analysis_id == analysis_id, AnnotationRow.target_type == target_type,
                                                  AnnotationRow.deleted_at.is_(None))).all()
    layers: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: (r.created_at, r.id)):
        layers.setdefault(row.target_id or "", {"status": None, "note": None, "annotation_ids": []})["annotation_ids"].append(row.id)
    for row in sorted((r for r in rows if r.examiner_status is not None), key=lambda r: (r.edited_at or r.created_at, r.id)):
        layers[row.target_id or ""].update(status=row.examiner_status, note=row.body)
    return layers
