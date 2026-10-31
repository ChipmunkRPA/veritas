"""Fusion context for benchmark items (ADR 0004; BUILD_SPEC H-4, H-7).

- **`manifest`** (v2.0 artefacts): context from the manifest columns, which describe the native file. `face_px_max` is never
  filled, and `jpeg_q_est` / `has_exif` describe bytes the detectors did not score: images are scored on the `norm`
  re-encode (JPEG, no EXIF).
- **`scored`**: the condition context that the production pipeline recorded while it scored those bytes. It is kept in the
  `v2.<kind>_fused@*.parquet` marker rows, `aux.context`: `face_px` from the face stage, `jpeg_q`, `min_side`,
  `has_camera_exif`. It is looked up per (item, variant, transform), so a robustness copy gets the context of its own bytes.
  Several marker files for one modality (different detector sets) record the same condition stage; the newest row wins.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.engine.fusion.features import Context

CONTEXT_SOURCES = ("manifest", "scored")
MARKERS = {"image": "v2.image_fused", "video": "v2.video_fused", "audio": "v2.audio_fused"}


def manifest_context(row: Mapping[str, Any], modality: str = "image") -> Context:
    """The fuser context from manifest columns (native file), as v2.0 `calibrate.py` built it."""
    width, height = row.get("width"), row.get("height")
    return Context(modality=modality, face_px=row.get("face_px_max"), jpeg_q=row.get("jpeg_q_est"),
                   min_side=min(width, height) if width and height else None, has_camera_exif=bool(row.get("has_exif")))


def _context(modality: str, raw: Mapping[str, Any]) -> Context:
    def whole(value):
        return None if value is None else int(value)

    return Context(modality=modality, face_px=whole(raw.get("face_px")), jpeg_q=whole(raw.get("jpeg_q")),
                   min_side=whole(raw.get("min_side")), has_camera_exif=bool(raw.get("has_camera_exif")))


class ScoredContexts:
    """(item_id, variant, transform) → Context from the modality's v2 fused marker rows in a score cache."""

    def __init__(self, scores_dir: Path, modality: str):
        import pyarrow.parquet as pq

        if modality not in MARKERS:
            raise ValueError(f"no scored context for modality {modality!r}")
        self.modality = modality
        best: dict[tuple[str, str, str], tuple[str, Context, str | None]] = {}
        self.files = sorted(p.name for p in Path(scores_dir).glob(f"{MARKERS[modality]}@*.parquet"))
        for name in self.files:
            columns = ["item_id", "variant", "transform", "aux", "scored_at"]
            for row in pq.read_table(Path(scores_dir) / name, columns=columns).to_pylist():
                try:
                    aux = json.loads(row["aux"]) if row["aux"] else {}
                    raw = aux.get("context")
                except json.JSONDecodeError:
                    raw = None
                if not isinstance(raw, Mapping):
                    continue
                key, stamp = (row["item_id"], row["variant"], row["transform"]), row["scored_at"] or ""
                if key not in best or stamp > best[key][0]:
                    best[key] = (stamp, _context(modality, raw), aux.get("slice_id"))
        self._contexts = {key: ctx for key, (_, ctx, _) in best.items()}
        self._slice_ids = {key: sid for key, (_, _, sid) in best.items()}

    def __len__(self) -> int:
        return len(self._contexts)

    def get(self, item_id: str, variant: str, transform: str = "none") -> Context | None:
        return self._contexts.get((item_id, variant, transform))

    def slice_id(self, item_id: str, variant: str, transform: str = "none") -> str | None:
        """The production condition slice of these exact bytes, including transformed inputs."""
        return self._slice_ids.get((item_id, variant, transform))
