"""ExhibitWriter: how detectors and stages emit exhibits (ARCHITECTURE §2.2 `RunContext.exhibits`, §4.3).

- **Where binaries go.** Payloads are written straight into the content-addressed store:
  - heatmaps as an 8-bit grayscale PNG, plus a reliability mask and an inferno preview (UX §8.3);
  - images as given.
  The writer keeps the pending artefact records and the exhibit models in memory. `run()` never touches the DB (§2.2).
  The runner indexes the artefacts with `persist_artefacts()` and projects the exhibits with the published result.
- **Ids are deterministic.** They are derived from (analysis id, created_by, kind, ordinal), so a resumed stage that
  re-emits the same exhibits reuses the same ids, and indexing is idempotent.
- **Exhibit `sha256`.** For binary exhibits (heatmap, image) it is the sha256 of the payload artefact. For JSON exhibits
  it is the canonical-JSON hash of the exhibit with the `sha256` field excluded.
"""
from __future__ import annotations

import hashlib
import io
import re
import uuid
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
from PIL import Image
from pydantic import TypeAdapter
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash
from app.schemas.evidence import CoordinateSpace, Exhibit
from app.storage.files import untrusted_strings, untrusted_text
from app.storage.objects import ObjectStore

PREVIEW_MAX_PX = 1024
COLORMAP = "inferno"  # UX §8.3: monotonic L*, CVD- and grayscale-safe
_HEX64_PLACEHOLDER = "0" * 64
_EXHIBIT = TypeAdapter(Exhibit)


RASTER_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class ExhibitBudgetExceeded(RuntimeError):
    """The analysis exceeded its artefact storage budget (ARCHITECTURE §4.3 storage budgets)."""


@dataclass(frozen=True)
class PendingArtefact:
    id: str
    sha256: str
    size_bytes: int
    media_type: str
    role: str
    created_by: str


@dataclass
class _WriterState:
    analysis_id: str
    store: ObjectStore
    budget_bytes: int | None
    exhibits: list[Any] = field(default_factory=list)
    artefacts: list[PendingArtefact] = field(default_factory=list)
    counters: dict[tuple[str, str], int] = field(default_factory=dict)
    region_ids: int = 0
    bytes_written: int = 0


@cache
def _lut() -> np.ndarray:
    from matplotlib import colormaps

    return np.rint(colormaps[COLORMAP](np.arange(256))[:, :3] * 255).astype(np.uint8)


def _png(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG", compress_level=6)  # no text chunks or timestamps: deterministic
    return buf.getvalue()


def exhibit_payload_hash(exhibit: Any) -> str:
    """Canonical hash of a JSON exhibit, excluding its own `sha256` field."""
    return canonical_hash(exhibit.model_dump(mode="json", exclude={"sha256"}))


class ExhibitWriter:
    def __init__(self, analysis_id: str, store: ObjectStore, *, created_by: str = "pipeline", budget_bytes: int | None = None,
                 _state: _WriterState | None = None):
        self._state = _state or _WriterState(analysis_id=analysis_id, store=store, budget_bytes=budget_bytes)
        self.created_by = created_by

    def bind(self, created_by: str) -> ExhibitWriter:
        """A view for one detector or stage that shares the analysis's exhibits, artefacts and budget."""
        return ExhibitWriter(self._state.analysis_id, self._state.store, created_by=created_by, _state=self._state)

    @property
    def exhibits(self) -> list[Any]:
        return list(self._state.exhibits)

    @property
    def artefacts(self) -> list[PendingArtefact]:
        return list(self._state.artefacts)

    @property
    def bytes_written(self) -> int:
        return self._state.bytes_written

    def artefact(self, artefact_id: str) -> PendingArtefact:
        for a in self._state.artefacts:
            if a.id == artefact_id:
                return a
        raise KeyError(artefact_id)

    # --- ids and storage -------------------------------------------------------------------------------------

    def _digest(self, prefix: str) -> bytes:
        key = (self.created_by, prefix)
        n = self._state.counters.get(key, 0)
        self._state.counters[key] = n + 1
        return hashlib.sha256(f"{self._state.analysis_id}\x1f{self.created_by}\x1f{prefix}\x1f{n}".encode()).digest()

    def _exhibit_id(self) -> str:
        return "ex_" + self._digest("exhibit").hex()[:24]

    def _region_id(self, region: dict[str, Any]) -> dict[str, Any]:
        """Localisation regions `R1…` are numbered once per analysis, in emission order, so a localiser's R1 and a segmented
        map's R1 cannot collide (D-26; DR P0-1). Stages run detectors sequentially, so the numbering is deterministic."""
        if not re.fullmatch(r"R\d+", str(region.get("display_id") or "")):
            return region
        self._state.region_ids += 1
        return {**region, "display_id": f"R{self._state.region_ids}"}

    def _put(self, data: bytes, *, media_type: str, role: str) -> PendingArtefact:
        s = self._state
        if s.budget_bytes is not None and s.bytes_written + len(data) > s.budget_bytes:
            raise ExhibitBudgetExceeded(
                f"artefacts for analysis {s.analysis_id} would exceed the {s.budget_bytes} byte budget")
        obj = s.store.put_bytes(data)
        pending = PendingArtefact(id=str(uuid.UUID(bytes=self._digest("artefact")[:16], version=4)), sha256=obj.sha256,
                                  size_bytes=obj.size_bytes, media_type=media_type, role=role, created_by=self.created_by)
        s.bytes_written += obj.size_bytes
        s.artefacts.append(pending)
        return pending

    def _record(self, exhibit: Any) -> str:
        self._state.exhibits.append(exhibit)
        return exhibit.id

    # --- exhibit kinds ---------------------------------------------------------------------------------------

    def binary_mask(self, values: np.ndarray) -> PendingArtefact:
        """Keep source-bound binary support as a PNG without creating a scored exhibit."""
        if values.ndim != 2 or not values.size or values.dtype != np.bool_:
            raise ValueError("a binary mask must be a non-empty Boolean pixel grid")
        return self._put(_png(values.astype(np.uint8) * 255), media_type="image/png", role="mask")

    def heatmap(self, values: Any, *, semantics: str, title: str, caption: str = "", calibrated: bool = False,
                value_range: tuple[float, float] | None = None, mask: Any = None, source_signal_ids: list[str] | None = None,
                media_ref: str = "analysis_copy", coordinate_space: CoordinateSpace | None = None,
                frame_pts: int | None = None) -> str:
        """Store a 2-D map as an 8-bit PNG (linear over `value_range`), with an optional reliability mask and a preview."""
        v = np.asarray(values, dtype=np.float64)
        if v.ndim != 2 or v.size == 0:
            raise ValueError("a heatmap must be a non-empty 2-D array")
        if not np.isfinite(v).all():
            raise ValueError("heatmap values must be finite")
        if calibrated and semantics != "p_manipulated":
            raise ValueError("only p_manipulated maps can carry calibrated probabilities (EA-04)")
        if value_range is None:
            lo, hi = (0.0, 1.0) if semantics == "p_manipulated" else (float(v.min()), float(v.max()))
            hi = hi if hi > lo else lo + 1.0  # a flat map quantises to 0
        else:
            lo, hi = float(value_range[0]), float(value_range[1])
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                raise ValueError("value_range must be finite with high > low")
        if calibrated and (lo, hi) != (0.0, 1.0):
            raise ValueError("a calibrated map must use value_range (0, 1)")
        q = np.clip(np.rint((v - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)

        mask_id = None
        if mask is not None:
            m = np.asarray(mask)
            if m.shape != v.shape:
                raise ValueError(f"mask shape {m.shape} does not match the heatmap {v.shape}")
            if m.dtype == np.bool_:
                m8 = np.where(m, 255, 0).astype(np.uint8)
            elif m.dtype == np.uint8:
                m8 = m
            else:
                mf = m.astype(np.float64)
                if not np.isfinite(mf).all() or mf.min() < 0 or mf.max() > 1:
                    raise ValueError("a float mask must lie in [0, 1]")
                m8 = np.rint(mf * 255.0).astype(np.uint8)
            mask_id = self._put(_png(m8), media_type="image/png", role="mask").id

        payload = self._put(_png(q), media_type="image/png", role="exhibit_payload")
        preview = Image.fromarray(_lut()[q])
        if max(preview.size) > PREVIEW_MAX_PX:
            preview.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX), Image.Resampling.BILINEAR)
        preview_art = self._put(_png(np.asarray(preview)), media_type="image/png", role="preview")

        exhibit = _EXHIBIT.validate_python({
            "kind": "heatmap", "id": self._exhibit_id(), "title": untrusted_text(title), "caption": untrusted_text(caption), "created_by": self.created_by,
            "source_signal_ids": list(source_signal_ids or []), "media_ref": media_ref, "frame_pts": frame_pts,
            "coordinate_space": coordinate_space, "sha256": payload.sha256, "semantics": semantics, "calibrated": calibrated,
            "grid": (int(v.shape[1]), int(v.shape[0])), "value_range": (lo, hi), "payload_artefact_id": payload.id,
            "reliability_mask_artefact_id": mask_id, "preview_artefact_id": preview_art.id,
        })
        return self._record(exhibit)

    def image(self, data: bytes, *, role: str, media_type: str, title: str, caption: str = "",
              source_signal_ids: list[str] | None = None, media_ref: str = "analysis_copy",
              coordinate_space: CoordinateSpace | None = None, frame_pts: int | None = None) -> str:
        """Store an image exhibit (still, crop, contact sheet, spectrum)."""
        if media_type not in RASTER_TYPES:  # N-10 (SEC-10): never a type a browser treats as a document (SVG, HTML)
            raise ValueError("an image exhibit is JPEG, PNG or WebP")
        payload = self._put(data, media_type=media_type, role="still" if role == "still" else "exhibit_payload")
        exhibit = _EXHIBIT.validate_python({
            "kind": "image", "id": self._exhibit_id(), "title": untrusted_text(title), "caption": untrusted_text(caption), "created_by": self.created_by,
            "source_signal_ids": list(source_signal_ids or []), "media_ref": media_ref, "frame_pts": frame_pts,
            "coordinate_space": coordinate_space, "sha256": payload.sha256, "role": role, "payload_artefact_id": payload.id,
        })
        return self._record(exhibit)

    def add(self, kind: str, *, title: str, media_ref: str, caption: str = "", source_signal_ids: list[str] | None = None,
            **fields: Any) -> str:
        """A JSON exhibit: regions, face_track, series, segments, table or text_spans."""
        if kind in ("heatmap", "image"):
            raise ValueError(f"use ExhibitWriter.{kind}() for binary exhibits")
        if kind == "regions":
            fields["regions"] = [self._region_id(r) for r in fields.get("regions") or []]
        # N-19p (SEC6-01): the title, the caption and the text of every kind may hold text from the file
        exhibit = _EXHIBIT.validate_python({
            "kind": kind, "id": self._exhibit_id(), "title": untrusted_text(title), "caption": untrusted_text(caption),
            "created_by": self.created_by, "source_signal_ids": list(source_signal_ids or []), "media_ref": media_ref,
            "sha256": _HEX64_PLACEHOLDER, **untrusted_strings(fields),
        })
        return self._record(exhibit.model_copy(update={"sha256": exhibit_payload_hash(exhibit)}))

    def table(self, columns: list[str], rows: list[list[Any]], *, title: str, media_ref: str = "original", **kw: Any) -> str:
        # N-19o (SEC5-01): a cell can hold text from the file (PNG text keys, a manifest's fields); `add` filters it
        return self.add("table", title=title, media_ref=media_ref, columns=columns, rows=rows, **kw)

    def regions(self, regions: list[dict[str, Any]], *, title: str, media_ref: str = "analysis_copy", **kw: Any) -> str:
        return self.add("regions", title=title, media_ref=media_ref, regions=regions, **kw)

    def series(self, series: list[dict[str, Any]], *, x_unit: str, title: str, media_ref: str = "frame", **kw: Any) -> str:
        return self.add("series", title=title, media_ref=media_ref, series=series, x_unit=x_unit, **kw)

    def segments(self, segments: list[dict[str, Any]], *, lane: str, title: str, media_ref: str = "frame", **kw: Any) -> str:
        return self.add("segments", title=title, media_ref=media_ref, segments=segments, lane=lane, **kw)

    # --- persistence (runner side) ---------------------------------------------------------------------------

    def persist_artefacts(self, session: Session) -> list[Any]:
        """Index the pending artefacts (idempotent on their deterministic ids). The caller commits."""
        from app.storage.artefacts import index_object

        return [index_object(session, a.sha256, a.size_bytes, media_type=a.media_type, role=a.role, created_by=a.created_by,
                             analysis_id=self._state.analysis_id, artefact_id=a.id) for a in self._state.artefacts]
