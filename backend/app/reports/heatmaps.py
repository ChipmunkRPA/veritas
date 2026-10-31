"""Frozen scalar-map bindings for report rendering; no live result or artifact-index lookups.

The existing snapshot image map stores new #scalar keys without adding default fields
that would change canonical hashes of older snapshots. The report service's template
identity gate preserves completed PDFs and refuses old pending snapshots.
"""
from __future__ import annotations

import math
import re
from typing import Any

from app.core import coords
from app.reports import figures
from app.reports.citations import finding_heatmaps
from app.reports.snapshot import ReportSnapshot

_HEX = re.compile(r"[0-9a-f]{64}")
_SEMANTICS = {"p_manipulated", "attribution", "residual", "error_level"}


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def _pair(value: Any, *, integers: bool = False) -> bool:
    try:
        return (isinstance(value, (tuple, list)) and len(value) == 2
                and all(type(v) is int and 0 < v <= 80_000_000 if integers else type(v) in (int, float) and math.isfinite(v) for v in value))
    except OverflowError:
        return False


def scalar_binding(snapshot: ReportSnapshot, exhibit: dict[str, Any]) -> dict[str, Any] | None:
    """Validate frozen identity and scalar metadata before reading any image bytes."""
    result = snapshot.result
    eid = exhibit.get("id")
    if (result.get("analysis_id") != snapshot.analysis.id or result.get("kind") != snapshot.analysis.kind
            or result.get("revision") != snapshot.analysis.revision or not isinstance(eid, str)
            or sum(isinstance(e, dict) and e.get("id") == eid for e in result.get("exhibits") or []) != 1
            or exhibit.get("kind") != "heatmap" or not isinstance(exhibit.get("semantics"), str)
            or exhibit["semantics"] not in _SEMANTICS
            or type(exhibit.get("calibrated")) is not bool):
        return None
    original = result.get("input")
    if not isinstance(original, dict) or original.get("sha256") != snapshot.media.sha256:
        return None
    scalar = snapshot.exhibit_images.get(f"{eid}#scalar")
    if not _sha(scalar) or scalar != exhibit.get("sha256") or not exhibit.get("payload_artefact_id"):
        return None
    grid, value_range = exhibit.get("grid"), exhibit.get("value_range")
    if not _pair(grid, integers=True) or not _pair(value_range) or value_range[1] <= value_range[0]:
        return None
    if exhibit["calibrated"] and (exhibit["semantics"] != "p_manipulated" or tuple(value_range) != (0, 1)):
        return None
    required = exhibit.get("reliability_mask_artefact_id") is not None
    mask = snapshot.exhibit_images.get(f"{eid}#reliability")
    if (required and (not exhibit.get("reliability_mask_artefact_id") or not _sha(mask))) or (not required and mask is not None):
        return None
    return {"payload_sha256": scalar, "grid": tuple(grid), "value_range": tuple(value_range),
            "reliability_sha256": mask, "require_reliability": required}


def overlay_source_size(snapshot: ReportSnapshot, exhibit: dict[str, Any]) -> tuple[int, int] | None:
    """Only whole-image mappings with a recorded extent within EA-04's two-pixel alignment bound (`coords.fits`). Returns the
    size in oriented pixels, the space the map and the decoded source share.

    Coordinate space can describe either the map grid or the larger analysis copy.
    Allow the canonical six-decimal scale rounding separately (half a unit in the
    last place times the recorded extent). Crops/frames or missing geometry are
    omitted; a map-cell-wide error is not sufficient evidence of pixel alignment.
    """
    if (snapshot.analysis.kind != "image" or exhibit.get("media_ref") not in ("original", "analysis_copy")
            or exhibit.get("frame_pts") is not None or snapshot.exhibit_images.get("source") != snapshot.media.sha256
            or not _sha(snapshot.media.sha256)):
        return None
    profile, space = snapshot.result.get("condition_profile"), exhibit.get("coordinate_space")
    if not isinstance(profile, dict) or not isinstance(space, dict):
        return None
    source_size = (profile.get("width"), profile.get("height"))
    extent = (space.get("width"), space.get("height"))
    scale = space.get("scale_to_original")
    if (not _pair(source_size, integers=True) or not _pair(extent, integers=True)
            or not _pair((scale, scale)) or scale <= 0):
        return None
    # N-4: exhibits are in EXIF-oriented pixels; a profile from before N-4 recorded the stored size of a rotated photo
    oriented = coords.oriented_source(space, source_size)
    return tuple(oriented) if coords.fits(space, oriented) else None


THUMBNAIL_NOTE = ("Miniatures show the full recorded range, not a decision threshold. "
                  "Dark areas do not rule out manipulation.")


def finding_thumbnails(snapshot: ReportSnapshot) -> dict[str, dict[str, str]]:
    """Map-only finding miniatures from frozen scalar evidence, never RGB previews.

    Selection precedes binding/byte validation: unusable evidence omits the chosen
    miniature instead of substituting another map. Captions distinguish attention
    from localisation and disclose whether reliability was recorded.
    """
    out: dict[str, dict[str, str]] = {}
    for finding_id, exhibit in finding_heatmaps(snapshot.result).items():
        bound = scalar_binding(snapshot, exhibit)
        if bound is None:
            continue
        uri = figures.scalar_map_plate(box=figures.THUMBNAIL_BOX, **bound)
        if uri is None:
            continue
        semantics = exhibit["semantics"]
        label = {"attribution": "Attention, not localisation", "residual": "Residual",
                 "error_level": "Error level"}.get(semantics)
        if semantics == "p_manipulated":
            label = "Calibrated manipulation probability" if exhibit["calibrated"] else "Uncalibrated anomaly score"
        low, high = bound["value_range"]
        # The canonical six-decimal values are displayed without percent conversion.
        extent = f"{str(low).removesuffix('.0')} to {str(high).removesuffix('.0')}"
        reliability = "reliability applied" if bound["require_reliability"] else "reliability not recorded"
        ident = exhibit.get("display_id") or exhibit.get("title") or "Evidence map"
        caption = f"{ident} · {label}; full range {extent}; {reliability}."
        title = exhibit.get("title") or ident
        out[finding_id] = {"uri": uri, "caption": caption, "alt": f"{title}. {caption} {THUMBNAIL_NOTE}",
                           "exhibit_id": exhibit["id"]}
    return out
