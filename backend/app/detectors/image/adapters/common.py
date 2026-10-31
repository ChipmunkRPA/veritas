"""Shared helpers for the image adapters: detector specs, the analysis copy, face boxes from the faces stage."""
from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from app.detectors.image.faces import Face
from app.engine.detector import (
    DetectorSpec,
    Hypothesis,
    InputUnit,
    Modality,
    Profile,
    Requirements,
    RunContext,
)

ALL_PROFILES = frozenset(Profile)
FULL_DEEP = frozenset({Profile.FULL, Profile.DEEP})
ANY_DEVICE = ("cuda", "mps", "cpu")
CPU_ONLY = ("cpu",)
H1 = Hypothesis.H1_WHOLE_CONTENT_SYNTHESIS
H2 = Hypothesis.H2_LOCALISED_MANIPULATION
H3 = Hypothesis.H3_PROVENANCE_INTEGRITY


def image_spec(det_id: str, name: str, family: str, unit: InputUnit, hypotheses: tuple[Hypothesis, ...],
               produces: tuple[str, ...] = (), *, model_ids: tuple[str, ...] = (), needs: tuple[str, ...] = (),
               lane: str = "accelerator", devices: tuple[str, ...] = ANY_DEVICE, timeout_s: float = 60.0,
               profiles: frozenset[Profile] = FULL_DEEP, version: str = "0.3.0") -> DetectorSpec:
    """Adapters keep version 0.3.0: pre- and post-processing are the v0.3 ones (the parity test holds them to it)."""
    return DetectorSpec(id=det_id, version=version, name=name, modality=Modality.IMAGE, input_unit=unit, family=family,
                        hypotheses=hypotheses, produces=produces,
                        requirements=Requirements(model_ids=model_ids, needs=needs, devices=devices, timeout_s=timeout_s,
                                                  lane=lane),  # type: ignore[arg-type]
                        profiles=profiles)


def analysis_rgb(ctx: RunContext) -> np.ndarray:
    pixels, _scale = ctx.media.analysis_copy()
    return np.ascontiguousarray(pixels)


def coordinate_space(ctx: RunContext) -> dict[str, Any]:
    pixels, scale = ctx.media.analysis_copy()
    return {"width": int(pixels.shape[1]), "height": int(pixels.shape[0]), "scale_to_original": float(scale)}


def face_boxes(ctx: RunContext) -> list[Face]:
    """Faces from the faces stage, in analysis-copy coordinates, largest first, with their landmarks when the stage has them.
    Missing output means no faces."""
    return [Face(int(round(f["x"])), int(round(f["y"])), int(round(f["w"])), int(round(f["h"])), float(f.get("score") or 0.0),
                 [(float(x), float(y)) for x, y in f.get("landmarks") or []])
            for f in (ctx.outputs.faces or [])]


def revisions(ctx: RunContext, *model_ids: str) -> dict[str, str]:
    return {m: str(ctx.lease.get(m).revision) for m in model_ids}


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG", compress_level=6)  # no text chunks: deterministic
    return buf.getvalue()


def original_bytes(ctx: RunContext) -> bytes:
    return bytes(ctx.media.original_bytes())
