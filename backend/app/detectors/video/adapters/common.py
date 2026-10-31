"""Shared helpers for the video adapters: detector specs, the sampled frames, faces per frame and the sound track.

On the v2 path the sampled frames come from the pts-accurate decode (`video/decode.py`, V-1): the v0.3 frame count,
uniform over the frame table, with `time_s = pts × time_base`. `pts_ms` is that time in milliseconds; region exhibits
and stills use it as `frame_pts`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.detectors.image.faces import Face
from app.engine.detector import (
    Applicability,
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
DEEP_ONLY = frozenset({Profile.DEEP})
ANY_DEVICE = ("cuda", "mps", "cpu")
CPU_ONLY = ("cpu",)
H1 = Hypothesis.H1_WHOLE_CONTENT_SYNTHESIS
H2 = Hypothesis.H2_LOCALISED_MANIPULATION
H3 = Hypothesis.H3_PROVENANCE_INTEGRITY
NO_FACE_REASON = "no face detected in the sampled frames"


@dataclass(frozen=True)
class SampledFrame:
    index: int  # position in the sampling plan (unit refs `frame:<index>`)
    frame_index: int  # frame-table index (decoder output order)
    time_s: float  # pts × time_base on the v2 path; frame_index / fps on the v1 path
    pts_ms: int
    rgb: np.ndarray
    pts: int | None = None  # stream pts (v2 path)
    time_base: str | None = None  # e.g. "1/30000"


@dataclass(frozen=True)
class AudioTrack:
    status: str  # absent | unreadable | short (< 1 s, no PCM) | silent | analysed
    pcm: np.ndarray | None = None
    samples: int = 0
    gate: tuple[tuple[float, float], ...] | None = None  # A-1: speech spans; voice windows at least half inside are scored
    max_windows: int | None = None  # A-1: window cap of a standalone recording (None: the sound-track cap)


def video_spec(det_id: str, name: str, unit: InputUnit, hypotheses: tuple[Hypothesis, ...], produces: tuple[str, ...] = (), *,
               family: str, model_ids: tuple[str, ...] = (), needs: tuple[str, ...] = ("frames",), lane: str = "accelerator",
               devices: tuple[str, ...] = ANY_DEVICE, timeout_s: float = 120.0, profiles: frozenset[Profile] = ALL_PROFILES,
               modality: Modality = Modality.VIDEO, version: str = "0.3.0") -> DetectorSpec:
    """Adapters keep version 0.3.0: pre- and post-processing are the v0.3 ones (the parity test holds them to it)."""
    return DetectorSpec(id=det_id, version=version, name=name, modality=modality, input_unit=unit, family=family,
                        hypotheses=hypotheses, produces=produces,
                        requirements=Requirements(model_ids=model_ids, needs=needs, devices=devices, timeout_s=timeout_s,
                                                  lane=lane),  # type: ignore[arg-type]
                        profiles=profiles)


def sampled_frames(ctx: RunContext) -> list[SampledFrame]:
    return list(ctx.outputs.frames or [])


def frame_faces(ctx: RunContext) -> list[list[dict[str, Any]]]:
    """v0.3 face dicts per sampled frame, largest first. Missing output means no faces."""
    faces = ctx.outputs.faces
    return [list(f) for f in faces] if faces else [[] for _ in sampled_frames(ctx)]


def to_face(d: dict[str, Any]) -> Face:
    return Face(int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]), float(d.get("score") or 0.0),
                [tuple(p) for p in d.get("landmarks") or []])


def has_frames(ctx: RunContext) -> Applicability:
    return Applicability(True) if ctx.outputs.frames else Applicability(False, "no decoded frames")


def has_faces(ctx: RunContext) -> Applicability:
    if not ctx.outputs.frames:
        return Applicability(False, "no decoded frames")
    return Applicability(True) if any(frame_faces(ctx)) else Applicability(False, NO_FACE_REASON)


def series_exhibit(ctx: RunContext, name: str, points: list[tuple[float, float]], title: str, caption: str = "") -> str:
    return ctx.exhibits.series([{"name": name, "points": points}], x_unit="s", title=title, media_ref="frame", caption=caption)
