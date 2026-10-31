"""v0.3 temporal and forensic video heuristics as v2 plugins (model-free, CPU).

- **Landmark jitter** and **face flicker** follow the primary face track (greedy IoU linking of the largest face,
  `video/temporal.py`). They report raw measurements; the v0.3 logistic mappings are applied by the legacy fuser.
- **Face/background noise** is the noise-residual ratio between the largest face and its surrounding ring per face frame.
"""
from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.detectors.image import forensics
from app.detectors.video import temporal
from app.detectors.video.adapters.common import (
    CPU_ONLY,
    H2,
    frame_faces,
    has_faces,
    sampled_frames,
    series_exhibit,
    to_face,
    video_spec,
)
from app.engine.detector import Applicability, Detector, DetectorResult, InputUnit, RunContext, UnitScore


def _numeric(d: dict) -> dict[str, float]:
    return {k: float(v) for k, v in d.items() if isinstance(v, int | float) and not isinstance(v, bool)}


class LandmarkJitterDetector(Detector):
    spec = video_spec("vid.temporal.landmark_jitter", "Facial-landmark temporal jitter (primary face track)", InputUnit.FRAME_BATCH,
                      (H2,), family="temporal-face", needs=("faces",), lane="cpu", devices=CPU_ONLY, timeout_s=30.0)

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_faces(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        faces = frame_faces(ctx)
        jitter = temporal.landmark_jitter(faces, temporal.track_primary_face(faces, settings.video_face_track_iou))
        if jitter.get("landmark_jitter") is None:
            return DetectorResult.not_applicable("fewer than three consecutive tracked frames with landmarks")
        j = jitter["landmark_jitter"]
        return DetectorResult(
            status="ok", aggregate=UnitScore("global", float(j), "score", quality={"frames_tracked": float(jitter["frames_tracked"])}),
            unit_scores=[UnitScore("global", float(j), "score")],
            observed=f"median box-normalised landmark shift={j} (p90 {jitter['landmark_jitter_p90']}); tracked "
                     f"{jitter['frames_tracked']} frames",
            reference="Natural motion at 2 fps sampling: median shift ≲ 0.03 of box size",
            reasoning="Measures how erratically eye, nose and mouth landmarks move relative to the face box between frames.",
            metrics=_numeric(jitter))


class FaceFlickerDetector(Detector):
    spec = video_spec("vid.temporal.face_flicker", "Face-region temporal flicker vs background (primary face track)",
                      InputUnit.FRAME_BATCH, (H2,), family="temporal-face", needs=("frames", "faces"), lane="cpu", devices=CPU_ONLY,
                      timeout_s=30.0)

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_faces(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        faces = frame_faces(ctx)
        track = temporal.track_primary_face(faces, settings.video_face_track_iou)
        flicker = temporal.face_flicker([f.rgb for f in sampled_frames(ctx)], faces, track)
        if flicker.get("face_flicker_ratio") is None:
            return DetectorResult.not_applicable("fewer than four consecutive tracked face frames")
        r = flicker["face_flicker_ratio"]
        return DetectorResult(
            status="ok", aggregate=UnitScore("global", float(r), "score"), unit_scores=[UnitScore("global", float(r), "score")],
            observed=f"face change / ring change = {r} (face {flicker['face_change']}, ring {flicker['background_change']})",
            reference="Real footage: face and surrounding ring change alike (ratio ≈ 0.7–1.4)",
            reasoning="Compares inter-frame change inside the face with change in the ring around it that moves with the head.",
            metrics=_numeric(flicker))


class FaceNoiseDetector(Detector):
    spec = video_spec("vid.forensic.face_noise", "Face/background noise-residual ratio per face frame", InputUnit.FRAME_BATCH, (H2,),
                      ("series",), family="forensic", needs=("frames", "faces"), lane="cpu", devices=CPU_ONLY, timeout_s=60.0)

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_faces(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        faces = frame_faces(ctx)
        units = []
        for f in sampled_frames(ctx):
            if not faces[f.index]:
                continue
            ctx.deadline.check("noise residual")
            metrics, _ = forensics.noise_map(f.rgb, face_box=to_face(faces[f.index][0]).box())
            ratio = metrics.get("noise_face_bg_ratio")
            if ratio:
                units.append(UnitScore(f"frame:{f.index}", float(ratio), "score", quality={"time_s": f.time_s}))
        if not units:
            return DetectorResult.not_applicable("no face frame produced a noise ratio")
        median = float(np.median([u.raw for u in units]))
        exhibit = series_exhibit(ctx, "noise_face_bg_ratio", [(u.quality["time_s"], u.raw) for u in units],
                                 "Face/background noise ratio per face frame", "1.0 means the face and its ring carry equal noise.")
        return DetectorResult(
            status="ok", unit_scores=units, aggregate=UnitScore("global", median, "score", quality={"frames": float(len(units))}),
            observed=f"median face/background noise ratio={median:.3f} over {len(units)} frames",
            reference="Single-source footage: face vs surrounding ring 0.6–1.6",
            reasoning="Compares the residual noise level of the face with the ring around it in each face frame.",
            metrics={"median": median, "frames": float(len(units))}, exhibit_ids=[exhibit])
