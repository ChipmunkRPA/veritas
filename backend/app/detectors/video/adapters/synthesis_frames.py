"""Synthesis lane, frame channel: whole-image detectors on native-resolution crops of clip frames (V-4; VF §5 item 1;
ML N6, §4.3).

The detect stage passes `ClipInputs` (see `synthesis.py`) in `outputs.clips`.

- **Frames.** Up to 8 frames per shot, evenly over the frames of that shot's plan clips, at most 48 per video evenly over
  the per-shot selections; decoded in one `FrameStore` pass at native resolution. This replaces the whole-frame average of
  resized frames (VF §5).
- **Crops.** The image detector's native crop grid (centre and four corners, never resampled; ML N2) on each frame; the
  frame logit is the mean crop logit, as the image adapter aggregates an image.
- **Scores.** One `frame@pts=<pts>` unit per scored frame; the aggregate is the mean frame logit, with the 90th percentile
  and the fraction of frames above 0 as quality (ML §4.3 clip features). Units and the time-axis exhibit follow frame-table
  order, independent of the FrameStore cache. Missing selected frames or unusable native crops fail the check rather than
  silently changing the sampled population.
- These video adapters are enabled by default in full and deep (H-8; served video fuser 2.1.1). Their image counterparts
  have separate admission and default-enable policies.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from app.detectors.image import commfor, dda, native_crops
from app.detectors.image.adapters.common import revisions
from app.detectors.video import sampling
from app.detectors.video.adapters.common import FULL_DEEP, H1, video_spec
from app.detectors.video.adapters.synthesis import ClipInputs
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    RunContext,
    UnitScore,
)

FRAMES_PER_SHOT = 8
MAX_FRAMES = 48


def select_frames(clips: Any) -> tuple[list[int], dict[int, int]]:
    """(frame-table indices in order, frame → shot): up to 8 per shot over its clips' frames, at most 48 per video."""
    by_shot: dict[int, list[int]] = {}
    for clip in clips or ():
        by_shot.setdefault(int(clip["shot"]), []).extend(clip.get("frames") or ())
    chosen: dict[int, int] = {}
    for shot, frames in sorted(by_shot.items()):
        unique = sorted(set(frames))
        for position in sampling.uniform_indices(len(unique), FRAMES_PER_SHOT):
            chosen[unique[position]] = shot
    ordered = sorted(chosen)
    if len(ordered) > MAX_FRAMES:
        ordered = [ordered[p] for p in sampling.uniform_indices(len(ordered), MAX_FRAMES)]
    return ordered, {i: chosen[i] for i in ordered}


class NativeFrameCropDetector(Detector):
    impl: ClassVar[Any]  # module exposing CROP, crop_origins(width, height), to_batch(pixels, origins), logits(model, batch, device)
    MODEL: ClassVar[str]
    LABEL: ClassVar[str]
    REFERENCE: ClassVar[str]

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL, len(native_crops.CROP_NAMES), "native frame crop",
                              f"One frame at a time, centre and corners: at most five {self.impl.CROP} px crops"),)

    def _small(self) -> str:
        return f"frames below {self.impl.CROP} px on the short side (native crops only, no upscaling)"

    def applicable(self, ctx: RunContext) -> Applicability:
        inputs = ctx.outputs.clips
        if not isinstance(inputs, ClipInputs) or not inputs.clips:
            return Applicability(False, "no clip in the sampling plan")
        if min(inputs.table.width, inputs.table.height) < self.impl.CROP:
            return Applicability(False, self._small())
        return Applicability(True) if select_frames(inputs.clips)[0] else Applicability(False, "no clip frame in the sampling plan")

    def run(self, ctx: RunContext) -> DetectorResult:
        inputs: ClipInputs = ctx.outputs.clips
        frames, shot_of = select_frames(inputs.clips)
        model = ctx.lease.get(self.MODEL).obj
        rows: list[tuple[int, float, int]] = []
        decoded: set[int] = set()
        unusable: set[int] = set()
        for index, rgb in inputs.store.iter_many(frames):
            decoded.add(index)
            origins = self.impl.crop_origins(int(rgb.shape[1]), int(rgb.shape[0]))
            if not origins:
                unusable.add(index)
                continue
            if ctx.deadline is not None:
                ctx.deadline.check(f"{self.LABEL} frame crops")
            crop_logits = self.impl.logits(model, self.impl.to_batch(rgb, origins), ctx.device)
            rows.append((index, float(np.mean(crop_logits)), len(origins)))
        missing = set(frames) - decoded
        if missing:
            return DetectorResult(status="failed", reason=f"{len(missing)} of {len(frames)} selected synthesis frames were not decoded",
                                  metrics={"frames_selected": float(len(frames)), "frames_decoded": float(len(decoded))})
        if unusable:
            return DetectorResult(status="failed",
                                  reason=f"{len(unusable)} selected synthesis frame(s) cannot supply native {self.impl.CROP} px crops",
                                  metrics={"frames_selected": float(len(frames)), "frames_unusable": float(len(unusable))})
        if not rows:
            return DetectorResult.not_applicable("no clip frame in the sampling plan")
        rows.sort(key=lambda row: row[0])

        table, crop = inputs.table, self.impl.CROP
        z = np.asarray([r[1] for r in rows])
        mean, p90, positive = float(z.mean()), float(np.quantile(z, 0.9)), float((z > 0).mean())
        shots = len({shot_of[i] for i, _, _ in rows})
        sid = f"{self.spec.id}:global"
        units = [UnitScore(f"frame@pts={table.pts[i]}", zi, "logit", quality={"time_s": table.seconds(i), "crops": float(n),
                                                                                "shot": float(shot_of[i])})
                 for i, zi, n in rows]
        exhibit_id = ctx.exhibits.series(
            [{"name": "logit", "hypothesis": H1.value, "points": [(round(table.seconds(i), 6), round(zi, 6)) for i, zi, _ in rows]}],
            x_unit="s", title=f"{self.LABEL} logit per frame", media_ref="frame", thresholds=[0.0],
            caption=(f"Mean logit of native-resolution {crop} px crops of each sampled frame. Uncalibrated; the decision boundary "
                     "is 0."), source_signal_ids=[sid])
        side = "synthetic" if mean >= 0 else "natural-image"
        reasoning = f"The classifier places the mean of its frame scores on the {side} side of its decision boundary."
        if z.min() < 0 <= z.max():
            reasoning += " Its frames fall on both sides of the boundary."
        reasoning += " It was trained on still images, so video compression and motion blur are outside its training conditions."
        return DetectorResult(
            status="ok", unit_scores=units,
            aggregate=UnitScore("global", mean, "logit", quality={"frames": float(len(rows)), "shots": float(shots), "p90_logit": p90,
                                                                  "fraction_positive": positive}),
            observed=(f"mean frame logit {mean:+.3f} over {len(rows)} native frame(s) in {shots} shot(s), each the mean of up to "
                      f"5 {crop} px crops; 90th percentile {p90:+.3f}; {round(positive * len(rows))} of {len(rows)} frame(s) at logit > 0"),
            reference=self.REFERENCE, reasoning=reasoning,
            metrics={"frames": float(len(rows)), "shots": float(shots), "logit_mean": mean, "logit_p90": p90, "logit_max": float(z.max()),
                     "logit_min": float(z.min()), "fraction_positive": positive},
            exhibit_ids=[exhibit_id], model_revisions=revisions(ctx, self.MODEL))


class FrameCommFor384Detector(NativeFrameCropDetector):
    impl = commfor
    MODEL = "commfor.vit_s16_384"
    LABEL = "Community Forensics"
    REFERENCE = ("ViT-S/16 trained on camera photographs and generator outputs of the Community Forensics training data, applied "
                 "to native 384 px crops of video frames; decision boundary at logit 0")
    spec = video_spec("vid.synth.frames.commfor384", "Synthesis lane: Community Forensics (ViT-S/16) on native 384 px crops of clip frames",
                      InputUnit.FRAME_BATCH, (H1,), ("series",), family="learned-whole-image", model_ids=(MODEL,), needs=("frames",),
                      timeout_s=120.0, profiles=FULL_DEEP, version="0.4.1")


class FrameDDADetector(NativeFrameCropDetector):
    impl = dda
    MODEL = "dda.dinov2_vitl14_lora"
    LABEL = "DDA"
    REFERENCE = ("DINOv2 ViT-L/14 with LoRA trained to separate MSCOCO photographs from their VAE reconstructions, applied to "
                 "native 336 px crops of video frames; decision boundary at logit 0")
    spec = video_spec("vid.synth.frames.dda", "Synthesis lane: DDA (DINOv2 ViT-L/14 + LoRA) on native 336 px crops of clip frames",
                      InputUnit.FRAME_BATCH, (H1,), ("series",), family="learned-whole-image", model_ids=(MODEL,), needs=("frames",),
                      timeout_s=180.0, profiles=FULL_DEEP, version="0.4.1")
