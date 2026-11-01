"""Face-swap lane per track as a v2 plugin (V-3; VF §3.3, §8.1; PRD DC-03; ARCHITECTURE §3.1 `detect`).

The detect stage passes `TrackInputs` in `outputs.tracks`: the frame table, the tracks stage summary, each track's
`face_track` frames (original px, stream pts) and the plan's sparse sampler, with the run's `FrameStore`.

- **Crops.** For each track, the windows of `video/swap.select_windows` (2–4 fps). Their native frames come from one
  `FrameStore` pass; each crop is aligned from the track's five landmarks at scale 1.3 (upstream `align_face`, ML N5) and
  scored by GenD, CLIP ViT-L/14, in batches of 16. Crops are kept only until their batch is scored.
- **Scores.** One `track:<k>` unit per scored track (mean of the top 30 % crop logits, flagged fraction, face size, the
  size gate) and one `track:<k>@pts=<pts>` unit per crop. The aggregate is the highest track score.
- **Exhibits.** Per track, a logit series on the time axis (threshold 0), and Grad-CAM on its top 3 crops (at most 12
  maps per video), warped from the aligned crop back to frame coordinates with the crop footprint as the reliability mask.
  Both cite the track signal `<detector_id>:track:<k>` that the video fuser builds (`track_text`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.detectors.image import gend
from app.detectors.image.adapters.common import revisions
from app.detectors.video import swap
from app.detectors.video.adapters.common import FULL_DEEP, H2, video_spec
from app.detectors.video.decode import FrameTable
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    RunContext,
    UnitScore,
)

BATCH = 16
NO_LANDMARKS = "no face track with five landmarks to align"
GRADCAM_TOP_K = 3
MAX_GRADCAM = 12
HEATMAP_MAX_PX = 640
REFERENCE = ("CLIP ViT-L/14 with fine-tuned LayerNorms and an L2-normalised linear head, trained on aligned face crops of "
             "face-forgery video datasets; decision boundary at logit 0 per crop; a track scores the mean of its top 30 % "
             "crop logits")


@dataclass(frozen=True)
class TrackInputs:
    table: FrameTable
    tracks: tuple[dict[str, Any], ...]  # tracks stage summary rows: track_id, identity, exhibit_id, …
    frames: dict[str, list[dict[str, Any]]]  # track_id → face_track exhibit frames
    sparse: tuple[int, ...]  # plan sparse sampler, frame-table indices
    sparse_fps: float | None
    store: Any  # FrameStore (iter_many, get_many)


def track_name(track: dict[str, Any]) -> str:
    """"Face A", or "Face track 3" for a track without an identity letter (the face_track exhibit titles)."""
    if track.get("identity"):
        return f"Face {track['identity']}"
    return f"Face track {int(str(track['track_id']).rpartition(':')[2]) + 1}"


def _side(logit: float) -> str:
    return "manipulated-face" if logit >= 0 else "unmanipulated-face"


def track_text(track: dict[str, Any], unit: dict[str, Any]) -> tuple[str, str]:
    """(observed, reasoning) of a track signal from its `track:<k>` unit."""
    q, raw, name = unit["quality"], float(unit["raw"]), track_name(track)
    n = int(q.get("windows", 0))
    flagged = round(q.get("flagged_fraction", 0.0) * n)
    fps = q.get("sparse_fps")
    observed = (f"mean of the top 30 % crop logits {raw:+.3f} over {n} aligned crop(s)" + (f" at {fps:g} fps" if fps else "")
                + f"; {flagged} of {n} crop(s) at logit ≥ 0; median face {q.get('median_face_px', 0.0):.0f} px")
    reasoning = f"The face-forgery detector places the upper tail of {name}'s crops on the {_side(raw)} side of its decision boundary."
    if q.get("small_face", 0.0) >= 0.5:
        reasoning += (f" The face is below {swap.SMALL_FACE_PX:.0f} px wide in most crops, so they were upscaled into the "
                      "detector input; face-swap detectors are unreliable at this size.")
    if q.get("fallback_window", 0.0) >= 0.5:
        reasoning += " The track is shorter than the sampler step, so only its middle detected frame was scored."
    return observed, reasoning


def _frame_map(cam: np.ndarray, matrix: np.ndarray, side: int, size: tuple[int, int], scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Patch-grid map on the aligned crop → (values, mask) on a frame grid of `size` (w, h) at `scale` original px per cell."""
    crop_map = cv2.resize(cam.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
    grid_to_crop = matrix @ np.diag([scale, scale, 1.0])  # crop = M · (scale · grid, 1)
    flags = cv2.WARP_INVERSE_MAP
    values = cv2.warpAffine(crop_map, grid_to_crop, size, flags=cv2.INTER_LINEAR | flags, borderValue=0.0)
    mask = cv2.warpAffine(np.full((side, side), 255, np.uint8), grid_to_crop, size, flags=cv2.INTER_NEAREST | flags, borderValue=0)
    return np.clip(values, 0.0, 1.0), mask


class GenDTrackDetector(Detector):
    MODEL = "gend.clip_l14"
    spec = video_spec("vid.swap.gend_clip_l14", "Face-swap lane: GenD (CLIP ViT-L/14) per face track", InputUnit.FACE_CROP, (H2,),
                      ("series", "heatmap"), family="learned-face", model_ids=(MODEL,), needs=("tracks",), timeout_s=300.0,
                      profiles=FULL_DEEP, version="0.4.0")

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL, BATCH, "224 px face crop", "Existing batches of sixteen aligned track crops",
                              ("Optional single-input Grad-CAM backward/retention has no separately validated peak estimate.",)),)

    def applicable(self, ctx: RunContext) -> Applicability:
        inputs = ctx.outputs.tracks
        if not isinstance(inputs, TrackInputs) or not inputs.tracks:
            return Applicability(False, "no face track")
        # the windows are chosen from the tracks alone, so a video without one takes no model (BUILD_SPEC N-14c, E-10)
        return Applicability(True) if self._plans(inputs) else Applicability(False, NO_LANDMARKS)

    @staticmethod
    def _plans(inputs: TrackInputs) -> list[tuple[Any, Any, bool, bool]]:
        """(track, windows, capped, fallback) of each track that has a window to score. No model and no decoding."""
        plans = []
        for track in inputs.tracks:
            selection = swap.select_windows(track["track_id"], inputs.frames.get(track["track_id"]) or [], inputs.sparse,
                                            inputs.table.seconds)
            if selection.windows:
                plans.append((track, selection.windows, selection.capped, selection.fallback))
        return plans

    def run(self, ctx: RunContext) -> DetectorResult:
        inputs: TrackInputs = ctx.outputs.tracks
        plans = self._plans(inputs)
        if not plans:
            return DetectorResult.not_applicable(NO_LANDMARKS)
        model = ctx.lease.get(self.MODEL).obj
        logits = [[math.nan] * len(p[1]) for p in plans]
        upscaled = [[False] * len(p[1]) for p in plans]
        by_frame: dict[int, list[tuple[int, int]]] = {}
        for t, (_, windows, _, _) in enumerate(plans):
            for k, window in enumerate(windows):
                by_frame.setdefault(window.frame_index, []).append((t, k))
        crops: list[np.ndarray] = []
        keys: list[tuple[int, int]] = []

        def flush() -> None:
            if ctx.deadline is not None:
                ctx.deadline.check("face-swap scoring")
            for (t, k), z in zip(keys, gend.logits(model, gend.to_batch(crops), ctx.device), strict=True):
                logits[t][k] = z
            crops.clear()
            keys.clear()

        for index, rgb in inputs.store.iter_many(by_frame):
            for t, k in by_frame[index]:
                transform = gend.align_transform(plans[t][1][k].landmarks)
                if transform is None:
                    continue
                crop = gend.warp(rgb, *transform)
                upscaled[t][k] = min(crop.shape[:2]) < gend.CROP
                crops.append(crop)
                keys.append((t, k))
                if len(crops) >= BATCH:
                    flush()
        if crops:
            flush()

        scored = []
        for t, (track, windows, capped, fallback) in enumerate(plans):
            keep = [k for k, z in enumerate(logits[t]) if math.isfinite(z)]
            if keep:
                kept = [windows[k] for k in keep]
                zs, ups = [logits[t][k] for k in keep], [upscaled[t][k] for k in keep]
                scored.append((track, kept, zs, capped, fallback, swap.score_track(track["track_id"], kept, zs, ups)))
        if not scored:
            return DetectorResult.not_applicable(NO_LANDMARKS)

        units: list[UnitScore] = []
        window_units: list[UnitScore] = []
        exhibit_ids: list[str] = []
        for track, kept, zs, capped, fallback, score in scored:
            sid = f"{self.spec.id}:{track['track_id']}"
            units.append(UnitScore(track["track_id"], score.top_mean, "logit", quality={
                "windows": float(score.windows), "flagged_fraction": score.flagged_fraction,
                "median_face_px": score.median_face_px, "small_face": float(score.small_face),
                "upscaled_fraction": score.upscaled_fraction, "interpolated_fraction": score.interpolated_fraction,
                "windows_capped": float(capped), "fallback_window": float(fallback),
                "start_s": kept[0].time_s, "end_s": kept[-1].time_s, **({"sparse_fps": inputs.sparse_fps} if inputs.sparse_fps else {})}))
            window_units += [UnitScore(f"{track['track_id']}@pts={w.pts}", z, "logit",
                                       quality={"time_s": w.time_s, "face_px": w.face_px, "interpolated": float(w.interpolated)})
                             for w, z in zip(kept, zs, strict=True)]
            exhibit_ids.append(ctx.exhibits.series(
                [{"name": "logit", "hypothesis": H2.value, "track_id": track["track_id"],
                  "points": [(round(w.time_s, 6), round(z, 6)) for w, z in zip(kept, zs, strict=True)]}],
                x_unit="s", title=f"GenD logit per aligned crop, {track_name(track)}", media_ref="frame", thresholds=[0.0],
                caption="Uncalibrated logit of each crop sampled from the track; the decision boundary is 0.",
                source_signal_ids=[sid]))

        maps = self._gradcam(ctx, model, inputs, scored)
        exhibit_ids += maps
        top_track, _, _, _, _, top = max(scored, key=lambda s: s[5].top_mean)
        total = sum(s[5].windows for s in scored)
        name = track_name(top_track)
        flagged_tracks = sum(1 for s in scored if s[5].top_mean >= 0)
        reasoning = f"The face-forgery detector places the upper tail of {name}'s crops on the {_side(top.top_mean)} side of its decision boundary."
        if len(scored) > 1:
            reasoning += f" {flagged_tracks} of {len(scored)} face tracks score on the manipulated-face side."
        if top.small_face:
            reasoning += f" {name}'s face is below {swap.SMALL_FACE_PX:.0f} px wide in most crops, so they were upscaled."
        return DetectorResult(
            status="ok", unit_scores=[*units, *window_units],
            aggregate=UnitScore("global", top.top_mean, "logit", quality={
                "tracks": float(len(scored)), "windows": float(total), "small_face": float(top.small_face),
                "median_face_px": top.median_face_px}),
            observed=(f"highest track score {top.top_mean:+.3f} for {name} (mean of the top 30 % of {top.windows} crop logits; "
                      f"{round(top.flagged_fraction * top.windows)} at logit ≥ 0); {len(scored)} face track(s) scored over "
                      f"{total} aligned crop(s)"),
            reference=REFERENCE, reasoning=reasoning,
            metrics={"tracks": float(len(inputs.tracks)), "tracks_scored": float(len(scored)), "windows": float(total),
                     "track_logit_max": top.top_mean, "track_logit_min": min(s[5].top_mean for s in scored),
                     "flagged_tracks": float(flagged_tracks), "small_face_tracks": float(sum(s[5].small_face for s in scored)),
                     "gradcam_maps": float(len(maps))},
            exhibit_ids=exhibit_ids, model_revisions=revisions(ctx, self.MODEL),
            details={"frame_pts_unit": "milliseconds", "source_time_base": inputs.table.time_base})

    def _gradcam(self, ctx: RunContext, model: Any, inputs: TrackInputs, scored: list) -> list[str]:
        chosen = []
        for track, kept, zs, *_rest in sorted(scored, key=lambda s: -s[5].top_mean):
            order = sorted(range(len(kept)), key=lambda k: -zs[k])[:GRADCAM_TOP_K]
            chosen += [(track, kept[k], zs[k]) for k in order]
        chosen = chosen[:MAX_GRADCAM]
        frames = inputs.store.get_many(sorted({w.frame_index for _, w, _ in chosen}))
        ids = []
        for track, window, z in chosen:
            rgb = frames.get(window.frame_index)
            transform = gend.align_transform(window.landmarks) if rgb is not None else None
            if transform is None:
                continue
            if ctx.deadline is not None:
                ctx.deadline.check("face-swap Grad-CAM")
            matrix, side = transform
            cam = gend.gradcam(model, gend.to_batch([gend.warp(rgb, matrix, side)]), ctx.device)
            if cam is None:
                continue
            height, width = rgb.shape[:2]
            scale = max(1.0, max(width, height) / HEATMAP_MAX_PX)
            size = (max(1, round(width / scale)), max(1, round(height / scale)))
            values, mask = _frame_map(cam, matrix, side, size, scale)
            ids.append(ctx.exhibits.heatmap(
                values, semantics="attribution", title=f"Grad-CAM, {track_name(track)}, {window.time_s:.2f} s (logit {z:+.2f})",
                caption=("Where GenD's decision on the aligned face crop drew its gradient (Grad-CAM, last encoder block, "
                         "positive part, normalised to its maximum), warped back onto the frame. Attribution, not a map of "
                         "manipulation; the mask marks the crop the detector saw."),
                value_range=(0.0, 1.0), mask=mask, source_signal_ids=[f"{self.spec.id}:{track['track_id']}"], media_ref="frame",
                coordinate_space={"width": size[0], "height": size[1], "scale_to_original": round(width / size[0], 6)},
                # The legacy exhibit/still API uses milliseconds. The immutable
                # video evidence anchors separately preserve exact stream PTS.
                frame_pts=round(window.time_s * 1000)))
        return ids
