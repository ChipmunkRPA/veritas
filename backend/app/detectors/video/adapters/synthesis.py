"""Synthesis lane: D3 per shot clip as a v2 plugin (V-4; VF §5; ARCHITECTURE §3.1 `detect`).

The detect stage passes `ClipInputs` in `outputs.clips`: the frame table, the plan's clips (up to 16 target slots on an
8 fps grid within one shot, up to 3 per shot) and the run's `FrameStore`.

- **Frames.** Every clip frame from one `FrameStore` pass, pre-processed as upstream (`video/d3.preprocess`). A clip scores
  16 frames, or 8 when a complete grid has 8–15 slots; one with fewer is not scored. Missing completeness metadata or
  collapsed target slots in any planned clip make the whole check inapplicable before model acquisition.
  Each clip's frames are encoded together by the X-CLIP B/16 vision tower once the last of them is decoded (the tower's message tokens span 8 consecutive frames). A VFR frame shared
  by two planned clips reaches both. Every selected frame must decode; incomplete planned clips fail the whole check.
- **Scores.** One `clip:<k>` unit per scored clip (σ of the second-order feature differences, with the clip's shot, time
  span and frame count). The aggregate is the median σ over the scored clips; the method reads a low σ as generated. The
  statistic is uncalibrated (`raw_kind` score).
- **Exhibits.** σ per clip on the time axis, and the second-order series of the clips with the lowest σ (at most 6). Both
  cite the signal `<detector_id>:global`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.detectors.image.adapters.common import revisions
from app.detectors.video import d3
from app.detectors.video.adapters.common import FULL_DEEP, H1, video_spec
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
from app.engine.fusion import derived

MAX_PROFILES = 6
REFERENCE = ("D3, training-free: X-CLIP B/16 features sampled on an 8 fps target grid within one shot; σ is the standard "
             "deviation of the second-order differences of the L2 distances between consecutive frame features. The method "
             "reads a flat second-order profile (low σ) as generated. σ is an uncalibrated feature; a local camera-footage "
             "threshold, when available, is described separately")
NO_CLIP = f"no shot long enough for a D3 clip ({d3.MIN_FRAMES} frames on an {d3.RATE_FPS:g} fps target grid)"


@dataclass(frozen=True)
class ClipInputs:
    table: FrameTable
    clips: tuple[dict[str, Any], ...]  # plan clips: shot, start_s, rate_fps (target), target_frame_count, frames
    shots: int  # shots in the plan
    shots_source: str  # detected | whole-video
    store: Any  # FrameStore (iter_many)


def sampling_issue(inputs: Any) -> str | None:
    """Refuse the whole check if any planned clip lost target slots or has unknown completeness.

    This is a target-slot check, not a calibrated tolerance for irregular source timestamps. Check the original
    planned clip before any minimum-length decision; otherwise a collapsed 16-slot grid can masquerade as a short shot.
    Only a complete known grid with fewer than eight target slots may be ignored as genuinely short.
    """
    if not isinstance(inputs, ClipInputs):
        return None
    for k, clip in enumerate(inputs.clips):
        frames = list(clip.get("frames") or ())
        prefix = f"D3 clip {k + 1}: "
        count = clip.get("target_frame_count")
        if type(count) is not int or not 1 <= count <= d3.CLIP_FRAMES:
            return prefix + "sampling completeness is unknown (missing or invalid target frame count)"
        if clip.get("rate_fps") != d3.RATE_FPS:
            return prefix + "target rate must be 8 fps"
        if any(type(i) is not int or i < 0 or i >= len(inputs.table) for i in frames):
            return prefix + "invalid sampling frame indices"
        if len(frames) != count or len(set(frames)) != count:
            return prefix + "target slots collapsed or do not match the selected frame count"
        if frames != sorted(frames) or any(inputs.table.pts[b] <= inputs.table.pts[a] for a, b in zip(frames, frames[1:])):
            return prefix + "selected source timestamps must increase strictly"
    return None


def cadence_quality(table: FrameTable, clip: dict[str, Any], frames: list[int]) -> dict[str, float]:
    """Actual source spacing for the scored frames; target rate is not measured source cadence."""
    intervals = np.diff([table.seconds(i) for i in frames])
    return {"rate_fps": d3.RATE_FPS,  # retained compatibility alias for the requested grid rate
            "target_rate_fps": d3.RATE_FPS, "target_frame_count": float(clip["target_frame_count"]),
            "effective_rate_fps": float((len(frames) - 1) / intervals.sum()),
            "frame_interval_min_s": float(intervals.min()), "frame_interval_max_s": float(intervals.max())}


def usable_clips(inputs: Any) -> list[tuple[int, dict[str, Any], list[int]]]:
    """Frames D3 scores, only when every planned clip has complete target slots."""
    if not isinstance(inputs, ClipInputs) or sampling_issue(inputs):
        return []
    out = []
    for k, clip in enumerate(inputs.clips):
        frames = list(clip.get("frames") or ())
        if d3.clip_length(len(frames)):
            out.append((k, clip, frames[:d3.clip_length(len(frames))]))
    return out


class D3Detector(Detector):
    MODEL = "xclip.base_p16"
    spec = video_spec("vid.synth.d3_xclip", "Synthesis lane: D3 second-order temporal statistic (X-CLIP B/16) per shot clip",
                      InputUnit.CLIP, (H1,), ("series",), family="synthetic-video", model_ids=(MODEL,), needs=("frames",),
                      timeout_s=120.0, profiles=FULL_DEEP, version="0.4.2")

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL, d3.CLIP_FRAMES, "224 px frame",
                              "One sixteen-frame clip (eight for short clips); coupled X-CLIP groups remain intact"),)

    def applicable(self, ctx: RunContext) -> Applicability:
        inputs = ctx.outputs.clips
        if not isinstance(inputs, ClipInputs) or not inputs.clips:
            return Applicability(False, "no clip in the sampling plan")
        if issue := sampling_issue(inputs):
            return Applicability(False, issue)
        return Applicability(True) if usable_clips(inputs) else Applicability(False, NO_CLIP)

    def run(self, ctx: RunContext) -> DetectorResult:
        inputs: ClipInputs = ctx.outputs.clips
        if issue := sampling_issue(inputs):
            return DetectorResult.not_applicable(issue)
        clips = usable_clips(inputs)
        encoder = ctx.lease.get(self.MODEL).obj
        owners: dict[int, list[int]] = {}
        for n, (_, _, frames) in enumerate(clips):
            for i in frames:
                owners.setdefault(i, []).append(n)
        pending: dict[int, dict[int, np.ndarray]] = {}
        decoded: set[int] = set()
        scored = []
        for index, rgb in inputs.store.iter_many(sorted(owners)):
            decoded.add(index)
            processed = d3.preprocess(rgb)
            for n in owners[index]:
                buffer = pending.setdefault(n, {})
                buffer[index] = processed
                k, clip, frames = clips[n]
                if len(buffer) == len(frames):
                    if ctx.deadline is not None:
                        ctx.deadline.check("D3 frame features")
                    scored.append((k, clip, frames, d3.second_order(encoder.features(np.stack([buffer[i] for i in frames])))))
                    del pending[n]
        missing = set(owners) - decoded
        if missing:
            return DetectorResult(status="failed", reason=f"{len(missing)} of {len(owners)} selected D3 frames were not decoded",
                                  metrics={"frames_selected": float(len(owners)), "frames_decoded": float(len(decoded))})
        if len(scored) != len(clips):
            return DetectorResult.failed("a planned D3 clip could not be scored completely")
        if not scored:
            return DetectorResult.not_applicable(NO_CLIP)
        scored.sort(key=lambda row: row[0])

        seconds = inputs.table.seconds
        sid = f"{self.spec.id}:global"
        units = [UnitScore(f"clip:{k}", stat.sigma, "score", quality={
            "shot": float(clip["shot"]), "start_s": seconds(frames[0]), "end_s": seconds(frames[-1]), "frames": float(len(frames)),
            **cadence_quality(inputs.table, clip, frames), "second_order_mean": stat.mean})
            for k, clip, frames, stat in scored]
        sigmas = [stat.sigma for *_, stat in scored]
        median = float(np.median(sigmas))
        lowest_k, _, lowest_frames, lowest = min(scored, key=lambda s: s[3].sigma)
        shots_scored = len({clip["shot"] for _, clip, _, _ in scored})
        by_time = sorted(scored, key=lambda s: seconds(s[2][0]))
        exhibit_ids = [ctx.exhibits.series(
            [{"name": "sigma", "hypothesis": H1.value,
              "points": [(round(seconds(frames[0]), 6), round(stat.sigma, 6)) for _, _, frames, stat in by_time]}],
            x_unit="s", title="D3 statistic σ per clip", media_ref="frame",
            caption=("Standard deviation of the second-order feature differences of each clip, at the clip's first frame. "
                     "Uncalibrated; the method reads a low σ as generated."), source_signal_ids=[sid])]
        flattest = sorted(scored, key=lambda s: s[3].sigma)[:MAX_PROFILES]
        exhibit_ids.append(ctx.exhibits.series(
            [{"name": f"clip {k + 1}", "hypothesis": H1.value,
              "points": [(round(seconds(frames[j + 1]), 6), round(v, 6)) for j, v in enumerate(stat.second_order)]}
             for k, _, frames, stat in sorted(flattest, key=lambda s: seconds(s[2][0]))],
            x_unit="s", title="D3 second-order feature differences, flattest clips", media_ref="frame",
            caption=("Change of the frame-to-frame feature distance at each step (X-CLIP B/16, 8 fps target grid), for the clips with "
                     "the lowest σ."), source_signal_ids=[sid]))

        reasoning = ("D3 reads a flat second-order profile of the frame features (a low σ) as a sign of generated video, and a "
                     "fluctuating profile as typical of camera footage. ")
        constants = derived.d3_sigma_constants()  # H-8: threshold set on compressed camera footage (scripts/d3_recalibrate.py)
        if constants is None:
            reasoning += "No threshold for σ has been calibrated on compressed footage, so the value is reported without a reading."
        else:
            threshold = constants[0]
            reasoning += (f"On compressed camera footage (native files and libx264 re-encodes), 5 % of clips have a median σ at or "
                          f"below {threshold:.4g}; this median σ is {'at or below' if median <= threshold else 'above'} that "
                          "threshold. The statistic separates generated from camera footage weakly on its own; the model card "
                          "gives the measured rates.")
        if inputs.shots_source != "detected":
            reasoning += " Shot cuts were not detected for this run, so a clip may span a cut, which raises σ."
        frames_scored = sum(len(frames) for _, _, frames, _ in scored)
        return DetectorResult(
            status="ok", unit_scores=units,
            aggregate=UnitScore("global", median, "score", quality={
                "clips": float(len(scored)), "shots": float(shots_scored), "sigma_min": lowest.sigma, "sigma_max": max(sigmas)}),
            observed=(f"median σ {median:.4g} over {len(scored)} clip(s) of up to {d3.CLIP_FRAMES} frames on an {d3.RATE_FPS:g} fps target grid "
                      f"in {shots_scored} shot(s); lowest σ {lowest.sigma:.4g} (clip {lowest_k + 1}, "
                      f"{seconds(lowest_frames[0]):.2f}–{seconds(lowest_frames[-1]):.2f} s)"),
            reference=REFERENCE, reasoning=reasoning,
            metrics={"clips": float(len(inputs.clips)), "clips_scored": float(len(scored)), "shots": float(inputs.shots),
                     "shots_scored": float(shots_scored), "frames_scored": float(frames_scored), "sigma_median": median,
                     "sigma_min": lowest.sigma, "sigma_max": max(sigmas)},
            exhibit_ids=exhibit_ids, model_revisions=revisions(ctx, self.MODEL))
