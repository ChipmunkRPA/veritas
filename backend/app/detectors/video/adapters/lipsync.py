"""Lip-sync lane per speaking face track as a v2 plugin (V-5; VF §4.1–4.3, §9.5; ARCHITECTURE §3.1 `detect`).

The detect stage passes `AVInputs` in `outputs.av`: the frame table, the tracks stage summary, each track's `face_track`
frames, the sound track (16 kHz mono) and the run's `FrameStore`. The model-free method is in `video/lipsync.py`.

- **Inputs.** Voice activity on the sound track; per track (at most 4, largest faces first) the 1 s windows of the 25 fps
  grid in which the face is present and there is speech. One `FrameStore` pass reads every window tick; per tick the lane
  keeps a grey mouth patch (speaker attribution) and the embedder's crop.
- **Scores.** Per attributed track, a `track:<k>` unit: the fraction of its windows inside an offset-step or
  confidence-collapse segment (0 when nothing is flagged, which is informational and not evidence for the face), with
  the baseline offset, median confidence, segment counts and times. Per window, `track:<k>@window=<j>` with the confidence
  and the offset. The aggregate is the highest track fraction. Every value is uncalibrated (`raw_kind` score).
- **Exhibits.** Per track, an offset and confidence series and, when something is flagged, a lip-sync segments exhibit;
  one speech-level series for the video. They cite the track signals `<detector_id>:track:<k>` the video fuser builds
  (`track_text`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.detectors.image.adapters.common import revisions
from app.detectors.video import lipsync, sampling
from app.detectors.video.adapters.common import FULL_DEEP, H2, video_spec
from app.detectors.video.adapters.swap import track_name
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

LANE = "lip_sync"
SEGMENT_PEAK = 0.6  # VF §4.2 heuristic for a step or collapse; not calibrated
AUDIO_REASONS = {"absent": "no audio track", "silent": "the sound track is silent", "unreadable": "the sound track could not be read",
                 "short": "the sound track is shorter than 1 s"}
NO_SPEECH = "no speech detected in the sound track"
NO_VISIBLE = "no face track with timestamp-matched video frames for a full 1 s window of speech"
NO_SPEAKER = ("no face track whose mouth movement follows the speech (voice-over, off-screen speaker, occluded mouth, or no "
              "crop the embedder could align)")
REFERENCE = ("SyncNet protocol per 1 s window of a speaking face track: the audio shift within ±15 frames at 25 fps with the "
             "smallest visual-audio embedding distance, and confidence = median − minimum distance. A constant offset is "
             "consistent with a mux delay; a change of offset or a collapse of confidence inside a track is consistent "
             "with an inserted or re-dubbed segment. Lip-sync generators optimise this agreement, so a high confidence is "
             "not evidence that the face is unmanipulated")


@dataclass(frozen=True)
class AVInputs:
    table: FrameTable
    tracks: tuple[dict[str, Any], ...]  # tracks stage summary rows: track_id, identity, exhibit_id, …
    frames: dict[str, list[dict[str, Any]]]  # track_id → face_track exhibit frames
    audio: Any  # AudioTrack
    store: Any  # FrameStore (iter_many)
    audio_mapping: dict[str, Any] | None = None


def source_audio_origin(inputs: AVInputs) -> float | str:
    """PCM tick zero on the source clock, or why it cannot be safely joined to video frames.

    Concatenating decoder blocks across a timestamp jump must never turn it into a measured lip-sync step.
    Such files remain eligible for independent voice checks; the AV lane currently requires continuous timestamps.
    """
    import hashlib

    from app.detectors.audio.decode import source_ranges

    mapping = inputs.audio_mapping
    pcm = getattr(inputs.audio, "pcm", None)
    if not mapping or pcm is None:
        return "audio-visual timing unavailable: no verified source audio timeline"
    if mapping.get("samples") != len(pcm) or mapping.get("pcm_sha256") != hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest():
        return "audio-visual timing unavailable: decoded audio differs from its source timeline"
    try:
        ranges = source_ranges(mapping, 0, len(pcm))
    except (ValueError, KeyError, TypeError):
        return "audio-visual timing unavailable: invalid source audio timeline"
    if len(ranges) != 1:
        return "audio-visual timing unavailable: discontinuous audio timestamps are not yet supported"
    return ranges[0]["source_start_pts"] / lipsync.SR


def _ms(ticks: float) -> int:
    return round(ticks * 1000.0 / lipsync.FPS)


def track_text(track: dict[str, Any], unit: dict[str, Any]) -> tuple[str, str]:
    """(observed, reasoning) of a lip-sync track signal from its `track:<k>` unit."""
    q, name = unit["quality"], track_name(track)
    windows, steps, collapses = int(q.get("windows", 0)), int(q.get("steps", 0)), int(q.get("collapses", 0))
    baseline = q.get("baseline_offset")
    observed = f"{windows} 1 s window(s) of speech; "
    observed += (f"baseline offset {int(baseline):+d} frame(s) ({_ms(baseline):+d} ms)" if baseline is not None
                 else "no stable offset")
    observed += f", median confidence {q.get('confidence_median', 0.0):.2f}; {steps} offset step(s), {collapses} confidence collapse(s)"
    parts = []
    if steps:
        parts.append(f"The audio-visual offset of {name} changes from {int(baseline):+d} to {int(q['step_offset']):+d} frames at "
                     f"{q['step_start_s']:.1f} s. This is consistent with an inserted, re-dubbed or re-timed segment, but "
                     "equally with frame drops or timestamp jitter in variable frame-rate recordings.")
    if collapses:
        parts.append(f"The sync confidence of {name} collapses in {int(q['collapse_windows'])} consecutive windows "
                     f"({q['collapse_start_s']:.1f}–{q['collapse_end_s']:.1f} s). This is consistent with a dubbed or "
                     "re-synthesised segment, but equally with an occluded or turned mouth, or speech from someone off screen.")
    if not parts:
        if baseline is None:
            parts.append(f"No stable offset could be established for {name}: the sync confidence is low in every window.")
        elif abs(baseline) >= lipsync.BENIGN_OFFSET_TICKS:
            side = "later" if baseline > 0 else "earlier"
            parts.append(f"The sound is {abs(_ms(baseline))} ms {side} than the picture throughout for {name}: {lipsync.BENIGN}.")
        else:
            parts.append(f"Sound and picture are in sync for {name}.")
        parts.append("Lip-sync generators optimise this agreement, so it is informational and not evidence that the face is "
                     "unmanipulated.")
    if q.get("small_face", 0.0) >= 0.5:
        parts.append(f"The face is below {lipsync.SMALL_FACE_PX:.0f} px in most windows, where SyncNet is unreliable.")
    return observed, " ".join(parts)


def segment_rows(inputs: AVInputs, grid: lipsync.TickGrid, tid: str, sync: lipsync.TrackSync, sid: str, peak: float) -> list[dict[str, Any]]:
    """`TimeSegment` rows of a track's offset steps and collapses, with the pts of the frames at their first and last ticks."""
    table = inputs.table
    out = []
    for seg in (*sync.steps, *sync.collapses):
        first, last = lipsync.ticks_to_frame(grid, seg.start_tick), lipsync.ticks_to_frame(grid, seg.end_tick - 1)
        out.append({"start_pts": int(table.pts[first]), "end_pts": int(table.pts[max(first, last)]), "time_base": table.time_base,
                    "start_s": round(grid.seconds(seg.start_tick), 6), "end_s": round(grid.seconds(seg.end_tick), 6),
                    "channel": "av", "track_id": tid, "peak": peak, "lane": LANE, "hypothesis": H2.value, "sources": [sid]})
    return sorted(out, key=lambda s: s["start_s"])


def speech_exhibit(ctx: RunContext, grid: lipsync.TickGrid, speech: lipsync.Speech, sids: list[str]) -> str:
    """The speech-level series of the sound track (0.2 s points) with the voice-activity decision."""
    step = 5  # 0.2 s points
    return ctx.exhibits.series(
        [{"name": "speech_level_db", "points": [
            (round(grid.seconds(k + step / 2.0), 6), round(float(np.mean(speech.level_db[k:k + step])), 3))
            for k in range(0, speech.ticks, step)]},
         {"name": "speech", "points": [(round(grid.seconds(k + step / 2.0), 6), float(np.mean(speech.speech[k:k + step]) >= 0.5))
                                      for k in range(0, speech.ticks, step)]}],
        x_unit="s", title="Speech activity on the sound track", media_ref="audio", thresholds=[round(speech.threshold_db, 3)],
        caption=("Level in dBFS per 0.2 s and the voice-activity decision (level at least 12 dB above the track's 10th "
                 "percentile and spectral flatness below 0.4); the threshold line is this track's level threshold."),
        source_signal_ids=sids)


class StableSyncNetDetector(Detector):
    MODEL = "syncnet.latentsync_stable"
    BATCH = 32
    spec = video_spec("vid.lipsync.stable_syncnet",
                      "Lip-sync lane: audio-visual offset and confidence (LatentSync StableSyncNet) per speaking face track",
                      InputUnit.AV_WINDOW, (H2,), ("series", "segments"), family="av-sync", model_ids=(MODEL,),
                      needs=("tracks", "audio"), timeout_s=300.0, profiles=FULL_DEEP, version="0.5.1")

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL, self.BATCH, "audio/visual window",
                              "At most 32 windows; visual window 16 frames at 128x256, audio window 80x52 mel bins; branches sequential",
                              ("Retained CPU crops, mel spectrogram and embeddings are not covered by a measured process peak.",)),)

    def applicable(self, ctx: RunContext) -> Applicability:
        inputs = ctx.outputs.av
        if not isinstance(inputs, AVInputs) or not inputs.tracks:
            return Applicability(False, "no face track")
        audio = inputs.audio
        if audio is None or audio.status != "analysed" or audio.pcm is None:
            return Applicability(False, AUDIO_REASONS.get(getattr(audio, "status", "absent"), "no audio track"))
        # speech and the windows in which a face is seen come from the sound track and the tracks alone, so a video
        # without them takes no model (BUILD_SPEC N-14c, E-10)
        plan = self._windows(inputs)
        return Applicability(False, plan) if isinstance(plan, str) else Applicability(True)

    @staticmethod
    def _windows(inputs: AVInputs) -> Any:
        """(speech, tick grid, frames by track, windows by track) of the tracks seen for a full window of speech, or the
        reason there is none. No model and no decoding."""
        pcm = inputs.audio.pcm
        origin = source_audio_origin(inputs)
        if isinstance(origin, str):
            return origin
        speech = lipsync.vad(pcm)
        if not speech.speech.any():
            return NO_SPEECH
        grid = lipsync.tick_grid(inputs.table.times(), sampling.duration_s(inputs.table), len(pcm), audio_start_s=origin)
        by_track = {t["track_id"]: {f["frame_index"]: f for f in inputs.frames.get(t["track_id"]) or []} for t in inputs.tracks}
        candidates = {tid: w for tid, frames in by_track.items() if (w := lipsync.track_windows(tid, frames, grid, speech))}
        return (speech, grid, by_track, candidates) if candidates else NO_VISIBLE

    def run(self, ctx: RunContext) -> DetectorResult:
        inputs: AVInputs = ctx.outputs.av
        pcm = inputs.audio.pcm
        plan = self._windows(inputs)
        if isinstance(plan, str):
            return DetectorResult.not_applicable(plan)
        speech, grid, by_track, candidates = plan
        largest = sorted(candidates, key=lambda tid: (-float(np.median([w.face_px for w in candidates[tid]])), tid))
        candidates = {tid: candidates[tid] for tid in largest[:lipsync.MAX_TRACKS]}

        embedder = ctx.lease.get(self.MODEL).obj
        needed: dict[int, list[tuple[str, int]]] = {}
        for tid, windows in candidates.items():
            for w in windows:
                for k in w.ticks:
                    needed.setdefault(grid.frames[k], []).append((tid, k))
        patches: dict[str, dict[int, np.ndarray]] = {tid: {} for tid in candidates}
        crops: dict[str, dict[int, Any]] = {tid: {} for tid in candidates}
        for index, rgb in inputs.store.iter_many(sorted(needed)):
            for tid, k in needed[index]:
                frame = by_track[tid][index]
                patches[tid][k] = lipsync.mouth_patch(rgb, frame)
                crops[tid][k] = embedder.crop(rgb, frame)
        if ctx.deadline is not None:
            ctx.deadline.check("lip-sync frames")
        scores = {tid: lipsync.speaker_score(w, patches[tid], speech.level) for tid, w in candidates.items()}
        kept = {tid: [w for w in ws if all(crops[tid].get(k) is not None for k in w.ticks)]
                for tid, ws in lipsync.attribute(candidates, scores).items()}
        kept = {tid: ws for tid, ws in kept.items() if ws}
        if not kept:
            return DetectorResult.not_applicable(NO_SPEAKER)

        frames_w = int(embedder.frames)
        starts = sorted({s for ws in kept.values() for w in ws
                         for s in lipsync.audio_starts(lipsync.visual_starts(w, frames_w), frames_w, grid.audio_ticks)})
        audio_rows = lipsync.batched(starts, self.BATCH, lambda part: embedder.audio(pcm, part))
        audio = {s: audio_rows[i] for i, s in enumerate(starts)}
        results = {}
        for tid, ws in kept.items():
            synced = []
            for w in ws:
                if ctx.deadline is not None:
                    ctx.deadline.check("lip-sync embeddings")
                clips = [np.stack([crops[tid][k] for k in range(s, s + frames_w)]) for s in lipsync.visual_starts(w, frames_w)]
                visual = lipsync.batched(clips, self.BATCH, lambda part: embedder.visual(np.stack(part)))
                synced.append(lipsync.window_sync(w, frames_w, visual, audio))
            results[tid] = lipsync.track_sync(synced, embedder)
        return self._result(ctx, inputs, grid, speech, scores, results, len(candidates))

    def _result(self, ctx: RunContext, inputs: AVInputs, grid: lipsync.TickGrid, speech: lipsync.Speech, scores: dict[str, float],
                results: dict[str, lipsync.TrackSync], candidates: int) -> DetectorResult:
        tracks = {t["track_id"]: t for t in inputs.tracks}
        units: list[UnitScore] = []
        window_units: list[UnitScore] = []
        exhibit_ids: list[str] = []
        sids = [f"{self.spec.id}:{tid}" for tid in results]
        centre = lipsync.WINDOW_TICKS / 2.0
        for tid, sync in sorted(results.items(), key=lambda kv: kv[1].windows[0].start_tick):
            sid, name = f"{self.spec.id}:{tid}", track_name(tracks[tid])
            quality = {"windows": float(len(sync.windows)), "speaker_score": round(scores[tid], 6),
                       "confidence_median": sync.confidence_median, "steps": float(len(sync.steps)),
                       "collapses": float(len(sync.collapses)), "flagged": float(sync.flagged),
                       "constant_offset": float(sync.reading == "constant_offset"), "in_sync": float(sync.reading == "in_sync"),
                       "start_s": grid.seconds(sync.windows[0].start_tick), "end_s": grid.seconds(sync.windows[-1].end_tick),
                       "median_face_px": float(np.median([w.window.face_px for w in sync.windows])),
                       "speech_fraction": float(np.mean([w.window.speech_fraction for w in sync.windows]))}
            quality["small_face"] = float(quality["median_face_px"] < lipsync.SMALL_FACE_PX)
            if sync.baseline is not None:
                quality["baseline_offset"] = float(sync.baseline)
            if sync.collapse_threshold is not None:
                quality["collapse_threshold"] = sync.collapse_threshold
            if sync.steps:
                step = sync.steps[0]
                quality |= {"step_start_s": grid.seconds(step.start_tick), "step_end_s": grid.seconds(step.end_tick),
                            "step_offset": float(step.offset)}
            if sync.collapses:
                collapse = sync.collapses[0]
                quality |= {"collapse_start_s": grid.seconds(collapse.start_tick), "collapse_end_s": grid.seconds(collapse.end_tick),
                            "collapse_windows": float(collapse.windows)}
            units.append(UnitScore(tid, sync.flagged_fraction, "score", quality=quality))
            window_units += [UnitScore(f"{tid}@window={w.window.index}", w.confidence, "score", quality={
                "start_s": grid.seconds(w.start_tick), "end_s": grid.seconds(w.end_tick), "offset_frames": float(w.offset),
                "min_distance": w.min_distance, "speech_fraction": w.window.speech_fraction, "face_px": w.window.face_px,
                "collapsed": float(down)}) for w, down in zip(sync.windows, sync.collapsed, strict=True)]
            exhibit_ids.append(ctx.exhibits.series(
                [{"name": "offset_frames", "hypothesis": H2.value, "track_id": tid,
                  "points": [(round(grid.seconds(w.start_tick + centre), 6), float(w.offset)) for w in sync.windows]},
                 {"name": "confidence", "hypothesis": H2.value, "track_id": tid,
                  "points": [(round(grid.seconds(w.start_tick + centre), 6), round(w.confidence, 6)) for w in sync.windows]}],
                x_unit="s", title=f"Sync offset and confidence per 1 s window, {name}", media_ref="frame", thresholds=[0.0],
                caption=("Offset in frames at 25 fps (positive: the sound is later than the picture; reference band 0 ± 1 frame) "
                         "and SyncNet confidence (median − minimum distance) per window. Uncalibrated."),
                source_signal_ids=[sid]))
            segments = segment_rows(inputs, grid, tid, sync, sid, SEGMENT_PEAK)
            if segments:
                exhibit_ids.append(ctx.exhibits.segments(
                    segments, lane=LANE, title=f"Lip-sync segments, {name}", media_ref="frame",
                    caption=("Offset steps and confidence collapses inside the track. The segment peak is the heuristic value "
                             f"{SEGMENT_PEAK:.2f}, not a calibrated probability."), source_signal_ids=[sid]))
        exhibit_ids.append(speech_exhibit(ctx, grid, speech, sids))

        top_tid, top = max(results.items(), key=lambda kv: (kv[1].flagged_fraction, -kv[1].windows[0].start_tick))
        top_unit = next(u for u in units if u.unit_ref == top_tid)
        observed, reasoning = track_text(tracks[top_tid], {"quality": top_unit.quality})
        if len(results) > 1:
            flagged = sum(1 for s in results.values() if s.flagged)
            reasoning += f" {flagged} of {len(results)} speaking face tracks show an offset step or a confidence collapse."
        windows = sum(len(s.windows) for s in results.values())
        return DetectorResult(
            status="ok", unit_scores=[*units, *window_units],
            aggregate=UnitScore("global", top.flagged_fraction, "score", quality={"tracks": float(len(results)), "windows": float(windows)}),
            observed=f"{track_name(tracks[top_tid])}: {observed}", reference=REFERENCE, reasoning=reasoning,
            metrics={"candidate_tracks": float(candidates), "speaking_tracks": float(len(results)), "windows": float(windows),
                     "speech_s": round(float(speech.speech.sum()) / lipsync.FPS, 3), "flagged_tracks": float(sum(s.flagged for s in results.values())),
                     "offset_steps": float(sum(len(s.steps) for s in results.values())),
                     "confidence_collapses": float(sum(len(s.collapses) for s in results.values())),
                     "flagged_fraction_max": top.flagged_fraction},
            exhibit_ids=exhibit_ids, model_revisions=revisions(ctx, self.MODEL))
