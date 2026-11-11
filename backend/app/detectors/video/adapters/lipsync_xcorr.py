"""Licence-clean lip-sync lane as a v2 plugin (V-5a; D-37; VF §4.1–4.3, §9.5). The method is in `video/xcorr.py`.

The detect stage passes the V-5 `AVInputs` in `outputs.av`: the frame table, the tracks stage summary, each track's
`face_track` frames, the sound track (16 kHz mono) and the run's `FrameStore`. No weights: the mouth corners come from
the face tracker's YuNet landmarks.

- **Not run.** No face track, or no sound track (absent, unreadable): not applicable. A silent sound track or no detected
  speech: status ok with no score (informational; there is no speech to align). No face track with five landmarks for a
  full 1 s window of speech, or no track whose mouth movement follows the speech (occluded, turned or absent mouth,
  voice-over): not applicable.
- **Scores.** Per attributed track (at most 4, largest faces first), a `track:<k>` unit: the hand-set probability of an
  offset step or correlation collapse with outcome weight 0.4, or, when neither is found, 0 as an uncalibrated score
  (informational). Per window, `track:<k>@window=<j>` with the peak correlation and the offset. The aggregate is the
  highest track probability, or none when nothing is flagged.
- **Exhibits.** Per track, the offset and peak-correlation series, the mouth-opening proxy and speech level (z-scores per
  25 Hz tick) and, when something is flagged, a lip-sync segments exhibit; one speech-activity series. They cite the track
  signals `<detector_id>:track:<k>` the video fuser builds (`track_text`).
"""
from __future__ import annotations

from collections import Counter
from itertools import pairwise
from typing import Any

import numpy as np

from app.detectors.video import lipsync, sampling, xcorr
from app.detectors.video.adapters.common import CPU_ONLY, FULL_DEEP, H2, video_spec
from app.detectors.video.adapters.lipsync import (
    AUDIO_REASONS,
    AVInputs,
    segment_rows,
    source_audio_origin,
    speech_exhibit,
)
from app.detectors.video.adapters.swap import track_name
from app.detectors.video.av_evidence import capture_correlation
from app.engine.detector import Applicability, Detector, DetectorResult, InputUnit, RunContext, UnitScore

NO_SPEECH = "no speech detected in the sound track"
SILENT = "the sound track is silent"
NO_MOUTH = "no face track with a visible mouth and timestamp-matched video frames for a full 1 s window of speech"
FLAT_MOUTH = "available speech windows had no varying mouth-opening measurement for timing comparison"
UNCHANGED_MOUTH = "the available speech windows had unchanged mouth crops, so timing could not be measured"
NO_SPEAKER = ("no face track whose mouth movement follows the speech (occluded, turned or closed mouth, voice-over or an "
              "off-screen speaker)")
MAX_SERIES_POINTS = 5000
REFERENCE = ("Mouth-opening proxy of a speaking face track (the darkness of the mouth centre between the two mouth corners) "
             "against the speech level at 25 Hz: per 1 s window every 0.5 s, the audio shift within ±15 frames with the "
             "highest correlation. A constant offset is consistent with a mux delay; a change of offset or a collapse of the "
             "correlation inside a track is consistent with an inserted or re-dubbed segment. Lip-sync generators produce "
             "well-synchronised mouths, so a high correlation is not evidence that the face is unmanipulated. Hand-set score "
             "within [0.55, 0.70], weight 0.4, uncalibrated")
INFORMATIONAL = ("Lip-sync was not checked because there is no speech to align with a mouth. This is informational and says "
                 "nothing about the faces.")


def _ms(ticks: float) -> int:
    return round(ticks * 1000.0 / lipsync.FPS)


def _mouth_status(window: lipsync.Window, patches: dict[int, np.ndarray], apertures: dict[int, float]) -> str:
    if any(tick not in apertures for tick in window.ticks):
        return "missing_mouth_input"
    values = np.asarray([apertures[tick] for tick in window.ticks], dtype=np.float64)
    if not np.isfinite(values).all():
        return "missing_mouth_input"
    if values.std() >= lipsync.CORRELATION_VARIATION_MIN:
        return "measurable"
    first = patches[window.start_tick]
    return ("unchanged_mouth_pixels" if all(np.array_equal(patches[tick], first) for tick in window.ticks[1:])
            else "flat_mouth_proxy")


def _track_sync(windows: list[lipsync.Window], apertures: dict[int, float], level: np.ndarray) -> lipsync.TrackSync:
    # An offset change needs observations on the same uninterrupted run of measured windows.
    gaps = tuple(index for before, after in pairwise(windows) for index in range(before.index + 1, after.index))
    return lipsync.track_sync([xcorr.window_sync(window, apertures, level) for window in windows],
                              xcorr.SETTINGS, unmeasured_indices=gaps)


def _longest_measured_run(windows: tuple[lipsync.WindowSync, ...]) -> int:
    longest = consecutive = 1
    for before, after in pairwise(windows):
        consecutive = consecutive + 1 if after.window.index == before.window.index + 1 else 1
        longest = max(longest, consecutive)
    return longest


def track_text(track: dict[str, Any], unit: dict[str, Any]) -> tuple[str, str]:
    """(observed, reasoning) of a lip-sync track signal from its `track:<k>` unit."""
    q, name = unit["quality"], track_name(track)
    windows, steps, collapses = int(q.get("windows", 0)), int(q.get("steps", 0)), int(q.get("collapses", 0))
    baseline = q.get("baseline_offset")
    offset_minimum = 2 * lipsync.MIN_RUN_WINDOWS
    minimum = min(offset_minimum, lipsync.COLLAPSE_MIN_TRACK_WINDOWS)
    longest = int(q.get("longest_measured_run", windows))
    short_track = 0 < windows < minimum and not (steps or collapses)
    split_track = windows >= minimum and longest < offset_minimum and not steps
    if windows == 1 and q.get("start_s") is not None and q.get("end_s") is not None:
        observed = f"Measured one 1-second speech window ({q['start_s']:.3f}–{q['end_s']:.3f} s on the source clock)."
    else:
        observed = f"{windows} measured 1 s speech window(s)."
    if short_track:
        observed += f" Timing change unassessed. This check needs at least {minimum} measured windows to assess a change."
        if baseline is not None:
            observed += f" Measured baseline offset {int(baseline):+d} frame(s) ({_ms(baseline):+d} ms)."
    else:
        offset = (f"Baseline offset {int(baseline):+d} frame(s) ({_ms(baseline):+d} ms)" if baseline is not None
                  else "No stable offset")
        observed += f" {offset}, median peak correlation {q.get('peak_median', 0.0):.2f}."
        if split_track:
            observed += (f" Offset changes unassessed. The longest measured run had {longest} consecutive windows "
                         f"and this comparison needs at least {offset_minimum}.")
            observed += f" {collapses} correlation collapse(s)."
        else:
            observed += f" {steps} offset step(s), {collapses} correlation collapse(s)."
    parts = []
    if steps:
        step_baseline = q.get("step_baseline", baseline)
        parts.append(f"The audio-visual offset of {name} changes from {int(step_baseline):+d} to {int(q['step_offset']):+d} frames at "
                     f"{q['step_start_s']:.1f} s. This is consistent with an inserted, re-dubbed or re-timed segment, but "
                     "equally with frame drops or timestamp jitter in variable frame-rate recordings.")
    if collapses:
        parts.append(f"The correlation between the mouth opening of {name} and the speech level collapses in "
                     f"{int(q['collapse_windows'])} consecutive windows ({q['collapse_start_s']:.1f}–{q['collapse_end_s']:.1f} s). "
                     "This is consistent with a dubbed or re-synthesised segment, but equally with an occluded or turned mouth, "
                     "or speech from someone off screen.")
    if short_track or (split_track and not parts):
        parts.append("The available measurements describe local voice and mouth timing only. "
                     "A well-synchronized mouth can occur in recorded or edited video.")
    elif not parts:
        if baseline is None:
            parts.append(f"No stable offset could be established for {name}: the mouth opening and the speech level do not "
                         "correlate clearly in any window.")
        elif abs(baseline) >= lipsync.BENIGN_OFFSET_TICKS:
            side = "later" if baseline > 0 else "earlier"
            parts.append(f"The measured sound offset is {abs(_ms(baseline))} ms {side} than the picture for {name}: {lipsync.BENIGN}.")
        else:
            parts.append(f"The measured windows show a near-zero sound and picture offset for {name}.")
        parts.append("Lip-sync generators produce well-synchronised mouths, so this is informational and not evidence that the "
                     "face is unmanipulated.")
    if q.get("small_face", 0.0) >= 0.5:
        parts.append(f"The face is below {lipsync.SMALL_FACE_PX:.0f} px in most windows, where the mouth region is unreliable.")
    excluded = int(q.get("unmeasured_mouth_windows", 0))
    if excluded:
        frozen = int(q.get("unchanged_mouth_pixels", 0))
        observed += f" {excluded} additional speech window(s) lacked measurable mouth input."
        if frozen:
            parts.append(f"The measured mouth crops stayed unchanged in {frozen} of the omitted windows.")
    return observed, " ".join(parts)


def _thin(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) <= MAX_SERIES_POINTS:
        return points
    keep = np.unique(np.linspace(0, len(points) - 1, num=MAX_SERIES_POINTS).round().astype(int))
    return [points[int(k)] for k in keep]


class LandmarkXcorrDetector(Detector):
    spec = video_spec("vid.lipsync.landmark_xcorr",
                      "Lip-sync lane: mouth opening vs speech level cross-correlation per speaking face track",
                      InputUnit.AV_WINDOW, (H2,), ("series", "segments"), family="av-sync", needs=("tracks", "audio"), lane="cpu",
                      devices=CPU_ONLY, timeout_s=120.0, profiles=FULL_DEEP, version="0.5.6")

    def applicable(self, ctx: RunContext) -> Applicability:
        inputs = ctx.outputs.av
        if not isinstance(inputs, AVInputs) or not inputs.tracks:
            return Applicability(False, "no face track")
        audio = inputs.audio
        status = getattr(audio, "status", "absent")
        if status in ("absent", "unreadable", "short") or (status == "analysed" and audio.pcm is None):
            return Applicability(False, AUDIO_REASONS.get(status, "no audio track"))
        return Applicability(True)

    def _informational(self, observed: str, speech_s: float) -> DetectorResult:
        return DetectorResult(status="ok", observed=observed, reference=REFERENCE, reasoning=INFORMATIONAL,
                              metrics={"speech_s": speech_s, "speaking_tracks": 0.0, "informational": 1.0})

    def run(self, ctx: RunContext) -> DetectorResult:
        inputs: AVInputs = ctx.outputs.av
        if inputs.audio.status == "silent" or inputs.audio.pcm is None:
            return self._informational("The sound track is silent", 0.0)
        pcm = inputs.audio.pcm
        origin = source_audio_origin(inputs)
        if isinstance(origin, str):
            return DetectorResult.not_applicable(origin)
        speech = lipsync.vad(pcm)
        if not speech.speech.any():
            return self._informational("No speech detected in the sound track", 0.0)
        grid = lipsync.tick_grid(inputs.table.times(), sampling.duration_s(inputs.table), len(pcm), audio_start_s=origin)
        by_track = {t["track_id"]: {f["frame_index"]: f for f in inputs.frames.get(t["track_id"]) or []
                                    if f.get("landmarks") and len(f["landmarks"]) >= 5} for t in inputs.tracks}
        candidates = {tid: w for tid, frames in by_track.items()
                      if (w := [x for x in lipsync.track_windows(tid, frames, grid, speech, hop=xcorr.HOP_TICKS, max_windows=xcorr.MAX_WINDOWS)
                                if xcorr.full_search(x, speech.ticks)])}
        if not candidates:
            return DetectorResult.not_applicable(NO_MOUTH)
        largest = sorted(candidates, key=lambda tid: (-float(np.median([w.face_px for w in candidates[tid]])), tid))
        candidates = {tid: candidates[tid] for tid in largest[:lipsync.MAX_TRACKS]}

        needed: dict[int, list[tuple[str, int]]] = {}
        for tid, windows in candidates.items():
            for k in sorted({k for w in windows for k in w.ticks}):
                needed.setdefault(grid.frames[k], []).append((tid, k))
        patches: dict[str, dict[int, np.ndarray]] = {tid: {} for tid in candidates}
        native_verified = getattr(inputs.store, "native_pts_verified", None)
        for index, rgb in inputs.store.iter_many(sorted(needed)):
            if index in inputs.table.interpolated or (native_verified is not None and not native_verified(index)):
                continue
            for tid, k in needed[index]:
                patches[tid][k] = lipsync.mouth_patch(rgb, by_track[tid][index])
        if ctx.deadline is not None:
            ctx.deadline.check("lip-sync frames")
        apertures = {tid: {tick: xcorr.aperture(patch) for tick, patch in rows.items()} for tid, rows in patches.items()}
        coverage = {tid: {} for tid in candidates}
        measurable = {}
        omitted = []
        for tid, windows in candidates.items():
            scored = []
            for window in windows:
                status = _mouth_status(window, patches[tid], apertures[tid])
                if status == "measurable":
                    scored.append(window)
                else:
                    coverage[tid][window.index] = status
                    omitted.append({"track_id": tid, "start_s": grid.seconds(window.start_tick),
                                    "end_s": grid.seconds(window.start_tick + lipsync.WINDOW_TICKS), "status": status})
            if scored:
                measurable[tid] = scored
        counts = Counter(row["status"] for row in omitted)
        if not measurable:
            reason = NO_MOUTH
            if counts["unchanged_mouth_pixels"] + counts["flat_mouth_proxy"]:
                reason = UNCHANGED_MOUTH if counts["unchanged_mouth_pixels"] == len(omitted) else FLAT_MOUTH
            result = DetectorResult.not_applicable(reason)
        else:
            scores = {tid: lipsync.speaker_score(w, patches[tid], speech.level) for tid, w in measurable.items()}
            kept = lipsync.attribute(measurable, scores)
            if not kept:
                result = DetectorResult.not_applicable(NO_SPEAKER)
            else:
                results = {tid: _track_sync(windows, apertures[tid], speech.level) for tid, windows in kept.items()}
                result = self._result(ctx, inputs, grid, speech, scores, results, apertures, len(candidates), coverage)
                result.details["av_input"] = capture_correlation(inputs, grid, speech, results, patches, apertures, by_track, scores)
        result.metrics.update({"unmeasured_mouth_windows": float(len(omitted)),
                               **{key: float(count) for key, count in counts.items()}})
        if omitted:
            result.details["unmeasured_mouth_windows"] = omitted
        return result

    def _result(self, ctx: RunContext, inputs: AVInputs, grid: lipsync.TickGrid, speech: lipsync.Speech, scores: dict[str, float],
                results: dict[str, lipsync.TrackSync], apertures: dict[str, dict[int, float]], candidates: int,
                coverage: dict[str, dict[int, str]]) -> DetectorResult:
        tracks = {t["track_id"]: t for t in inputs.tracks}
        units: list[UnitScore] = []
        window_units: list[UnitScore] = []
        exhibit_ids: list[str] = []
        sids = [f"{self.spec.id}:{tid}" for tid in results]
        centre = lipsync.WINDOW_TICKS / 2.0
        probabilities: dict[str, float] = {}
        for tid, sync in sorted(results.items(), key=lambda kv: kv[1].windows[0].start_tick):
            sid, name = f"{self.spec.id}:{tid}", track_name(tracks[tid])
            p = xcorr.probability(sync)
            quality = {"windows": float(len(sync.windows)), "longest_measured_run": float(_longest_measured_run(sync.windows)),
                       "speaker_score": round(scores[tid], 6),
                       "peak_median": round(sync.confidence_median, 6), "steps": float(len(sync.steps)),
                       "collapses": float(len(sync.collapses)), "flagged": float(sync.flagged),
                       "flagged_fraction": round(sync.flagged_fraction, 6),
                       "constant_offset": float(sync.reading == "constant_offset"), "in_sync": float(sync.reading == "in_sync"),
                       "no_stable_offset": float(sync.reading == "no_stable_offset"),
                       "start_s": grid.seconds(sync.windows[0].start_tick), "end_s": grid.seconds(sync.windows[-1].end_tick),
                       "median_face_px": float(np.median([w.window.face_px for w in sync.windows])),
                       "speech_fraction": float(np.mean([w.window.speech_fraction for w in sync.windows])),
                       "unmeasured_mouth_windows": float(len(coverage[tid])),
                       "unchanged_mouth_pixels": float(sum(s == "unchanged_mouth_pixels" for s in coverage[tid].values()))}
            quality["small_face"] = float(quality["median_face_px"] < lipsync.SMALL_FACE_PX)
            if sync.baseline is not None:
                quality["baseline_offset"] = float(sync.baseline)
            if sync.collapse_threshold is not None:
                quality["collapse_threshold"] = round(sync.collapse_threshold, 6)
            if sync.steps:
                step = sync.steps[0]
                quality |= {"step_start_s": grid.seconds(step.start_tick), "step_end_s": grid.seconds(step.end_tick),
                            "step_offset": float(step.offset), "step_baseline": float(step.baseline)}
            if sync.collapses:
                collapse = sync.collapses[0]
                quality |= {"collapse_start_s": grid.seconds(collapse.start_tick), "collapse_end_s": grid.seconds(collapse.end_tick),
                            "collapse_windows": float(collapse.windows)}
            if p is None:
                units.append(UnitScore(tid, 0.0, "score", quality=quality | {"informational": 1.0}))
            else:
                probabilities[tid] = p
                units.append(UnitScore(tid, p, "probability", quality=quality | {"outcome_weight": xcorr.OUTCOME_WEIGHT}))
            window_units += [UnitScore(f"{tid}@window={w.window.index}", round(w.confidence, 6), "score", quality={
                "start_s": grid.seconds(w.start_tick), "end_s": grid.seconds(w.end_tick), "offset_frames": float(w.offset),
                "speech_fraction": w.window.speech_fraction, "face_px": w.window.face_px, "collapsed": float(down)})
                for w, down in zip(sync.windows, sync.collapsed, strict=True)]
            exhibit_ids.append(ctx.exhibits.series(
                [{"name": "offset_frames", "hypothesis": H2.value, "track_id": tid,
                  "points": [(round(grid.seconds(w.start_tick + centre), 6), float(w.offset)) for w in sync.windows]},
                 {"name": "peak_correlation", "hypothesis": H2.value, "track_id": tid,
                  "points": [(round(grid.seconds(w.start_tick + centre), 6), round(w.confidence, 6)) for w in sync.windows]}],
                x_unit="s", title=f"Sync offset and peak correlation per 1 s window, {name}", media_ref="frame", thresholds=[0.0],
                caption=("Offset in frames at 25 fps (positive: the sound is later than the picture; reference band 0 ± 1 frame) "
                         "and the peak correlation between the mouth-opening proxy and the speech level per window, every 0.5 s. "
                         "Uncalibrated."), source_signal_ids=[sid]))
            ticks = sorted({k for w in sync.windows for k in w.window.ticks})
            mouth, level = xcorr.zscore([apertures[tid][k] for k in ticks]), xcorr.zscore(speech.level[ticks])
            exhibit_ids.append(ctx.exhibits.series(
                [{"name": "mouth_opening_z", "track_id": tid,
                  "points": _thin([(round(grid.seconds(k), 6), round(float(v), 4)) for k, v in zip(ticks, mouth, strict=True)])},
                 {"name": "speech_level_z", "track_id": tid,
                  "points": _thin([(round(grid.seconds(k), 6), round(float(v), 4)) for k, v in zip(ticks, level, strict=True)])}],
                x_unit="s", title=f"Mouth opening and speech level, {name}", media_ref="frame",
                caption=("The mouth-opening proxy (darkness of the mouth centre) and the speech RMS level per 25 Hz tick in the "
                         "track's speech windows, each as a z-score over those ticks. In synchronised speech the two rise and fall "
                         "together."), source_signal_ids=[sid]))
            segments = segment_rows(inputs, grid, tid, sync, sid, p if p is not None else xcorr.P_COLLAPSE)
            if segments:
                exhibit_ids.append(ctx.exhibits.segments(
                    segments, lane="lip_sync", title=f"Lip-sync segments, {name}", media_ref="frame",
                    caption=("Offset steps and correlation collapses inside the track. The segment peak is the hand-set explanatory "
                             "score of the track, not a calibrated probability."), source_signal_ids=[sid]))
        exhibit_ids.append(speech_exhibit(ctx, grid, speech, sids))

        top_tid = max(results, key=lambda tid: (probabilities.get(tid, 0.0), results[tid].flagged_fraction, -results[tid].windows[0].start_tick))
        top_unit = next(u for u in units if u.unit_ref == top_tid)
        observed, reasoning = track_text(tracks[top_tid], {"quality": top_unit.quality})
        if len(results) > 1:
            flagged = sum(1 for s in results.values() if s.flagged)
            reasoning += f" {flagged} of {len(results)} speaking face tracks show an offset step or a correlation collapse."
        windows = sum(len(s.windows) for s in results.values())
        aggregate = (UnitScore("global", max(probabilities.values()), "probability",
                               quality={"outcome_weight": xcorr.OUTCOME_WEIGHT, "tracks": float(len(results)), "windows": float(windows)})
                     if probabilities else None)
        return DetectorResult(
            status="ok", unit_scores=[*units, *window_units], aggregate=aggregate,
            observed=f"{track_name(tracks[top_tid])}: {observed}", reference=REFERENCE, reasoning=reasoning,
            metrics={"candidate_tracks": float(candidates), "speaking_tracks": float(len(results)), "windows": float(windows),
                     "speech_s": round(float(speech.speech.sum()) / lipsync.FPS, 3),
                     "flagged_tracks": float(sum(s.flagged for s in results.values())),
                     "offset_steps": float(sum(len(s.steps) for s in results.values())),
                     "correlation_collapses": float(sum(len(s.collapses) for s in results.values())),
                     "peak_median": round(float(np.median([w.confidence for s in results.values() for w in s.windows])), 6)},
            exhibit_ids=exhibit_ids)
