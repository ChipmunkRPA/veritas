"""Exercise existing AV timing arithmetic with deterministic authored signals."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256

import numpy as np

from app.detectors.video import lipsync, xcorr
from scripts.mouth_phone_timing_core import AudioClock, LipFrame, PhoneEvent, TimingRules, review

FPS = 25
AUDIO_RATE = 16_000
HOP = AUDIO_RATE // FPS
LAG = 8
LOCAL_EDIT = (77, 120)
COLLAPSE_EDIT = (77, 152)


@dataclass(frozen=True)
class TraceCase:
    name: str
    ticks: int
    mode: str
    lag: int = 0
    edit: tuple[int, int] | None = None


CASES = (
    TraceCase("same_time", 190, "copy"),
    TraceCase("global_late", 190, "copy", LAG),
    TraceCase("local_late", 190, "copy", LAG, LOCAL_EDIT),
    TraceCase("local_early", 190, "copy", -LAG, LOCAL_EDIT),
    TraceCase("local_flat_correlation", 227, "flat", edit=COLLAPSE_EDIT),
    TraceCase("speech_unavailable", 190, "copy"),
)


def signal_value(tick: int) -> float:
    """Nonperiodic positive reference with deterministic pre-roll and look-ahead."""
    raw = sha256(f"av-timing-engineering-fixture/1:{tick}".encode()).digest()
    return 0.125 + 0.5 * int.from_bytes(raw[:4], "big") / 2**32


def traces(case: TraceCase) -> tuple[dict[int, float], np.ndarray]:
    mouth = {tick: signal_value(tick) for tick in range(case.ticks)}
    level = np.fromiter(mouth.values(), dtype=np.float64, count=case.ticks)
    start, end = case.edit if case.edit else (0, case.ticks)
    if case.mode == "flat":
        level[start:end] = 0.375
    else:
        level[start:end] = [signal_value(tick - case.lag) for tick in range(start, end)]
    return mouth, level


def window_context(start: int, end: int, case: TraceCase) -> dict:
    """Describe the 31-lag search footprint independently of the measured result."""
    a, b = start - lipsync.VSHIFT, end + lipsync.VSHIFT
    phase = "no_local_edit"
    exact_lag = case.lag if case.edit is None else None
    if case.edit is not None:
        first, last = case.edit
        if b <= first:
            phase = "before"
        elif last <= a:
            phase = "after"
        elif first <= a and b <= last:
            phase = "inside_edit"
        else:
            phase = "cross_edit"

        if case.mode == "copy" and first <= start + case.lag and end + case.lag <= last:
            exact_lag = case.lag
        elif end <= first or last <= start:
            exact_lag = 0
    return {
        "core_ticks": [start, end],
        "full_search_ticks": [a, b],
        "full_search_phase": phase,
        "exact_copy_lag_ticks": exact_lag,
        "wholly_flat_search": case.mode == "flat" and phase == "inside_edit",
    }


def _time(tick: int | float) -> Fraction:
    return Fraction(str(tick)) / FPS


def _segment(segment: lipsync.Segment, case: TraceCase) -> dict:
    start, end = _time(segment.start_tick), _time(segment.end_tick)
    row = {
        "kind": segment.kind,
        "start_tick": segment.start_tick,
        "end_tick": segment.end_tick,
        "start_seconds": str(start),
        "end_seconds": str(end),
        "windows": segment.windows,
        "offset_ticks": segment.offset,
        "baseline_ticks": segment.baseline,
        "minimum_peak": segment.confidence,
    }
    if case.edit is not None and case.mode == "copy" and segment.kind == "offset_step":
        first, last = (Fraction(tick, FPS) for tick in case.edit)
        intersection = max(Fraction(), min(end, last) - max(start, first))
        union = (end - start) + (last - first) - intersection
        row["comparison_to_authored_audio_copy"] = {
            "authored_audio_copy_seconds_half_open": [str(first), str(last)],
            "flagged_timing_seconds_half_open": [str(start), str(end)],
            "temporal_iou": str(intersection / union),
            "signed_onset_error_seconds": str(start - first),
            "signed_end_error_seconds": str(end - last),
        }
    return row


def measure(case: TraceCase) -> dict:
    mouth, level = traces(case)
    # Deliberately supplied availability. This isolates the unchanged
    # correlation and change rules from real face detection, VAD and attribution.
    frames = {tick: {"bbox": {"w": 128, "h": 128}} for tick in range(case.ticks)}
    has_speech = case.name != "speech_unavailable"
    speech = lipsync.Speech(
        20 * np.log10(level),
        level,
        np.zeros(case.ticks, dtype=np.float64),
        np.full(case.ticks, has_speech, dtype=bool),
        -60,
    )
    grid = lipsync.tick_grid(
        np.arange(case.ticks, dtype=np.float64) / FPS,
        case.ticks / FPS,
        case.ticks * HOP,
        audio_start_s=0.0,
    )
    if grid.frames != tuple(range(case.ticks)) or grid.audio_ticks != case.ticks:
        raise ValueError("Authored trace must have an exact one-picture, one-tick routing")
    candidates = lipsync.track_windows(
        "authored-face-support", frames, grid, speech, hop=xcorr.HOP_TICKS, max_windows=xcorr.MAX_WINDOWS
    )
    eligible = tuple(window for window in candidates if xcorr.full_search(window, speech.ticks))
    measured = tuple(xcorr.window_sync(window, mouth, level) for window in eligible)
    result = lipsync.track_sync(measured, xcorr.SETTINGS)

    window_rows = []
    for window, collapsed in zip(result.windows, result.collapsed, strict=True):
        core = window_context(window.start_tick, window.end_tick, case)
        lags = [1 - float(distance) for distance in window.distances[0]]
        window_rows.append(
            {
                **core,
                "selected_lag_ticks": window.offset,
                "peak_correlation": window.confidence,
                "all_lag_correlations": lags,
                "relative_collapse": bool(collapsed),
            }
        )
    change_map = None
    if case.edit is not None and case.mode == "copy":
        a, b = case.edit
        change_map = {
            "output_samples_half_open": [a * HOP, b * HOP],
            "input_samples_half_open": [(a - case.lag) * HOP, (b - case.lag) * HOP],
            "sample_rate": AUDIO_RATE,
            "authored_lag_ticks": case.lag,
        }
    return {
        "name": case.name,
        "native_picture_count": case.ticks,
        "authored_audio_ticks": grid.audio_ticks,
        "implied_pcm_samples": case.ticks * HOP,
        "observed_last_picture_seconds": str(_time(case.ticks - 1)),
        "edit_ticks_half_open": list(case.edit) if case.edit else None,
        "copy_mapping": change_map,
        "scheduled_starts_before_speech_and_full_search": [
            int(index * xcorr.HOP_TICKS)
            for index in range(case.ticks)
            if int(index * xcorr.HOP_TICKS) + lipsync.WINDOW_TICKS <= grid.ticks
        ],
        "speech_supported_starts": [window.start_tick for window in candidates],
        "full_search_starts": [window.start_tick for window in eligible],
        "reading": result.reading if measured else "unassessed",
        "baseline_ticks": result.baseline,
        "relative_collapse_threshold": result.collapse_threshold,
        "segments": [_segment(segment, case) for segment in (*result.steps, *result.collapses)],
        "windows": window_rows,
    }


def articulation() -> dict:
    """Independent six-event /m/ authored geometry check on the same 25 Hz clock."""
    clock = AudioClock(Fraction(), AUDIO_RATE, 190 * HOP)
    rules = TimingRules(
        closed_gap=0.015,
        open_gap=0.06,
        minimum_eye_px=16,
        maximum_frame_gap=Fraction(1, 20),
        maximum_transition_gap=Fraction(2, 25),
        maximum_lag=Fraction(2, 5),
        articulation_slack=Fraction(1, 100),
        hold_duration_slack=Fraction(1, 20),
        maximum_event_gap=Fraction(8, 5),
        minimum_local_shift=Fraction(3, 25),
        minimum_closed_frames=2,
        flank_events=2,
        local_events=2,
    )
    supported = (22, 52, 82, 110, 139, 168)
    wide_gap = (20, 40, 80, 100, 140, 160)
    outcomes = {}
    for name, centers, local_lag in (
        ("same_time", supported, 0),
        ("local_late", supported, LAG),
        ("wide_gap_local_late", wide_gap, LAG),
    ):
        closed = {center + delta for center in centers for delta in (-1, 0, 1)}
        frames = [
            LipFrame(tick, tick, 0, (0.008,) * 3 if tick in closed else (0.1,) * 3, 50.0, True)
            for tick in range(190)
        ]
        events = []
        for index, center in enumerate(centers):
            moved_center = (center + (local_lag if index in {2, 3} else 0)) * HOP
            events.append(PhoneEvent("m", moved_center - 960, moved_center + 960, 160, True, True))
        result = review(clock, Fraction(1, FPS), events, frames, rules)
        outcomes[name] = {
            "state": result.state,
            "authored_lip_center_ticks": list(centers),
            "authored_local_lag_ticks": local_lag,
            "matched_events": sum(event.state == "matched" for event in result.events),
            "comparisons": [
                {
                    "state": row.state,
                    "before": list(row.before),
                    "during": list(row.during),
                    "after": list(row.after),
                    "change_seconds": [str(value) for value in row.change_s] if row.change_s else None,
                    "observed_audio_intervals_seconds": [
                        [str(value) for value in span] for span in row.observed_audio_intervals_s
                    ],
                }
                for row in result.comparisons
            ],
        }
    return outcomes


def run_fixture() -> dict:
    return {
        "schema": "chipmunk.av-timing-engineering-fixture/1",
        "input_kind": "authored numeric traces and timing support",
        "picture_ticks_per_second": FPS,
        "implied_pcm_sample_rate": AUDIO_RATE,
        "lag_search_ticks": list(range(-lipsync.VSHIFT, lipsync.VSHIFT + 1)),
        "xcorr": [measure(case) for case in CASES],
        "articulation": articulation(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", action="store_true", help="Include every measured window and lag curve")
    args = parser.parse_args()
    result = run_fixture()
    if not args.windows:
        for case in result["xcorr"]:
            del case["windows"]
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
