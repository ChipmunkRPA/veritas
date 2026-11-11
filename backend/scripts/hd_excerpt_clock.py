"""Select the earliest exact native picture and PCM boundaries in a qualified shot.

The caller supplies ordered decoded metadata from one continuous source read,
with shot bounds and picture support established independently. Decoded audio
PTS and their declared precision constrain a conditional PCM sample lattice.
Quantized PTS residuals remain visible alongside the exact sample indices.
Extraction must still verify decoded sample order and the output clock. This
experiment-construction helper sets no policy for app uploads or visual checks.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from fractions import Fraction
from math import ceil
from typing import Literal

MIN_DURATION = Fraction(6)
MAX_DURATION = Fraction(36, 5)
Exact = Fraction | int | str
AudioQuantization = Literal["unspecified", "nearest_half_open", "nearest_closed", "exact"]
SampleBoundaryPolicy = Literal["inside", "coincident"]


def _exact(value: Exact) -> Fraction:
    if type(value) not in {Fraction, int, str}:
        raise ValueError("Use exact rational source times and time bases")
    return Fraction(value)


def _unavailable(reason: str, **clock_details) -> dict:
    return {"status": "unavailable", "reason": reason, "excerpt": None, **clock_details}


def _audio_clock(
    frames: Sequence[tuple[int, int]], base: Fraction, rate: int, quantization: AudioQuantization
) -> dict:
    closed = quantization in {"nearest_closed", "exact"}
    phase_width = Fraction() if quantization == "exact" else base

    def outside_precision(error: Fraction) -> bool:
        return abs(error) > phase_width if closed else abs(error) >= phase_width

    origin_pts = frames[0][0]
    origin = origin_pts * base
    sample_counts = [0]
    smallest = largest = previous_residual = Fraction()
    previous_pts = None
    for index, (pts, samples) in enumerate(frames):
        if type(pts) is not int or type(samples) is not int or samples <= 0:
            raise ValueError("Audio frames require integer PTS and positive native sample counts")
        if previous_pts is not None and pts < previous_pts:
            return _unavailable("audio_pts_out_of_order", audio_frame_index=index)
        start = pts * base
        residual = start - (origin + Fraction(sample_counts[-1], rate))
        adjacent = residual - previous_residual
        if outside_precision(adjacent):
            return _unavailable(
                "audio_gap_exceeds_quantization" if adjacent > 0 else "audio_overlap_exceeds_quantization",
                audio_frame_index=index,
                adjacent_error_seconds=str(adjacent),
                audio_time_base=str(base),
            )
        smallest, largest = min(smallest, residual), max(largest, residual)
        if outside_precision(largest - smallest):
            return _unavailable(
                "audio_quantization_phase_drift",
                audio_frame_index=index,
                pts_residual_range_seconds=[str(smallest), str(largest)],
                audio_time_base=str(base),
            )
        sample_counts.append(sample_counts[-1] + samples)
        previous_pts, previous_residual = pts, residual

    # An unspecified convention allows a conservative one-tick absolute error.
    # Declared nearest rounding gives half-tick bounds at each decoded frame.
    radius = base if quantization == "unspecified" else phase_width / 2
    lower, upper = origin + largest - radius, origin + smallest + radius
    lower_inclusive = quantization != "unspecified"
    origin_range = {
        "seconds": [str(lower), str(upper)],
        "lower_inclusive": lower_inclusive,
        "upper_inclusive": closed,
    }
    context = {
        "decoded_origin_pts": origin_pts,
        "decoded_origin_seconds": str(origin),
        "timestamp_quantization": quantization,
        "compatible_origin_bounds": origin_range,
        "pts_residual_range_seconds": [str(smallest), str(largest)],
    }
    # Use the reported decoded origin when compatible. If quantization uniquely
    # determines a different origin, retain both without rounding the sample grid.
    if (lower < origin or lower_inclusive and lower == origin) and (
        origin < upper or closed and origin == upper
    ):
        lattice_origin = origin
        origin_basis = "reported_decoded_pts"
    elif closed and lower == upper:
        lattice_origin = lower
        origin_basis = "unique_quantization_intersection"
    else:
        return _unavailable("audio_origin_phase_unresolved", **context)

    reported_end = start + Fraction(samples, rate)
    lattice_end = lattice_origin + Fraction(sample_counts[-1], rate)
    return {
        "status": "ready",
        **context,
        "conditional_lattice_origin_seconds": str(lattice_origin),
        "conditional_lattice_origin_basis": origin_basis,
        "reported_frame_end_seconds": str(reported_end),
        "conditional_lattice_end_seconds": str(lattice_end),
        "common_end_seconds": str(min(reported_end, lattice_end)),
        "native_sample_count": sample_counts[-1],
        "sample_boundaries": sample_counts,
    }


def admit_excerpt(
    video_pts: Sequence[int],
    video_time_base: Exact,
    audio_frames: Sequence[tuple[int, int]],
    audio_time_base: Exact,
    sample_rate: int,
    shot_bounds: tuple[Exact, Exact],
    *,
    audio_quantization: AudioQuantization = "unspecified",
    sample_boundary_policy: SampleBoundaryPolicy = "inside",
) -> dict:
    """Return JSON-ready exact bounds for the earliest 6 through 7.2 second excerpt.

    All output intervals are half-open. Shot bounds are absolute source seconds.
    An observed end picture supplies the exclusion boundary. No terminal frame
    duration is inferred. Native audio frames are (decoded PTS, sample count).
    Origin uncertainty has its own open or closed bounds. Nearest modes use
    half-tick absolute rounding error. Only a declared closed nearest convention
    admits a phase spread of exactly one tick. Exact mode admits no PTS error.
    The inside policy retains the native video boundaries and all audio sample
    starts in that range. The coincident policy requires both video boundaries
    to fall exactly on the conditional audio lattice.
    """
    video_base, audio_base = _exact(video_time_base), _exact(audio_time_base)
    shot_start, shot_end = map(_exact, shot_bounds)
    if video_base <= 0 or audio_base <= 0 or type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("Positive native time bases and an integral sample rate are required")
    if shot_start >= shot_end:
        raise ValueError("Qualified shot bounds must be strictly ordered")
    if audio_quantization not in {"unspecified", "nearest_half_open", "nearest_closed", "exact"}:
        raise ValueError("An explicit supported audio timestamp quantization convention is required")
    if sample_boundary_policy not in {"inside", "coincident"}:
        raise ValueError("Choose inside or coincident audio sample boundaries")
    if any(type(pts) is not int for pts in video_pts) or any(
        a >= b for a, b in zip(video_pts, video_pts[1:])
    ):
        raise ValueError("Decoded video PTS must be strictly increasing integers")
    if len(video_pts) < 2:
        return _unavailable("insufficient_observed_video_boundaries")
    if not audio_frames:
        return _unavailable("no_decoded_audio_frames")
    if type(audio_frames[0][0]) is not int:
        raise ValueError("First decoded audio PTS must be an integer")
    clock = _audio_clock(audio_frames, audio_base, sample_rate, audio_quantization)
    if clock["status"] != "ready":
        return clock
    boundaries = clock.pop("sample_boundaries")
    origin = Fraction(clock["conditional_lattice_origin_seconds"])
    start_bound = max(shot_start, origin)
    end_bound = min(shot_end, Fraction(clock["common_end_seconds"]))
    points = []
    for index, pts in enumerate(video_pts):
        time = pts * video_base
        if start_bound <= time <= end_bound:
            sample = (time - origin) * sample_rate
            if sample_boundary_policy == "inside" or sample.denominator == 1:
                points.append((time, index, ceil(sample)))

    context = {
        "video_time_base": str(video_base),
        "audio_time_base": str(audio_base),
        "native_sample_rate": sample_rate,
        "sample_boundary_policy": sample_boundary_policy,
        "shot_seconds_half_open": [str(shot_start), str(shot_end)],
        "decoded_audio_clock": clock,
    }
    times = [point[0] for point in points]
    for position, (start, first_index, first_sample) in enumerate(points):
        stop = bisect_left(times, start + MIN_DURATION, lo=position + 1)
        if stop == len(points):
            break
        end, last_index, last_sample = points[stop]
        if end - start > MAX_DURATION:
            continue
        first_audio_frame = bisect_right(boundaries, first_sample) - 1
        last_audio_frame = bisect_left(boundaries, last_sample)
        audio_start, audio_end = (
            origin + Fraction(sample, sample_rate) for sample in (first_sample, last_sample)
        )
        return {
            "status": "ready",
            "reason": None,
            **context,
            "excerpt": {
                "source_seconds_half_open": [str(start), str(end)],
                "duration_seconds": str(end - start),
                "video_pts_half_open": [video_pts[first_index], video_pts[last_index]],
                "video_frame_indices_half_open": [first_index, last_index],
                "audio_sample_indices_half_open": [first_sample, last_sample],
                "audio_frame_indices_half_open": [first_audio_frame, last_audio_frame],
                "first_audio_frame_sample_offset": first_sample - boundaries[first_audio_frame],
                "last_audio_frame_sample_end_offset": last_sample - boundaries[last_audio_frame - 1],
                "conditional_audio_sample_times_half_open": [str(audio_start), str(audio_end)],
                "audio_boundary_offsets_from_video_seconds": [str(audio_start - start), str(audio_end - end)],
            },
        }
    reason = (
        "no_exact_common_boundaries_within_duration"
        if sample_boundary_policy == "coincident"
        else "no_common_boundaries_within_duration"
    )
    return _unavailable(reason, **context)
