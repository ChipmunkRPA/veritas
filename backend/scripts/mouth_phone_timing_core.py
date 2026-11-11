"""Experimental articulation timing from observed lip frames and aligned PCM events.

The caller binds both inputs to the same file, speaker track and source clock.
The PCM timeline must be contiguous. Gaps and rejected pose observations break
visual support. Event and comparison times remain Fractions for private export.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

TARGET_PHONES = frozenset({"m", "b", "p"})
TimeRange = tuple[Fraction, Fraction]


@dataclass(frozen=True)
class AudioClock:
    """First decoded PCM sample on the same rational clock as the video."""

    origin_s: Fraction
    sample_rate: int
    samples: int

    def __post_init__(self):
        if type(self.origin_s) is not Fraction or any(
            type(value) is not int or value <= 0 for value in (self.sample_rate, self.samples)
        ):
            raise ValueError("A rational origin and a positive integral PCM clock are required")

    def time(self, sample: int | Fraction) -> Fraction:
        return self.origin_s + Fraction(sample) / self.sample_rate


@dataclass(frozen=True)
class PhoneEvent:
    """One aligned sound on the decoded output PCM, with an independent error bound.

    /m/ uses the midpoint of the aligned phone interval. /b/ and /p/ require
    an explicitly located acoustic release. An aligner's phone end alone does
    not establish that release.
    """

    phone: str
    start_sample: int
    end_sample: int
    uncertainty_samples: int
    timing_supported: bool
    speaker_bound: bool
    release_sample: int | None = None


@dataclass(frozen=True)
class LipFrame:
    """Three inner-lip gaps normalized by eye separation, on one native track."""

    index: int
    pts: int
    shot: int
    gaps: tuple[float, float, float] | None
    eye_distance_px: float | None
    pose_supported: bool


@dataclass(frozen=True)
class TimingRules:
    """Exploratory settings supplied explicitly by the caller."""

    closed_gap: float
    open_gap: float
    minimum_eye_px: float
    maximum_frame_gap: Fraction
    maximum_transition_gap: Fraction
    maximum_lag: Fraction
    articulation_slack: Fraction
    hold_duration_slack: Fraction
    maximum_event_gap: Fraction
    minimum_local_shift: Fraction
    minimum_closed_frames: int
    flank_events: int
    local_events: int

    def __post_init__(self):
        gaps = self.closed_gap, self.open_gap, self.minimum_eye_px
        if (
            not all(type(value) in {int, float} and math.isfinite(value) for value in gaps)
            or not 0 <= self.closed_gap < self.open_gap
            or self.minimum_eye_px <= 0
        ):
            raise ValueError("Finite ordered mouth-state thresholds and eye scale are required")
        positive_times = (
            self.maximum_frame_gap,
            self.maximum_transition_gap,
            self.maximum_lag,
            self.maximum_event_gap,
            self.minimum_local_shift,
        )
        slack_times = self.articulation_slack, self.hold_duration_slack
        if (
            any(type(value) is not Fraction or value <= 0 for value in positive_times)
            or any(type(value) is not Fraction or value < 0 for value in slack_times)
            or self.minimum_local_shift >= 2 * self.maximum_lag
        ):
            raise ValueError("Explicit rational timing supports and a reachable local shift are required")
        if any(
            type(value) is not int or value < 2
            for value in (self.minimum_closed_frames, self.flank_events, self.local_events)
        ):
            raise ValueError("Require multiple native pictures and events per group")


@dataclass(frozen=True)
class Picture:
    index: int
    pts: int
    time_s: Fraction
    mouth_state: str


@dataclass(frozen=True)
class Closure:
    before: Picture
    first_closed: Picture
    last_closed: Picture
    after: Picture

    @property
    def hold_midpoint(self) -> TimeRange:
        return (
            (self.before.time_s + self.last_closed.time_s) / 2,
            (self.first_closed.time_s + self.after.time_s) / 2,
        )

    @property
    def hold_duration(self) -> TimeRange:
        return (
            self.last_closed.time_s - self.first_closed.time_s,
            self.after.time_s - self.before.time_s,
        )

    @property
    def release(self) -> TimeRange:
        return self.last_closed.time_s, self.after.time_s


@dataclass(frozen=True)
class EventReading:
    event_index: int
    phone: str
    audio_samples: tuple[int, int]
    audio_times: TimeRange
    state: str
    audio_anchor_sample: Fraction | None = None
    audio_anchor_times: TimeRange | None = None
    visual_run: int | None = None
    closure: Closure | None = None
    visual_anchor_times: TimeRange | None = None
    offset_s: TimeRange | None = None
    possible_closures: int = 0


@dataclass(frozen=True)
class LocalComparison:
    before: tuple[int, ...]
    during: tuple[int, ...]
    after: tuple[int, ...]
    state: str
    flank_offset_s: TimeRange | None
    local_offset_s: TimeRange | None
    change_s: TimeRange | None
    observed_audio_intervals_s: tuple[TimeRange, ...]


@dataclass(frozen=True)
class TimingReview:
    state: str
    events: tuple[EventReading, ...]
    comparisons: tuple[LocalComparison, ...]

    @property
    def timing_flags(self) -> tuple[LocalComparison, ...]:
        return tuple(
            comparison for comparison in self.comparisons if comparison.state == "local_timing_change"
        )


@dataclass(frozen=True)
class VisualRun:
    bounds: TimeRange
    closures: tuple[Closure, ...]


def _mouth_state(frame: LipFrame, rules: TimingRules) -> str | None:
    if frame.gaps is None:
        if frame.eye_distance_px is not None:
            raise ValueError("An unavailable geometry record must omit both gaps and eye distance")
        return None
    if (
        len(frame.gaps) != 3
        or any(
            type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in frame.gaps
        )
        or type(frame.eye_distance_px) not in {int, float}
        or not math.isfinite(frame.eye_distance_px)
        or frame.eye_distance_px <= 0
    ):
        raise ValueError(
            "Mouth observations must contain finite nonnegative gaps and a positive eye distance"
        )
    if not frame.pose_supported or frame.eye_distance_px < rules.minimum_eye_px:
        return None
    if max(frame.gaps) <= rules.closed_gap:
        return "closed"
    if min(frame.gaps) >= rules.open_gap:
        return "open"
    return "uncertain"


def _closures(pictures: list[Picture], rules: TimingRules) -> tuple[Closure, ...]:
    def open_neighbor(position: int, step: int) -> Picture | None:
        origin = pictures[position].time_s
        i = position + step
        while 0 <= i < len(pictures) and abs(pictures[i].time_s - origin) <= rules.maximum_transition_gap:
            state = pictures[i].mouth_state
            if state != "uncertain":
                return pictures[i] if state == "open" else None
            i += step
        return None

    result = []
    i = 0
    while i < len(pictures):
        if pictures[i].mouth_state != "closed":
            i += 1
            continue
        start = i
        while i < len(pictures) and pictures[i].mouth_state == "closed":
            i += 1
        if i - start < rules.minimum_closed_frames:
            continue
        before, after = open_neighbor(start, -1), open_neighbor(i - 1, 1)
        if before is not None and after is not None:
            result.append(Closure(before, pictures[start], pictures[i - 1], after))
    return tuple(result)


def _visual_runs(
    frames: Sequence[LipFrame], time_base: Fraction, rules: TimingRules
) -> tuple[VisualRun, ...]:
    previous = None
    spans: list[list[Picture]] = []
    current: list[Picture] = []
    for frame in frames:
        if (
            type(frame.index) is not int
            or frame.index < 0
            or type(frame.pts) is not int
            or type(frame.shot) is not int
            or frame.shot < 0
            or type(frame.pose_supported) is not bool
        ):
            raise ValueError("Use native integral frame indices, PTS, shot IDs and explicit pose support")
        if previous and (
            frame.index <= previous.index or frame.pts <= previous.pts or frame.shot < previous.shot
        ):
            raise ValueError("Native pictures must preserve strict frame and PTS order")
        state = _mouth_state(frame, rules)
        if current and (
            state is None
            or frame.shot != previous.shot
            or frame.index != previous.index + 1
            or (frame.pts - previous.pts) * time_base > rules.maximum_frame_gap
        ):
            spans.append(current)
            current = []
        if state is not None:
            current.append(Picture(frame.index, frame.pts, frame.pts * time_base, state))
        previous = frame
    if current:
        spans.append(current)
    return tuple(VisualRun((span[0].time_s, span[-1].time_s), _closures(span, rules)) for span in spans)


def _validate_event(event: PhoneEvent, previous_end: int, clock: AudioClock) -> str:
    if (
        type(event.phone) is not str
        or not event.phone.strip()
        or type(event.start_sample) is not int
        or type(event.end_sample) is not int
        or not previous_end <= event.start_sample < event.end_sample <= clock.samples
        or type(event.uncertainty_samples) is not int
        or event.uncertainty_samples < 0
        or type(event.timing_supported) is not bool
        or type(event.speaker_bound) is not bool
        or (
            event.release_sample is not None
            and (
                type(event.release_sample) is not int
                or not event.start_sample <= event.release_sample <= event.end_sample
            )
        )
    ):
        raise ValueError("Phone events require ordered nonoverlapping PCM intervals and explicit support")
    return event.phone.strip().lower()


def _contained(span: TimeRange, bounds: TimeRange) -> bool:
    return bounds[0] <= span[0] <= span[1] <= bounds[1]


def _subtract(left: TimeRange, right: TimeRange) -> TimeRange:
    return left[0] - right[1], left[1] - right[0]


def _event_reading(
    clock: AudioClock,
    event: PhoneEvent,
    index: int,
    phone: str,
    video_extent: TimeRange | None,
    runs: tuple[VisualRun, ...],
    rules: TimingRules,
) -> EventReading:
    times = clock.time(event.start_sample), clock.time(event.end_sample)
    base = EventReading(index, phone, (event.start_sample, event.end_sample), times, "unsupported_phone")
    if phone not in TARGET_PHONES:
        return base
    if not event.speaker_bound:
        return replace(base, state="unattributed_speech")
    if not event.timing_supported:
        return replace(base, state="unsupported_audio_timing")
    if phone != "m" and event.release_sample is None:
        return replace(base, state="unverified_release")
    anchor = (
        Fraction(event.start_sample + event.end_sample, 2) if phone == "m" else Fraction(event.release_sample)
    )
    timing_uncertainty = Fraction(event.uncertainty_samples, clock.sample_rate)
    estimate = clock.time(anchor)
    audio_anchor = estimate - timing_uncertainty, estimate + timing_uncertainty
    base = replace(base, audio_anchor_sample=anchor, audio_anchor_times=audio_anchor)
    if not _contained(audio_anchor, (clock.origin_s, clock.time(clock.samples))):
        return replace(base, state="audio_edge")
    context = (
        audio_anchor[0] - rules.maximum_lag - rules.maximum_transition_gap,
        audio_anchor[1] + rules.maximum_lag + rules.maximum_transition_gap,
    )
    if video_extent is None or not _contained(context, video_extent):
        return replace(base, state="video_edge")
    support = next(((i, run) for i, run in enumerate(runs) if _contained(context, run.bounds)), None)
    if support is None:
        return replace(base, state="visual_gap_cut_or_pose")
    run_id, run = support
    # Allow physical coarticulation uncertainty only in the offset interval.
    possible = []
    limit = -rules.maximum_lag, rules.maximum_lag
    acoustic_duration = times[1] - times[0]
    for closure in run.closures:
        visual_anchor = closure.hold_midpoint if phone == "m" else closure.release
        offset = _subtract(audio_anchor, visual_anchor)
        if offset[0] > limit[1] or offset[1] < limit[0]:
            continue
        if phone == "m":
            duration = closure.hold_duration
            slack = rules.hold_duration_slack + 2 * timing_uncertainty
            if not duration[0] - slack <= acoustic_duration <= duration[1] + slack:
                continue
        if not _contained((closure.before.time_s, closure.after.time_s), context):
            return replace(base, state="closure_crosses_search_edge", visual_run=run_id)
        possible.append((closure, visual_anchor, offset))
    base = replace(base, visual_run=run_id, possible_closures=len(possible))
    if not possible:
        return replace(base, state="no_supported_closure")
    if len(possible) > 1:
        return replace(base, state="ambiguous_closures")
    closure, visual_anchor, offset = possible[0]
    if not _contained(offset, limit):
        return replace(base, state="lag_search_edge")
    slack = rules.articulation_slack
    return replace(
        base,
        state="matched",
        closure=closure,
        visual_anchor_times=visual_anchor,
        offset_s=(offset[0] - slack, offset[1] + slack),
    )


def _unique_in_order(events: list[EventReading]) -> tuple[EventReading, ...]:
    matched = [reading for reading in events if reading.state == "matched"]
    uses = Counter((reading.visual_run, reading.closure.first_closed.index) for reading in matched)
    crossed = set()
    maximum, minimum = {}, {}
    for reading in matched:
        run, frame = reading.visual_run, reading.closure.first_closed.index
        if frame < maximum.get(run, frame):
            crossed.add(reading.event_index)
        maximum[run] = max(frame, maximum.get(run, frame))
    for reading in reversed(matched):
        run, frame = reading.visual_run, reading.closure.first_closed.index
        if frame > minimum.get(run, frame):
            crossed.add(reading.event_index)
        minimum[run] = min(frame, minimum.get(run, frame))
    for reading in matched:
        shared = uses[reading.visual_run, reading.closure.first_closed.index] > 1
        if shared or reading.event_index in crossed:
            events[reading.event_index] = replace(
                reading,
                state="ambiguous_shared_closure" if shared else "ambiguous_event_order",
                closure=None,
                visual_anchor_times=None,
                offset_s=None,
            )
    return tuple(events)


def _intersection(spans: Sequence[TimeRange]) -> TimeRange | None:
    left, right = max(span[0] for span in spans), min(span[1] for span in spans)
    return (left, right) if left <= right else None


def _compare(group: list[EventReading], rules: TimingRules) -> LocalComparison:
    n, m = rules.flank_events, rules.local_events
    left, middle, right = group[:n], group[n : n + m], group[n + m :]
    flank = _intersection([item.offset_s for item in (*left, *right)])
    local = _intersection([item.offset_s for item in middle])
    change = _subtract(local, flank) if flank is not None and local is not None else None
    mixed_anchors = len({item.phone == "m" for item in group}) > 1
    if mixed_anchors:
        state = "mixed_anchor_types"
        flank = local = change = None
    elif flank is None:
        state = "flanks_disagree"
    elif local is None:
        state = "local_events_disagree"
    elif change[0] > rules.minimum_local_shift or change[1] < -rules.minimum_local_shift:
        state = "local_timing_change"
    elif _contained(change, (-rules.minimum_local_shift, rules.minimum_local_shift)):
        state = "measured_no_local_change"
    else:
        state = "shift_uncertain"
    return LocalComparison(
        tuple(item.event_index for item in left),
        tuple(item.event_index for item in middle),
        tuple(item.event_index for item in right),
        state,
        flank,
        local,
        change,
        tuple(item.audio_times for item in middle),
    )


def _local_comparisons(events: tuple[EventReading, ...], rules: TimingRules) -> tuple[LocalComparison, ...]:
    span = 2 * rules.flank_events + rules.local_events
    result = []
    run = []
    for event in events:
        if event.state == "unsupported_phone":
            continue
        if event.state != "matched":
            run.clear()
            continue
        if run and (
            event.visual_run != run[-1].visual_run
            or event.audio_times[0] - run[-1].audio_times[1] > rules.maximum_event_gap
        ):
            run.clear()
        run.append(event)
        if len(run) == span:
            result.append(_compare(run, rules))
            run.pop(0)
    return tuple(result)


def review(
    audio_clock: AudioClock,
    video_time_base: Fraction,
    phones: Sequence[PhoneEvent],
    frames: Sequence[LipFrame],
    rules: TimingRules,
) -> TimingReview:
    """Report local timing comparisons and their exact observed support.

    Positive offsets mean sound later than visible motion. Comparison support
    is a list of observed phone intervals. Gaps between them are unmeasured.
    A whole-track constant delay is outside the local-change endpoint.
    """
    if type(video_time_base) is not Fraction or video_time_base <= 0:
        raise ValueError("A positive exact rational video time base is required")
    runs = _visual_runs(frames, video_time_base, rules)
    extent = (frames[0].pts * video_time_base, frames[-1].pts * video_time_base) if frames else None
    events = []
    previous_end = 0
    for index, phone in enumerate(phones):
        label = _validate_event(phone, previous_end, audio_clock)
        events.append(_event_reading(audio_clock, phone, index, label, extent, runs, rules))
        previous_end = phone.end_sample
    observed = _unique_in_order(events)
    comparisons = _local_comparisons(observed, rules)
    states = {item.state for item in comparisons}
    if "local_timing_change" in states:
        state = "local_timing_change"
    elif "measured_no_local_change" in states:
        state = "no_change_in_measured_comparisons"
    else:
        state = "unassessed"
    return TimingReview(state, observed, comparisons)
