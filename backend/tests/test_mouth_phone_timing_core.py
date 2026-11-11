from dataclasses import replace
from fractions import Fraction as F

import pytest

from scripts.mouth_phone_timing_core import AudioClock, LipFrame, PhoneEvent, TimingRules, review

CLOCK = AudioClock(F(1, 10), 16_000, 144_000)
TIME_BASE = F(1, 90_000)
TICKS_PER_PICTURE = 3_600
CENTERS = (25, 55, 85, 115, 145, 175)
RULES = TimingRules(
    closed_gap=0.015,
    open_gap=0.06,
    minimum_eye_px=16,
    maximum_frame_gap=F(1, 20),
    maximum_transition_gap=F(2, 25),
    maximum_lag=F(2, 5),
    articulation_slack=F(1, 100),
    hold_duration_slack=F(1, 20),
    maximum_event_gap=F(8, 5),
    minimum_local_shift=F(3, 25),
    minimum_closed_frames=2,
    flank_events=2,
    local_events=2,
)


def pictures(centers=CENTERS, count=225):
    closed = {i + step for i in centers for step in (-1, 0, 1)}
    return [
        LipFrame(i, 9_000 + i * TICKS_PER_PICTURE, 0, (0.008,) * 3 if i in closed else (0.1,) * 3, 50.0, True)
        for i in range(count)
    ]


def phone(i, *, shift=0, label="M", release=False, uncertainty=160):
    center = 640 * (i + shift)
    return PhoneEvent(
        label,
        center - 960,
        center + 960,
        uncertainty,
        True,
        True,
        release_sample=center if release else None,
    )


def read(events, frames=None, rules=RULES):
    return review(CLOCK, TIME_BASE, events, pictures() if frames is None else frames, rules)


def test_local_delay_is_bracketed_and_only_claims_observed_intervals():
    phones = [phone(i, shift=8 if slot in {2, 3} else 0) for slot, i in enumerate(CENTERS)]
    result = read(phones)
    assert result.state == "local_timing_change"
    assert len(result.timing_flags) == 1
    flag = result.timing_flags[0]
    assert (flag.before, flag.during, flag.after) == ((0, 1), (2, 3), (4, 5))
    assert flag.flank_offset_s == (F(-1, 25), F(1, 25))
    assert flag.local_offset_s == (F(7, 25), F(9, 25))
    assert flag.change_s == (F(6, 25), F(2, 5))
    assert flag.observed_audio_intervals_s == tuple(
        (CLOCK.time(event.start_sample), CLOCK.time(event.end_sample)) for event in phones[2:4]
    )
    assert flag.observed_audio_intervals_s[0][1] < flag.observed_audio_intervals_s[1][0]
    evidence = result.events[2]
    assert evidence.audio_anchor_sample == F(59_520)
    assert evidence.closure.before.index == 83
    assert evidence.closure.first_closed.index == 84
    assert evidence.closure.last_closed.index == 86
    assert evidence.closure.after.index == 87
    assert evidence.closure.after.time_s == F(179, 50)


def test_early_sound_has_negative_change():
    result = read([phone(i, shift=-8 if slot in {2, 3} else 0) for slot, i in enumerate(CENTERS)])
    assert result.timing_flags[0].change_s == (F(-2, 5), F(-6, 25))


@pytest.mark.parametrize("global_shift", [0, 8])
def test_same_time_and_global_offset_do_not_create_a_local_change(global_shift):
    result = read([phone(i, shift=global_shift) for i in CENTERS])
    assert result.state == "no_change_in_measured_comparisons"
    assert result.timing_flags == ()
    assert result.comparisons[0].change_s == (F(-2, 25), F(2, 25))
    assert result.events[0].offset_s[0] == F(global_shift - 1, 25)


def test_missing_clean_flanks_and_inconsistent_flanks_are_unassessed():
    events = [phone(i, shift=8 if slot in {2, 3} else 0) for slot, i in enumerate(CENTERS)]
    no_flank = read(events[:-1])
    assert no_flank.state == "unassessed"
    assert no_flank.comparisons == ()
    disagree = read([*events[:4], phone(CENTERS[4], shift=-4), phone(CENTERS[5], shift=-4)])
    assert disagree.state == "unassessed"
    assert disagree.comparisons[0].state == "flanks_disagree"


def test_periodic_visual_closures_abstain_instead_of_selecting_nearest():
    result = read([phone(25)], pictures((20, 25, 30)))
    assert result.state == "unassessed"
    assert result.events[0].state == "ambiguous_closures"
    assert result.events[0].possible_closures == 3
    assert result.events[0].offset_s is None


def test_one_closure_cannot_be_claimed_by_two_phones():
    result = read([phone(25), phone(30)], pictures((25,)))
    assert [event.state for event in result.events] == ["ambiguous_shared_closure"] * 2
    assert all(event.closure is None for event in result.events)


@pytest.mark.parametrize(
    "perturb",
    [
        lambda frames: [frame for frame in frames if frame.index != 85],
        lambda frames: [
            replace(frame, pose_supported=False) if frame.index == 85 else frame for frame in frames
        ],
        lambda frames: [
            replace(frame, gaps=None, eye_distance_px=None) if frame.index == 85 else frame
            for frame in frames
        ],
        lambda frames: [replace(frame, shot=1) if frame.index >= 85 else frame for frame in frames],
    ],
)
def test_missing_picture_geometry_pose_and_cuts_break_support(perturb):
    result = read([phone(i) for i in CENTERS], perturb(pictures()))
    assert result.state == "unassessed"
    assert result.events[2].state == "visual_gap_cut_or_pose"
    assert result.events[2].offset_s is None
    assert result.comparisons == ()


def test_known_timing_edges_and_unverified_audio_support_abstain():
    result = read(
        [
            phone(2, uncertainty=1600),
            phone(6),
            replace(phone(25), speaker_bound=False),
            replace(phone(55), timing_supported=False),
            phone(85, shift=10),
        ]
    )
    assert [event.state for event in result.events] == [
        "audio_edge",
        "video_edge",
        "unattributed_speech",
        "unsupported_audio_timing",
        "lag_search_edge",
    ]
    assert result.state == "unassessed"


def test_stop_phones_require_independent_release_and_do_not_mix_with_hold_midpoints():
    b = phone(25, label="B")
    p = phone(55, label="P", release=True)
    result = read([b, p])
    assert result.events[0].state == "unverified_release"
    assert result.events[1].state == "matched"
    assert result.events[1].visual_anchor_times == (CLOCK.time(640 * 56), CLOCK.time(640 * 57))
    mixed = read(
        [phone(i, label="P", release=True) if k in {2, 3} else phone(i) for k, i in enumerate(CENTERS)]
    )
    assert mixed.comparisons[0].state == "mixed_anchor_types"
    assert mixed.state == "unassessed"


def test_ordinary_geometry_perturbation_that_preserves_transitions_is_stable():
    events = [phone(i) for i in CENTERS]
    original = read(events)
    perturbed = [
        replace(frame, gaps=tuple(0.92 * gap + 0.002 for gap in frame.gaps), eye_distance_px=48.0)
        for frame in pictures()
    ]
    assert read(events, perturbed) == original


def test_broad_but_supported_time_uncertainty_is_not_a_measured_negative():
    events = [phone(i, uncertainty=800) for i in CENTERS]
    result = read(events)
    assert result.comparisons[0].state == "shift_uncertain"
    assert result.state == "unassessed"
    assert all(event.state == "matched" for event in result.events)


def test_invalid_clock_and_overlapping_audio_fail_closed():
    with pytest.raises(ValueError, match="rational"):
        review(CLOCK, 0.001, [phone(25)], pictures(), RULES)
    with pytest.raises(ValueError, match="nonoverlapping"):
        read([phone(25), phone(26)])
    with pytest.raises(ValueError, match="frame and PTS order"):
        read([phone(25)], [pictures()[2], pictures()[1]])
