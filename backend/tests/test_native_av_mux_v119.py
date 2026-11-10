from fractions import Fraction

import pytest

from scripts import native_av_mux_v119 as mux


def test_native_jitter_and_gap_are_retained_in_clock_validation():
    pts = [27, 60, 94, 127, 260]
    assert mux.validate_clock(pts, Fraction(0)) == [33, 34, 33, 133]
    assert pts == [27, 60, 94, 127, 260]


@pytest.mark.parametrize(
    "pts,origin",
    [
        ([27], Fraction(0)),
        ([27, 27], Fraction(0)),
        ([60, 27], Fraction(0)),
        ([-1, 27], Fraction(0)),
        ([True, 27], Fraction(0)),
        ([27, 60], Fraction(-1, 1000)),
        ([27, 60], Fraction(1, 16000)),
        ([27, 60], 0.0),
    ],
)
def test_unsupported_clock_cannot_be_silently_rebased(pts, origin):
    with pytest.raises(ValueError):
        mux.validate_clock(pts, origin)


def test_fixed_data_worker_routing_preserves_arguments(monkeypatch):
    args = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", "private", "0"]
    calls = []
    monkeypatch.setattr(mux.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    mux.routed_popen(args)
    assert args[3] == "scripts.video_appearance_native_v108"
    assert calls == [[*args[:3], "scripts.native_av_mux_v119", *args[4:]]]
