from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.video import lipsync, xcorr
from app.detectors.video.adapters.lipsync_xcorr import LandmarkXcorrDetector
from app.engine.stages import common, video


class Exhibits:
    def __init__(self):
        self.rows = []

    def series(self, series, **metadata):
        return self._save("series", "series", series, metadata)

    def segments(self, segments, **metadata):
        return self._save("segments", "segments", segments, metadata)

    def _save(self, kind, field, values, metadata):
        exhibit_id = f"exhibit:{len(self.rows) + 1}"
        self.rows.append({"id": exhibit_id, "kind": kind, field: values, **metadata})
        return exhibit_id


def result_and_signal(offset_peaks):
    track_id = "track:0"
    track = {"track_id": track_id, "identity": "A", "exhibit_id": "face-track"}
    windows = []
    for index, (offset, peak) in enumerate(offset_peaks, start=2):
        window = lipsync.Window(track_id, index, 1.0, 120.0, xcorr.HOP_TICKS)
        distances = np.ones((1, 31))
        distances[0, offset + 15] = 1 - peak
        windows.append(lipsync.WindowSync(window, (window.start_tick,), distances, offset, 1 - peak, peak))
    sync = lipsync.track_sync(windows, xcorr.SETTINGS)
    ticks = tuple(range(120))
    grid = lipsync.TickGrid(0.0, ticks, len(ticks))
    levels = np.linspace(0.1, 1, len(ticks))
    speech = lipsync.Speech(
        20 * np.log10(levels), levels, np.zeros(len(ticks)), np.ones(len(ticks), dtype=bool), -45.0
    )
    inputs = SimpleNamespace(tracks=[track], table=SimpleNamespace(pts=ticks, time_base="1/25"))
    exhibits = Exhibits()
    detector = LandmarkXcorrDetector()
    result = detector._result(
        SimpleNamespace(exhibits=exhibits),
        inputs,
        grid,
        speech,
        {track_id: 0.8},
        {track_id: sync},
        {track_id: dict(zip(ticks, levels, strict=True))},
        1,
    )
    entry = common.detector_entry(detector.spec, result, None)
    ctx = SimpleNamespace(
        state={},
        outputs={"tracks": {"tracks": [track]}, "detect": {"detectors": [entry], "exhibits": exhibits.rows}},
    )
    [signal] = video.track_signals(ctx, 1)
    assert result.observed == "Face A: " + signal["observed"]
    assert result.reasoning == signal["reasoning"]
    assert signal["exhibit_ids"] == ["face-track", *result.exhibit_ids]
    return result, signal, exhibits.rows


@pytest.mark.parametrize("window_count, offset", [(1, 0), (2, 5), (3, 0)])
def test_short_support_preserves_measured_lag_and_shows_coverage(window_count, offset):
    result, signal, exhibits = result_and_signal([(offset, 0.8)] * window_count)

    assert "too few eligible speech windows to assess timing changes" in signal["observed"]
    assert "median peak correlation 0.80" in signal["observed"]
    assert "Too few eligible speech windows" in signal["reasoning"]
    assert "only the sampled windows" in signal["reasoning"]
    assert "0 offset step(s)" not in signal["observed"]
    assert "in sync" not in signal["reasoning"]
    assert "throughout" not in signal["reasoning"]
    assert result.status == "ok"
    assert result.aggregate is None
    assert result.metrics["windows"] == window_count
    assert signal["raw_score"] == signal["effective_weight"] == 0.0
    assert signal["legacy"]["direction"] == "inconclusive"
    assert signal["segments"] == []
    assert [unit.raw for unit in result.unit_scores[1:]] == [0.8] * window_count
    assert [unit.quality["offset_frames"] for unit in result.unit_scores[1:]] == [
        float(offset)
    ] * window_count
    assert [point[1] for point in exhibits[0]["series"][0]["points"]] == [float(offset)] * window_count


def test_unstable_lags_can_have_clear_correlations():
    _, signal, _ = result_and_signal([(offset, 0.8) for offset in (0, 4, -4, 8)])
    assert "baseline offset unresolved" in signal["reasoning"]
    assert "do not correlate clearly" not in signal["reasoning"]
    assert "too few eligible speech windows" not in signal["observed"]
    assert signal["metrics"]["no_stable_offset"] == 1.0


@pytest.mark.parametrize("offset, phrase", [(0, "within 40 ms"), (5, "200 ms later")])
def test_stable_offset_is_scoped_to_measured_speech(offset, phrase):
    _, signal, _ = result_and_signal([(offset, 0.8)] * 4)
    assert phrase in signal["reasoning"]
    assert "stable measured speech windows" in signal["reasoning"]
    assert "throughout" not in signal["reasoning"]
    assert "in sync" not in signal["reasoning"]


@pytest.mark.parametrize(
    "offset_peaks, probability, phrase",
    [
        ([(0, 0.8), (0, 0.8), (5, 0.8), (5, 0.8)], 0.65, "offset of Face A changes"),
        ([(0, 0.1), (0, 0.1), (0, 0.9), (0, 0.9)], 0.55, "speech level collapses"),
    ],
)
def test_flagged_measurements_keep_their_scores_and_linked_segments(offset_peaks, probability, phrase):
    result, signal, exhibits = result_and_signal(offset_peaks)
    assert result.aggregate.raw == signal["raw_score"] == probability
    assert result.aggregate.quality["outcome_weight"] == signal["metrics"]["outcome_weight"] == 0.4
    assert signal["raw_kind"] == "probability"
    assert phrase in signal["reasoning"]
    [segment] = signal["segments"]
    [record] = [row for row in exhibits if row["kind"] == "segments"]
    assert segment["peak"] == record["segments"][0]["peak"] == probability
    assert segment["sources"] == [signal["id"]]
