import numpy as np

from app.detectors.audio.phone_events import collapse
from app.evidence import video_viseme_ranked as ranked
from app.evidence import video_viseme_timing as timing
from scripts.video_lip_dense_native import nearest_25
from scripts.video_lip_dense_posteriors import acoustic, predict


def test_phone_collapse_preserves_peak_time_across_consecutive_hops():
    posterior = np.array([[.12, .81, .07], [.05, .92, .03], [.93, .04, .03], [.1, .11, .79]])

    assert collapse(posterior, {1: "open", 2: "closure"}) == [["open", 520, .92], ["closure", 1160, .79]]


def test_dense_prediction_withholds_the_edges_and_quiet_hops():
    groups = [[0, 0, 0, 0, 30_000]] * 18
    assert all(value is None for value in predict(groups, [0.5] + [0.] * 18))
    groups = [[100_000, 200_000, 100_000, 0, 100_000]] * 18
    assert acoustic(groups)[6] == [.2, .4, .2, 0., .2, .5]
    assert predict(groups, [0.5] + [0.] * 18) == [None] * 5 + [500_000] * 8 + [None] * 5


def test_virtual_clock_keeps_the_original_picture_commitments():
    source = {"time_base": "1/1000", "frames": [
        {"pts": pts, "index": i, "rgb_sha256": str(i)} for i, pts in enumerate((0, 33, 67, 100, 133, 167))]}
    new = nearest_25(source)

    assert new["time_base"] == "1/16000"
    assert [f["pts"] for f in new["frames"]] == [0, 640, 1280, 1920, 2560, 3200]
    assert [f["recorded_pts"] for f in new["frames"]] == [0, 33, 67, 133, 167, 167]
    assert [f["rgb_sha256"] for f in new["frames"]] == ["0", "1", "2", "4", "5", "5"]


def test_ranked_aperture_compares_order_without_changing_picture_indices():
    row = {"frames": [{"sample": i * timing.STEP, "gap": (i % 6) / 20, "index": i} for i in range(75)]}
    prediction = [None if i % 11 == 0 else (i % 17) * 10_000 for i in range(150)]
    stretched = [None if value is None else 2 * value + 20_000 for value in prediction]
    original = ranked.scores(row, prediction)
    assert original and original == ranked.scores(row, stretched)
    assert all(0 <= frame < len(row["frames"]) for _, _, _, frames in original for frame in frames)


def test_wider_search_uses_its_own_lag_coordinates():
    lags = tuple(range(-6400, 6401, timing.STEP))
    positions = [3200, 8960, 9600, 10880, 13440, 15360, 16000, 21120, 24320]
    targets = [4480, 4480, 1280, 1280, 1280, 1280, 4480, 4480, 4480]
    events = [(pos, "aperture", [-abs(lag - target) * timing.SCORE_UNITS for lag in lags],
               list(range(len(lags)))) for pos, target in zip(positions, targets, strict=True)]
    chosen = timing.hypotheses(events, 32000, lags=lags)
    assert chosen["status"] == "measured"
    assert chosen["baseline_lag"] == 4480 and chosen["change"] == -3200
