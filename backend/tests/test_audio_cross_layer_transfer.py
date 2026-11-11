import numpy as np
import pytest

from scripts import audio_cross_layer_room_points as natural
from scripts import audio_cross_layer_transfer as trial


def test_projection_and_relation_share_a_fixed_equal_size_layer_basis():
    rng = np.random.default_rng(21)
    hidden = rng.normal(size=(3, 1024))
    projection = trial.projection()
    assert np.array_equal(projection, trial.projection())
    before = trial.project(hidden, projection)
    hidden[1] *= -20
    assert np.array_equal(before, trial.project(hidden, projection))
    a, b = before
    paired = trial._columns(before[None], "relation")[0]
    direct = trial._columns(before[None], "concatenated")[0]
    assert len(paired) == len(direct) == 128
    assert np.array_equal(paired, np.r_[abs(a-b), a*b])
    assert np.array_equal(direct, np.r_[a, b])


def test_calibration_uses_genuine_group_maxima_and_keeps_incumbent_coverage():
    calibration = [{"role": "cal", "label": 0, "group": f"g{i}"} for i in range(20)]
    calibration.extend(({"role": "cal", "label": 0, "group": "g19"},
                        {"role": "cal", "label": 1, "group": "synthetic"}))
    point = trial.point(calibration, [*range(20), 24, 300])
    assert point == {"value": 24, "genuine_groups": 20, "rank": 20}
    with pytest.raises(ValueError, match="separate calibration"):
        trial.point([{**calibration[0], "role": "fit"}, *calibration[1:]], [*range(20), 24, 300])
    transfer = [{"group": f"t{i}", "condition": "room", "label": label, "decision": direction}
                for i, (label, direction) in enumerate(((1, "original"), (0, "inconclusive"),
                                                        (1, "unassessed"), (1, "inconclusive")))]
    observed = trial.summarize(transfer, [35, 24, None, 25], point["value"])
    assert observed["rescued_synthetic_groups"] == 1
    assert observed["added_genuine_groups"] == 0
    counts = observed["conditions"]["room"]
    assert counts["synthetic_candidate_original"] == counts["synthetic_candidate_unassessed"] == 1


def test_natural_room_points_exclude_clean_and_incumbent_directional_rows():
    rows = [{"condition": "mostly_single_interferer", "label": 0, "decision": "inconclusive", "group": f"g{i}"}
            for i in range(8)]
    rows += [{**rows[7], "condition": "overlapping_interferers"}, {**rows[0], "condition": "clean"},
             {**rows[1], "decision": "original"}, {**rows[2], "label": 1}]
    result = natural.room_point(rows, [*range(8), 24, 50, 50, 50])
    assert result == {"value": 6, "genuine_groups": 8, "rank": 7}
