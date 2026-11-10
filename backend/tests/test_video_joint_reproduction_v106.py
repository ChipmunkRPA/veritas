import pytest

from scripts.reproduce_video_joint_v106 import compare


def test_small_float_differences_allowed_but_counts_remain_exact():
    compare(
        {"count": 160, "loss": 0.1, "missing": None, "status": "complete"},
        {"count": 160, "loss": 0.1 + 1e-10, "missing": None, "status": "complete"},
    )


@pytest.mark.parametrize(
    "actual",
    [
        {"count": 159, "loss": 0.1},
        {"count": 160, "loss": 0.1001},
        {"count": True, "loss": 0.1},
        {"count": 160, "loss": float("nan")},
        {"count": 160},
        {"count": 160, "loss": 0.1, "extra": 0},
    ],
)
def test_changed_counts_missing_fields_status_types_or_nonfinite_values_refused(actual):
    with pytest.raises(ValueError):
        compare({"count": 160, "loss": 0.1}, actual)
