from copy import deepcopy

import pytest

from scripts.video_joint_scalar_v106 import (
    VIEWS,
    complete_genuine_groups,
    features,
    fit,
    point,
    predict,
    raw_appearance,
    summarize,
)


def head():
    return {"status": "fitted", "center": [0.0], "scale": [1.0], "coefficients": [1.0], "intercept": 0.0}


def file(values=(2.0, 3.0, 4.0), role="fusion_fit", label=1, group="group", name="file"):
    return {
        "id": name,
        "group": group,
        "role": role,
        "label": label,
        "units": [
            {"specialist": view, "status": "ok", "raw": value}
            for view, value in zip(VIEWS, values, strict=True)
        ],
    }


def heads():
    return {view + "|single-face": head() for view in VIEWS}


def examples():
    rows = []
    for role in ("fusion_fit", "fusion_cal", "confirmation"):
        for i in range(4):
            for y in (0, 1):
                rows.append(
                    file(
                        values=(0.5 + 0.01 * i, 2 * y + 0.03 * i, 0.1 * y),
                        role=role,
                        label=y,
                        group=f"{role}-{i}",
                        name=f"{role}-{i}-{y}",
                    )
                )
    return rows


def test_per_view_features_retain_identity_instead_of_one_family_maximum():
    assert features(file(), heads(), VIEWS) == [2.0, 3.0, 4.0]
    assert features(file(), heads(), ("face-appearance",)) == [3.0]
    assert features(file(values=(-2.0, -3.0, -4.0)), heads(), VIEWS) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("problem", ["missing", "failed", "unfitted", "nonfinite"])
def test_missing_or_invalid_required_evidence_never_becomes_zero(problem):
    row = file()
    models = heads()
    if problem == "missing":
        row["units"] = row["units"][1:]
    elif problem == "failed":
        row["units"][0].update(status="unavailable", raw=None)
    elif problem == "unfitted":
        models[VIEWS[0] + "|single-face"]["status"] = "insufficient_support"
    else:
        row["units"][0]["raw"] = float("nan")
    if problem == "nonfinite":
        with pytest.raises(ValueError):
            features(row, models, VIEWS)
    else:
        assert features(row, models, VIEWS) is None


def test_appearance_only_does_not_require_an_optional_crop_response():
    row = file()
    row["units"][0].update(status="failed", raw=None)
    assert features(row, heads(), ("face-appearance",)) == [3.0]
    assert raw_appearance(row) == 1.0
    row["units"] = [u for u in row["units"] if u["specialist"] != "face-appearance"]
    assert raw_appearance(row) is None


def test_confirmation_changes_cannot_change_response_fit_or_calibration():
    rows = examples()
    before = deepcopy(rows)
    model = fit(rows, heads(), VIEWS, minimum=3)
    for row in rows:
        if row["role"] == "confirmation":
            row["label"] = 1 - row["label"]
            for unit in row["units"]:
                unit["raw"] = 7.0
    assert fit(rows, heads(), VIEWS, minimum=3) == model
    assert model["fitting_files"] == model["calibration_files"] == 8
    assert all(c >= 0 for c in model["response"]["coefficients"])
    assert 0 < predict(before[-1], heads(), model) < 1
    broken = deepcopy(before[-1])
    broken["units"][0]["status"] = "failed"
    assert predict(broken, heads(), model) is None


def test_low_source_support_does_not_produce_a_fitted_model():
    with pytest.raises(ValueError):
        fit(examples(), heads(), VIEWS, minimum=25)


def test_genuine_group_requires_all_its_files_and_excludes_other_roles():
    rows = [
        file(role="fusion_cal", label=0, group="a", name="a1"),
        file(role="fusion_cal", label=0, group="a", name="a2"),
        file(role="fusion_cal", label=0, group="b", name="b1"),
        file(role="confirmation", label=0, group="c", name="c1"),
    ]
    assert complete_genuine_groups(rows, {"a1": 0.1, "a2": None, "b1": 0.3, "c1": 0.9}) == {"b": 0.3}


def test_research_point_is_chosen_on_common_complete_genuine_groups():
    values = {"a": 0.1, "b": 0.2, "c": 0.3, "outside": 0.99}
    result = point(values, {"a", "b", "c"}, budget=1, minimum=3)
    assert result["threshold"] == 0.2 and result["groups_flagged"] == 1
    assert point({"a": 0.2, "b": 0.2, "c": 0.2}, {"a", "b", "c"}, budget=1, minimum=3)["groups_flagged"] == 0
    with pytest.raises(ValueError):
        point({"a": None}, {"a"}, minimum=1, budget=0)


def test_inconclusive_ai_counts_as_missed_and_is_not_a_genuine_verdict():
    rows = [file(label=1, name="ai"), file(label=0, name="real", group="real")]
    result = summarize(rows, {"ai": None, "real": 0.9}, 0.5)
    assert result["inconclusive"] == result["missed_ai_including_inconclusive"] == result["false_ai"] == 1
    assert result["ai"] == result["genuine"] == 1 and result["ai_flagged"] == 0
