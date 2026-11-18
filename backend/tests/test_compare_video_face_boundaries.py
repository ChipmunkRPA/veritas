from pathlib import Path

import pytest

from scripts.compare_video_face_boundaries import panel, prepare, score

SOURCE = Path(__file__).parents[2] / "docs/evaluation-panels/video-face-single-boundary-2026-11-18.json.gz"


def test_existing_predictions_keep_known_ordinary_cuts_in_the_control_denominator():
    _, document, cases = panel(SOURCE, "held")
    predictions = prepare(SOURCE, "held")
    arm = document["thresholds"]["selected"]
    point = document["thresholds"]["options"][arm]["point"]
    for prediction in predictions["predictions"]:
        saved = cases[prediction["sha256"]]["candidates"][arm]
        if saved is None:
            prediction["status"] = "unassessed"
        elif saved[arm] > point:
            prediction.update(status="localized", frame=saved["boundary"])
        else:
            prediction["status"] = "no_match"

    comparison = score(SOURCE, predictions, "held")
    assert comparison["cases"] == 216
    assert comparison["total"]["located"] == 81
    assert comparison["total"]["exact"] == 80
    assert comparison["total"]["added_locations"] == comparison["total"]["lost_locations"] == 0
    assert comparison["by_control"]["different_shot"]["ordinary"] == 24
    assert comparison["total"]["ordinary"] == 120 and comparison["total"]["ordinary_alerts"] == 0

    # An ordinary cut has an authored boundary but is still a false alert for this check.
    cut = next(row for row in predictions["predictions"] if cases[row["sha256"]]["role"] == "different_shot")
    cut.update(status="localized", frame=cases[cut["sha256"]]["authored_edge"])
    alternative = score(SOURCE, predictions, "held")
    assert alternative["by_control"]["different_shot"]["new_ordinary_alerts"] == 1
    assert alternative["total"]["located"] == 81 and alternative["total"]["new_ordinary_alerts"] == 1

    predictions["predictions"].remove(cut)
    with pytest.raises(ValueError, match="explicit result for each exact input"):
        score(SOURCE, predictions, "held")
