import copy

import numpy as np
import pytest
from PIL import Image

from scripts import evaluate_native_mouth_v127 as evaluation


def metadata(group=0, label=0, condition="original", status="extracted"):
    return {
        "asset_index": group,
        "source_group": str(group),
        "study_role": "selection",
        "condition": condition,
        "label": label,
        "input_status": status,
    }


def case(group=0, label=0, flag=False, assessed=True, status="extracted"):
    condition = "ai_aligned" if label else "original"
    return evaluation.outcome(
        metadata(group, label, condition, status),
        assessed,
        [["1", "2"]] if flag else [],
        0.8 if assessed else None,
        ["1", "2"] if label else None,
    )


def test_missing_scope_is_separate_from_missed_assessed_ai_and_genuine_flags():
    rows = [
        case(0, 1, False, True),
        case(1, 1, False, False),
        case(2, 0, True),
        case(3, 0, False, False),
        case(4, 1, False, False, "unavailable"),
    ]
    result = evaluation.counts(rows)
    assert result["ready_ai"] == 2 and result["assessed_ai"] == 1
    assert result["missed_assessed_ai"] == 1 and result["unassessed_ready_ai"] == 1
    assert result["unavailable_cases"] == 1 and result["inconclusive_ready_cases"] == 2
    assert result["genuine_false_flags"] == 1 and result["assessed_genuine"] == 1
    assert result["false_alarm_rate_assessed_genuine"] == 1
    assert result["false_alarm_rate_ready_genuine"] == 0.5


def test_unavailable_scope_cannot_be_assessed_or_get_probability():
    with pytest.raises(ValueError, match="Unavailable"):
        evaluation.outcome(metadata(status="unavailable"), True, [], 0.6, None)
    with pytest.raises(ValueError, match="Unsupported"):
        evaluation.outcome(metadata(), False, [["1", "2"]], None, None)


def test_one_class_condition_slice_keeps_unweighted_reliability_without_invented_binary_balance():
    rows = [case(0, 1, True), case(1, 1, False)]
    result = evaluation.counts(rows)
    assert result["clip_reliability_unweighted"]["brier"] == pytest.approx(0.04)
    assert result["clip_reliability_balanced"]["brier"] is None
    assert result["false_alarm_rate_ready_genuine"] is None


def test_all_unassessed_has_no_rate_denominator_or_original_verdict():
    rows = [case(0, 0, False, False), case(0, 1, False, False)]
    result = evaluation.counts(rows)
    assert result["inconclusive_rate_ready"] == 1 and result["assessed_cases"] == 0
    assert result["false_alarm_rate_assessed_genuine"] is None
    assert result["clip_reliability_unweighted"]["n"] == 0


def test_connected_group_bootstrap_keeps_variants_together_and_has_known_paired_delta():
    reference = [case(g, y, False) for g in range(4) for y in (0, 1)]
    candidate = [case(g, y, bool(y)) for g in range(4) for y in (0, 1)]
    result = evaluation.paired(candidate, reference)
    detection = result["rate_differences"]["ai_detection_rate_ready"]
    assert detection["candidate_minus_reference"] == 1 and detection["group_bootstrap_ci95"] == [1, 1]
    assert result["additional_localized_ai"] == 4 and result["additional_genuine_flags"] == 0
    wrong = copy.deepcopy(reference)
    wrong[0]["condition"] = "blur2"
    with pytest.raises(ValueError, match="same-input"):
        evaluation.paired(candidate, wrong)


def test_paired_summary_shows_lost_locations_and_extra_false_flags():
    reference = [case(g, y, bool(y)) for g in range(3) for y in (0, 1)]
    candidate = [case(g, y, not bool(y)) for g in range(3) for y in (0, 1)]
    result = evaluation.paired(candidate, reference)
    assert result["lost_localized_ai"] == 3 and result["additional_genuine_flags"] == 3
    assert result["rate_differences"]["ai_localization_rate_ready"]["candidate_minus_reference"] == -1
    assert result["rate_differences"]["false_alarm_rate_ready_genuine"]["candidate_minus_reference"] == 1


def test_inspected_roi_is_compared_with_actual_mask_without_claiming_predicted_pixel_mask(tmp_path):
    file = tmp_path / "mask.png"
    mask = np.zeros((10, 10), np.uint8)
    mask[3:5, 3:5] = 255
    Image.fromarray(mask).save(file)
    feature = {"frames": [{"region_bbox": [2, 2, 6, 6]}]}
    construction = {
        "edit_proofs": [
            {
                "frame_index": 0,
                "mask_file": str(file),
                "mask_sha256": evaluation.sha(file),
                "changed_pixels": 4,
            }
        ]
    }
    result = evaluation.mask_overlap(feature, construction, {"centre_frame_index": 0})
    assert result["pixel_iou"] == 0.25 and result["changed_pixel_coverage"] == 1
    assert "not predicted" in result["region_role"]
    construction["edit_proofs"] = []
    assert evaluation.mask_overlap(feature, construction, {"centre_frame_index": 0})["pixel_iou"] == 0
    assert evaluation.mask_overlap(feature, construction, None) is None


def test_training_or_unknown_stage_is_not_dispatched():
    with pytest.raises(ValueError):
        evaluation.run("fusion_fit")
    with pytest.raises(ValueError):
        evaluation.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract"])
