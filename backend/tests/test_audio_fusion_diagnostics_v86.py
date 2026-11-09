import pytest

from scripts.audio_fusion_diagnostics_v86 import background_exposure, count_outcomes, joint_table


def rows():
    return [{"base_id": str(index), "group": "shared" if index < 2 else str(index),
             "kind": "stock" if label else "genuine", "label": label, "condition": "noise_0"}
            for index, label in enumerate((0, 0, 1, 1))]


def test_joint_cells_keep_both_wrong_directions_and_all_empty_cells():
    predictions = {"current": ["original", "ai", "inconclusive", "ai"],
                   "candidate": ["ai", "original", "original", "unassessed"]}
    result = joint_table(rows(), predictions)["conditions"]
    assert sum(len(cells) for cells in result.values()) == 36
    genuine = result["noise_0"]["genuine"]["candidate"]
    assert genuine["false_ai"] == 1 and genuine["genuine_correct"] == 1
    assert genuine["new_correct_vs_current"] == genuine["lost_correct_vs_current"] == 1
    ai = result["noise_0"]["stock"]["candidate"]
    assert ai["false_original"] == ai["unassessed"] == 1
    assert ai["missed_ai_rate_including_unassessed"] == 1.0
    assert result["clean"]["clone"]["candidate"]["cases"] == 0
    assert result["clean"]["clone"]["candidate"]["false_ai_rate"] is None


def test_inconclusive_and_unassessed_are_distinct_from_both_verdicts():
    result = count_outcomes(rows(), ["inconclusive", "unassessed", "inconclusive", "unassessed"], ["original"] * 4)
    assert result["inconclusive"] == result["unassessed"] == 2
    assert result["false_ai"] == result["false_original"] == result["genuine_correct"] == result["ai_found"] == 0
    assert result["connected_foreground_groups"] == 3


def test_missing_method_rows_or_implicit_none_decisions_cannot_be_dropped():
    with pytest.raises(ValueError, match="same complete"):
        joint_table(rows(), {"current": ["ai"] * 3})
    with pytest.raises(ValueError, match="explicit outcomes"):
        count_outcomes(rows(), [None] * 4, ["ai"] * 4)


def prepared_metadata():
    manifest, plan = {"foregrounds": [], "backgrounds": []}, []
    for role in ("fusion_fit", "confidence_cal", "selection", "confirmation"):
        for kind in ("noise", "music", "speech"):
            for ordinal in range(4):
                manifest["backgrounds"].append({"raw_sha256": f"{role}-{kind}-{ordinal}", "role": role, "kind": kind})
        counts = {0: 0, 1: 0}
        for family, count in (("genuine", 56), ("clone", 20), ("expressive", 24), ("stock", 16)):
            label = int(family != "genuine")
            for number in range(count):
                key = f"{role}-{family}-{number}"
                manifest["foregrounds"].append({"raw_sha256": key, "role": role, "kind": family, "label": label})
                plan.append({"source_sha256": key, "role": role,
                             "backgrounds": {kind: {"raw_sha256": f"{role}-{kind}-{counts[label] % 4}"}
                                             for kind in ("noise", "music", "speech")}})
                counts[label] += 1
    return manifest, plan


def test_background_histograms_distinguish_class_balance_from_family_balance():
    manifest, plan = prepared_metadata()
    clone, expressive = plan[56], plan[77]
    clone["backgrounds"]["noise"], expressive["backgrounds"]["noise"] = expressive["backgrounds"]["noise"], clone["backgrounds"]["noise"]
    hist = background_exposure(manifest, plan)["sorted_foreground_use_counts_per_four_background_sources"]["fusion_fit"]
    assert hist["genuine"]["noise"] == [14] * 4
    assert hist["all_ai"]["noise"] == [15] * 4
    assert hist["clone"]["noise"] == [4, 5, 5, 6]


def test_background_metadata_review_rejects_cross_role_source():
    manifest, plan = prepared_metadata()
    plan[0]["backgrounds"]["noise"]["raw_sha256"] = "confidence_cal-noise-0"
    with pytest.raises(ValueError, match="crosses assigned role"):
        background_exposure(manifest, plan)
