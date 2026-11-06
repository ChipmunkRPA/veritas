import math

import pytest

from scripts import audio_guard_v7 as method


def source(identifier, split, voice, text=None):
    return {
        "id": identifier,
        "split": split,
        "cluster_id": voice,
        "source_group": identifier,
        "transcript": text,
        "dataset": "example",
        "label": 0,
    }


@pytest.mark.parametrize("shared", ["voice", "text", "source"])
def test_connected_sources_cannot_cross_learning_and_calibration(shared):
    a, b = source("a", "fit", "v1", "A sentence."), source("b", "cal", "v2", "Different words")
    if shared == "voice":
        b["cluster_id"] = a["cluster_id"]
    elif shared == "source":
        b["source_group"] = a["source_group"]
    else:
        b["transcript"] = "a SENTENCE!"
    with pytest.raises(ValueError, match="crosses roles"):
        method.source_roles([a, b])


def test_paired_real_generated_voice_remains_one_evaluation_group():
    a, b = source("a", "test", "shared"), source("b", "test", "shared")
    b["label"] = 1
    assignments, groups = method.source_roles([a, b])
    assert len(groups) == 1 and groups[0]["class_counts"] == [1, 1]
    assert assignments["a"] == assignments["b"]
    assert assignments["a"]["role"] == "development"


def guards():
    return {
        str(label): [{"speaker": str(i), "maximum": 0.2, "rows": 6} for i in range(10)] for label in (0, 1)
    }


def test_missing_or_unstable_probe_only_withholds_original():
    g = guards()
    assert method.initial(0.01, g) == "original"
    assert method.final(0.01, None, g)["decision"] == "inconclusive"
    result = method.final(0.01, 0.1, g)
    assert result["original_stability_veto"]
    assert result["stability_log_odds_change"] > math.log(2)
    assert method.final(0.01, 0.012, g)["decision"] == "original"
    assert method.final(0.99, None, g)["decision"] == "ai"
    assert method.final(None, 0.01, g)["decision"] == "inconclusive"


def test_inadequate_calibration_groups_withhold_all_directions():
    g = guards()
    g["1"] = g["1"][:8]
    assert method.initial(0.99, g) == method.initial(0.01, g) == "inconclusive"


def test_wrong_roles_are_refused_before_learning_or_calibration():
    with pytest.raises(ValueError, match="Only fitting"):
        method.fit([{"role": "cal"}], None)
    with pytest.raises(ValueError, match="Only all eighty"):
        method.calibrate([{"role": "fit"}], {})


def test_calibration_includes_all_variants_and_keeps_missing_as_infinity():
    rows = []
    for label in (0, 1):
        for i in range(40):
            for c in method.CONDITIONS:
                rows.append(
                    {
                        "role": "cal",
                        "base_id": f"{label}:{i}",
                        "group": f"{label}:{i}",
                        "label": label,
                        "raw": (0.95 if label else 0.05),
                        "condition": c,
                    }
                )
    rows[0]["raw"] = None
    result = method.calibrate(rows, {"transform": None, "coefficients": [1.0, 0.0]})
    assert result["0"][0]["maximum"] is None
    assert result["0"][0]["rows"] == 6
    assert len(result["0"]) == len(result["1"]) == 40
