from copy import deepcopy
from hashlib import sha256
from statistics import median

import pytest

from scripts.compare_whole_clip_face_scores import ARMS, evaluate, fit


def _case(key, *, group=None, role="development", family="original_videos"):
    return {"id": key, "group": group or key, "role": role, "family": family,
            "positive": family != "original_videos", "sha256": sha256(key.encode()).hexdigest()}


def _receipt(case, values):
    measured = [value for value in values if value is not None]
    return {"id": case["id"], "source": case["sha256"], "time_base": "1/30", "measured": len(measured),
            "median": median(measured) if len(measured) >= 4 else None,
            "samples": [{"index": i, "pts": i + 100, "rgb_sha256": str(i) * 64, "box": [1, 2, 3, 4], "score": value}
                        for i, value in enumerate(values)]}


@pytest.fixture
def paired():
    cases = [_case(f"development-{i}") for i in range(10)]
    cases += [_case("prior-positive", family="uniface"), _case("prior-confirmation", role="confirmation")]
    reference = [_receipt(row, [float(i)] * 6) for i, row in enumerate(cases)]
    candidate = [_receipt(row, [float(9 - i)] * 6) for i, row in enumerate(cases)]
    return {"cases": cases}, reference, candidate


def test_only_development_originals_set_the_joint_point(paired):
    plan, reference, candidate = paired
    locked = fit(*paired)
    assert locked["development_originals"] == 10
    assert locked["points"]["reference_median"] == 8.0
    assert locked["points"]["candidate_median"] == 8.0
    assert locked["points"]["paired_max_median"] > 1.0
    assert set(locked["development_false_flags"]) == set(ARMS)
    assert max(locked["development_false_flags"].values()) <= 1
    for index in (-2, -1):
        reference[index] = _receipt(plan["cases"][index], [1e6] * 6)
        candidate[index] = _receipt(plan["cases"][index], [-1e6] * 6)
    assert fit(*paired) == locked


def test_missing_pair_stays_unassessed_and_top_two_detects_concentrated_response(paired):
    cases = [_case(f"fresh-{family}", group="new", role="confirmation", family=family)
             for family in ("original_videos", "inswapper", "uniface")]
    first = [_receipt(row, values) for row, values in zip(cases, [[8] * 6, [0] * 6, [0] * 6], strict=True)]
    second = [_receipt(row, values) for row, values in zip(cases,
              [[0] * 6, [0, 0, 0, 0, 10, 10], [0, 0, 0, None, None, None]], strict=True)]
    result = evaluate(fit(*paired), paired[0], {"cases": cases}, first, second)
    assert result["negative_flag_groups"]["confirmation"] == dict.fromkeys(ARMS, 0)
    assert result["results"]["confirmation/inswapper"]["candidate_top_two"]["flagged"] == 1
    assert result["results"]["confirmation/inswapper"]["candidate_median"]["below"] == 1
    assert result["results"]["confirmation/uniface"]["paired_max_median"]["unassessed"] == 1


def test_same_file_alone_cannot_substitute_for_matching_native_inputs(paired):
    plan, reference, candidate = deepcopy(paired)
    candidate[0]["samples"][0]["box"] = [0, 1, 3, 4]
    with pytest.raises(ValueError, match="Paired detector inputs differ"):
        fit(plan, reference, candidate)


def test_reused_group_and_incomplete_triplet_fail_closed(paired):
    row = _case("new-file", group=paired[0]["cases"][0]["group"], role="confirmation")
    locked = fit(*paired)
    with pytest.raises(ValueError, match="overlaps"):
        evaluate(locked, paired[0], {"cases": [row]}, [], [])
    row["group"] = "new-group"
    with pytest.raises(ValueError, match="matched original and both swaps"):
        evaluate(locked, paired[0], {"cases": [row]}, [], [])
