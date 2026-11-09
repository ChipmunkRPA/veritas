import copy
from collections import Counter

import pytest

from scripts.audio_calibration_roles_v71 import ROLES
from scripts.audio_confidence_mixtures_v82 import make_plan


def panel():
    foregrounds, backgrounds = [], []
    for role in ROLES:
        for label, count in ((0, 56), (1, 60)):
            for number in range(count):
                key = f"{role}-foreground-{label}-{number}"
                foregrounds.append({"raw_sha256": key, "role": role, "label": label,
                                    "study_group": key, "samples": 64000, "status": "ready", "kind": "genuine" if label == 0 else "ai"})
        for kind in ("noise", "speech", "music"):
            for number in range(4):
                key = f"{role}-{kind}-{number}"
                backgrounds.append({"raw_sha256": key, "role": role, "kind": kind,
                                    "study_group": key, "samples": 160000})
    return {"foregrounds": foregrounds, "backgrounds": backgrounds}


def test_plan_retains_complete_roles_and_balances_backgrounds_within_truth():
    jobs = make_plan(panel())
    assert len(jobs) == 464
    for role in ROLES:
        for label, count in ((0, 56), (1, 60)):
            selected = [job for job in jobs if job["role"] == role and job["label"] == label]
            assert len(selected) == count
            for kind in ("noise", "speech", "music"):
                assert set(Counter(job["backgrounds"][kind]["raw_sha256"] for job in selected).values()) == {count // 4}
                for job in selected:
                    bg = job["backgrounds"][kind]
                    assert 0 <= bg["start_sample"] < bg["end_sample"] <= 160000
                    assert bg["end_sample"] - bg["start_sample"] == 64000


def test_fixed_hash_order_is_independent_of_manifest_row_order():
    body = panel()
    permuted = copy.deepcopy(body)
    permuted["foregrounds"].reverse()
    permuted["backgrounds"].reverse()
    assert make_plan(body) == make_plan(permuted)


def test_missing_source_or_short_prefix_cannot_trigger_substitution():
    body = panel()
    body["foregrounds"].pop()
    with pytest.raises(ValueError, match="All qualified"):
        make_plan(body)
    body = panel()
    body["foregrounds"][0]["samples"] = 63999
    with pytest.raises(ValueError, match="four-second"):
        make_plan(body)


def test_background_cannot_cross_roles_or_reuse_foreground_group():
    body = panel()
    body["backgrounds"][0]["role"] = ROLES[1]
    with pytest.raises(ValueError, match="Complete source counts"):
        make_plan(body)
    body = panel()
    group = "shared-ancestry"
    for row in body["foregrounds"]:
        if row["role"] == ROLES[0]:
            row["study_group"] = group
    for row in body["backgrounds"]:
        if row["role"] == ROLES[0]:
            row["study_group"] = group
    with pytest.raises(ValueError, match="separate qualified background"):
        make_plan(body)
