import copy
import hashlib
import json

import pytest

from scripts.audio_calibration_roles_v71 import ROLES
from scripts.audio_combined_confidence_v84 import check_compact, validate_panel


def panel():
    return {"variants": [{"id": f"{role}-{number}", "role": role, "label": number % 2,
                          "samples": 64000, "sample_rate": 16000,
                          "group": role, "sha256": f"bytes-{role}-{number}", "pcm_sha256": f"pcm-{role}-{number}"}
                         for role in ROLES for number in range(1044)]}


def test_whole_panel_requires_complete_roles_clocks_and_alias_separation():
    body = panel()
    assert len(validate_panel(body)) == 4176
    for field, value, message in (("group", ROLES[0], "alias crosses"),
                                  ("sha256", body["variants"][0]["sha256"], "alias crosses"),
                                  ("samples", 63999, "four-second")):
        changed = copy.deepcopy(body)
        changed["variants"][1044][field] = value
        with pytest.raises(ValueError, match=message):
            validate_panel(changed)
    changed = copy.deepcopy(body)
    changed["variants"][0]["role"] = ROLES[1]
    with pytest.raises(ValueError, match="fixed order"):
        validate_panel(changed)


def test_compaction_is_exactly_bound_to_private_full_features_and_missing_scores():
    full = {"id": "authored", "raw": None, "candidates": {"complement": {"p": None}}, "embedding": [0.1, 0.2]}
    line = json.dumps(full) + "\n"
    compact = {key: full[key] for key in ("id", "raw", "candidates")}
    compact["full_record_sha256"] = hashlib.sha256(line.encode()).hexdigest()
    check_compact(compact, line)
    changed = copy.deepcopy(compact)
    changed["raw"] = 0
    with pytest.raises(ValueError, match="Compact scores"):
        check_compact(changed, line)
    with pytest.raises(ValueError, match="complete private"):
        check_compact(compact, line + " ")
