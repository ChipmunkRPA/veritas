import copy
import json

import pytest

from scripts.characterize_mouth_panel_v7 import freeze


@pytest.mark.parametrize("corruption", ["duplicate", "condition", "label"])
def test_current_check_freeze_refuses_incomplete_or_relabelled_denominators(tmp_path, corruption):
    conditions = ("muse-resample", "local-blur-2", "local-blur-4", "muse-ai-aligned", "muse-ai-earlier")
    rows = [
        {"id": f"{i}:{c}", "group": str(i), "condition": c, "label": int(c.startswith("muse-ai-"))}
        for i in range(8)
        for c in conditions
    ]
    altered = copy.deepcopy(rows)
    if corruption == "duplicate":
        altered[-1] = copy.deepcopy(altered[-2])
    elif corruption == "condition":
        altered[-1]["condition"] = "muse-ai-aligned"
    else:
        altered[-1]["label"] = 0
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"items": altered}))
    output = tmp_path / "lock.json"
    with pytest.raises(ValueError):
        freeze(path, output)
    assert not output.exists()
