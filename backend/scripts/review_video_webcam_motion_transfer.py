# SPDX-License-Identifier: Apache-2.0
"""Audit source-interval and movement decision arithmetic without model calls."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(folder):
    freeze, result = read(folder / "freeze.json"), read(folder / "aggregate.json")
    rules = freeze["rules"]
    journals = {arm: list(map(json.loads, (folder / f"{arm}.jsonl").open())) for arm in ("camera", "gros", "positive")}
    assert {arm: len(rows) for arm, rows in journals.items()} == result["views"]
    for arm in ("gros", "positive"):
        assert [row["group"] for row in journals[arm]] == list(range(1, len(journals[arm]) + 1))
    assert result["freeze_sha256"] == sha(folder / "freeze.json")
    assert result["journal_sha256"] == {arm: sha(folder / f"{arm}.jsonl") for arm in journals}
    assert result["coverage"] == {arm: dict(Counter(row["status"] for row in rows)) for arm, rows in journals.items()}
    used_clocks, native_swaps, decisions = 0, [], 0
    for arm, rows in journals.items():
        for row in rows:
            if row["incumbent_extent"]:
                a, b = map(Fraction, row["incumbent_extent"])
                assert 0 <= a < b
                used_clocks += 1
                if arm == "positive":
                    c, d = map(Fraction, row["truth"])
                    assert 0 <= c < d
                    used_clocks += 1
                    if 2 * max(0, min(b, d) - max(a, c)) >= max(b, d) - min(a, c):
                        native_swaps.append(row)
            if row["status"] != "measured":
                assert not (row.get("translation_context") or row.get("translation_roll_context"))
                continue
            n = row["scheduled_frames"] - 4
            minimum = math.ceil(n * rules["minimum_middle_frame_fraction"])
            minimum_roll = math.ceil(n * rules["minimum_middle_roll_fraction"])
            assert row["minimum_support"] == minimum and row["minimum_roll_support"] == minimum_roll
            assert row["registered_frames"] == row["scheduled_frames"]
            assert row["scene_max_px_at_320"] <= rules["maximum_scene_displacement_px_at_320"]
            assert row["translation_context"] == (row["translation_support"] >= minimum and
                row["translation_flank_gap_box_widths"] <= rules["maximum_flank_gap_box_widths"])
            assert row["translation_roll_context"] == (row["translation_context"] and row["roll_support"] >= minimum_roll
                and row["roll_flank_gap_degrees"] <= rules["maximum_flank_roll_gap_degrees"])
            frames = [row["representative_inputs"][role] for role in ("before", "during", "after")]
            assert all(len(frame["decoded_rgb_sha256"]) == 64 for frame in frames)
            assert sorted({frame["pts"] for frame in frames}) == [frame["pts"] for frame in frames]
            assert sorted({frame["frame_index"] for frame in frames}) == [frame["frame_index"] for frame in frames]
            decisions += 2
    for arm, expected in result["arms"].items():
        webcam = [row for row in journals["camera"] if row.get(arm)]
        fresh = [row for row in journals["gros"] if row.get(arm)]
        lost = sum(bool(row.get(arm)) for row in native_swaps)
        groups = len({row["group"] for row in webcam})
        surveyed = sum(row["status"] != "incumbent_unassessed" for row in journals["gros"])
        swap_groups = len({row["group"] for row in native_swaps})
        qualifies = len(webcam) >= 4 and groups >= 2 and lost == 0 and len(native_swaps) >= 4 and swap_groups >= 3 and surveyed >= 3
        assert expected == {"camera_contextualized": len(webcam), "camera_groups_contextualized": groups,
                            "gros_contextualized": len(fresh), "useful_swap_ranges_contextualized": lost, "advances": qualifies}
    return {"status": "verified", "rows": sum(len(rows) for rows in journals.values()), "exact_intervals": used_clocks,
            "geometry_arm_decisions": decisions, "journal_sha256": result["journal_sha256"],
            "freeze_sha256": result["freeze_sha256"]}


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    verified = audit(folder)
    with (folder / "independent-audit.json").open("x") as stream:
        json.dump(verified, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(verified, sort_keys=True))
