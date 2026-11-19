"""Recount the scalar-only 29.97 fps ordinary-control supplement."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ("untouched", "second_window", "jump3", "jump4", "zoom3", "blur4", "cutaway4", "hard_cut")
CLOCKS = {"1/30000": 1001, "1/90000": 3003}
SOURCE_CLOCKS = ("native_ntsc", "one_to_one_30_as_ntsc")
STATUSES = ("localized", "no_range", "unassessed")
BASE_FIELDS = {
    "group",
    "role",
    "delivery_time_base",
    "camera_clock",
    "physical_pairs",
    "physical_target_support",
    "entered_scoring",
    "author_frames",
}
OBSERVED_FIELDS = {"new_status", "existing_return", "cut_at_authored_edge"}
ALERT_FIELDS = {"output_frames", "output_pts", "independently_replayed_frames"}


def recount(path):
    data = json.loads(path.read_bytes())
    assert data["schema"] == "chipmunk.video-short-ntsc-ordinary-supplement/1"
    rows = data["rows"]
    expected = {(group, role) for group in range(1, 8) for role in ROLES}
    summary = defaultdict(Counter)
    exact_blur = 0
    witnesses = 0
    assessed_groups = set()
    selected_groups = set()
    source_groups = defaultdict(set)
    for row in rows:
        group, role = row["group"], row["role"]
        expected.remove((group, role))
        clock = row["delivery_time_base"]
        source = row["camera_clock"]
        assert type(group) is int and clock in CLOCKS and source in SOURCE_CLOCKS
        source_groups[source].add(group)
        assert clock == tuple(CLOCKS)[(group - 1 + ROLES.index(role)) % 2]
        pairs = row["physical_pairs"]
        assert type(pairs) is int and 0 <= pairs <= 32
        selected = row["entered_scoring"]
        assert type(selected) is bool and (selected or pairs == 0)
        targeted = row["physical_target_support"]
        assert type(targeted) is bool and (not targeted or pairs > 0)
        authored = row["author_frames"]
        if role in ("second_window", "untouched"):
            assert authored is None
        else:
            a, b = authored
            length = {"jump3": 3, "zoom3": 3, "jump4": 4, "blur4": 4, "cutaway4": 4}.get(role)
            assert 0 < a < b <= 120
            assert b - a == length if length else b == 120
        groups = ("all", clock, role, source)
        for bucket in groups:
            summary[bucket]["authored"] += 1
            summary[bucket]["physical"] += bool(pairs)
        if not selected:
            assert set(row) == BASE_FIELDS
            continue
        selected_groups.add(group)
        status = row["new_status"]
        assert status in STATUSES
        is_alert = status == "localized"
        assert set(row) == BASE_FIELDS | OBSERVED_FIELDS | (ALERT_FIELDS if is_alert else set())
        assert type(row["existing_return"]) is bool and type(row["cut_at_authored_edge"]) is bool
        assert not row["cut_at_authored_edge"] or role in ("hard_cut", "cutaway4")
        if status != "unassessed":
            assessed_groups.add(group)
        if is_alert:
            a, b = row["output_frames"]
            assert 0 < a < b < 120 and b - a in (3, 4)
            assert row["output_pts"] == [a * CLOCKS[clock], b * CLOCKS[clock]]
            assert row["independently_replayed_frames"] == 3
            exact_blur += role == "blur4" and authored == [a, b]
            witnesses += 3
        for bucket in groups:
            summary[bucket]["selected"] += 1
            summary[bucket]["assessed"] += status != "unassessed"
            summary[bucket][status] += 1
            summary[bucket]["existing_return"] += row["existing_return"]
            summary[bucket]["authored_cut_detected"] += row["cut_at_authored_edge"]
    assert not expected
    assert len(source_groups["native_ntsc"]) == 1 and len(source_groups["one_to_one_30_as_ntsc"]) == 6
    assert not (source_groups["native_ntsc"] & source_groups["one_to_one_30_as_ntsc"])
    assert data["witnesses"] == {
        "source_picture_mapping_records_checked": 120 * len(rows),
        "native_crop_checks_replayed": witnesses,
    }
    assert summary["all"]["selected"] <= 20
    assert all(summary[role]["selected"] >= 2 for role in ROLES)
    return {
        "scope": "Recount of the released scalar observations",
        "totals": summary["all"],
        "selected_groups": len(selected_groups),
        "assessed_groups": len(assessed_groups),
        "exact_local_blur_alerts": exact_blur,
        "verified_native_crops": witnesses,
        "by_role": {k: summary[k] for k in ROLES},
        "by_delivery_clock": {k: summary[k] for k in CLOCKS},
        "by_input_clock_class": {k: summary[k] for k in SOURCE_CLOCKS},
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: review_video_face_short_ntsc_ordinary.py RECEIPT.json")
    print(json.dumps(recount(Path(sys.argv[1])), indent=2, sort_keys=True))
