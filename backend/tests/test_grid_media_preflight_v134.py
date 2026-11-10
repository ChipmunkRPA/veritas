from dataclasses import replace

import pytest

from app.detectors.video.decode import FrameTable
from scripts import grid_media_preflight_v134 as preflight


def test_member_selection_is_fixed_excludes_forks_and_prevents_reused_utterances():
    rows = [{"name": f"source/utterance{i:03d}.mpg", "bytes": 1000, "crc32": i} for i in range(12)]
    chosen = preflight.select_members(rows)
    assert preflight.select_members(list(reversed(rows)) + [{"name": "__MACOSX/source/._clip.mpg"}]) == chosen
    used = [row["name"].split("/")[-1][:-4] for row in chosen]
    assert all(row not in chosen for row in preflight.select_members(rows, used))
    with pytest.raises(ValueError):
        preflight.select_members(rows + [{"name": "../escape.mpg"}])
    with pytest.raises(ValueError):
        preflight.select_members(rows[:1])


def test_native_timestamps_are_checked_without_assuming_zero_origin_or_integer_fps():
    table = FrameTable(
        time_base="1/90000",
        pts=(45045, 48048, 51051),
        key_frames=(0,),
        pict_types="IPB",
        width=360,
        height=288,
        coded_width=360,
        coded_height=288,
    )
    assert preflight.video_supported(table)
    for bad in (
        replace(table, pts=(45045, 45045, 51051)),
        replace(table, interpolated=(1,)),
        replace(table, truncated=True),
        replace(table, pts=()),
        replace(table, width=1920),
        replace(table, pts=(0, 900001)),
    ):
        assert not preflight.video_supported(bad)


def test_probe_cannot_dispatch_a_model_stage(monkeypatch):
    calls = []
    monkeypatch.setattr(preflight.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    preflight.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", "out", "0"])
    assert calls[0][3] == "scripts.grid_media_preflight_v134"
    with pytest.raises(ValueError):
        preflight.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored"])
