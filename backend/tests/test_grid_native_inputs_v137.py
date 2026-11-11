import copy

import pytest

from app.detectors.video.decode import FrameTable
from scripts import grid_native_inputs_v137 as native


def test_full_archive_member_correspondence_rejects_changed_size_crc_or_source():
    member = {"name": "source/utterance.mpg", "bytes": 2000, "crc32": 12345}
    second = {**member, "name": "source/other.mpg"}
    record = {"ordinal": 3, "real_video_members": [member, second]}
    clips = [{"source_ordinal": 3, "member": member}, {"source_ordinal": 3, "member": second}]
    assert native.selected_members(record, clips) == [member, second]
    for key, changed in (("crc32", 999), ("bytes", 2100), ("name", "replacement.mpg")):
        bad = copy.deepcopy(clips)
        bad[0]["member"][key] = changed
        with pytest.raises(ValueError):
            native.selected_members(record, bad)
    with pytest.raises(ValueError):
        native.selected_members({**record, "ordinal": 2}, clips)


def test_source_origins_preserve_observed_av_offset_instead_of_resetting_to_zero():
    table = FrameTable(
        time_base="1/90000",
        pts=(48600, 51600),
        key_frames=(0,),
        pict_types="IP",
        width=360,
        height=288,
        coded_width=360,
        coded_height=288,
    )
    mapping = {"timeline_time_base": "1/16000", "timeline": [{"source_pts": 8464}]}
    assert native.source_origins(table, mapping) == {
        "video_origin_seconds": "27/50",
        "audio_origin_seconds": "529/1000",
    }
    with pytest.raises(ValueError):
        native.source_origins(table, {**mapping, "timeline": []})
    with pytest.raises(ValueError):
        native.source_origins(table, {**mapping, "timeline_time_base": "1/90000"})


def test_only_original_source_specific_decoder_stage_is_dispatched(monkeypatch):
    calls = []
    monkeypatch.setattr(native.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    output = native.TRIAL / "source-03/bind-00/output"
    argv = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", str(output), "0"]
    native.routed_popen(3, argv)
    assert calls[0][3:] == ["scripts.grid_native_inputs_v137", "bind", str(output), "3"]
    with pytest.raises(ValueError):
        native.routed_popen(3, [*argv[:4], "authored", *argv[5:]])
    for ordinal in (True, -1, 16):
        with pytest.raises(ValueError):
            native.run(ordinal)
