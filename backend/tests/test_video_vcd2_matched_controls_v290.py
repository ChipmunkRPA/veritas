import json

import pytest

from scripts import video_vcd2_matched_controls_v290 as study


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(study.source, "TRIAL", tmp_path.resolve())
    clock = {
        "pts": [17, 3017, 6017, 9017],
        "frame_count": 4,
        "time_base": "1/1000",
        "ordinary_frame_interval": [1, 2],
        "ordinary_picture_pts": [3017, 6017],
        "packet_durations": [3000, 3000, 3000, 3000],
    }
    stream = {
        "codec_type": "video",
        "codec_name": "ffv1",
        "width": 1280,
        "height": 720,
        "pix_fmt": "yuv420p",
        "time_base": "1/1000",
    }
    cases, slots = [], []
    for slot in study.SLOTS:
        attempt = f"slot-{slot:02d}-attempt-01"
        folder = study.source.TRIAL / attempt
        folder.mkdir()
        (folder / "acquisition.json").write_text(json.dumps({"bytes": slot * 5}))
        qualified = folder / "qualification.json"
        qualified.write_text(json.dumps({"clock": clock, "header": {"streams": [stream]}}))
        cases.append({"slot": slot, "resolution": "1280x720"})
        slots.append(
            {
                "slot": slot,
                "status": "qualified",
                "attempts": [{"attempt": attempt, "status": "qualified"}],
                "selected": {"attempt": attempt, "qualification_sha256": study.sha(qualified)},
            }
        )
    return cases, slots


def test_every_qualified_source_binds_to_the_bundle_without_media(tmp_path, monkeypatch):
    cases, slots = fixture(tmp_path, monkeypatch)
    resolutions, bindings = study.source_bindings(cases, slots)
    assert resolutions == {"1280x720": 8}
    assert [row[:2] for row in bindings] == [(n * 5, 4) for n in study.SLOTS]
    with pytest.raises(ValueError, match="eight ordered"):
        study.source_bindings(cases[:-1], slots[:-1])
    slots[0]["attempts"].append({"attempt": "slot-01-attempt-02", "status": "incomplete"})
    with pytest.raises(ValueError, match="Exact v287 qualification attempt"):
        study.source_bindings(cases, slots)


def test_changed_declared_window_refuses_preparation(tmp_path, monkeypatch):
    cases, slots = fixture(tmp_path, monkeypatch)
    path = study.source.TRIAL / "slot-04-attempt-01" / "qualification.json"
    record = json.loads(path.read_text())
    record["clock"]["ordinary_frame_interval"] = [1, 3]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Frozen frame interval"):
        study.source_bindings(cases, slots)
