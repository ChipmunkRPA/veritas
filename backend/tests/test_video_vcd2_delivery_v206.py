import hashlib
import shutil
from copy import deepcopy
from fractions import Fraction

import pytest

from scripts import video_vcd2_controls_v200 as controls
from scripts import video_vcd2_delivery_v206 as delivery


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools required")
@pytest.mark.parametrize(
    ("size", "sar", "limit", "status"),
    [
        ("64x48", "1/1", delivery.MAX_UPLOAD_BYTES, "prepared"),
        ("96x64", "8/9", 1, "unsupported_upload_size"),
    ],
)
def test_fixed_delivery_preserves_irregular_clock_and_paired_rgb(
    tmp_path, monkeypatch, size, sar, limit, status
):
    monkeypatch.setattr(delivery, "MAX_UPLOAD_BYTES", limit)
    source = tmp_path / "authored.mkv"
    controls.ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=s={size}:r=10:d=8",
        "-vf",
        f"settb=1/1000,setpts=N*100+floor(N/3),format=yuv422p,setsar={sar}",
        "-c:v",
        "ffv1",
        "-threads",
        "1",
        "-enc_time_base",
        "1/1000",
        "-fps_mode",
        "passthrough",
        str(source),
    )
    measured = controls.BASE.media_record(source, size)
    metadata = controls.BASE.ffprobe(
        source,
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,color_range,color_space,time_base,sample_aspect_ratio",
    )
    raw_plan = controls.recipe(measured, metadata["streams"][0])
    inputs = tmp_path / "controls"
    output = tmp_path / "delivery"
    inputs.mkdir()
    output.mkdir()
    prior = controls.construct(source, inputs, raw_plan)
    sha256 = controls.sha(source)
    entry = {
        "slot": 1,
        "selected": {
            "source_sha256": sha256,
            "rgb_picture_bytes": raw_plan.width * raw_plan.height * len(raw_plan.pts) * 3,
        },
    }
    receipt = {
        "slot": 1,
        "source_sha256": sha256,
        "model_calls": 0,
        "frame_count": len(raw_plan.pts),
        "width": raw_plan.width,
        "height": raw_plan.height,
        "time_base": str(raw_plan.tb),
        "source_origin_pts": raw_plan.origin_pts,
        "last_duration": raw_plan.last_duration,
        "relative_pts": list(raw_plan.pts),
        "frame_interval": list(raw_plan.interval),
        "source_color": {"sample_aspect_ratio": sar},
        **prior,
    }
    plan = delivery.receipt_plan(entry, receipt)
    assert plan.pts == raw_plan.pts and plan.last_duration == raw_plan.last_duration
    assert len(set(b - a for a, b in zip(plan.pts, plan.pts[1:]))) == 2
    assert plan.sar == Fraction(sar)
    report = delivery.construct(inputs, output, receipt, plan)
    assert list(report["cases"]) == list(delivery.ROLES)
    baseline = report["cases"]["unchanged"]
    assert baseline["changed_frames"] == []
    assert len(baseline["delivery_encode_changed_frames"]) == len(plan.pts)
    for role, case in report["cases"].items():
        assert case["status"] == status
        assert controls.sha(inputs / case["file"]) == prior["cases"][role]["sha256"] == case["input_sha256"]
        assert controls.sha(output / case["file"]) == case["sha256"]
        table = controls.verify_clock(output / case["file"], plan, edited=True)
        assert table.key_frames == tuple(range(len(plan.pts))) and set(table.pict_types) == {"I"}
        pair = delivery.diff_indices(baseline["rgb_sha256"], case["rgb_sha256"])
        assert pair == [row["index"] for row in case["changed_frames"]]
        assert role == "unchanged" or pair
        assert all(plan.interval[0] <= index < plan.interval[1] for index in pair)
        assert all(
            row["rgb_mae"] > 0 and row["pts"] == plan.pts[row["index"]]
            for row in case["delivery_encode_changed_frames"]
        )
        assert sorted(pair + case["suppressed_pair_change_indices"]) == [
            row["index"] for row in prior["cases"][role]["changed_frames"]
        ]
    assert report["lossy_codec_changed_indices"]
    assert (
        sorted(report["lossy_codec_changed_indices"] + report["suppressed_lossy_codec_change_indices"])
        == (prior["lossy_codec_changed_indices"])
    )
    damaged = deepcopy(receipt)
    damaged_chroma = damaged["cases"]["chroma_conversion_only"]
    damaged_chroma["rgb_sha256"][0] = "0" * 64
    damaged_chroma["changed_frames"].insert(0, {"index": 0, "pts": 0, "rgb_mae": 1})
    with pytest.raises(ValueError, match="paired ordinary changes"):
        delivery.receipt_plan(entry, damaged)


def test_inventory_preserves_attempts_and_unavailable_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(controls, "TRIAL", tmp_path)
    raw = hashlib.sha256(b"authored").hexdigest()
    cases = {role: {"sha256": raw, "bytes": 8} for role in delivery.ROLES}
    attempts = ("slot-01-attempt-01", "slot-01-attempt-02", "slot-02-attempt-01")
    for name in attempts:
        (tmp_path / name).mkdir()
    complete = tmp_path / attempts[1]
    (complete / "qualification.json").touch()
    monkeypatch.setattr(
        delivery, "input_selection", lambda entry, folder: {"attempt": folder.name, "cases": cases}
    )
    entries = [{"slot": slot, "status": "qualified"} for slot in (1, 2)]
    entries.append({"slot": 3, "status": "unavailable"})
    waiting = delivery.inventory(entries, set())
    assert [row["status"] for row in waiting] == ["prepared", "preparation_pending", "source_unavailable"]
    assert [row["status"] for row in waiting[0]["attempts"]] == ["incomplete", "prepared"]
    assert waiting[0]["selected"]["attempt"] == complete.name
    assert delivery.inventory(entries, {2})[1]["status"] == "preparation_unavailable"
    with pytest.raises(ValueError, match="completed input"):
        delivery.inventory(entries, {1})
