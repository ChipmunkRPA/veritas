import shutil
from fractions import Fraction

import pytest

from scripts import video_vcd2_controls_v200 as controls


def test_recipe_keeps_source_presentation_instants_and_honors_tags():
    pts = [12, 2312, 3012, 5012, 6012, 7067]
    clock = {
        "pts": pts,
        "time_base": "1/1000",
        "frame_count": len(pts),
        "ordinary_frame_interval": [2, 4],
        "ordinary_picture_pts": [3012, 6012],
        "packet_durations": [1000] * len(pts),
    }
    video = {
        "width": 720,
        "height": 480,
        "pix_fmt": "yuv422p",
        "time_base": "1/1000",
        "color_range": "pc",
        "color_space": "bt470bg",
        "sample_aspect_ratio": "8:9",
    }
    plan = controls.recipe({"clock": clock}, video)
    assert plan.pts == (0, 2300, 3000, 5000, 6000, 7055)
    assert plan.origin_pts == 12 and plan.last_duration == 1000 and plan.interval == (2, 4)
    assert (plan.source_range, plan.source_matrix, plan.sar) == ("pc", "bt601", Fraction(8, 9))
    assert not any((plan.assumed_range, plan.assumed_matrix, plan.assumed_sar))
    clock["ordinary_frame_interval"] = [1, 4]
    with pytest.raises(ValueError, match="Frozen frame interval differs"):
        controls.recipe({"clock": clock}, video)


def test_expected_edit_keeps_outside_frames_exact():
    base, edit = list("abcdef"), list("uvwxyz")
    assert controls.expected_edit(base, edit, (2, 4)) == list("abwxef")
    with pytest.raises(ValueError, match="Invalid source-aligned replacement"):
        controls.expected_edit(base, edit[1:], (2, 4))


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools required")
@pytest.mark.parametrize("size,tagged", [("64x48", False), ("96x64", True)])
def test_constructs_lossless_rgb_controls_with_irregular_source_timestamps(tmp_path, size, tagged):
    source = tmp_path / "synthetic-fixture.mkv"
    # FFmpeg-native synthetic geometry is only a codec and time-base fixture.
    tag_args = ["-color_range", "pc", "-colorspace", "bt470bg"] if tagged else []
    sar = ",setsar=8/9" if tagged else ""
    controls.ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=s={size}:r=10:d=8",
        "-vf",
        f"settb=1/1000,setpts=N*100+floor(N/3),format=yuv422p{sar}",
        "-c:v",
        "ffv1",
        "-threads",
        "1",
        "-enc_time_base",
        "1/1000",
        "-fps_mode",
        "passthrough",
        *tag_args,
        str(source),
    )
    verified = controls.BASE.media_record(source, size)
    header = controls.BASE.ffprobe(
        source,
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,color_range,color_space,time_base,sample_aspect_ratio",
    )
    plan = controls.recipe(verified, header["streams"][0])
    assert len(set(b - a for a, b in zip(plan.pts, plan.pts[1:]))) > 1
    assert plan.assumed_range == plan.assumed_matrix == (not tagged)
    assert plan.sar == (Fraction(8, 9) if tagged else Fraction(1))
    result = controls.construct(source, tmp_path, plan)
    assert tuple(result["cases"]) == controls.ROLES
    assert not result["cases"]["unchanged"]["changed_frames"]
    for role in controls.ROLES[1:]:
        changed = result["cases"][role]["changed_frames"]
        assert changed and all(plan.interval[0] <= row["index"] < plan.interval[1] for row in changed)
        assert all(row["rgb_mae"] > 0 and row["pts"] == plan.pts[row["index"]] for row in changed)
    assert result["lossy_codec_changed_indices"]


def test_inventory_preserves_incomplete_attempt_and_explicitly_unavailable_slot(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(controls.source, "TRIAL", root)
    failed = root / "slot-01-attempt-01"
    failed.mkdir()
    qualified = root / "slot-01-attempt-02"
    qualified.mkdir()
    second = root / "slot-02-attempt-01"
    second.mkdir()
    file = qualified / "source.mkv"
    file.write_bytes(b"fixture")
    path = "benchmarking/mkv/TH/example.mkv"
    sha256 = controls.sha(file)
    controls.write(
        qualified / "acquisition.json",
        {
            "source_url": controls.BASE.AZURE + path,
            "bytes": file.stat().st_size,
            "sha256": sha256,
        },
    )
    controls.write(
        qualified / "qualification.json",
        {
            "slot": 1,
            "attempt": 2,
            "sha256": sha256,
            "clock": {"frame_count": 306},
        },
    )
    cases = [{"slot": 1, "benchmark_path": path, "resolution": "1280x720"}, {"slot": 2}]
    rows = controls.inventory(cases, {2})
    assert [row["status"] for row in rows] == ["qualified", "unavailable"]
    assert [item["status"] for item in rows[0]["attempts"]] == ["incomplete", "qualified"]
    assert rows[0]["selected"]["attempt"] == qualified.name
    assert rows[1]["selected"] is None and list(second.iterdir()) == []
    assert controls.storage_projection(rows)["retained_rgb_equivalent_bytes"] == 1280 * 720 * 306 * 3 * 3
    with pytest.raises(ValueError, match="explicit unavailability"):
        controls.inventory(cases, set())
