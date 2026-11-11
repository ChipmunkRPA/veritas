import shutil
import subprocess
from fractions import Fraction

import pytest

from scripts import video_faceswapai_matched_v186 as study


def test_case_uses_the_frozen_half_open_source_span():
    reference = [f"input:{index}" for index in range(study.FRAME_COUNT)]
    swap = [f"publisher:{index}" for index in range(study.FRAME_COUNT)]
    expected = study.expected_case(reference, swap)
    assert expected[71:74] == ["input:71", "publisher:72", "publisher:73"]
    assert expected[143:146] == ["publisher:143", "input:144", "input:145"]
    assert study.changed_frames(reference, expected, "swap_tv_bt601") == list(range(72, 144))
    assert not study.changed_frames(reference, study.expected_case(reference), "unchanged")
    expected[144] = swap[144]
    with pytest.raises(ValueError, match="confined"):
        study.changed_frames(reference, expected, "swap_tv_bt601")
    with pytest.raises(ValueError, match="complete native clock"):
        study.expected_case(reference, swap[:-1])


def test_unknown_output_color_keeps_all_four_assumptions():
    assert set(study.SWAP_COLORS.values()) == {
        (range_name, matrix) for range_name in ("pc", "tv") for matrix in ("bt601", "bt709")
    }
    for color in study.SWAP_COLORS.values():
        graph = study.filter_graph(color)
        assert "in_range=pc:out_range=pc:in_color_matrix=bt601" in graph
        assert f"in_range={color[0]}:out_range=pc:in_color_matrix={color[1]}" in graph
        assert "concat=n=3:v=1:a=0,settb=1/12288,setpts=N*512" in graph
    with pytest.raises(ValueError, match="Explicit YUV"):
        study.normalize(("auto", "bt709"))


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Local video codec tools required"
)
def test_real_codec_paths_replay_exact_small_generated_pictures(tmp_path, monkeypatch):
    for key, value in {
        "FRAME_COUNT": 12,
        "WIDTH": 64,
        "HEIGHT": 48,
        "START": 4,
        "STOP": 8,
        "PTS": list(range(0, 12 * 512, 512)),
    }.items():
        monkeypatch.setattr(study, key, value)
    original, swap = (tmp_path / f"{name}.mp4" for name in ("original", "swap"))
    for output, filter_text, color in (
        (
            original,
            "testsrc=size=64x48:rate=24",
            ["-pix_fmt", "yuvj420p", "-color_range", "pc", "-colorspace", "bt470bg"],
        ),
        (swap, "testsrc2=size=64x48:rate=24", ["-pix_fmt", "yuv420p"]),
    ):
        subprocess.run(
            [
                study.tool("ffmpeg"),
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                filter_text,
                "-frames:v",
                "12",
                *color,
                "-c:v",
                "libx264",
                "-bf",
                "0",
                "-threads",
                "1",
                "-video_track_timescale",
                "12288",
                str(output),
            ],
            check=True,
            timeout=20,
        )
    base = study.frame_hashes(original, study.ORIGINAL_COLOR)
    ordinary = tmp_path / "ordinary.mp4"
    study.render(ordinary, original, recompress=True)
    intermediate = study.stream_metadata(ordinary, rendered=True)
    assert (intermediate["pix_fmt"], intermediate["color_range"], intermediate["color_space"]) == (
        "yuv420p",
        "tv",
        "bt470bg",
    )
    arms = [(None, None), (ordinary, study.RECOMPRESSION_COLOR)]
    arms.extend((swap, color) for color in study.SWAP_COLORS.values())
    for index, (replacement, color) in enumerate(arms):
        target = tmp_path / f"result-{index}.mp4"
        study.render(target, original, replacement, color)
        stream = study.stream_metadata(target, rendered=True)
        assert (stream["pix_fmt"], stream["color_range"]) == ("gbrp", "pc")
        hashes = study.frame_hashes(replacement, color) if replacement else None
        expected = study.expected_case(base, hashes)
        table, pictures = study.verify_rgb(target, expected, Fraction(1, 24), (3, 5, 8))
        study.check_clock(table)
        assert set(pictures) == {3, 5, 8}
