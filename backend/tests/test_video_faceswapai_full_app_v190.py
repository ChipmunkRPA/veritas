import subprocess
import sys

from scripts import video_faceswapai_full_app_v190 as trial


def test_worker_import_keeps_app_settings_uninitialized():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; from scripts import video_faceswapai_full_app_v190; "
            "assert 'app.core.config' not in sys.modules",
        ],
        check=True,
    )


def test_classifier_interval_retains_its_native_representatives(monkeypatch):
    from app.evidence import face_interval_binding

    calls = []
    monkeypatch.setattr(
        face_interval_binding, "validate_packet", lambda *args, **kwargs: calls.append(kwargs)
    )
    frames = [
        {
            "interval_role": role,
            "anchor": {"frame_index": i, "pts": trial.PICTURE_PTS[i], "decoded_rgb_sha256": str(i)},
        }
        for role, i in [("before", 71), ("during", 100), ("after", 144)]
    ]
    packet = {
        "semantics": face_interval_binding.METHOD,
        "frames": frames,
        "face_interval": {"samples": {"during": [{"frame_index": 72}, {"frame_index": 143}]}},
    }
    result = trial.review_packet(packet, {}, {"rgb_sha256": list(map(str, range(240)))})
    assert calls == [{"table_pts": tuple(trial.PICTURE_PTS)}]
    assert result["start_seconds"] == 3
    assert result["core_last_frame"] == 143
    assert result["verified_native_rgb_inputs"] == 3
    assert [row["native_frame"] for row in result["representatives"]] == [71, 100, 144]
