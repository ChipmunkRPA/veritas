from __future__ import annotations

import io
import shutil
import wave
import zipfile

import pytest

from scripts import audio_native_stereo_preflight_v274 as preflight


def wave_payload(samples, rate=16_000):
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setparams((1, 2, rate, len(samples) // 2, "NONE", "not compressed"))
        wav.writeframes(samples)
    return out.getvalue()


def test_tape_roles_and_window_assignment_are_fixed():
    plan = preflight.plan()
    assert plan["roles"] == {
        "development": {"washing_room": "DWASHING_16k.zip", "river": "NRIVER_16k.zip"},
        "holdout": {"kitchen": "DKITCHEN_16k.zip", "traffic": "STRAFFIC_16k.zip"},
    }
    assert len(set(plan["window_starts_samples"])) == 16
    assert plan["window_samples"] == 48_000
    assert preflight.window_assignment(0) == (200 * 16_000, 0)
    assert preflight.window_assignment(15) == (290 * 16_000, 1)
    assert preflight.window_assignment(16) == preflight.window_assignment(0)
    with pytest.raises(ValueError, match="nonnegative"):
        preflight.window_assignment(-1)


def test_archive_member_rejects_duplicate_and_wrong_header(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "TAPE_SAMPLES", 4)
    expected = b"\0\1" * 4
    path = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("first/ch02.wav", wave_payload(expected))
        archive.writestr("second/ch02.wav", wave_payload(expected))
        archive.writestr("first/ch16.wav", wave_payload(expected, 8000))
    with zipfile.ZipFile(path) as archive:
        with pytest.raises(ValueError, match="One bounded"):
            preflight.tape_member(archive, "ch02.wav")
        with pytest.raises(ValueError, match="header"):
            preflight.tape_member(archive, "ch16.wav")


def test_tape_commitment_covers_exact_windows_and_rejects_equal_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "TAPE_SAMPLES", 4)
    monkeypatch.setattr(preflight, "SAMPLES", 1)
    monkeypatch.setattr(preflight, "WINDOWS", (0, 2))
    left, right = b"\0\1\0\2\0\3\0\4", b"\1\0\2\0\3\0\4\0"
    path = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("tape/ch02.wav", wave_payload(left))
        archive.writestr("tape/ch16.wav", wave_payload(right))
    receipt = preflight.tape_readout(path)
    assert receipt["members"]["ch02.wav"] == {
        "pcm_sha256": preflight.digest(left), "windows_sha256": preflight.digest(b"\0\1\0\3"),
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("tape/ch02.wav", wave_payload(left))
        archive.writestr("tape/ch16.wav", wave_payload(left))
    with pytest.raises(ValueError, match="Distinct recorded channels"):
        preflight.tape_readout(path)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required")
def test_authored_fixtures_match_real_production_downmix_and_demux(tmp_path):
    rows = preflight.authored_checks(tmp_path)
    assert len(rows) == 5
    assert rows["signed_impulses"]["channel_pcm_sha256"] == rows["swapped_channels"]["channel_pcm_sha256"][::-1]
    assert rows["silent_left"]["mixed_pcm_sha256"] == rows["silent_right"]["mixed_pcm_sha256"]
    assert rows["opposite_phase"]["mixed_pcm_sha256"] == preflight.digest(b"\0" * 96_000)
    assert all(row["source_timeline"] == [{"start_sample": 0, "end_sample": 48_000, "source_pts": 0}]
               for row in rows.values())
