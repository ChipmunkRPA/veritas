from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from scripts import audio_ami_native_stereo_v291 as intake


def mono(values, rate=16_000):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setparams((1, 2, rate, len(values), "NONE", "not compressed"))
        wav.writeframes(np.array(values, dtype="<i2").tobytes())
    stream.seek(0)
    return wave.open(stream, "rb")


def test_first_party_fixed_sources_and_clock_plan():
    plan = intake.plan()
    assert len(plan["sources"]) == 4
    assert set(plan["sources"]) == {f"{session}.Array1-{index:02d}.wav"
                                    for session in ("ES2004a", "IS1009a") for index in (1, 5)}
    assert all(url.startswith("https://groups.inf.ed.ac.uk/ami/") for url in plan["sources"].values())
    assert plan["window_starts"][0] == 60 * 16_000
    assert plan["window_starts"][-1] == 390 * 16_000
    assert plan["window_samples"] == 48_000 and plan["planned_cases"] == 24


def test_exact_source_indices_are_shared_without_padding_or_alignment(monkeypatch):
    monkeypatch.setattr(intake, "WINDOWS", (0, 2))
    monkeypatch.setattr(intake, "SAMPLES", 2)
    with mono([5, 6, -7, 8]) as left, mono([-20, 10, -30, 1]) as right:
        assert intake.pair_geometry((left, right)) == 4
        assert [channel.tolist() for channel in intake.paired_interval((left, right), 2)] == [[-7, 8], [-30, 1]]
        assert intake.channel_digest(left, 4) == intake.stereo.digest(np.array([5, 6, -7, 8], dtype="<i2").tobytes())
    with mono([1, 2, 3, 4]) as left, mono([1, 2, 3]) as right:
        with pytest.raises(ValueError, match="share one complete"):
            intake.pair_geometry((left, right))
    with mono([1, 2, 3, 4]) as left, mono([1, 2, 3, 4], rate=44100) as right:
        with pytest.raises(ValueError, match="16 kHz"):
            intake.pair_geometry((left, right))


def test_foreign_host_is_refused_before_download_or_write(tmp_path):
    dest = tmp_path / "never-created"
    with pytest.raises(ValueError, match="first-party"):
        intake.receive("https://groups.inf.ed.ac.uk.example.com/ami/audio.wav", dest, 40, 1000)
    assert not dest.exists()
