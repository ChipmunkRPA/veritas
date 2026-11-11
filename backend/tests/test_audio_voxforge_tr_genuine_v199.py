import io
import subprocess
import wave

import numpy as np
import pytest

from scripts import audio_voxforge_tr_genuine_v199 as lab


def indexed(names):
    return "\n".join(f'<a href="{name}">{name}</a>' for name in names).encode()


def pulse_pcm(rate=lab.RATE):
    time = np.arange(7 * rate) / rate
    on = ((time >= 0.8) & (time < 3.84)) | ((time >= 4.7) & (time < 5.3))
    return (0.2 * np.sin(2 * np.pi * 320 * time) * on).astype(np.float32)


def waveform(values, rate=lab.RATE, channels=1):
    memory = io.BytesIO()
    with wave.open(memory, "wb") as wav:
        wav.setnchannels(channels)
        wav.setframerate(rate)
        wav.setsampwidth(2)
        wav.writeframes(np.rint(np.repeat(values, channels) * 32768).astype("<i2").tobytes())
    return memory.getvalue()


def test_source_groups_fix_two_sessions_and_exclude_seen_or_anonymous():
    aliases = [f"contributor{i}" for i in range(30)] + ["anonymous0", "guest4"]
    archives = [f"{alias}-2026010{index + 1}-abc.tgz" for alias in aliases for index in range(3)]
    exclude = {"contributor1", "contributor2"}
    roster, available = lab.source_plan(indexed(archives), exclude)
    assert lab.source_plan(indexed(reversed(archives)), exclude) == (roster, available)
    assert available == 28
    assert len(roster) == len({row["contributor"] for row in roster}) == 24
    assert all(row["contributor"] not in exclude and len(row["sessions"]) == 2 for row in roster)
    with pytest.raises(ValueError, match="repeated session URLs"):
        lab.source_plan(indexed([*archives, archives[0]]), exclude)


def test_natural_short_keeps_all_samples_and_long_crop_has_quiet_boundaries():
    pcm = pulse_pcm()
    short = pcm[lab.RATE // 2 : 4 * lab.RATE + 17]
    bounds, density = lab.select_crop(short)
    assert bounds == (0, len(short))
    assert density["selection"] == "natural_short_full"
    assert density["active_frame_share"] >= 0.6
    bounds, density = lab.select_crop(pcm)
    start, end = bounds
    assert 0 < start < end < len(pcm)
    assert end - start == 56_000
    assert density["selection"] == "pause_bounded_crop"
    assert density["active_frame_share"] >= 0.6
    assert not np.any(pcm[start - 640 : start + 640])
    assert not np.any(pcm[end - 640 : end + 640])
    assert lab.select_crop(pcm) == (bounds, density)


def test_crop_fails_silence_and_long_speech_without_pause_boundaries():
    with pytest.raises(ValueError, match="activity proxy minimum"):
        lab.select_crop(np.zeros(lab.MIN_SAMPLES, dtype=np.float32))
    time = np.arange(7 * lab.RATE) / lab.RATE
    tone = np.sin(2 * np.pi * 320 * time).astype(np.float32) * 0.2
    with pytest.raises(ValueError, match="No pause-bounded"):
        lab.select_crop(tone)


@pytest.mark.parametrize("extension", [".wav", ".flac"])
def test_production_non_native_rate_crop_round_trips_and_excludes_pcm(extension, tmp_path):
    rate = 22_050
    original = waveform(pulse_pcm(rate), rate=rate)
    if extension == ".flac":
        original = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "wav", "-i", "pipe:0", "-c:a", "flac", "-f", "flac", "pipe:1"],
            input=original,
            capture_output=True,
            check=True,
        ).stdout
        # Streaming FLAC lacks the declared sample count. The publisher archives contain regular files.
        with pytest.raises(ValueError, match="three to twenty seconds"):
            lab.native_header(original, extension)
        path = tmp_path / "fixture.flac"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "wav", "-i", "pipe:0", "-c:a", "flac", str(path)],
            input=waveform(pulse_pcm(rate), rate=rate),
            capture_output=True,
            check=True,
        )
        original = path.read_bytes()
    wav, receipt, keys = lab.production_crop(original, extension, tmp_path, set())
    assert receipt["native_rate"] == rate
    assert receipt["full_samples"] == 7 * lab.RATE
    assert receipt["source_range"]["source_start_pts"] > 0
    assert receipt["wav_range"] == lab.sample_range(0, receipt["samples"])
    assert receipt["pcm_sha256"] == lab.pcm_sha(lab.read_wav(wav))
    assert f"pcm:{receipt['full_pcm_sha256']}" in keys
    assert f"bytes:{receipt['source_sha256']}" in keys
    with pytest.raises(ValueError, match="Known crop or original"):
        lab.production_crop(original, extension, tmp_path, {f"pcm:{receipt['pcm_sha256']}"})


def test_native_stereo_and_truncation_are_unavailable():
    pcm = pulse_pcm()
    with pytest.raises(ValueError, match="Uncompressed mono WAV"):
        lab.native_header(waveform(pcm, channels=2), ".wav")
    with pytest.raises(ValueError, match="Uncompressed mono WAV"):
        lab.native_header(waveform(pcm)[:-1], ".wav")


def test_unavailable_slots_stay_in_aggregate():
    slots = [{"slot": f"genuine-{index:02d}"} for index in range(lab.GROUPS)]
    rows = [{**row, "status": "unavailable", "language": "tr", "label": 0} for row in slots]
    result = lab.aggregation(rows, [{"status": "session_unavailable"}], slots, [])
    assert result["planned_cases"] == lab.GROUPS
    assert result["case_outcomes"] == {"unavailable": lab.GROUPS}
    assert result["qualified_licenses"] == result["selection_outcomes"] == {}
    with pytest.raises(ValueError, match="must be accounted for"):
        lab.aggregation(rows[:-1], [], slots, [])
