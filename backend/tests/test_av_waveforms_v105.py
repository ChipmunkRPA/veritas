import hashlib
import subprocess
from fractions import Fraction

import pytest

from scripts.prepare_av_waveforms_v105 import (
    audio_command,
    check_phase,
    coordinates,
    parse_hash,
    pcm,
    phase_fixture,
    sample_count,
    save_wav,
    values,
)


def test_signed16_encoding_and_sample_frames_preserve_stereo():
    data = pcm([12000, -12000, 32767, -32768])
    assert list(values(data)) == [12000, -12000, 32767, -32768]
    assert sample_count(data, 2) == 2 and sample_count(data, 1) == 4


@pytest.mark.parametrize(
    "data,channels", [(b"", 1), (b"\x00", 1), (b"\x00\x00", 2), (b"\x00\x00", True), (b"\x00\x00", 3)]
)
def test_empty_partial_or_unsupported_pcm_refused(data, channels):
    with pytest.raises(ValueError):
        sample_count(data, channels)


def test_wav_header_and_samples_are_exact_and_existing_file_is_preserved(tmp_path):
    path = tmp_path / "native.wav"
    data = pcm([1, 2, -4, 8])
    result = save_wav(path, data, 44100, 2)
    assert result["sample_frames"] == 2 and result["channels"] == 2
    assert result["pcm_sha256"] == hashlib.sha256(data).hexdigest()
    original = path.read_bytes()
    with pytest.raises(ValueError):
        save_wav(path, pcm([0, 0]), 16000, 1)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "bad", [b"SHA256=abc", b"SHA256=" + b"a" * 64 + b"\nSHA256=" + b"b" * 64, b"MD5=" + b"a" * 64]
)
def test_native_repeat_hash_must_be_one_complete_sha256(bad):
    with pytest.raises(ValueError):
        parse_hash(bad)


def test_native_repeat_hash_parser_accepts_only_the_exact_hash():
    assert parse_hash(b"SHA256=" + b"a" * 64 + b"\n") == "a" * 64


def test_picture_coordinates_keep_audio_origin_and_fractional_sample_support():
    clock = {
        "source_audio_origin_seconds": "1/100",
        "native_audio_clock_quantization_allowance_seconds": "1/1000",
        "audio_frame_mappings": [
            {"video_frame_index": 3, "source_time_seconds": "27/1000"},
            {"video_frame_index": 4, "source_time_seconds": "1001/30000"},
        ],
    }
    rows = coordinates(clock, 1600)
    assert rows[0]["bracketing_samples"] == [272, 272]
    assert Fraction(rows[1]["sample_position_fraction"]) == (Fraction(1001, 30000) - Fraction(1, 100)) * 16000
    assert rows[1]["bracketing_samples"] == [373, 374]
    assert all(r["status"] == "supported" for r in rows)


def test_outside_decoded_audio_is_unavailable_not_padded_or_clamped():
    clock = {
        "source_audio_origin_seconds": "1/10",
        "native_audio_clock_quantization_allowance_seconds": "1/1000",
        "audio_frame_mappings": [
            {"video_frame_index": 0, "source_time_seconds": "0"},
            {"video_frame_index": 1, "source_time_seconds": "1/5"},
        ],
    }
    rows = coordinates(clock, 1000)
    assert [r["status"] for r in rows] == ["unavailable", "unavailable"]
    assert rows[0]["bracketing_samples"] == [-1600, -1600]
    assert rows[1]["bracketing_samples"] == [1600, 1600]


def test_phase_checker_rejects_truncated_silent_and_moved_responses():
    with pytest.raises(ValueError):
        check_phase(pcm([0] * 100), [13230])
    with pytest.raises(ValueError):
        check_phase(pcm([0] * 32000), [13230])
    moved = [0] * 32000
    moved[4804] = 1000
    with pytest.raises(ValueError):
        check_phase(pcm(moved), [13230])


@pytest.mark.parametrize("channels", [1, 2])
def test_authored_actual_ffmpeg_resampling_and_native_hash(channels, tmp_path):
    # Authored pulses only: no project source, detector, model or network use.
    data, impulses = phase_fixture(channels)
    path = tmp_path / f"phase-{channels}.wav"
    save_wav(path, data, 44100, channels)
    resampled = subprocess.run(
        audio_command(path, rate=16000, channels=1), capture_output=True, check=True, timeout=10
    )
    assert not resampled.stderr
    assert len(check_phase(resampled.stdout, impulses)) == 5
    native = subprocess.run(audio_command(path), capture_output=True, check=True, timeout=10)
    assert native.stdout == data and not native.stderr
    hashed = subprocess.run(
        audio_command(path, hash_output=True), capture_output=True, check=True, timeout=10
    )
    assert parse_hash(hashed.stdout) == hashlib.sha256(data).hexdigest() and not hashed.stderr
