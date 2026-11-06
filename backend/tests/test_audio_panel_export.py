import numpy as np
import pytest
import soundfile as sf

from scripts.export_audio_development_panel import SOURCE_FIELDS, choose, public_reference, wav_metadata
from scripts.rebuild_audio_development_panel import restore_container_timestamp


def test_export_uses_an_allowlist_and_keeps_source_licensing():
    row = {
        "id": "public-case",
        "url": "https://huggingface.co/datasets/example/source/resolve/abc/audio.wav",
        "license": "CC-BY-4.0",
        "path": "/Users/private/data/audio.wav",
        "api_key": "must-not-export",
        "transcript": "Do not publish full transcripts",
        "transcript_sha256": "a" * 64,
    }
    out = choose(row, SOURCE_FIELDS)
    assert set(out) == {"id", "url", "license", "transcript_sha256"}


@pytest.mark.parametrize(
    "url",
    [
        "file:///private/customer.wav",
        "http://huggingface.co/a",
        "https://example.com/a",
        "https://token@huggingface.co/a",
        "https://huggingface.co/a?token=private",
        "https://huggingface.co/a#private",
    ],
)
def test_nonpublic_or_signed_references_are_refused(url):
    with pytest.raises(ValueError, match="unsigned public URL"):
        public_reference(url)


def test_reconstruction_restores_only_container_clock_not_audio_payload(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(path, np.linspace(-0.2, 0.2, 1600, dtype=np.float32), 16000, subtype="FLOAT")
    original = path.read_bytes()
    before = wav_metadata(original)
    restore_container_timestamp(path, 12345)
    after = wav_metadata(path.read_bytes())
    assert after["float_payload_sha256"] == before["float_payload_sha256"]
    assert after["container_peak_timestamp"] == 12345
    peak = original.index(b"PEAK")
    assert original[: peak + 12] == path.read_bytes()[: peak + 12]
    assert original[peak + 16 :] == path.read_bytes()[peak + 16 :]


def test_unrecorded_container_format_is_not_silently_reencoded(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(path, np.ones(1600, np.float32), 16000, subtype="PCM_16")
    with pytest.raises(ValueError, match="float WAV"):
        wav_metadata(path.read_bytes())
