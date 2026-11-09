import pytest

from scripts.audio_history_v58 import records, row_keys, text_keys, voice_key


def test_publisher_clone_and_genuine_speaker_are_the_same_known_voice():
    genuine = {"dataset": "LibriSpeech", "id": "x", "speaker": "123", "sha256": "a" * 64}
    clone = {"dataset": "DuplexGen/duplexgen-spoken", "id": "y", "user_prompt_speaker_id": "123", "sha256": "b" * 64}
    assert voice_key(genuine) == voice_key(clone) == "librispeech:123"
    assert row_keys(genuine) & row_keys(clone) == {"voice:librispeech:123"}


def test_stock_voice_aliases_and_unicode_transcripts_are_connected():
    a = {"generator": "Kokoro", "speaker": "AF_HEART"}
    b = {"generator_engine": "Kokoro-82M", "generator_voice": "kokoro:af_heart"}
    assert voice_key(a) == voice_key(b)
    assert text_keys("Ｔｈｅ SAME text, again!") == text_keys("the same text again")


def test_pcm_hash_aliases_and_known_openings_connect_old_schemas():
    a = {"decoded_pcm_sha256": "c" * 64, "transcript": "one two three four five six seven eight nine"}
    b = {"prefix_pcm_sha256": "c" * 64, "transcript": "one two three four five six seven eight ten"}
    assert {"pcm:" + "c" * 64, "prefix8:one two three four five six seven eight"}.issubset(row_keys(a) & row_keys(b))


def test_noise_has_no_assumed_voice_and_unresolved_known_speaker_is_explicit():
    assert voice_key({"kind": "noise", "dataset": "sounds"}) is None
    with pytest.raises(ValueError):
        voice_key({"dataset": "LibriSpeech", "id": "unknown"})
    with pytest.raises(ValueError):
        voice_key({"generator": "Kokoro", "speaker": "unknown"})


def test_nested_source_records_are_not_lost_and_invalid_hashes_are_not_fingerprints():
    row = {"sha256": "d" * 64, "dataset": "test", "id": "one"}
    assert list(records({"sources": [row], "nested": {"original": row}})) == [("/sources/0", row), ("/nested/original", row)]
    assert not row_keys({"sha256": "not-a-hash"})
