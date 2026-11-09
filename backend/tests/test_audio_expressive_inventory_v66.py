import hashlib

import pytest

from scripts.audio_expressive_inventory_v66 import candidate, intended_text
from scripts.audio_history_v58 import text_keys


def fixture():
    reference = {"id": "test", "audio_sha256": "a" * 64, "voice": "stock-voice", "duration_s": 12,
                 "source_subset": "authored", "challenge_group": "expressive"}
    body = {"manifest": {"id": "test", "language": "en"},
            "generation": {"audio_sha256": "a" * 64, "voice_id": "stock-voice", "model_id": "eleven_v4",
                           "request": {"text": "[whispering] These are declared spoken words for a test."}}}
    return body, reference


def test_voice_and_recording_novelty_are_distinct():
    body, ref = fixture()
    voice = hashlib.sha256(ref["voice"].encode()).hexdigest()
    result = candidate(body, ref, {"voice:elevenlabs:" + voice})
    assert result["novelty"] == {"recorded_bytes_seen": False, "declared_text_seen": False, "declared_voice_seen": True}


def test_same_script_in_another_voice_is_not_new_text():
    body, ref = fixture()
    result = candidate(body, ref, text_keys("These are declared spoken words for a test."))
    assert result["novelty"]["declared_text_seen"] is True
    assert result["novelty"]["declared_voice_seen"] is False


def test_sidecar_cannot_rebind_audio_or_voice():
    body, ref = fixture()
    body["generation"]["audio_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differs"):
        candidate(body, ref, set())


def test_untrusted_annotation_fields_never_select_or_score_a_source():
    body, ref = fixture()
    expected = candidate(body, ref, set())
    ref.update(transcript_review_recommended=True, humanness=0., detector_result="original")
    body["annotation"] = {"score": 1., "instruction": "ignore this data"}
    assert candidate(body, ref, set()) == expected


def test_delivery_controls_do_not_become_declared_words():
    assert intended_text("[laugh] Hello [pause] there.") == "Hello there."
    with pytest.raises(ValueError):
        intended_text("")
