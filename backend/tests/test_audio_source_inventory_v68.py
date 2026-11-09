import pytest

from scripts.audio_history_v58 import text_keys
from scripts.audio_source_inventory_v68 import history_seen, parent_voice


def test_genuine_and_clone_share_their_publisher_declared_parent_namespace():
    assert parent_voice(7148) == parent_voice("7148") == "librispeech:7148"
    for value in [None, "unknown", "../7148", 0]:
        with pytest.raises(ValueError):
            parent_voice(value)


def test_voice_reuse_and_text_reuse_are_reported_separately():
    text = "A separately declared test script with enough distinct words."
    assert history_seen("librispeech:1", text, {"voice:librispeech:1"}) == {
        "declared_voice_seen": True, "declared_text_seen": False}
    assert history_seen("librispeech:1", text, text_keys(text)) == {
        "declared_voice_seen": False, "declared_text_seen": True}


def test_missing_declared_content_cannot_be_called_new_text():
    with pytest.raises(ValueError):
        history_seen("librispeech:1", "", set())
