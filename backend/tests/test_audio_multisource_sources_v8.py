import json

import pytest

from scripts import acquire_genuine_audio_v8 as genuine
from scripts.materialize_reserved_audio_v8 import spoken_request


def test_provider_delivery_controls_are_separate_from_intended_spoken_text():
    assert spoken_request("[whispering] Please wait. [short pause] I am here.") == "Please wait. I am here."
    assert spoken_request("Plain spoken words.") == "Plain spoken words."


@pytest.mark.parametrize("invalid", ["revision", "speaker", "text", "origin"])
def test_genuine_asset_requires_pinned_revision_row_speaker_and_text(monkeypatch, invalid):
    expected = {"id": "17-20-0000", "speaker_id": 17, "row_index": 10, "text": "KNOWN SOURCE TEXT"}
    row = {
        "id": expected["id"],
        "speaker_id": 17,
        "text": expected["text"],
        "audio": [{"src": "https://datasets-server.huggingface.co/assets/pinnedrevision/audio.wav"}],
    }
    if invalid == "revision":
        row["audio"][0]["src"] = row["audio"][0]["src"].replace("pinnedrevision", "different")
    elif invalid == "speaker":
        row["speaker_id"] = 18
    elif invalid == "text":
        row["text"] = "DIFFERENT TEXT"
    elif invalid == "origin":
        row["audio"][0]["src"] = "https://example.org/pinnedrevision/audio.wav"
    monkeypatch.setattr(genuine, "request", lambda *_args: json.dumps({"rows": [{"row": row}]}).encode())
    with pytest.raises(ValueError):
        genuine.source_asset(expected, "pinnedrevision")
