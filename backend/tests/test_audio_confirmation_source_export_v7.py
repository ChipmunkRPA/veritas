import json

from scripts.export_audio_confirmation_sources_v7 import catalogue


def test_source_export_keeps_publisher_binding_without_private_paths_or_text():
    row = {
        "id": "el:one",
        "generator": "ElevenLabs",
        "role": "foreground",
        "label": 1,
        "transcript": "A full source transcript that must be omitted.",
        "path": "/private/source.audio",
        "sha256": "a" * 64,
        "production_input": {"path": "/private/source.audio"},
        "asset": {"url": "https://huggingface.co/public.wav"},
        "voice_id": "private-unhashed-voice-identifier",
    }
    prepared = {
        "sources": [row],
        "backgrounds": [],
        "assignments": {row["id"]: "connected:one"},
        "groups": [],
        "counts": {},
        "foreground_plan_sha256": "b" * 64,
        "preparation_code_sha256": "c" * 64,
        "limitations": [],
    }
    body = catalogue(prepared)
    serialized = json.dumps(body)
    assert row["transcript"] not in serialized
    assert "/private/" not in serialized
    assert row["voice_id"] not in serialized
    public = body["sources"][0]
    assert public["sha256"] == row["sha256"]
    assert public["connected_group"] == "connected:one"
    assert len(public["known_text_key_sha256"]) == 2
    assert public["url"] == row["asset"]["url"]
    assert body["detector_inference_run"] is False
