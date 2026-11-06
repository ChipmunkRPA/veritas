import json

import pytest

from scripts.export_audio_multisource_v8 import public_source


def test_source_export_uses_allowlist_and_hashed_connections():
    row = {
        "id": "provider-private-profile",
        "sha256": "a" * 64,
        "voice_key": "voice-private-profile",
        "source_key": "script-private",
        "transcript": "private words are not public application content",
        "source_path": "/Users/person/private/source.audio",
        "provider_api_token": "not-real-but-never-export",
        "unknown_future_sensitive_field": "secret",
        "dataset": "fixture",
        "role": "confirmation",
    }
    out = public_source(row)
    serialized = json.dumps(out)
    for text in (
        row["id"],
        row["transcript"],
        row["source_path"],
        row["provider_api_token"],
        row["source_key"],
    ):
        assert text not in serialized
    assert len(out["voice_key_sha256"]) == 64
    assert out["text_connection_keys"]


def test_public_source_refuses_presigned_or_nonpublisher_url():
    for url in (
        "https://huggingface.co/a?token=example",
        "https://example.com/private",
        "http://huggingface.co/a",
    ):
        with pytest.raises(ValueError, match="pinned publisher URL"):
            public_source({"id": "a", "url": url})
