import json

import pytest

from scripts.export_audio_identity_v9 import source
from scripts.review_audio_identity_plan_v9 import check_components


def row(identifier, role="fit", group=None, canonical=None):
    return {
        "sha256": identifier,
        "role": role,
        "group": group or identifier,
        "canonical_speaker_sha256": canonical or identifier,
        "connection_fingerprints": [identifier],
    }


def test_public_review_catches_old_namespaces_hiding_shared_parent_speaker():
    with pytest.raises(ValueError, match="crosses roles"):
        check_components(
            [row("real", "fit", canonical="same-parent"), row("clone", "cal", canonical="same-parent")], []
        )


def test_known_same_speaker_cannot_be_two_statistical_groups():
    with pytest.raises(ValueError, match="split across statistical groups"):
        check_components([row("a", canonical="same"), row("b", canonical="same")], [])


def test_unrelated_sources_cannot_be_merged_into_one_declared_group():
    with pytest.raises(ValueError, match="merges unrelated"):
        check_components([row("a", group="wrong"), row("b", group="wrong")], [])


def test_public_source_exports_only_fingerprints_of_identity_and_text():
    original = {
        "id": "private-provider-recording",
        "sha256": "a" * 64,
        "dataset": "openslr/librispeech_asr",
        "speaker": "librispeech:2843",
        "role": "confirmation",
        "group": "safe-group",
        "label": 0,
        "generator": "human",
        "transcript": "these source words must never be copied into public connection fields",
        "canonical_speaker": "librispeech:2843",
        "source_path": "/Users/person/private.audio",
    }
    result = source(original)
    text = json.dumps(result)
    for value in (original["id"], original["speaker"], original["transcript"], original["source_path"]):
        assert value not in text
    assert len(result["canonical_speaker_sha256"]) == 64
    assert all(len(value) == 64 for value in result["connection_fingerprints"])
