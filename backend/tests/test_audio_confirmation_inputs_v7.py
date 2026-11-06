import pytest

from scripts.prepare_audio_confirmation_inputs_v7 import group_sources, materialize, store


def row(identifier, generator="human", source=None, text="individual words", role="foreground"):
    return {
        "id": identifier,
        "generator": generator,
        "source_group": source or identifier,
        "transcript": text,
        "role": role,
    }


def test_known_voice_and_shared_opening_variants_form_one_uncertainty_group():
    a = row("a", "Chatterbox", text="first distinct prompt")
    b = row("b", "Chatterbox", text="second distinct prompt")
    a["prompt_speaker_id"] = b["prompt_speaker_id"] = "17"
    c = row("c", "ElevenLabs", text="The flow of genetic information in cells from DNA to RNA.")
    d = row("d", "ElevenLabs", text="The flow of genetic information in cells from DNA, um, to RNA.")
    assignments, groups = group_sources([a, b, c, d, row("bg", role="background")])
    assert len(groups) == 2
    assert assignments["a"] == assignments["b"]
    assert assignments["c"] == assignments["d"]
    assert assignments["a"] != assignments["c"]
    assert "bg" not in assignments


def test_duplicate_foreground_identity_is_refused():
    with pytest.raises(ValueError, match="Repeated"):
        group_sources([row("same"), row("same")])


def test_changed_materialized_source_is_never_silently_overwritten(tmp_path):
    result = store(tmp_path, b"original permitted source")
    (tmp_path / result["path"]).write_bytes(b"modified bytes")
    with pytest.raises(ValueError, match="changed"):
        store(tmp_path, b"original permitted source")


def test_fixed_elevenlabs_checksum_is_required_after_acquisition(tmp_path):
    acquired = tmp_path / "fresh-sources/elevenlabs"
    acquired.mkdir(parents=True)
    (acquired / "source.wav").write_bytes(b"changed publisher content")
    plan = {
        "sources": [
            {
                "id": "el",
                "generator": "ElevenLabs",
                "asset": {"name": "source.wav", "sha256": "0" * 64, "bytes": 25},
            }
        ]
    }
    with pytest.raises(ValueError, match="checksum"):
        materialize(plan, tmp_path / "confirmation")
