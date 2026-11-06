import pytest

from scripts.audio_source_identity_v9 import connected_groups, role_violations, speaker_key


def genuine(identifier, role="fit", **extra):
    return {
        "id": identifier,
        "dataset": "openslr/librispeech_asr",
        "speaker": "librispeech:2843",
        "sha256": identifier,
        "label": 0,
        "generator": "human",
        "role": role,
        "group": identifier,
        **extra,
    }


def clone(identifier, role="cal", **extra):
    return {
        "id": identifier,
        "dataset": "DuplexGen/duplexgen-spoken",
        "prompt_speaker_id": "2843",
        "sha256": identifier,
        "label": 1,
        "generator": "Chatterbox",
        "role": role,
        "group": identifier,
        **extra,
    }


def test_publisher_clone_and_genuine_ids_are_one_speaker_across_dataset_prefixes():
    assert speaker_key(genuine("a")) == speaker_key(clone("b"))
    result = role_violations([genuine("a"), clone("b")])
    assert len(result) == 1 and result[0]["cross_role"]


def test_paired_truth_classes_form_one_connected_group():
    groups = connected_groups([genuine("a"), clone("b", "fit")])
    assert len(groups) == 1 and groups[0]["sources"] == ["a", "b"]
    assert groups[0]["class_counts"] == {"0": 1, "1": 1}


def test_same_voice_cannot_be_foreground_and_background_even_in_one_role():
    result = role_violations([genuine("a"), clone("b", "fit", kind="speech")])
    assert result[0]["foreground_background_shared"]


def test_unknown_clone_id_is_not_treated_as_an_independent_voice():
    with pytest.raises(ValueError, match="numeric LibriSpeech"):
        speaker_key(clone("b", prompt_speaker_id=None))


def test_kokoro_namespace_change_does_not_create_a_new_voice():
    old = {
        "dataset": "ductai199x/dictation-speech",
        "generator": "kokoro-82m",
        "speaker": "kokoro-82m:af_heart",
    }
    new = {
        "dataset": "ductai199x/dictation-speech",
        "generator": "Kokoro-82M",
        "voice_key": "kokoro:af_heart",
    }
    assert speaker_key(old) == speaker_key(new)


def test_shared_numeric_id_in_unrelated_corpus_is_not_invented_as_same_person():
    unrelated = {"dataset": "different/public-corpus", "speaker": "2843"}
    assert speaker_key(unrelated) != speaker_key(genuine("a"))


def test_grouping_preserves_all_sources_and_is_order_independent():
    rows = [genuine("a"), clone("b"), genuine("c", speaker="librispeech:999")]
    assert connected_groups(rows) == connected_groups(list(reversed(rows)))
    assert sum(len(r["sources"]) for r in connected_groups(rows)) == 3
