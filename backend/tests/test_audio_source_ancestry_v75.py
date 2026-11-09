from scripts.audio_source_ancestry_v75 import components


def test_different_cloned_voices_in_same_original_dialogue_stay_connected():
    rows = [{"canonical_speaker": "speaker-a", "source_group": "dialogue-one"},
            {"canonical_speaker": "speaker-b", "source_group": "dialogue-one"},
            {"canonical_speaker": "speaker-c", "source_group": "dialogue-two"}]
    assert sorted(sorted(group) for group in components(rows)) == [[0, 1], [2]]


def test_genuine_and_synthesis_with_same_declared_words_share_group():
    text = "A matched source passage with enough distinct words."
    rows = [{"canonical_speaker": "genuine", "transcript": text},
            {"canonical_speaker": "synthetic", "transcript": text}]
    assert components(rows) == [[0, 1]]


def test_byte_and_pcm_aliases_join_different_metadata_identities():
    rows = [{"canonical_speaker": "a", "pcm_sha256": "known"},
            {"canonical_speaker": "b", "first4_pcm_sha256": "known"}]
    assert components(rows) == [[0, 1]]


def test_indirect_source_connections_are_transitive():
    rows = [{"canonical_speaker": "a", "source_group": "one"},
            {"canonical_speaker": "b", "source_group": "one"},
            {"canonical_speaker": "b", "source_group": "two"}]
    assert components(rows) == [[0, 1, 2]]
