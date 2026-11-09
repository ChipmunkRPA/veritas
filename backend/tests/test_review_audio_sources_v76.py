from scripts.review_audio_sources_v76 import grouped


def test_breadth_first_review_follows_context_and_voice_edges_transitively():
    rows = [{"canonical_speaker": "one", "source_group": "a"},
            {"canonical_speaker": "two", "source_group": "a"},
            {"canonical_speaker": "two", "source_group": "b"},
            {"canonical_speaker": "three", "source_group": "c"}]
    assert grouped(rows) == [[0, 1, 2], [3]]


def test_review_recognizes_pcm_aliases_across_different_record_schemas():
    rows = [{"pcm_sha256": "same", "transcript": "distinct original words"},
            {"first4_pcm_sha256": "same", "transcript": "another source passage"}]
    assert grouped(rows) == [[0, 1]]
