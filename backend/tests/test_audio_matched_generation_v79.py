from types import SimpleNamespace

import pytest

from scripts.audio_matched_generation_v79 import complete_chunks, source_seed


def token(text, phonemes, whitespace=" "):
    return SimpleNamespace(text=text, phonemes=phonemes, whitespace=whitespace)


class Pipeline:
    def __init__(self, drop=False, duplicate=False):
        self.tokens = [token("First", "a"), token("second", "b", "")]
        self.drop, self.duplicate = drop, duplicate

    def g2p(self, text):
        return "a b", self.tokens

    def en_tokenize(self, tokens):
        selected = tokens[:1] if self.drop else tokens + tokens[:1] if self.duplicate else tokens
        phonemes = "".join(t.phonemes + (" " if t.whitespace else "") for t in selected).strip()
        yield "First second", phonemes, selected


def test_complete_pinned_chunks_preserve_every_preprocessed_source_token():
    assert complete_chunks("First second", Pipeline(), set("ab ")) == [{"phonemes": "a b", "token_count": 2}]


def test_dropped_or_duplicated_source_word_cannot_be_silently_rendered():
    for pipeline in (Pipeline(drop=True), Pipeline(duplicate=True)):
        with pytest.raises(ValueError, match="dropped, duplicated"):
            complete_chunks("First second", pipeline, set("ab "))


def test_unknown_phoneme_is_rejected_before_generation():
    with pytest.raises(ValueError, match="Unknown or overlong"):
        complete_chunks("First second", Pipeline(), set("a "))


def test_g2p_cannot_drop_a_spoken_token_before_chunking():
    pipeline = Pipeline()
    pipeline.tokens[0].phonemes = None
    with pytest.raises(ValueError, match="spoken source token"):
        complete_chunks("First second", pipeline, set("ab "))


def test_source_seed_binds_content_identity_independently_of_assigned_role():
    first = {"canonical_speaker": "stock-profile", "matched_genuine_source_id": "original", "role": "fusion_fit"}
    assert source_seed(first) == source_seed({**first, "role": "confirmation"})
    assert source_seed(first) != source_seed({**first, "matched_genuine_source_id": "another"})
