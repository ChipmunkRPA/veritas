# SPDX-License-Identifier: MIT
"""Verify the exact bounds of the word anchor and fragment experiment."""

from collections import Counter

from scripts import writing_word_anchor_transfer as study


def test_two_four_word_fragments_use_adjacent_interior_sentences():
    sentences = [0] * 4 + [1] * 6 + [2] * 6 + [3] * 4
    peak_indices = [6, 7, 8, 9, 11, 12, 13, 14]
    words = [(i * 2, i * 2 + 1, sentence) for i, sentence in enumerate(sentences)]
    scores = [5 if i in peak_indices else -20 for i in range(len(words))]

    shapes = study.best_by_shape(words, scores, 4)

    assert shapes["two/4"] == {"indices": peak_indices, "score": 5.0}
    assert "one/4" not in shapes
    assert all(0 < words[index][2] < 3 for shape in shapes.values() for index in shape["indices"])


def test_only_exact_words_are_anchors_within_their_sentence():
    before = [(0, 1, 0, "start"), (2, 3, 1, "the"), (4, 5, 1, "draft"), (6, 7, 1, "closes")]
    after = [(0, 1, 0, "start"), (2, 3, 1, "the"), (4, 5, 1, "revised"), (6, 7, 1, "closes"), (8, 9, 2, "draft")]

    assert study.word_alignment(before, after) == ([0, 1, 3], [0, 1, 3])


def test_mixed_revision_counts_from_serialized_indices():
    record = {"delivery": "edited", "family": "published", "target": [1], "changed": [0, 1],
              "words": [[0, 2, 1], [3, 5, 1], [6, 9, 1]]}
    counts = Counter()
    study.add_count(counts, record, {"indices": [1, 2]})
    assert counts["useful"] == 1
    assert counts["matched_sentences"] == 1

    ordinary = Counter()
    study.add_count(ordinary, record | {"family": "editorial"}, {"indices": [1, 2]})
    assert ordinary["ordinary_flags"] == ordinary["genuine_revision_words_flagged"] == 1
    assert ordinary["useful"] == 0
