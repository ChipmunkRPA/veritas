from scripts.writing_attention_attribution import token_selection, winner


def test_ablation_covers_the_entire_unicode_sentence_without_touching_other_words():
    text = "X. Café déjà. Y."
    offsets = [(0, 0), (0, 2), (2, 7), (7, 12), (12, 13), (13, 16), (0, 0)]
    special = [1, 0, 0, 0, 0, 0, 1]

    assert token_selection(text, offsets, special, (3, 13)) == [2, 3, 4]
    assert token_selection(text, [*offsets[:3], (8, 11), *offsets[4:]], special, (3, 13)) == []
    assert token_selection(text, [*offsets[:4], (12, 15), *offsets[5:]], special, (3, 13)) == []


def test_agreement_uses_the_same_sentence_and_withholds_ties_and_missing_views():
    refs = {(key, index): dict(enumerate(range(8))) for key in ("isolated", "ablation") for index in (1, 2, 3)}
    row = {"sha256": "one"}
    disagreeing = {"one": {"isolated": [8, 7, 0], "ablation": [0, 7, 8]}}
    tied = {"one": {"isolated": [7, 7, 0], "ablation": [8, 8, 7]}}
    incomplete = {"one": {"isolated": [8, 5, 0], "ablation": []}}

    assert winner(row, disagreeing, refs) == {"isolated": (8888, 1), "ablation": (8888, 3), "both": (7777, 2)}
    assert winner(row, tied, refs) == {"isolated": (0, None), "ablation": (0, None), "both": (0, None)}
    assert winner(row, incomplete, refs) == {"isolated": (8888, 1), "ablation": (0, None), "both": (0, None)}
