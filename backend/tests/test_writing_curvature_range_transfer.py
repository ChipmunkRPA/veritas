from scripts.writing_curvature_range_transfer import calibration_points, ranked_ranges, word_tokens


def test_word_moments_require_complete_unshared_predicted_pieces():
    words = [(0, 3, 0), (4, 7, 0), (8, 12, 0), (13, 16, 1), (17, 19, 1)]
    offsets = [(0, 3), (3, 5), (5, 7), (8, 9), (10, 12), (12, 18), (18, 19)]

    assert word_tokens(words, offsets) == [(), [1, 2], (), (), ()]


def test_local_effect_and_both_flanks_point_to_the_same_native_words():
    words = [(i * 2, i * 2 + 1, i // 5) for i in range(25)]
    values = [[5. if i in {11, 12, 13} else -1., 1.] for i in range(25)]
    choices, by_sentence, assessed = ranked_ranges({"words": words}, {"words": values})

    assert set(choices) == {"absolute", "two_flanks", "agreement"}
    assert all(choice["words"] == [11, 12, 13] for choice in choices.values())
    assert choices == by_sentence[2]
    assert all(assessed.values())


def test_point_selection_uses_the_maximum_per_ordinary_file():
    controls = [{"absolute": {"score": score}} for score in (10., 8., 6., 4., 2., 0., -1., -3.)]
    points = calibration_points(controls)

    assert points["absolute"] == {"10pct": 10., "25pct": 6.}
    assert points["two_flanks"] == points["agreement"] == {"10pct": 0., "25pct": 0.}
