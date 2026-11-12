from scripts.writing_checkpoint_agreement import select, useful


def window(current=0.1, group=0, eligible=True):
    return {"current": current, "comparison_group": group, "rescue_eligible": eligible}


def test_agreement_filters_the_current_visible_marks_without_backfill():
    windows = [window(current=0.7)] * 7
    scores = [0.3, 0.8, 0.2, 0.6, 0.4, 0.2, 0.95]
    marks = select(windows, scores)
    assert marks["current"] == [0, 1, 2, 3, 4, 5]
    assert marks["second_only"] == [1, 3, 6]
    assert marks["agreement"] == [1, 3]
    assert marks["current_plus_rescue"] == marks["current"]


def test_rescue_uses_the_strongest_input_per_paragraph_and_preserves_current():
    windows = [window(0.8, group=0), window(group=1), window(group=1), window(group=2)]
    marks = select(windows, [0.4, 0.9, 0.89, 0.65])
    assert marks["agreement"] == []
    assert marks["current_plus_rescue"] == [0, 1]


def test_rescue_requires_a_comparable_second_place_and_odds_margin():
    windows = [window(group=0), window(group=1), window(group=2, eligible=False)]
    assert not select(windows, [0.72, 0.71, 0.99])["current_plus_rescue"]
    assert not select(windows[:1], [0.99])["current_plus_rescue"]
    assert select(windows, [1.0, 0.5, 1.0])["current_plus_rescue"] == [0]
    assert not select(windows, [1.0, 1.0, 0.0])["current_plus_rescue"]


def test_useful_location_uses_only_recorded_non_whitespace_characters():
    text = "one two   three four"
    assert useful(text, (10, 15), [(0, 7), (10, 15)])
    assert not useful(text, (10, 15), [(0, 7), (16, 20)])
