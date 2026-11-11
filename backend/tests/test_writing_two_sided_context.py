from app.detectors.text.extract import from_string
from app.detectors.text.pipeline import windows
from app.detectors.text.segment import segment
from scripts.writing_two_sided_context import evaluate_regions, features, geometry, native_input


def test_left_and_right_views_share_the_exact_native_focus():
    text = ("The first source sentence describes the instrumentation. "
            "The longer middle sentence contains enough words to qualify for the two sided check. "
            "The following sentence gives the recorded results.\n\n"
            "This separate paragraph has no within paragraph neighbors.")
    rows = geometry(text)
    assert [native_input(text, row.full) for row in rows] == windows(segment(from_string(text)), 3)
    assert [row.pairs is not None for row in rows] == [False, True, False, False]
    left, right, flank_left, flank_right = rows[1].pairs
    focus = rows[1].position
    assert left == (rows[0].position, focus)
    assert right == (focus, rows[2].position)
    assert flank_left == (rows[0].position,)
    assert flank_right == (rows[2].position,)


def test_one_strong_flank_cannot_raise_the_weaker_contrast():
    strong = features(.7, .9, .8, .2, .1)
    weak = features(.7, .9, .8, .2, .9)
    assert weak[:2] == strong[:2]
    assert strong[2] > 0 > weak[2]


def test_non_whitespace_geometry_distinguishes_wrong_location_from_narrow_coverage():
    text = "alpha beta gamma"
    assert evaluate_regions(text, [(0, 5)], [(11, 16)], []) == {
        "flagged": True, "useful": False, "wrong_location": True, "ordinary_edit_overlap": False,
    }
    assert evaluate_regions(text, [(6, 11)], [(6, 10)], [(6, 10)]) == {
        "flagged": True, "useful": True, "wrong_location": False, "ordinary_edit_overlap": True,
    }
