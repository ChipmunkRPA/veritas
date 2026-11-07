"""Evidence position boundaries and character-level provenance, no classifier."""
from types import SimpleNamespace

import pytest

from scripts.prepare_writing_focus_v3 import focus_record, project_labels


def document(text, ranges):
    return SimpleNamespace(text=text, source_map=[{"source_ranges": ranges}])


def test_unicode_offsets_and_nonwhitespace_authorship():
    doc = document("A😀 B", [(0, 4, 0, 4)])
    labels = project_labels(doc, ["User", "Bot", "User", "Bot"])
    focus = focus_record(doc, labels, 0, 4, "A😀 B. Context.")
    assert focus["focus"] == "A😀 B"
    assert focus["human_characters"] == 1 and focus["ai_characters"] == 2
    assert focus["focus_ai_fraction"] == 2 / 3
    assert focus["offset_unit"] == "Unicode code points, end exclusive"
    assert focus["focus_sha256"] != focus["context_sha256"]


def test_whitespace_normalization_preserves_only_supported_projection():
    doc = document("A\nB", [(0, 1, 0, 1), (1, 2, 1, 3), (2, 3, 3, 4)])
    assert project_labels(doc, ["User", "User", "Bot", "Bot"]) == {0: "User", 2: "Bot"}
    with pytest.raises(ValueError):
        project_labels(document("A!B", doc.source_map[0]["source_ranges"]), ["User"] * 4)


@pytest.mark.parametrize("ranges", [[(0, 2, 0, 2)], [(0, 4, 0, 4)], [(0, 3, 0, 2)],
                                     [(0, 3, 0, 3), (0, 1, 1, 2)]])
def test_missing_outside_or_conflicting_provenance_is_refused(ranges):
    with pytest.raises(ValueError):
        project_labels(document("ABC", ranges), ["User", "Bot", "User"])


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 4), (2, 2), (False, 1), (0., 1)])
def test_invalid_focus_positions_are_not_evidence(start, end):
    with pytest.raises(ValueError):
        focus_record(document("ABC", []), {i: "User" for i in range(3)}, start, end, "ABC")


def test_whitespace_focus_has_no_authorship_fact():
    with pytest.raises(ValueError):
        focus_record(document(" \t", []), {}, 0, 2, "Context")
