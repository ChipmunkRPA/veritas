import copy
import hashlib
import subprocess
import sys

import pytest

from scripts.review_text_format_v99 import compare, equal, mapping, methods, project


def view(text, positions, labels):
    return {
        "text": text,
        "source_positions": positions,
        "labels": labels,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def test_reconstructs_unicode_original_character_correspondence():
    first = view("é x", [4, None, 7], [1, -1, 0])
    alternate = view("é\n x", [4, None, None, 7], [1, -1, -1, 0])
    assert mapping(first, alternate) == {4: 1, 7: 0}


@pytest.mark.parametrize("change", ["content", "position", "label", "token", "bool", "hash", "length"])
def test_changed_correspondence_is_refused(change):
    first = view("a b", [1, None, 3], [1, -1, 0])
    alternate = copy.deepcopy(first)
    if change == "content":
        alternate["text"] = "z b"
    elif change == "position":
        alternate["source_positions"][2] = 4
    elif change == "label":
        alternate["labels"][2] = 1
    elif change == "token":
        alternate = view("ab", [1, 3], [1, 0])
    elif change == "bool":
        alternate["source_positions"][0] = True
    elif change == "hash":
        alternate["text_sha256"] = "0" * 64
    else:
        alternate["labels"].pop()
    with pytest.raises(ValueError):
        mapping(first, alternate)


def test_projection_keeps_uncovered_positions_missing_and_uses_actual_maximum():
    text = view("a b c", [0, None, 2, None, 4], [1, -1, 0, -1, 1])
    proposals = [{"start": 0, "end": 1, "score": 0.7}, {"start": 0, "end": 3, "score": 0.4}]
    assert project(text, proposals, 0.5) == ({0}, {0: 0.7, 2: 0.4})
    result = compare(({0}, {0: 0.7}), ({4}, {4: 0.8}), {0: 1, 2: 0, 4: 1})
    assert result[0]["shared_scored_characters"] == 0
    assert result[0]["added_ai_characters"] == result[0]["lost_ai_characters"] == 1
    assert result[2] is None


@pytest.mark.parametrize(
    "proposal",
    [
        {"start": -1, "end": 1, "score": 0.5},
        {"start": 0, "end": 4, "score": 0.5},
        {"start": 0, "end": 1, "score": True},
        {"start": 0, "end": 1, "score": float("nan")},
    ],
)
def test_invalid_responses_fail(proposal):
    with pytest.raises(ValueError):
        project(view("a", [0], [0]), [proposal], 0.5)


def test_unassessed_is_not_empty_assessed_agreement():
    assert compare(None, (set(), {}), {}) is None
    assert compare((set(), {}), (set(), {}), {}) == (
        {
            "highlight_changed": 0,
            "original_highlighted": 0,
            "view_highlighted": 0,
            "added_ai_characters": 0,
            "lost_ai_characters": 0,
            "added_human_characters": 0,
            "removed_human_characters": 0,
            "shared_scored_characters": 0,
            "only_original_scored_characters": 0,
            "only_view_scored_characters": 0,
            "absolute_score_change_sum": 0,
        },
        1,
        None,
    )


def test_combined_masks_have_no_invented_combined_probability():
    text = view("ab", [0, 1], [1, 0])
    response = {
        "status": "ok",
        "arms": {
            "current": [{"start": 0, "end": 1, "score": 0.8}],
            "interval": [{"start": 1, "end": 2, "score": 0.7}],
        },
    }
    result = methods(text, response, {"current": {"cut": 0.5}, "interval": {"cut": 0.5}})
    assert result["union"] == ({0, 1}, {}) and result["intersection"] == (set(), {})
    missing = methods(
        text, {"status": "unassessed", "arms": None, "reason": "unsupported_native_character_coverage"}, {}
    )
    assert all(value is None for value in missing.values())


def test_aggregate_counts_are_exact_but_float_roundoff_is_permitted():
    equal({"count": 1, "sum": 0.1}, {"count": 1, "sum": 0.1 + 1e-12})
    for wrong in ({"count": True, "sum": 0.1}, {"count": 2, "sum": 0.1}, {"count": 1, "sum": 0.1001}):
        with pytest.raises(ValueError):
            equal({"count": 1, "sum": 0.1}, wrong)


def test_reviewer_import_needs_no_application_or_scientific_runtime():
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(('app','torch','numpy','scipy','sklearn','transformers','scripts.text_format_comparison','scripts.text_format_views','scripts.prepare_text')):
            raise RuntimeError('Forbidden reviewer dependency: '+fullname)
sys.meta_path.insert(0,Block())
import scripts.review_text_format_v99
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
